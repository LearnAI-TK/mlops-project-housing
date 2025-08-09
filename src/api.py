"""
FastAPI application for California Housing Price Prediction
- Loads MLflow-registered model via @alias (e.g., @staging)
- Downloads preprocessing artifacts
- Full logging and health checks
- SQLite logging, Prometheus metrics, retraining trigger, and system monitoring
"""

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import psutil
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from mlflow.tracking import MlflowClient

# Prometheus Client
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)
from pydantic import BaseModel, Field, validator
# ======================================
# Config (env-driven)
# ======================================
LOG_DIR = os.getenv("LOG_DIR", "logs")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# For Docker
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# For local: MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = os.getenv("MODEL_NAME", "CaliforniaHousingRegressor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "preprocessing"), exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(LOG_DIR, "predictions.db"))

# ======================================
# Prometheus metrics
# ======================================

PRED_VALUE = Summary("california_housing_prediction_value", "Predicted value (final)")

PREDICTION_COUNT = Counter(
    "california_housing_predictions_total",
    "Total number of predictions made",
    ["model_version"],
)
PREDICTION_DURATION = Histogram(
    "california_housing_prediction_duration_seconds",
    "Time spent processing predictions",
)
REQUEST_COUNT = Counter(
    "california_housing_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "california_housing_request_duration_seconds",
    "Time spent processing requests",
    ["method", "endpoint"],
)

# System gauges
CPU_PERCENT = Gauge("cpu_percent", "CPU usage percentage")
MEMORY_PERCENT = Gauge("memory_percent", "Memory usage percentage")
DISK_PERCENT = Gauge("disk_percent", "Disk usage percentage")

# GPU gauges (if available)
if GPU_AVAILABLE:
    GPU_LOAD_PERCENT = Gauge("gpu_load_percent", "GPU load %", ["gpu_id"])
    GPU_MEMORY_PERCENT = Gauge("gpu_memory_percent", "GPU memory %", ["gpu_id"])
    GPU_TEMPERATURE = Gauge("gpu_temperature_celsius", "GPU temp C", ["gpu_id"])

# ======================================
# Logging
# ======================================
LOG_FILE = os.path.join(LOG_DIR, "api_predictions.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ======================================
# Optional training funcs
# ======================================
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    from src.model_training import train_and_log_model  # , validate_data

    TRAINING_MODULE_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import training module functions: {e}")
    TRAINING_MODULE_AVAILABLE = False

# Default data paths
DEFAULT_TRAIN_FEATURES_PATH = os.getenv(
    "TRAIN_FEATURES_PATH", "data/processed/train_features.csv"
)
DEFAULT_TRAIN_TARGET_PATH = os.getenv(
    "TRAIN_TARGET_PATH", "data/processed/train_target.csv"
)
DEFAULT_TEST_FEATURES_PATH = os.getenv(
    "TEST_FEATURES_PATH", "data/processed/test_features.csv"
)
DEFAULT_TEST_TARGET_PATH = os.getenv(
    "TEST_TARGET_PATH", "data/processed/test_target.csv"
)

# Resolve relative paths to working dir
if not os.path.isabs(DEFAULT_TRAIN_FEATURES_PATH):
    BASE_DIR = os.getcwd()
    DEFAULT_TRAIN_FEATURES_PATH = os.path.join(BASE_DIR, DEFAULT_TRAIN_FEATURES_PATH)
    DEFAULT_TRAIN_TARGET_PATH = os.path.join(BASE_DIR, DEFAULT_TRAIN_TARGET_PATH)
    DEFAULT_TEST_FEATURES_PATH = os.path.join(BASE_DIR, DEFAULT_TEST_FEATURES_PATH)
    DEFAULT_TEST_TARGET_PATH = os.path.join(BASE_DIR, DEFAULT_TEST_TARGET_PATH)

# ======================================
# Globals
# ======================================
ml_model = None
scaler = None
feature_transformer = None
feature_names = None
target_transformer = None
model_version_str = "unknown"


# ======================================
# SQLite init
# ======================================
def init_db():
    """Initialize SQLite database for prediction logs."""
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL, -- JSON string
                    prediction_raw REAL NOT NULL,
                    prediction_final REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    processing_time REAL,
                    client_ip TEXT
                )
                """
            )
            # Non-destructive schema evolution
            for col, typ in [("processing_time", "REAL"), ("client_ip", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        logger.info(f"Initialized SQLite database at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise


# ======================================
# Metrics helpers
# ======================================
def get_system_metrics():
    """Get CPU/memory/disk metrics via psutil."""
    metrics = {}
    try:
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        try:
            load_avg = psutil.getloadavg()
            (
                metrics["cpu_load_1min"],
                metrics["cpu_load_5min"],
                metrics["cpu_load_15min"],
            ) = load_avg
        except AttributeError:
            pass
        mem = psutil.virtual_memory()
        metrics["memory_total_bytes"] = mem.total
        metrics["memory_available_bytes"] = mem.available
        metrics["memory_used_bytes"] = mem.used
        metrics["memory_percent"] = mem.percent
        try:
            disk = psutil.disk_usage("/")
        except Exception:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            disk = psutil.disk_usage(script_dir)
        metrics["disk_total_bytes"] = disk.total
        metrics["disk_used_bytes"] = disk.used
        metrics["disk_free_bytes"] = disk.free
        metrics["disk_percent"] = disk.percent
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        metrics["system_metrics_error"] = str(e)
    return metrics


def get_gpu_metrics():
    """Get GPU metrics via GPUtil."""
    gpu_metrics = {}
    if not GPU_AVAILABLE:
        gpu_metrics["gpu_available"] = False
        return gpu_metrics
    try:
        gpus = GPUtil.getGPUs()
        gpu_metrics["gpu_available"] = True
        gpu_metrics["gpus"] = []
        for gpu in gpus:
            gpu_metrics["gpus"].append(
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load_percent": round(gpu.load * 100, 2),
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_percent": (
                        round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2)
                        if gpu.memoryTotal > 0
                        else 0
                    ),
                    "temperature_celsius": gpu.temperature,
                }
            )
    except Exception as e:
        logger.error(f"Error collecting GPU metrics: {e}")
        gpu_metrics["gpu_metrics_error"] = str(e)
    return gpu_metrics


def get_prediction_metrics():
    """Aggregate prediction metrics from DB."""
    pred_metrics = {}
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cur.fetchone()[0]
            cur = conn.execute("SELECT AVG(processing_time) FROM predictions")
            avg_processing_time = cur.fetchone()[0]
            cur = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE timestamp > datetime('now', '-1 hour')"
            )
            recent_predictions = cur.fetchone()[0]
            pred_metrics = {
                "total_predictions": total_predictions,
                "model_version": model_version_str,
                "average_processing_time_seconds": avg_processing_time,
                "predictions_last_hour": recent_predictions,
            }
    except Exception as e:
        logger.error(f"Error fetching prediction metrics: {e}")
        pred_metrics["prediction_metrics_error"] = str(e)
    return pred_metrics


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent prediction rows from DB."""
    out: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT request_id, timestamp, input_data, prediction_raw, prediction_final,
                       model_version, processing_time, client_ip
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            for r in rows:
                out.append(
                    {
                        "request_id": r["request_id"],
                        "timestamp": r["timestamp"],
                        "input_data": json.loads(r["input_data"]),
                        "prediction_raw": r["prediction_raw"],
                        "prediction_final": r["prediction_final"],
                        "model_version": r["model_version"],
                        "processing_time": r["processing_time"],
                        "client_ip": r["client_ip"],
                    }
                )
    except Exception as e:
        logger.error(f"Error fetching recent predictions: {e}")
    return out


def get_all_metrics():
    m = {}
    m.update(get_prediction_metrics())
    m.update(get_system_metrics())
    m.update(get_gpu_metrics())
    m["recent_predictions"] = get_recent_predictions(limit=10)
    return m


def update_prometheus_system_metrics():
    try:
        CPU_PERCENT.set(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        MEMORY_PERCENT.set(mem.percent)
        try:
            disk = psutil.disk_usage("/")
        except Exception:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            disk = psutil.disk_usage(script_dir)
        DISK_PERCENT.set(disk.percent)
    except Exception as e:
        logger.error(f"Error updating Prometheus system metrics: {e}")


def update_prometheus_gpu_metrics():
    if not GPU_AVAILABLE:
        return
    try:
        for gpu in GPUtil.getGPUs():
            gid = str(gpu.id)
            GPU_LOAD_PERCENT.labels(gpu_id=gid).set(round(gpu.load * 100, 2))
            mem_pct = (
                round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2)
                if gpu.memoryTotal
                else 0
            )
            GPU_MEMORY_PERCENT.labels(gpu_id=gid).set(mem_pct)
            GPU_TEMPERATURE.labels(gpu_id=gid).set(gpu.temperature)
    except Exception as e:
        logger.error(f"Error updating Prometheus GPU metrics: {e}")


# --- helper: efficient tail of a file ---
def _tail_lines(path: str, max_lines: int):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=max_lines))
    except FileNotFoundError:
        return None


# --- helper: best-effort JSON extraction from a log line ---
_json_obj_re = re.compile(r"\{.*\}")


def _extract_json(line: str):
    try:
        # support lines like: 'PRED|{"a":1,"b":2}' or pure JSON lines
        m = _json_obj_re.search(line)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None


# ======================================
# Lifespan (init model + artifacts)
# ======================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, scaler, feature_transformer, feature_names, target_transformer, model_version_str, GPU_AVAILABLE
    try:
        init_db()

        if not GPU_AVAILABLE:
            logger.info("GPUtil not found. GPU metrics will not be available.")
        else:
            logger.info("GPUtil found. GPU metrics will be available.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        logger.info(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")

        # Resolve model version from alias
        try:
            logger.info(f"🔍 Looking up model '{MODEL_NAME}' alias='{MODEL_ALIAS}'")
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            if not mv:
                raise Exception(f"No model version for alias '{MODEL_ALIAS}'")
            run_id, version = mv.run_id, mv.version
            model_version_str = f"{MODEL_NAME}_v{version}"
            logger.info(f"🎯 Using version {version} (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Alias lookup failed: {e} → falling back to latest")
            mvs = client.search_model_versions(f"name='{MODEL_NAME}'", max_results=1)
            if not mvs:
                raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")
            mv = mvs[0]
            run_id, version = mv.run_id, mv.version
            model_version_str = f"{MODEL_NAME}_v{version}"
            logger.info(f"🎯 Fallback to version {version} (run_id={run_id})")

        # Download preprocessing artifacts
        local_artifact_dir = os.path.join(MODEL_DIR, "preprocessing")
        artifact_path = "preprocessing"
        logger.info(
            f"📥 Downloading artifacts from run {run_id}, path='{artifact_path}'"
        )
        try:
            client.download_artifacts(
                run_id, artifact_path, dst_path=local_artifact_dir
            )
            logger.info(f"✅ Artifacts downloaded to {local_artifact_dir}")
        except Exception as e:
            logger.error(f"Failed to download '{artifact_path}': {e}")
            alt_path = f"artifacts/{artifact_path}"
            logger.info(f"Retrying with alt path: {alt_path}")
            client.download_artifacts(run_id, alt_path, dst_path=local_artifact_dir)
            logger.info(f"✅ Success using alt path '{alt_path}'")

        # Load preprocessing
        def load_joblib_safe(filename: str, description: str):
            path = os.path.join(local_artifact_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{description} not found: {path}")
            return joblib.load(path)

        scaler = load_joblib_safe("scaler.pkl", "Scaler")
        feature_transformer = load_joblib_safe(
            "power_transformer.pkl", "Power Transformer"
        )
        feature_names = load_joblib_safe("feature_names.pkl", "Feature Names")
        target_transformer_path = os.path.join(
            local_artifact_dir, "target_transformer.pkl"
        )
        target_transformer = (
            joblib.load(target_transformer_path)
            if os.path.exists(target_transformer_path)
            else None
        )
        logger.info("✅ Preprocessing components loaded.")

        # Load model via alias
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info(f"📥 Loading model: {model_uri}")
        ml_model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ Model loaded.")
    except Exception as e:
        logger.critical(f"❌ API init failed: {e}", exc_info=True)
        ml_model = None
        scaler = feature_transformer = feature_names = target_transformer = None
    yield
    logger.info("🛑 API shutting down.")


# ======================================
# FastAPI app
# ======================================
app = FastAPI(
    title="California Housing Price Predictor API",
    description=(
        "Predict housing prices with a full preprocessing pipeline. "
        "Uses MLflow @alias. Logging, metrics, retraining."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ======================================
# Middleware (logs + Prometheus)
# ======================================
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "unknown")
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        status = 500
        raise
    finally:
        dur = time.time() - start
        logger.info(
            f"REQ: {request.method} {request.url.path} | Status: {status} | "
            f"Duration: {dur:.4f}s | IP: {client_ip} | UA: {ua}"
        )
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status=str(status)
        ).inc()
        REQUEST_DURATION.labels(
            method=request.method, endpoint=request.url.path
        ).observe(dur)


# ======================================
# Schemas
# ======================================
class PredictionRequest(BaseModel):
    MedInc: float = Field(
        ..., description="Median income in block group", example=3.87, gt=0
    )
    HouseAge: float = Field(
        ..., description="Median house age in block group", example=25.0, ge=0
    )
    AveRooms: float = Field(
        ..., description="Average rooms per household", example=5.0, gt=0
    )
    AveBedrms: float = Field(
        ..., description="Average bedrooms per household", example=1.0, gt=0
    )
    Population: float = Field(
        ..., description="Block group population", example=1200.0, ge=0
    )
    AveOccup: float = Field(
        ..., description="Average household members", example=2.5, gt=0
    )
    Latitude: float = Field(..., description="Latitude", example=34.0, ge=32, le=42)
    Longitude: float = Field(
        ..., description="Longitude", example=-118.0, ge=-124, le=-114
    )

    @validator("AveBedrms")
    def bedrooms_less_than_rooms(cls, v, values, **kwargs):
        if "AveRooms" in values and v >= values["AveRooms"]:
            raise ValueError("AveBedrms must be less than AveRooms")
        return v


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_raw: float
    timestamp: str
    model_version: str
    status: str = "success"


class RecentPredictionsResponse(BaseModel):
    predictions: List[Dict[str, Any]]


# ---- helper: robust CSV reader (handles UTF-16/Windows encodings) ----
def read_csv_smart(path, **kwargs):
    import pandas as pd

    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise ValueError(
        f"Failed to read CSV '{path}' using encodings {encodings}. Last error: {last_err}"
    )


# ======================================
# Endpoints
# ======================================
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the California Housing Price Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    status_ok = bool(ml_model)
    health_status = {
        "status": "OK" if status_ok else "LOADING",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "mlflow_connection": {"status": "OK", "tracking_uri": MLFLOW_TRACKING_URI},
            "model": {
                "status": "LOADED" if ml_model else "NOT_LOADED",
                "version": model_version_str,
                "type": type(ml_model).__name__ if ml_model else None,
            },
            "preprocessing": {
                "scaler": "LOADED" if scaler else "MISSING",
                "feature_transformer": "LOADED" if feature_transformer else "MISSING",
                "feature_names": "LOADED" if feature_names else "MISSING",
            },
        },
    }
    return JSONResponse(content=health_status, status_code=200 if status_ok else 503)


@app.get("/model/info")
async def model_info():
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "status": "loading",
                "health_check": "/health",
            },
        )
    try:
        model_details = {
            "status": "loaded",
            "model_type": type(ml_model).__name__,
            "model_version": model_version_str,
            "loaded_at": datetime.now().isoformat(),
            "features": feature_names,
            "preprocessing_components": {
                "scaler": bool(scaler),
                "feature_transformer": bool(feature_transformer),
                "target_transformer": bool(target_transformer),
            },
        }
        if hasattr(ml_model, "get_params"):
            model_details["parameters"] = ml_model.get_params()
        return model_details
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Could not retrieve model information",
                "exception": str(e),
            },
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, prediction_request: PredictionRequest):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    if ml_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")
    if not all([scaler, feature_transformer, feature_names]):
        raise HTTPException(status_code=500, detail="Preprocessing components missing.")

    try:
        input_data = prediction_request.dict()
        input_df = pd.DataFrame([input_data])

        if set(input_df.columns) != set(feature_names):
            missing = set(feature_names) - set(input_df.columns)
            extra = set(input_df.columns) - set(feature_names)
            msg = []
            if missing:
                msg.append(f"Missing: {sorted(list(missing))}")
            if extra:
                msg.append(f"Extra: {sorted(list(extra))}")
            raise ValueError("Input features do not match training. " + " ".join(msg))

        input_df = input_df[feature_names]
        X = feature_transformer.transform(input_df)
        X = scaler.transform(X)
        prediction_raw = float(ml_model.predict(X)[0])

        prediction_final = prediction_raw
        if target_transformer is not None:
            prediction_final = float(
                target_transformer.inverse_transform([[prediction_raw]])[0][0]
            )

        processing_time = time.time() - start_time

        request_id = str(uuid.uuid4())
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction_raw": prediction_raw,
            "prediction_final": prediction_final,
            "model_version": model_version_str,
            "processing_time": processing_time,
            "client_ip": client_ip,
        }

        logger.info(json.dumps(log_entry, indent=2))

        try:
            with sqlite3.connect(DB_PATH, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT INTO predictions (
                        request_id, timestamp, input_data, prediction_raw, prediction_final,
                        model_version, processing_time, client_ip
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_entry["request_id"],
                        log_entry["timestamp"],
                        json.dumps(log_entry["input"]),
                        log_entry["prediction_raw"],
                        log_entry["prediction_final"],
                        log_entry["model_version"],
                        log_entry["processing_time"],
                        log_entry["client_ip"],
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store prediction log in SQLite: {e}")

        # Prometheus
        PRED_VALUE.observe(prediction_final)
        PREDICTION_DURATION.observe(processing_time)
        PREDICTION_COUNT.labels(model_version=model_version_str).inc()
        update_prometheus_system_metrics()
        update_prometheus_gpu_metrics()

        return PredictionResponse(
            predicted_price=prediction_final
            * 100_000,  # keep if your target is in 100k units
            predicted_price_raw=prediction_final,
            timestamp=datetime.now().isoformat(),
            model_version=model_version_str,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---- Metrics endpoints ----
@app.get("/metrics")  # Prometheus default scrape path
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/json")  # Human-friendly, rich JSON metrics
async def metrics_json():
    return JSONResponse(content=get_all_metrics())


# ---- Recent predictions ----
@app.get("/predictions/recent", response_model=RecentPredictionsResponse)
async def get_recent_predictions_endpoint(limit: int = 50):
    limit = min(limit, 1000)
    return {"predictions": get_recent_predictions(limit=limit)}


# ---- corrected /retrain ----
@app.post("/retrain")
async def retrain():
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Training module is not available.")

    try:
        t0 = time.time()
        logger.info("=== RETRAINING TRIGGERED ===")

        # 1) Load base/train/test sets (encoding-robust)
        base_X = read_csv_smart(DEFAULT_TRAIN_FEATURES_PATH)
        base_y = read_csv_smart(DEFAULT_TRAIN_TARGET_PATH).squeeze()
        X_test = read_csv_smart(DEFAULT_TEST_FEATURES_PATH)
        y_test = read_csv_smart(DEFAULT_TEST_TARGET_PATH).squeeze()

        # 2) Start with base; optionally append new labeled data
        X_train, y_train = base_X.copy(), base_y.copy()
        new_rows = 0
        new_path = os.path.join("data", "new", "new_labeled.csv")

        if os.path.exists(new_path) and os.path.getsize(new_path) > 0:
            try:
                new_df = read_csv_smart(new_path)
                if "target" not in new_df.columns:
                    logger.warning(
                        f"'target' column missing in {new_path}; skipping new data."
                    )
                else:
                    # Check required feature columns
                    missing = set(base_X.columns) - set(new_df.columns)
                    if missing:
                        raise ValueError(
                            f"New data missing feature columns: {sorted(list(missing))}"
                        )

                    # Keep only known features + target; drop rows with NA in any feature/target
                    use_cols = list(base_X.columns) + ["target"]
                    new_df = new_df[use_cols].copy()
                    for c in base_X.columns:
                        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
                    new_df["target"] = pd.to_numeric(new_df["target"], errors="coerce")
                    new_df = new_df.dropna(subset=use_cols)

                    extra_X = new_df[base_X.columns]
                    extra_y = new_df["target"]

                    # Deduplicate vs base (best-effort)
                    try:
                        key_base = base_X.astype(str).agg("|".join, axis=1)
                        key_new = extra_X.astype(str).agg("|".join, axis=1)
                        mask_new = ~key_new.isin(set(key_base))
                        extra_X = extra_X[mask_new]
                        extra_y = extra_y[mask_new]
                    except Exception:
                        pass

                    if len(extra_X):
                        X_train = pd.concat([X_train, extra_X], ignore_index=True)
                        y_train = pd.concat([y_train, extra_y], ignore_index=True)
                        new_rows = int(len(extra_X))

            except Exception as e:
                logger.error(
                    f"Failed to incorporate new labeled data from {new_path}: {e}",
                    exc_info=True,
                )

        # Optional: shuffle
        if len(X_train) > 1:
            idx = X_train.sample(frac=1.0, random_state=42).index
            X_train = X_train.loc[idx].reset_index(drop=True)
            y_train = y_train.loc[idx].reset_index(drop=True)

        logger.info(
            f"Training size: {len(X_train)} (base {len(base_X)} + new {new_rows})"
        )

        # 3) Train candidates
        logger.info("Training Linear Regression (Retrain)...")
        lr_run_id, lr_rmse, lr_r2 = train_and_log_model(
            "LinearRegression", LinearRegression, {}, X_train, y_train, X_test, y_test
        )

        logger.info("Training Decision Tree (Retrain)...")
        dt_params = {"max_depth": 10, "random_state": 42}
        dt_run_id, dt_rmse, dt_r2 = train_and_log_model(
            "DecisionTreeRegressor",
            DecisionTreeRegressor,
            dt_params,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        # 4) Pick winner by RMSE
        if lr_rmse <= dt_rmse:
            best_run_id, best_model_name, best_rmse, best_r2 = (
                lr_run_id,
                "LinearRegression",
                lr_rmse,
                lr_r2,
            )
        else:
            best_run_id, best_model_name, best_rmse, best_r2 = (
                dt_run_id,
                "DecisionTreeRegressor",
                dt_rmse,
                dt_r2,
            )

        logger.info(
            f"🏆 Winner: {best_model_name} (run {best_run_id}) | RMSE={best_rmse:.4f}, R²={best_r2:.4f}"
        )

        # 5) Promote winner to alias 'staging'
        client = MlflowClient()
        version = None
        for _ in range(10):
            try:
                mvs = client.search_model_versions(
                    f"run_id='{best_run_id}' and name='{MODEL_NAME}'"
                )
                if mvs:
                    version = str(mvs[0].version)
                    break
            except Exception:
                pass
            time.sleep(2)

        if not version:
            raise RuntimeError("No model version found for best retrained run.")

        try:
            client.set_registered_model_alias(MODEL_NAME, "staging", version)
            promotion_message = f"Model {MODEL_NAME} v{version} promoted to staging"
            logger.info(f"✅ {promotion_message}")
        except Exception as alias_e:
            logger.warning(f"Alias API failed: {alias_e} → tagging fallback")
            client.set_model_version_tag(MODEL_NAME, version, "alias", "staging")
            promotion_message = f"Tagged version {version} with alias=staging"

        dur = time.time() - t0
        return {
            "status": "success",
            "message": "Model retraining completed",
            "best_model": best_model_name,
            "best_run_id": best_run_id,
            "rmse": best_rmse,
            "r2_score": best_r2,
            "promotion": promotion_message,
            "duration_seconds": round(dur, 2),
            "new_rows_added": new_rows,
            "train_rows_total": int(len(X_train)),
        }

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger retraining: {str(e)}"
        )


# --- endpoint: raw log lines ---
@app.get("/logs", response_class=PlainTextResponse)
def get_logs(limit: int = 100, contains: Optional[str] = None):
    """
    Return the last `limit` lines from the API predictions log.
    Optional: filter lines containing `contains`.
    """
    limit = max(1, min(limit, 1000))  # cap to avoid huge responses
    lines = _tail_lines(LOG_FILE, limit)
    if lines is None:
        return "Log file not found."

    if contains:
        contains_lower = contains.lower()
        lines = [ln for ln in lines if contains_lower in ln.lower()]

    return "".join(lines)


class FeedbackItem(BaseModel):
    features: dict  # same keys as feature_names
    target: float


@app.post("/feedback")
def feedback(item: FeedbackItem):
    os.makedirs("data/new", exist_ok=True)
    path = "data/new/new_labeled.csv"
    df = pd.DataFrame([{**item.features, "target": item.target}])
    # append / create
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)
    return {"status": "ok", "rows_appended": 1}


# ======================================
# Entrypoint
# ======================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
