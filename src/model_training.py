"""
Model training and MLflow tracking for California Housing dataset.
Trains multiple models and registers the best one.
Promotes best model to 'staging' using MLflow aliases.
Saves preprocessing artifacts robustly for Docker compatibility.
Logs system resource usage metrics.
- Enhanced to log EDA report to MLflow artifacts.
"""

import shutil
import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib
from ydata_profiling import ProfileReport
sys.path.append(os.path.dirname(__file__))



# Add project root to path (relative to this script's location)
# This helps ensure consistent path resolution regardless of the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root) # Insert at beginning for priority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- System Metrics Collection ---
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.info("GPUtil not found. GPU metrics will not be collected during training.")

def collect_system_metrics(prefix=""):
    """
    Collects system metrics (CPU, Memory, Disk, optionally GPU).
    Prefix is added to metric names (e.g., 'training_start_', 'training_end_').
    Returns a dictionary of metrics suitable for mlflow.log_metrics().
    """
    metrics = {}
    # CPU Metrics
    cpu_percent = psutil.cpu_percent(interval=1) # 1 second sample for better accuracy
    metrics[f'{prefix}cpu_percent'] = cpu_percent
    # Load average (Unix-like systems)
    try:
        load_avg = psutil.getloadavg()
        metrics[f'{prefix}cpu_load_1min'] = load_avg[0]
        metrics[f'{prefix}cpu_load_5min'] = load_avg[1]
        metrics[f'{prefix}cpu_load_15min'] = load_avg[2]
    except AttributeError:
        # Not available on all systems (e.g., Windows)
        pass
    # Memory Metrics
    memory = psutil.virtual_memory()
    # Convert bytes to MB for easier readability in MLflow UI
    metrics[f'{prefix}memory_total_mb'] = memory.total / (1024 * 1024)
    metrics[f'{prefix}memory_available_mb'] = memory.available / (1024 * 1024)
    metrics[f'{prefix}memory_used_mb'] = memory.used / (1024 * 1024)
    metrics[f'{prefix}memory_percent'] = memory.percent
    # Disk Metrics (for the disk where the project root is)
    try:
        disk = psutil.disk_usage(project_root)
        metrics[f'{prefix}disk_total_mb'] = disk.total / (1024 * 1024)
        metrics[f'{prefix}disk_used_mb'] = disk.used / (1024 * 1024)
        metrics[f'{prefix}disk_free_mb'] = disk.free / (1024 * 1024)
        metrics[f'{prefix}disk_percent'] = disk.percent
    except Exception as e:
        logger.warning(f"Could not get disk usage for {project_root}: {e}")
    # GPU Metrics (if available)
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            # Log metrics for the first GPU found, or aggregate if needed
            if gpus:
                gpu = gpus[0] # Log first GPU
                metrics[f'{prefix}gpu_{gpu.id}_load_percent'] = gpu.load * 100
                metrics[f'{prefix}gpu_{gpu.id}_memory_used_mb'] = gpu.memoryUsed
                metrics[f'{prefix}gpu_{gpu.id}_memory_total_mb'] = gpu.memoryTotal
                if gpu.memoryTotal > 0:
                    metrics[f'{prefix}gpu_{gpu.id}_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                metrics[f'{prefix}gpu_{gpu.id}_temperature_celsius'] = gpu.temperature
                # If you have multiple GPUs and want metrics for all:
                # for i, gpu in enumerate(gpus):
                #     metrics[f'{prefix}gpu_{i}_load_percent'] = gpu.load * 100
                #     ... etc
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
    return metrics
# --- End System Metrics Collection ---

# Set MLflow tracking URI and experiment (Make configurable)
# Use environment variable, defaulting to localhost:5000 for local dev
# For Docker Compose, MLFLOW_TRACKING_URI should be set to http://mlflow:5000 via environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("California Housing Regression")
logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

def validate_data(X, y, dataset_name="Dataset"):
    """Validate basic data quality."""
    logger.info(f"Validating {dataset_name} data...")
    assert not X.isnull().any().any(), f"NaN values found in {dataset_name} features"
    assert not y.isnull().any(), f"NaN values found in {dataset_name} target"
    assert X.shape[0] == y.shape[0], f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}"
    assert len(X.columns) > 0, "No features found"
    logger.info(f"{dataset_name} data validated ✅")

def train_and_log_model(model_name, model_class, params, X_train, y_train, X_test, y_test):
    if mlflow.active_run():
        mlflow.end_run()
    try:
        with mlflow.start_run(run_name=model_name) as run:
            logger.info(f"Starting training for model: {model_name} (Run ID: {run.info.run_id})")
            # --- Log System Metrics BEFORE Training ---
            # --- Log EDA Report to MLflow ---

            raw_path = os.path.join(project_root, "data", "raw", "california_housing.csv")
            raw_df = pd.read_csv(raw_path)

            report_dir = os.path.join(project_root, "reports")
            os.makedirs(report_dir, exist_ok=True)

            eda_report_path = os.path.join(report_dir, "eda_ydata.html")

            profile = ProfileReport(raw_df, title="EDA Report", minimal=True)
            profile.to_file(eda_report_path)

            # Upload the whole 'reports' folder -> shows as Artifacts/reports/eda_ydata.html
            mlflow.log_artifact(eda_report_path, artifact_path="model/eda")
            mlflow.log_artifacts(report_dir, artifact_path="reports")
            logger.info(f"EDA report logged to MLflow: {eda_report_path}")


            logger.info("Collecting system metrics before training...")
            pre_training_metrics = collect_system_metrics("training_start_")
            mlflow.log_metrics(pre_training_metrics)
            logger.info(f"Logged pre-training system metrics: {list(pre_training_metrics.keys())}")
            start_train_time = time.time() # For calculating training duration
            # Fit preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            power_transformer = PowerTransformer()
            X_train_transformed = power_transformer.fit_transform(X_train_scaled)
            X_test_transformed = power_transformer.transform(X_test_scaled)
            feature_names = X_train.columns.tolist()
            logger.debug(f"Feature names: {feature_names}")
            # --- Saving Artifacts Locally (Robust path relative to project root) ---
            # Ensure we save artifacts to a known location relative to the project root.
            # This makes it consistent whether run locally or inside Docker container.
            preprocessing_dir = os.path.join(project_root, "models", "preprocessing")
            logger.info(f"Attempting to save preprocessing artifacts to: {preprocessing_dir}")
            os.makedirs(preprocessing_dir, exist_ok=True)
            # Define paths for the artifacts
            scaler_path = os.path.join(preprocessing_dir, "scaler.pkl")
            transformer_path = os.path.join(preprocessing_dir, "power_transformer.pkl")
            features_path = os.path.join(preprocessing_dir, "feature_names.pkl")
            # Save artifacts using the defined paths
            joblib.dump(scaler, scaler_path)
            joblib.dump(power_transformer, transformer_path)
            joblib.dump(feature_names, features_path)
            logger.info(f"✅ Preprocessing artifacts saved locally to: {preprocessing_dir}")
            # --- Verify Artifacts Exist Locally Before Logging ---
            expected_artifacts = {
                "scaler.pkl": scaler_path,
                "power_transformer.pkl": transformer_path,
                "feature_names.pkl": features_path,
            }
            missing_artifacts = []
            for name, path in expected_artifacts.items():
                if not os.path.exists(path):
                    logger.error(f"❌ Expected artifact file NOT found: {path}")
                    missing_artifacts.append(name)
                else:
                    logger.debug(f"✅ Confirmed artifact file exists: {path}")
            if missing_artifacts:
                error_msg = f"Failed to create required local artifact files: {missing_artifacts}"
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)
            else:
                 logger.info("✅ All preprocessing artifacts confirmed locally before MLflow logging.")
            # --- Logging Artifacts to MLflow (Directly into 'preprocessing' path) ---
            logger.info("📥 Logging preprocessing artifacts to MLflow under 'preprocessing' path...")
            logged_count = 0
            for artifact_name, local_file_path in expected_artifacts.items():
                try:
                    # Log each file directly into the 'preprocessing' artifact directory in MLflow
                    mlflow.log_artifact(local_file_path, artifact_path="preprocessing")
                    logger.info(f"  📤 Successfully logged {artifact_name} to MLflow 'preprocessing' path.")
                    logged_count += 1
                except Exception as log_err:
                    logger.error(f"  ❌ Failed to log {artifact_name} to MLflow: {log_err}")
                    # Depending on your policy, you might want to raise an error here
                    # if logging artifacts is critical. For now, we'll log the error.
            if logged_count == 0:
                 error_msg = "Failed to log any preprocessing artifacts to MLflow."
                 logger.error(f"❌ {error_msg}")
                 raise RuntimeError(error_msg)
            elif logged_count < len(expected_artifacts):
                 logger.warning(f"⚠️  Only {logged_count}/{len(expected_artifacts)} artifacts were successfully logged to MLflow.")
            else:
                 logger.info("✅ All preprocessing artifacts successfully logged to MLflow 'preprocessing' path.")
            # Log model metadata
            mlflow.log_params(params)
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("features_count", X_train.shape[1])
            mlflow.log_param("train_samples", X_train.shape[0])
            # Train model
            model = model_class(**params)
            logger.debug("Fitting model...")
            model.fit(X_train_transformed, y_train)
            end_train_time = time.time()
            training_duration = end_train_time - start_train_time
            mlflow.log_metric("training_duration_seconds", training_duration)
            logger.info(f"Model training completed in {training_duration:.2f} seconds.")
            logger.debug("Making predictions...")
            predictions = model.predict(X_test_transformed)
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            # --- Log System Metrics AFTER Training ---
            logger.info("Collecting system metrics after training...")
            post_training_metrics = collect_system_metrics("training_end_")
            # Calculate resource deltas (change during training)
            delta_metrics = {}
            for key, end_val in post_training_metrics.items():
                # Only calculate delta for metrics that likely change and were also logged before
                if key.startswith("training_end_") and key.replace("training_end_", "training_start_") in pre_training_metrics:
                    start_key = key.replace("training_end_", "training_start_")
                    if start_key in pre_training_metrics:
                        try:
                            delta = end_val - pre_training_metrics[start_key]
                            delta_metrics[key.replace("training_end_", "delta_")] = delta
                        except TypeError:
                            # Handle cases where subtraction isn't straightforward (e.g., strings)
                            pass
            # Log all metrics to MLflow
            mlflow.log_metrics({"rmse": rmse, "r2_score": r2, "training_duration_seconds": training_duration})
            mlflow.log_metrics(post_training_metrics) # Log end state metrics
            mlflow.log_metrics(delta_metrics) # Log changes in metrics
            logger.info(f"Logged post-training system metrics and deltas.")
            # Infer signature and example
            signature = infer_signature(X_test, predictions)
            input_example = X_test.iloc[:2]  # Use original DataFrame
            # Log model to MLflow and register
            logger.debug("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="CaliforniaHousingRegressor",
                signature=signature,
                input_example=input_example,
            )
            mlflow.set_tag("run_status", "success")
            logger.info(f"✅ Logged {model_name} with MLflow. Run ID: {run.info.run_id}")
            logger.info(f"📈 Metrics - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            logger.info(f"🔗 View in UI: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
            return run.info.run_id, rmse, r2
    except Exception as e:
        mlflow.set_tag("run_status", "failed")
        logger.error(f"❌ Failed to train/log model {model_name}: {e}", exc_info=True)
        raise e

def _get_model_version_for_run(client: MlflowClient, model_name: str, run_id: str):
    """Return the model version (string) that was created for this run_id."""
    # More robust filter
    filter_string = f"run_id='{run_id}' and name='{model_name}'"
    logger.debug(f"Searching for model versions with filter: {filter_string}")
    mvs = client.search_model_versions(filter_string)
    if mvs:
        version = str(mvs[0].version)
        logger.debug(f"Found model version {version} for run_id {run_id}")
        return version
    logger.debug(f"No model version found for run_id {run_id} and name {model_name}")
    return None

def _promote_with_alias(client: MlflowClient, model_name: str, version: str, alias: str = "staging"):
    """
    Promote model version using alias (preferred). Falls back to tag if alias API not available.
    """
    logger.info(f"Attempting to promote model {model_name} version {version} to alias '{alias}'")
    try:
        client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"✅ Set alias '{alias}' → version {version} for model '{model_name}'.")
        return True
    except Exception as e:
        logger.warning(f"Alias API not available or failed ({e}). Falling back to tag.")
        try:
            client.set_model_version_tag(model_name, version, "alias", alias)
            logger.info(f"Tagged version {version} with alias='{alias}' for model '{model_name}'.")
            return True
        except Exception as e2:
            logger.error(f"Failed to set alias or tag: {e2}")
            return False

def main(
    train_features_path="data/processed/train_features.csv",
    train_target_path="data/processed/train_target.csv",
    test_features_path="data/processed/test_features.csv",
    test_target_path="data/processed/test_target.csv",
    model_name="CaliforniaHousingRegressor"
):
    logger.info("🚀 Starting model training pipeline...")
    logger.info(f"Project root determined to be: {project_root}")
    # --- Log System Metrics for Overall Run ---
    logger.info("Collecting system metrics for the overall training run...")
    overall_start_metrics = collect_system_metrics("pipeline_start_")
    mlflow.log_metrics(overall_start_metrics)
    logger.info(f"Logged overall start system metrics: {list(overall_start_metrics.keys())}")
    pipeline_start_time = time.time()
    # Resolve data paths relative to the project root for consistency
    train_features_path = os.path.join(project_root, train_features_path)
    train_target_path = os.path.join(project_root, train_target_path)
    test_features_path = os.path.join(project_root, test_features_path)
    test_target_path = os.path.join(project_root, test_target_path)
    logger.info("Loading data...")
    # Load data
    logger.debug(f"Loading training features from {train_features_path}")
    X_train = pd.read_csv(train_features_path)
    logger.debug(f"Loading training target from {train_target_path}")
    y_train = pd.read_csv(train_target_path).squeeze()
    logger.debug(f"Loading test features from {test_features_path}")
    X_test = pd.read_csv(test_features_path)
    logger.debug(f"Loading test target from {test_target_path}")
    y_test = pd.read_csv(test_target_path).squeeze()
    validate_data(X_train, y_train, "Training")
    validate_data(X_test, y_test, "Testing")
    # Train models
    logger.info("Training Linear Regression...")
    lr_run_id, lr_rmse, lr_r2 = train_and_log_model(
        "LinearRegression", LinearRegression, {}, X_train, y_train, X_test, y_test
    )
    dt_params = {"max_depth": 10, "random_state": 42}
    logger.info("Training Decision Tree...")
    dt_run_id, dt_rmse, dt_r2 = train_and_log_model(
        "DecisionTreeRegressor", DecisionTreeRegressor, dt_params, X_train, y_train, X_test, y_test
    )
    # Select best model (prioritizing lower RMSE)
    if lr_rmse <= dt_rmse:
        best_run_id, best_model_name, best_rmse, best_r2 = lr_run_id, "LinearRegression", lr_rmse, lr_r2
    else:
        best_run_id, best_model_name, best_rmse, best_r2 = dt_run_id, "DecisionTreeRegressor", dt_rmse, dt_r2
    logger.info(f"🏆 Best model: {best_model_name} (Run ID: {best_run_id})")
    logger.info(f"   RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")
    # Promote to @staging
    logger.info("Initializing MLflow Client for promotion...")
    client = MlflowClient()
    version = None
    for attempt in range(10):
        logger.debug(f"Attempt {attempt+1}/10 to find model version for run {best_run_id}")
        version = _get_model_version_for_run(client, model_name, best_run_id)
        if version:
            logger.debug(f"Model version {version} found.")
            break
        logger.warning(f"Waiting for model version... (attempt {attempt+1}/10)")
        time.sleep(2) # Consider if this wait is necessary or too long
    if not version:
        error_msg = "❌ No model version found for best run."
        logger.error(error_msg)
        # Raising an exception is generally better than sys.exit in library code/functions
        # but sys.exit is okay in a main script if you want to stop the whole process.
        # Let's raise for now to make the error clearer in logs if called as a module.
        # sys.exit(1)
        raise RuntimeError(error_msg)
    success = _promote_with_alias(client, model_name, version, alias="staging")
    if success:
        logger.info(f"✅ Model {model_name} v{version} promoted to alias 'staging'")
    else:
        logger.error(f"❌ Failed to promote model {model_name} v{version}")
        # Depending on requirements, you might want to raise an error here too.
        # raise RuntimeError(f"Failed to promote model {model_name} v{version}")




    # --- Log System Metrics for Overall Run END ---
    pipeline_end_time = time.time()
    overall_duration = pipeline_end_time - pipeline_start_time
    mlflow.log_metric("pipeline_duration_seconds", overall_duration)
    logger.info("Collecting system metrics for the overall training run end...")
    overall_end_metrics = collect_system_metrics("pipeline_end_")
    mlflow.log_metrics(overall_end_metrics)
    logger.info(f"Logged overall end system metrics: {list(overall_end_metrics.keys())}")
    # Calculate and log overall deltas
    overall_delta_metrics = {}
    for key, end_val in overall_end_metrics.items():
        if key.startswith("pipeline_end_") and key.replace("pipeline_end_", "pipeline_start_") in overall_start_metrics:
            start_key = key.replace("pipeline_end_", "pipeline_start_")
            if start_key in overall_start_metrics:
                try:
                    delta = end_val - overall_start_metrics[start_key]
                    overall_delta_metrics[key.replace("pipeline_end_", "pipeline_delta_")] = delta
                except TypeError:
                    pass
    mlflow.log_metrics(overall_delta_metrics)
    logger.info(f"Logged overall system metric deltas.")
    mlflow.log_metric("pipeline_total_duration_seconds", overall_duration)
    # --- End Overall Metrics Logging ---
    logger.info("🎉 Training complete.")

if __name__ == "__main__":
    main()
