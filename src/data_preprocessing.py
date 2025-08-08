# src/preprocess.py
"""
Preprocessing for California Housing dataset:
- (Optional) Download raw data if missing
- Train/test split
- Power transform (features [+ optional target])
- Standard scale
- Save processed CSVs and preprocessing artifacts
"""

import os
import logging
import joblib
import yaml
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_CSV = "data/raw/california_housing.csv"
PROC_DIR = "data/processed"
ART_DIR = "models/preprocessing"

def load_params(path="params.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {"test_size": 0.2, "random_state": 42, "transform_target": True}

def fetch_raw(raw_csv=RAW_CSV):
    """Download dataset and write RAW_CSV if missing."""
    if os.path.exists(raw_csv):
        logger.info(f"Raw exists → {raw_csv}")
        return
    os.makedirs(os.path.dirname(raw_csv), exist_ok=True)
    logger.info("Fetching California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target.rename("target")
    df = pd.concat([X, y], axis=1)
    df.to_csv(raw_csv, index=False)
    logger.info(f"💾 Raw saved: {raw_csv}")

def validate_data(X, y, name):
    assert not X.isnull().values.any(), f"NaN in {name} X"
    assert not y.isnull().values.any(), f"NaN in {name} y"
    assert len(X) == len(y), f"len mismatch in {name}"

def preprocess(
    raw_csv=RAW_CSV,
    processed_dir=PROC_DIR,
    art_dir=ART_DIR,
    test_size=0.2,
    random_state=42,
    transform_target=True,
):
    logger.info("🚀 Preprocessing start")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    df = pd.read_csv(raw_csv)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    validate_data(X_train, y_train, "train raw")
    validate_data(X_test, y_test, "test raw")

    # Power transform features
    logger.info("🔧 PowerTransformer(Yeo-Johnson) on features")
    feat_pt = PowerTransformer(method="yeo-johnson")
    X_train_tf = pd.DataFrame(feat_pt.fit_transform(X_train), columns=X.columns)
    X_test_tf  = pd.DataFrame(feat_pt.transform(X_test), columns=X.columns)

    # Scale
    logger.info("📐 StandardScaler on features")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_tf), columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test_tf), columns=X.columns)

    # Optional: transform target
    tgt_pt = None
    if transform_target:
        logger.info("🎯 PowerTransformer on target")
        tgt_pt = PowerTransformer(method="yeo-johnson", standardize=True)
        y_train_out = pd.DataFrame({"target_transformed": tgt_pt.fit_transform(y_train.values.reshape(-1,1)).squeeze()})
        y_test_out  = pd.DataFrame({"target_transformed": tgt_pt.transform(y_test.values.reshape(-1,1)).squeeze()})
    else:
        y_train_out = pd.DataFrame({"target": y_train})
        y_test_out  = pd.DataFrame({"target": y_test})

    # Save processed
    X_train_sc.to_csv(f"{processed_dir}/train_features.csv", index=False)
    X_test_sc.to_csv(f"{processed_dir}/test_features.csv", index=False)
    y_train_out.to_csv(f"{processed_dir}/train_target.csv", index=False)
    y_test_out.to_csv(f"{processed_dir}/test_target.csv", index=False)
    logger.info(f"✅ Processed saved → {processed_dir}")

    # Save artifacts
    joblib.dump(scaler, f"{art_dir}/scaler.pkl")
    joblib.dump(feat_pt, f"{art_dir}/power_transformer.pkl")
    joblib.dump(X.columns.tolist(), f"{art_dir}/feature_names.pkl")
    if tgt_pt is not None:
        joblib.dump(tgt_pt, f"{art_dir}/target_transformer.pkl")
    logger.info(f"📦 Artifacts saved → {art_dir}")

def main():
    p = load_params()
    fetch_raw(RAW_CSV)  # no-op if already present
    preprocess(
        raw_csv=RAW_CSV,
        processed_dir=PROC_DIR,
        art_dir=ART_DIR,
        test_size=p.get("test_size", 0.2),
        random_state=p.get("random_state", 42),
        transform_target=p.get("transform_target", True),
    )
    logger.info("🎉 Preprocessing done")

if __name__ == "__main__":
    main()
