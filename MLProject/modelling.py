import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import os
import json

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./BreastCancer_preprocessing")
args = parser.parse_args()

# Load data
X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")
X_test = pd.read_csv(f"{args.data_dir}/X_test.csv")
y_train = pd.read_csv(f"{args.data_dir}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{args.data_dir}/y_test.csv").values.ravel()

# Model Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    metrics = {
        "accuracy_train": accuracy_score(y_train, predictions_train),
        "f1_train": f1_score(y_train, predictions_train, average='weighted'),
        "roc_auc_train": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) if len(set(y_train)) == 2 else roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr'),
        "recall_train": recall_score(y_train, predictions_train, average='weighted'),
        "precision_train": precision_score(y_train, predictions_train, average='weighted'),
        "accuracy_test": accuracy_score(y_test, predictions_test),
        "f1_test": f1_score(y_test, predictions_test, average='weighted'),
        "roc_auc_test": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) == 2 else roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'),
        "recall_test": recall_score(y_test, predictions_test, average='weighted'),
        "precision_test": precision_score(y_test, predictions_test, average='weighted')
    }

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Tambahkan signature dan input_example
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:1]

    # Log model dengan signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Hapus penyimpanan model manual

    # Simpan metrics ke JSON dan log
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact("metrics.json")

    # Simpan dataset train+test sebagai artifact
    X_train_full = X_train.copy()
    X_train_full['target'] = y_train
    X_train_full.to_csv("BreastCancer_train.csv", index=False)
    mlflow.log_artifact("BreastCancer_train.csv")

    X_test_full = X_test.copy()
    X_test_full['target'] = y_test
    X_test_full.to_csv("BreastCancer_test.csv", index=False)
    mlflow.log_artifact("BreastCancer_test.csv")

    print("Model dan dataset berhasil dilogging ke MLflow.")
