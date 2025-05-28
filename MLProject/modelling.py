import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import json

# --- Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_estimators", type=int, default=100)
parser.add_argument("-d", "--max_depth", type=int, default=None)
args = parser.parse_args()

# --- Data loading
data = pd.read_csv("breast_cancer_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

# --- Model training
clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
clf.fit(X, y)
y_pred = clf.predict(X)

# --- Metrics calculation
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred),
    "f1": f1_score(y, y_pred),
    "roc_auc": roc_auc_score(y, y_pred)
}

# --- Start or use active MLflow run ---
run = mlflow.active_run()
if run is None:
    run = mlflow.start_run(run_name=f"RandomForest_n{args.n_estimators}_d{args.max_depth}")

# --- Logging params and metrics ---
mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
mlflow.log_metrics(metrics)

# --- Save model locally and log to MLflow ---
joblib.dump(clf, "model.pkl")
mlflow.sklearn.log_model(clf, "model")

# --- Save metrics.json ---
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# --- End run if we started it manually ---
if run and run.info.run_id == mlflow.active_run().info.run_id:
    mlflow.end_run()
