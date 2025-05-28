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
import os

# --- Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_estimators", type=int, default=100)
parser.add_argument("-d", "--max_depth", type=int, default=None)
args = parser.parse_args()

# --- Data loading
X_train = pd.read_csv('BreastCancer_preprocessing/X_train.csv')
X_test = pd.read_csv('BreastCancer_preprocessing/X_test.csv')
y_train = pd.read_csv('BreastCancer_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('BreastCancer_preprocessing/y_test.csv').values.ravel()

# --- Model training
clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
clf.fit(X_train, y_train)

# --- Predictions and metrics
y_pred = clf.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

# --- Start or use active MLflow run ---
run = mlflow.active_run()
if run is None:
    run = mlflow.start_run(run_name=f"RandomForest_n{args.n_estimators}_d{args.max_depth}")

# --- Logging params and metrics ---
mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
mlflow.log_metrics(metrics)

# --- Save model locally and log it ---
model_path = "model.pkl"
joblib.dump(clf, model_path)
mlflow.sklearn.log_model(clf, "model")

# --- Save metrics as JSON and log it ---
metrics_path = "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
mlflow.log_artifact(metrics_path)

# --- Log dataset files as artifacts ---
dataset_dir = 'BreastCancer_preprocessing'
for filename in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
    filepath = os.path.join(dataset_dir, filename)
    mlflow.log_artifact(filepath)

# --- End run if started manually ---
if run and run.info.run_id == mlflow.active_run().info.run_id:
    mlflow.end_run()
