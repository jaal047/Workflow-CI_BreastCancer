import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import os
import json

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# Load data
X_train = pd.read_csv('BreastCancer_preprocessing/X_train.csv')
X_test = pd.read_csv('BreastCancer_preprocessing/X_test.csv')
y_train = pd.read_csv('BreastCancer_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('BreastCancer_preprocessing/y_test.csv').values.ravel()

# Model Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Cek apakah data biner atau multi-class
    if len(set(y_test)) == 2:  
        # Biner, ambil kolom kedua (probabilitas kelas 1)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        # Multi-class, gunakan multi_class='ovr'
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    # Log parameters and metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("ROC AUC", roc_auc)

    if os.path.exists("MLProject/artifact"):
        if not os.path.isdir("MLProject/artifact"):
            print("⚠️ 'artifact' is not a directory, deleting it...")
            os.remove("MLProject/artifact")
    os.makedirs("MLProject/artifact", exist_ok=True)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }
    with open("MLProject/artifact/metrics.json", "w") as outfile:
        json.dump(metrics, outfile)

    # Log artefak ke MLflow
    mlflow.log_artifact("MLProject/artifact/metrics.json")

    # ================================
    # Save model
    model_path = "MLProject/artifact/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
