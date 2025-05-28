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

# Set experiment name
mlflow.set_experiment("BreastCancer_RF_Model")

# Load data
X_train = pd.read_csv('BreastCancer_preprocessing/X_train.csv')
X_test = pd.read_csv('BreastCancer_preprocessing/X_test.csv')
y_train = pd.read_csv('BreastCancer_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('BreastCancer_preprocessing/y_test.csv').values.ravel()

# Model Training
with mlflow.start_run(run_name=f"RandomForest_n{args.n_estimators}_d{args.max_depth}"):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # ROC AUC calculation
    if len(set(y_test)) == 2:  
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    # Log parameters and metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model with MLflow (ini yang penting untuk serving)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="BreastCancer_RF_Model",
        input_example=X_train.iloc[:5],  # contoh input
        signature=mlflow.models.infer_signature(X_train, predictions)
    )
    
    # Save model locally juga (untuk backward compatibility)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    
    # Log model performance summary
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    mlflow.log_artifact("metrics.json")
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print run info untuk serving
    run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"Model URI: runs:/{run_id}/model")