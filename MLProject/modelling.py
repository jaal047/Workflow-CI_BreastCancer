import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import json
from mlflow.models.signature import infer_signature

def load_data():
    try:
        X_train = pd.read_csv('BreastCancer_preprocessing/X_train.csv')
        X_test = pd.read_csv('BreastCancer_preprocessing/X_test.csv')
        y_train = pd.read_csv('BreastCancer_preprocessing/y_train.csv').values.ravel()
        y_test = pd.read_csv('BreastCancer_preprocessing/y_test.csv').values.ravel()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")

def train_evaluate_model(X_train, y_train, X_test, y_test, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
        "recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, predictions, average='weighted', zero_division=0)
    }

    # Tambahkan handling untuk multiclass dan error
    try:
        if len(set(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    except Exception:
        metrics["roc_auc"] = None

    return model, metrics, predictions

def save_metrics(metrics, filepath="metrics.json"):
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

def main(n_estimators, max_depth):
    mlflow.set_experiment("BreastCancer_RF_Model")
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=f"RandomForest_n{n_estimators}_d{max_depth}") as run:
        model, metrics, predictions = train_evaluate_model(X_train, y_train, X_test, y_test, n_estimators, max_depth)

        mlflow.log_params({
            "n_estimators": n_estimators, 
            "max_depth": max_depth, 
            "model_type": "RandomForestClassifier"
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="BreastCancer_RF_Model",
            input_example=X_train.iloc[:5],
            signature=signature
        )

        # Simpan model dan metrik
        model_filename = "model.pkl"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)

        metrics_filename = "metrics.json"
        save_metrics(metrics, metrics_filename)
        mlflow.log_artifact(metrics_filename)

        # Print hasil evaluasi
        print("Model trained successfully!")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: N/A")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the trees.")
    args = parser.parse_args()

    try:
        main(args.n_estimators, args.max_depth)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
