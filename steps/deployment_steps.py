"""
Deployment steps for production pipeline
handles validation, model registration, and artifact saving
"""
import dagshub
import mlflow
import mlflow.sklearn
import logging
import json
import os
import joblib
from datetime import datetime
from zenml import step
from typing import Annotated, Dict, Tuple, Any, Optional
from sklearn.base import BaseEstimator
import pandas as pd

from src.evaluation_util import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import roc_auc_score

# Initialize Dagshub MLflow
dagshub.init(repo_owner='Amanuel-1', repo_name='churn-pipeline', mlflow=True)

EXPERIMENT_NAME = "churn_production"
MODEL_REGISTRY_NAME = "churn_predictor"


@step
def validate_training_data(
    data_frame: Annotated[pd.DataFrame, "raw data"]
) -> Tuple[
    Annotated[pd.DataFrame, "validated data"],
    Annotated[Dict[str, Any], "data stats"]
]:
    """
    production data validation - more strict than training
    logs stats for monitoring data drift later
    """
    required_cols = ['Gender', 'Age', 'Tenure', 'Usage Frequency', 
                     'Support Calls', 'Payment Delay', 'Subscription Type',
                     'Contract Length', 'Total Spend', 'Last Interaction', 'Churn']
    
    missing = [c for c in required_cols if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing columns for production: {missing}")
    
    # check data quality
    null_pct = data_frame.isnull().sum() / len(data_frame) * 100
    high_null_cols = null_pct[null_pct > 5].to_dict()
    if high_null_cols:
        logging.warning(f"High null percentage: {high_null_cols}")
    
    # compute stats for drift detection
    stats = {
        "n_samples": len(data_frame),
        "n_features": len(data_frame.columns),
        "churn_rate": float(data_frame['Churn'].mean()),
        "age_mean": float(data_frame['Age'].mean()),
        "tenure_mean": float(data_frame['Tenure'].mean()),
        "timestamp": datetime.now().isoformat()
    }
    
    logging.info(f"Data validated: {stats['n_samples']} samples, churn rate: {stats['churn_rate']:.2%}")
    
    return data_frame, stats


@step
def train_production_model(
    X_train: Annotated[pd.DataFrame, "training features"],
    y_train: Annotated[pd.Series, "training labels"],
    model_name: str = "GradientBoosting",
    hyperparams: Dict[str, Any] = None
) -> Tuple[
    Annotated[BaseEstimator, "trained model"],
    Annotated[Dict[str, Any], "model params"]
]:
    """
    train model for production - uses best known config by default
    """
    from steps.config import ModelConfig, HyperParams
    
    if hyperparams is None:
        hyperparams = {
            "gb_n_estimators": 100,
            "gb_learning_rate": 0.1,
            "gb_max_depth": 4
        }
    
    params = HyperParams(**hyperparams)
    model_trainer = ModelConfig.get_model(model_name, params)
    
    logging.info(f"Training production model: {model_name}")
    trained_model = model_trainer.train(X_train, y_train)
    model_params = model_trainer.get_params()
    model_params["model_type"] = model_name
    
    return trained_model, model_params


@step
def evaluate_for_deployment(
    model: Annotated[BaseEstimator, "model"],
    X_test: Annotated[pd.DataFrame, "test features"],
    y_test: Annotated[pd.Series, "test labels"],
    min_accuracy: float = 0.85
) -> Tuple[
    Annotated[Dict[str, float], "metrics"],
    Annotated[bool, "passed quality gate"]
]:
    """
    evaluate model against deployment threshold
    only models meeting accuracy threshold will be deployed
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': Accuracy().evaluate(y_test, y_pred),
        'precision': Precision().evaluate(y_test, y_pred),
        'recall': Recall().evaluate(y_test, y_pred),
        'f1_score': F1Score().evaluate(y_test, y_pred)
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # quality gate - only accuracy threshold
    passed = metrics['accuracy'] >= min_accuracy
    
    if passed:
        logging.info(f"✓ PASSED: accuracy {metrics['accuracy']:.4f} >= {min_accuracy}")
    else:
        logging.error(f"✗ FAILED: accuracy {metrics['accuracy']:.4f} < {min_accuracy}")
    
    return metrics, passed


@step
def deploy_model(
    model: Annotated[BaseEstimator, "model"],
    metrics: Annotated[Dict[str, float], "metrics"],
    model_params: Annotated[Dict[str, Any], "params"],
    data_stats: Annotated[Dict[str, Any], "data stats"],
    passed_quality_gate: Annotated[bool, "quality gate result"]
) -> Annotated[str, "deployment status"]:
    """
    deploy model only if it passed quality gate
    registers to mlflow and saves artifacts locally
    """
    if not passed_quality_gate:
        logging.warning("Model did NOT pass quality gate - skipping deployment")
        return "not_deployed"
    
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    model_type = model_params.get("model_type", "unknown")
    
    with mlflow.start_run(run_name=f"deploy_{model_type}") as run:
        # log params
        for k, v in model_params.items():
            if v is not None:
                mlflow.log_param(k, v)
        
        # log metrics
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)
        
        mlflow.log_param("training_samples", data_stats["n_samples"])
        mlflow.log_param("churn_rate", data_stats["churn_rate"])
        
        # register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        
        mlflow.set_tag("pipeline", "deployment")
        mlflow.set_tag("status", "deployed")
        mlflow.set_tag("model_type", model_type)
        
        run_id = run.info.run_id
    
    # save locally too
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = f"models/deployed_{timestamp}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(artifact_dir, "model.joblib"))
    
    metadata = {
        "model_params": model_params,
        "metrics": metrics,
        "data_stats": data_stats,
        "mlflow_run_id": run_id,
        "timestamp": timestamp
    }
    with open(os.path.join(artifact_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Model DEPLOYED to: {artifact_dir}")
    logging.info(f"MLflow run: {run_id}")
    
    return f"deployed:{artifact_dir}"
