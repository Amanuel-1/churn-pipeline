"""
Deployment Pipeline - for production model deployment
this is the real deal - validates, trains, registers, and deploys
"""
import mlflow
import mlflow.sklearn
import logging
import json
import os
import joblib
from datetime import datetime
from zenml import pipeline, step
from typing import Annotated, Dict, Tuple, Any, Optional
from sklearn.base import BaseEstimator
import pandas as pd

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


MLFLOW_TRACKING_URI = "mlruns"
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
    
    # check columns
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
    
    # default to best performing config if no params given
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
def evaluate_production_model(
    model: Annotated[BaseEstimator, "model"],
    X_test: Annotated[pd.DataFrame, "test features"],
    y_test: Annotated[pd.Series, "test labels"],
    min_accuracy: float = 0.85,
    min_f1: float = 0.80
) -> Tuple[
    Annotated[Dict[str, float], "metrics"],
    Annotated[bool, "passed quality gate"]
]:
    """
    evaluate model against production quality gates
    stricter thresholds than training
    """
    from src.evaluation_util import Accuracy, Precision, Recall, F1Score
    from sklearn.metrics import roc_auc_score
    
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
    
    # quality gate check
    passed = metrics['accuracy'] >= min_accuracy and metrics['f1_score'] >= min_f1
    
    if passed:
        logging.info(f"✓ Model PASSED quality gate: acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")
    else:
        logging.error(f"✗ Model FAILED quality gate: acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")
    
    return metrics, passed


@step
def register_model(
    model: Annotated[BaseEstimator, "model"],
    metrics: Annotated[Dict[str, float], "metrics"],
    model_params: Annotated[Dict[str, Any], "params"],
    data_stats: Annotated[Dict[str, Any], "data stats"],
    passed_quality_gate: Annotated[bool, "quality gate result"]
) -> Annotated[Optional[str], "model version"]:
    """
    register model to mlflow model registry if it passed quality gate
    """
    if not passed_quality_gate:
        logging.warning("Model did not pass quality gate - not registering")
        return None
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    model_type = model_params.get("model_type", "unknown")
    
    with mlflow.start_run(run_name=f"production_{model_type}") as run:
        # log everything
        for k, v in model_params.items():
            if v is not None:
                mlflow.log_param(k, v)
        
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)
        
        # log data stats as params
        mlflow.log_param("training_samples", data_stats["n_samples"])
        mlflow.log_param("churn_rate", data_stats["churn_rate"])
        
        # register model
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        
        # tags for production tracking
        mlflow.set_tag("pipeline", "production")
        mlflow.set_tag("environment", "staging")
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("quality_gate", "passed")
        
        logging.info(f"Model registered: {MODEL_REGISTRY_NAME}")
        
        return run.info.run_id


@step
def save_model_artifacts(
    model: Annotated[BaseEstimator, "model"],
    model_params: Annotated[Dict[str, Any], "params"],
    metrics: Annotated[Dict[str, float], "metrics"],
    passed_quality_gate: Annotated[bool, "quality gate"]
) -> Annotated[str, "artifact path"]:
    """
    save model artifacts locally for deployment
    creates versioned directory with model + metadata
    """
    if not passed_quality_gate:
        return "not_saved"
    
    # create artifacts dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = f"models/production_{timestamp}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # save model
    model_path = os.path.join(artifact_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # save metadata
    metadata = {
        "model_params": model_params,
        "metrics": metrics,
        "timestamp": timestamp,
        "model_path": model_path
    }
    
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Artifacts saved to: {artifact_dir}")
    
    return artifact_dir


@pipeline(name="deployment_pipeline")
def deployment_pipeline(
    file_path: str,
    model_name: str = "GradientBoosting",
    hyperparams: Dict[str, Any] = None,
    min_accuracy: float = 0.85,
    min_f1: float = 0.80
):
    """
    Production deployment pipeline
    
    1. Validates data quality
    2. Trains model with production config
    3. Evaluates against quality gates
    4. Registers to model registry (if passed)
    5. Saves artifacts for deployment
    """
    # data ingestion and validation
    raw_data = ingest_data(file_path)
    validated_data, data_stats = validate_training_data(raw_data)
    
    # preprocessing
    X_train, X_test, y_train, y_test = clean_data(validated_data)
    
    # training
    model, model_params = train_production_model(X_train, y_train, model_name, hyperparams)
    
    # evaluation with quality gates
    metrics, passed = evaluate_production_model(model, X_test, y_test, min_accuracy, min_f1)
    
    # registration and artifact saving
    run_id = register_model(model, metrics, model_params, data_stats, passed)
    artifact_path = save_model_artifacts(model, model_params, metrics, passed)
    
    return passed
