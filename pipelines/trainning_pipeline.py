"""
Training Pipeline - for model development and experimentation
use this for training new models and comparing different approaches
"""
import mlflow
import mlflow.sklearn
import logging
import json
import os
from datetime import datetime
from zenml import pipeline, step, get_step_context
from typing import Annotated, Dict, Tuple, Any
from sklearn.base import BaseEstimator
import pandas as pd

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "churn_training"


@step
def validate_data(
    data_frame: Annotated[pd.DataFrame, "raw data"]
) -> Annotated[pd.DataFrame, "validated data"]:
    """
    validate incoming data before training
    checks for missing cols, data types, etc
    """
    required_cols = ['Gender', 'Age', 'Tenure', 'Churn']
    
    missing = [c for c in required_cols if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # check for nulls in critical columns
    null_counts = data_frame[required_cols].isnull().sum()
    if null_counts.any():
        logging.warning(f"Found nulls: {null_counts[null_counts > 0].to_dict()}")
    
    # basic stats logging
    logging.info(f"Data shape: {data_frame.shape}")
    logging.info(f"Churn distribution: {data_frame['Churn'].value_counts().to_dict()}")
    
    return data_frame


@step
def log_training_run(
    model: Annotated[BaseEstimator, "trained model"],
    metrics: Annotated[Dict[str, float], "metrics"],
    model_params: Annotated[Dict[str, Any], "params"],
    run_name: str = "training_run"
) -> Annotated[str, "run_id"]:
    """log training run to mlflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    model_type = model_params.get("model_type", "unknown")
    
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in model_params.items():
            if v is not None:
                mlflow.log_param(k, v)
        
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)
        
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("pipeline", "training")
        mlflow.set_tag("model_type", model_type)
        
        logging.info(f"Training logged: {run.info.run_id}")
        return run.info.run_id


@pipeline(name="training_pipeline")
def training_pipeline(
    file_path: str,
    model_name: str = "RandomForest",
    hyperparams: Dict[str, Any] = None,
    run_name: str = "training_run"
):
    """
    Training pipeline for model development
    - validates data
    - trains model
    - evaluates performance
    - logs to mlflow
    """
    # ingest and validate
    raw_data = ingest_data(file_path)
    validated_data = validate_data(raw_data)
    
    # preprocess and split
    X_train, X_test, y_train, y_test = clean_data(validated_data)
    
    # train
    model, model_params = train_model(X_train, y_train, model_name, hyperparams)
    
    # evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # log
    run_id = log_training_run(model, metrics, model_params, run_name)
    
    return metrics
