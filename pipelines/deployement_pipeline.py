"""
Deployment Pipeline - deploys model only if it meets accuracy threshold
"""
from zenml import pipeline
from typing import Dict, Any

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.deployment_steps import (
    validate_training_data,
    train_production_model,
    evaluate_for_deployment,
    deploy_model
)


@pipeline(name="deployment_pipeline")
def deployment_pipeline(
    file_path: str,
    model_name: str = "GradientBoosting",
    hyperparams: Dict[str, Any] = None,
    min_accuracy: float = 0.85
):
    """
    Production deployment pipeline
    
    Only deploys if model meets the specified accuracy threshold
    
    Steps:
    1. Ingest and validate data
    2. Preprocess and split
    3. Train model
    4. Evaluate against accuracy threshold
    5. Deploy only if passed
    """
    # ingest and validate
    raw_data = ingest_data(file_path)
    validated_data, data_stats = validate_training_data(raw_data)
    
    # preprocess
    X_train, X_test, y_train, y_test = clean_data(validated_data)
    
    # train
    model, model_params = train_production_model(X_train, y_train, model_name, hyperparams)
    
    # evaluate with threshold
    metrics, passed = evaluate_for_deployment(model, X_test, y_test, min_accuracy)
    
    # deploy only if passed
    status = deploy_model(model, metrics, model_params, data_stats, passed)
    
    return status
