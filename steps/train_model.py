import pandas as pd
import logging
from zenml import step
from typing import Annotated, Dict, Any, Tuple
from sklearn.base import BaseEstimator
from steps.config import ModelConfig, HyperParams


@step(enable_cache=False)  # disable cache for experiment tracking
def train_model(
    X_train: Annotated[pd.DataFrame, "Training features"],
    y_train: Annotated[pd.Series, "Training labels"],
    model_name: Annotated[str, "Model name"] = "RandomForest",
    hyperparams: Annotated[Dict[str, Any], "Hyperparameters"] = None
) -> Tuple[
    Annotated[BaseEstimator, "Trained model"],
    Annotated[Dict[str, Any], "Model params"]
]:
    """
    Train a ml model with custom hyperparameters
    returns the model and the params used
    """
    
    # build hyperparams from dict if provided
    if hyperparams:
        params = HyperParams(**hyperparams)
    else:
        params = HyperParams()
    
    logging.info(f"Training {model_name} model...")
    logging.info(f"Training data shape: {X_train.shape}")
    
    # get model with hyperparams
    model_trainer = ModelConfig.get_model(model_name, params)
    
    # train it
    trained_model = model_trainer.train(X_train, y_train)
    model_params = model_trainer.get_params()
    model_params["model_type"] = model_name
    
    logging.info(f"{model_name} training done with params: {model_params}")
    
    return trained_model, model_params
