import pandas as pd
import logging
from zenml import step
from typing import Annotated
from sklearn.base import BaseEstimator
from src.model_util import RandomForest, LogisticRegression, SVMS


@step(enable_cache=True)
def train_model(
    X_train: Annotated[pd.DataFrame, "Training features"],
    y_train: Annotated[pd.Series, "Training labels"],
    model_name: Annotated[str, "Model name"] = "RandomForest"
) -> Annotated[BaseEstimator, "Trained model"]:
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of model to train ('RandomForest', 'LogisticRegression', 'SVMS')
        
    Returns:
        Trained model
    """
    
    # Model mapping
    models = {
        "RandomForest": RandomForest(),
        "LogisticRegression": LogisticRegression(),
        "SVMS": SVMS()
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    logging.info(f"Training {model_name} model...")
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Training labels shape: {y_train.shape}")
    
    # Train the model
    model_trainer = models[model_name]
    trained_model = model_trainer.train(X_train, y_train)
    
    logging.info(f"{model_name} training completed successfully!")
    
    return trained_model


if __name__ == "__main__":
    # Test the train_model step
    from sklearn.datasets import make_classification
    import pandas as pd
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y)
    
    # Test RandomForest
    rf_trainer = RandomForest()
    model = rf_trainer.train(X_df, y_series)
    print(f"RandomForest training test passed! Model type: {type(model)}")
    
    # Test LogisticRegression
    lr_trainer = LogisticRegression()
    model = lr_trainer.train(X_df, y_series)
    print(f"LogisticRegression training test passed! Model type: {type(model)}")