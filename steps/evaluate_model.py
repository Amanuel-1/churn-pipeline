import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Annotated
from zenml import step
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from src.evaluation_util import Accuracy, Precision, Recall, F1Score


@step(enable_cache=True)
def evaluate_model(
    model: Annotated[BaseEstimator, "Trained model"],
    X_test: Annotated[pd.DataFrame, "Test features"],
    y_test: Annotated[pd.Series, "Test labels"]
) -> Annotated[Dict[str, float], "Evaluation metrics"]:
    """
    Evaluate model performance using classification metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Initialize metrics
    metrics = {
        'accuracy': Accuracy().evaluate(y_test, y_pred),
        'precision': Precision().evaluate(y_test, y_pred),
        'recall': Recall().evaluate(y_test, y_pred),
        'f1_score': F1Score().evaluate(y_test, y_pred)
    }
    
    # Add ROC-AUC if model supports probabilities
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]  # Positive class probability
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = None
    
    # Log results
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Test individual metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    # Test each metric
    print(f"Accuracy: {Accuracy().evaluate(y_true, y_pred):.4f}")
    print(f"Precision: {Precision().evaluate(y_true, y_pred):.4f}")
    print(f"Recall: {Recall().evaluate(y_true, y_pred):.4f}")
    print(f"F1-Score: {F1Score().evaluate(y_true, y_pred):.4f}")
    print("Evaluation metrics test passed!")