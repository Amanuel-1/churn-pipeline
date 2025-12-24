from sklearn.base import BaseEstimator
from src.model_util import RandomForest, LogisticRegression, SVMS, Model


class ModelConfig:
    model_name: str = "RandomForest"  # Default model
    
    @staticmethod
    def get_model(model_name: str) -> Model:
        """Get model instance by name"""
        models = {
            "RandomForest": RandomForest(),
            "LogisticRegression": LogisticRegression(),
            "SVMS": SVMS()
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]
    
    @staticmethod
    def get_available_models():
        """Get list of available model names"""
        return ["RandomForest", "LogisticRegression", "SVMS"]