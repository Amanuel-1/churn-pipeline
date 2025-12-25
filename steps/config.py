from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.model_util import RandomForest, LogisticRegression, SVMS, GradientBoosting, Model


@dataclass
class HyperParams:
    """hyperparameter config for different models"""
    # random forest params
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    # logistic regression params
    lr_C: float = 1.0
    lr_max_iter: int = 100
    lr_solver: str = "lbfgs"
    
    # svm params
    svm_C: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    
    # gradient boosting params
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3


class ModelConfig:
    model_name: str = "RandomForest"
    
    @staticmethod
    def get_model(model_name: str, params: HyperParams = None) -> Model:
        """Get model instance with hyperparams"""
        if params is None:
            params = HyperParams()
        
        models = {
            "RandomForest": RandomForest(
                n_estimators=params.rf_n_estimators,
                max_depth=params.rf_max_depth,
                min_samples_split=params.rf_min_samples_split
            ),
            "LogisticRegression": LogisticRegression(
                C=params.lr_C,
                max_iter=params.lr_max_iter,
                solver=params.lr_solver
            ),
            "SVMS": SVMS(
                C=params.svm_C,
                kernel=params.svm_kernel,
                gamma=params.svm_gamma
            ),
            "GradientBoosting": GradientBoosting(
                n_estimators=params.gb_n_estimators,
                learning_rate=params.gb_learning_rate,
                max_depth=params.gb_max_depth
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]
    
    @staticmethod
    def get_available_models():
        """Get list of available model names"""
        return ["RandomForest", "LogisticRegression", "SVMS", "GradientBoosting"]
