import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from typing import Dict, Any


class Model(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> BaseEstimator:
        """Train a model and return the trained estimator"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """return the hyperparameters used"""
        pass


class RandomForest(Model):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestClassifier:
        """Train using Random Forest"""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        }


class LogisticRegression(Model):
    def __init__(self, C: float = 1.0, max_iter: int = 100, solver: str = "lbfgs"):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> SklearnLogisticRegression:
        """Train using Logistic Regression"""
        model = SklearnLogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver
        }


class SVMS(Model):
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> SVC:
        """Train using Support Vector Machine"""
        model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=42,
            probability=True
        )
        model.fit(X_train, y_train)
        return model
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma
        }


class GradientBoosting(Model):
    # adding gradient boosting cuz why not
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> GradientBoostingClassifier:
        """Train using Gradient Boosting"""
        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth
        }
