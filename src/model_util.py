
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator


class Model(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> BaseEstimator:
        """Train a model and return the trained estimator"""
        pass


class RandomForest(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestClassifier:
        """Train using Random Forest"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model


class LogisticRegression(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> SklearnLogisticRegression:
        """Train using Logistic Regression"""
        model = SklearnLogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        return model


class SVMS(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> SVC:
        """Train using Support Vector Machine"""
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(X_train, y_train)
        return model


