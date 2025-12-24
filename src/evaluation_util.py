
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class Evaluation(ABC):
    """Base class for evaluation metrics"""
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric"""
        pass


class Accuracy(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


class Precision(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred, average='weighted', zero_division=0)


class Recall(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred, average='weighted', zero_division=0)


class F1Score(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)


class RMSE(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))


class MSE(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)


class MAE(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)


class R2Score(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)

