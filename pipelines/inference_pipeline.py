"""
Inference Pipeline - for making predictions in production
loads trained model and runs predictions on new data
"""
import mlflow
import logging
import joblib
import json
import os
from datetime import datetime
from zenml import pipeline, step
from typing import Annotated, Dict, List, Any, Optional
import pandas as pd
import numpy as np


MLFLOW_TRACKING_URI = "mlruns"
MODEL_REGISTRY_NAME = "churn_predictor"


@step
def load_production_model(
    model_path: str = None
) -> Annotated[Any, "loaded model"]:
    """
    load model from local path or mlflow registry
    defaults to latest registered model
    """
    if model_path and os.path.exists(model_path):
        # load from local file
        logging.info(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    
    # load from mlflow registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logging.info(f"Loaded model from registry: {MODEL_REGISTRY_NAME}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


@step
def preprocess_inference_data(
    data_frame: Annotated[pd.DataFrame, "raw inference data"]
) -> Annotated[pd.DataFrame, "preprocessed data"]:
    """
    preprocess data for inference - same transforms as training
    but without the target column
    """
    df = data_frame.copy()
    
    # store customer ids if present
    customer_ids = None
    if 'CustomerID' in df.columns:
        customer_ids = df['CustomerID'].copy()
        df = df.drop('CustomerID', axis=1)
    
    # drop target if accidentally included
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    
    # same encoding as training
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    if 'Subscription Type' in df.columns:
        df['Subscription Type'] = df['Subscription Type'].map({
            'Basic': 0, 'Standard': 1, 'Premium': 2
        })
    
    if 'Contract Length' in df.columns:
        df['Contract Length'] = df['Contract Length'].map({
            'Monthly': 0, 'Quarterly': 1, 'Annual': 2
        })
    
    # feature engineering
    if 'Total Spend' in df.columns and 'Tenure' in df.columns:
        df['Spend_per_Tenure'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
    
    if 'Support Calls' in df.columns and 'Tenure' in df.columns:
        df['Support_Call_Rate'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
    
    if 'Usage Frequency' in df.columns and 'Tenure' in df.columns:
        df['Usage_per_Tenure'] = df['Usage Frequency'] / df['Tenure'].replace(0, 1)
    
    # handle nulls
    df = df.fillna(df.median(numeric_only=True))
    
    logging.info(f"Preprocessed {len(df)} samples for inference")
    
    return df


@step
def make_predictions(
    model: Annotated[Any, "model"],
    data: Annotated[pd.DataFrame, "preprocessed data"]
) -> Annotated[pd.DataFrame, "predictions"]:
    """
    run predictions and return results with probabilities
    """
    predictions = model.predict(data)
    
    results = pd.DataFrame({
        'prediction': predictions,
        'churn_label': ['Churn' if p == 1 else 'No Churn' for p in predictions]
    })
    
    # add probabilities if available
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(data)
        results['churn_probability'] = probas[:, 1]
        results['confidence'] = np.maximum(probas[:, 0], probas[:, 1])
    
    logging.info(f"Predictions made: {len(results)} samples")
    logging.info(f"Churn rate: {results['prediction'].mean():.2%}")
    
    return results


@step
def save_predictions(
    predictions: Annotated[pd.DataFrame, "predictions"],
    output_path: str = "predictions"
) -> Annotated[str, "output file"]:
    """
    save predictions to file
    """
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}/predictions_{timestamp}.csv"
    
    predictions.to_csv(filename, index=False)
    logging.info(f"Predictions saved to: {filename}")
    
    return filename


@pipeline(name="inference_pipeline")
def inference_pipeline(
    data_path: str,
    model_path: str = None,
    output_path: str = "predictions"
):
    """
    Production inference pipeline
    
    1. Loads trained model
    2. Preprocesses new data
    3. Makes predictions
    4. Saves results
    """
    from steps.ingest_data import ingest_data
    
    # load model
    model = load_production_model(model_path)
    
    # load and preprocess data
    raw_data = ingest_data(data_path)
    processed_data = preprocess_inference_data(raw_data)
    
    # predict
    predictions = make_predictions(model, processed_data)
    
    # save
    output_file = save_predictions(predictions, output_path)
    
    return predictions
