from zenml import pipeline
from steps.clean_data import clean_data
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.train_model import train_model


@pipeline
def training_pipeline(file_path: str, model_name: str = "RandomForest"):
    """
    Complete training pipeline for churn prediction.
    
    Args:
        file_path: Path to the data file
        model_name: Name of the model to train
    """
    # Ingest data
    data_frame = ingest_data(file_path)
    
    # Clean and split data
    X_train, X_test, y_train, y_test = clean_data(data_frame)
    
    # Train model
    model = train_model(X_train, y_train, model_name)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    return metrics