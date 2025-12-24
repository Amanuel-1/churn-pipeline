import pandas as pd
import logging
from zenml import step
from typing import Annotated, Tuple
from src.clean_util import DataCleaning, PreprocessStrategy, DataSplitStrategy


@step(enable_cache=True)
def clean_data(
    data_frame: Annotated[pd.DataFrame, "Raw customer churn dataset"]
) -> Tuple[
    Annotated[pd.DataFrame, "X_train: Training features for model"],
    Annotated[pd.DataFrame, "X_test: Testing features for evaluation"], 
    Annotated[pd.Series, "y_train: Training labels (churn/no churn)"],
    Annotated[pd.Series, "y_test: Testing labels (churn/no churn)"]
]:
    """ 
    Clean and split the customer churn data into training and testing sets.
    
    This step:
    1. Removes unnecessary columns (CustomerID)
    2. Splits data into 80% training, 20% testing
    3. Ensures stratified split to maintain class balance
    
    Args:
        data_frame: Raw customer churn dataset
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) for model training and evaluation
    """
    
    # Initialize strategies
    process_strategy = PreprocessStrategy()
    data_split_strategy = DataSplitStrategy() 

    # First preprocess the data (remove CustomerID, clean values)
    preprocess = DataCleaning(data_frame, process_strategy)
    cleaned_data = preprocess.handle_data()

    # Then split the cleaned data into train/test sets
    datasplit = DataCleaning(cleaned_data, data_split_strategy)
    x_train, x_test, y_train, y_test = datasplit.handle_data()

    # Log information about the split
    logging.info(f"Data split completed:")
    logging.info(f"Training set: {x_train.shape[0]} samples, {x_train.shape[1]} features")
    logging.info(f"Testing set: {x_test.shape[0]} samples, {x_test.shape[1]} features")
    logging.info(f"Training churn rate: {y_train.mean():.2%}")
    logging.info(f"Testing churn rate: {y_test.mean():.2%}")

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # Test the clean_data step
    import pandas as pd
    
    # Create sample data
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Feature2': [0.5, 1.5, 2.5, 3.5, 4.5],
        'Churn': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Test preprocessing
    preprocess_strategy = PreprocessStrategy()
    cleaned = preprocess_strategy.handle_data(df)
    print("Preprocessing test passed!")
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Cleaned columns: {cleaned.columns.tolist()}")