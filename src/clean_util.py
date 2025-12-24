import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Annotated
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data_frame: pd.DataFrame) -> Union[
        Annotated[pd.DataFrame, "Preprocessed customer data"],
        Tuple[
            Annotated[pd.DataFrame, "Training features"],
            Annotated[pd.DataFrame, "Testing features"], 
            Annotated[pd.Series, "Training labels"],
            Annotated[pd.Series, "Testing labels"]
        ]
    ]:
        """
        Each implementation will handle the specified strategy and apply it on the dataframe and then return it.
        """
        pass

class PreprocessStrategy(DataStrategy):
    
    def handle_data(self, data_frame: pd.DataFrame) -> Annotated[pd.DataFrame, "Preprocessed customer data with CustomerID removed"]:
        """
        Preprocess the data by removing unnecessary columns and cleaning values.
        """
        df = data_frame.copy()
        
        # Drop CustomerID - not predictive
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Handle missing values
        df = df.dropna()
        
        # Binary encoding for Gender
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
        
        # Ordinal encoding for Subscription Type (hierarchy: Basic < Standard < Premium)
        if 'Subscription Type' in df.columns:
            df['Subscription Type'] = df['Subscription Type'].map({
                'Basic': 0, 'Standard': 1, 'Premium': 2
            })
        
        # Ordinal encoding for Contract Length (commitment level)
        if 'Contract Length' in df.columns:
            df['Contract Length'] = df['Contract Length'].map({
                'Monthly': 0, 'Quarterly': 1, 'Annual': 2
            })
        
        # Feature engineering (avoid division by zero)
        if 'Total Spend' in df.columns and 'Tenure' in df.columns:
            df['Spend_per_Tenure'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
        
        if 'Support Calls' in df.columns and 'Tenure' in df.columns:
            df['Support_Call_Rate'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
        
        if 'Usage Frequency' in df.columns and 'Tenure' in df.columns:
            df['Usage_per_Tenure'] = df['Usage Frequency'] / df['Tenure'].replace(0, 1)
        
        return df

class DataSplitStrategy(DataStrategy):

    def handle_data(self, df: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "Training features (80% of data)"],
        Annotated[pd.DataFrame, "Testing features (20% of data)"], 
        Annotated[pd.Series, "Training labels for churn prediction"],
        Annotated[pd.Series, "Testing labels for churn prediction"]
    ]:
        """
        Split the dataframe into train and test divisions with 80/20 split.
        """
        X = df.drop('Churn', axis=1)  # Features
        y = df['Churn']               # Target variable
        
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Ensure balanced split for churn classes
        )

        return x_train, x_test, y_train, y_test

        

class DataCleaning:
    def __init__(self, data_frame: pd.DataFrame, strategy: DataStrategy):
        self.strategy = strategy
        self.data = data_frame

    def handle_data(self) -> Union[
        Annotated[pd.DataFrame, "Processed data"],
        Tuple[
            Annotated[pd.DataFrame, "Training features"],
            Annotated[pd.DataFrame, "Testing features"], 
            Annotated[pd.Series, "Training labels"],
            Annotated[pd.Series, "Testing labels"]
        ]
    ]:
        """
        Execute the data handling strategy and return the processed data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"An error occurred while handling the data: {str(e)}")
            raise e
