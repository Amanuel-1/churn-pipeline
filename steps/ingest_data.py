import pandas as pd
from zenml import step
from typing import Annotated
from src.ingest_util import DataIngestor


@step(enable_cache=True)
def ingest_data(
    file_path: Annotated[str, "Path to the data file"]
) -> Annotated[pd.DataFrame, "Raw ingested data"]:
    """
    Ingest data from a file path.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with ingested data
    """
    
    try:
        data_ingestor = DataIngestor(file_path)
        df = data_ingestor.get_data()
        
        print(f"Data ingested successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Data ingestion failed: {str(e)}")


if __name__ == "__main__":
    # Test data ingestion
    test_path = "test_data.csv"
    print("Ingest data step ready for testing!")