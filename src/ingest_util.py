import pandas as pd

class DataIngestor:
    def __init__(self,file_path:str):
        self.file_path = file_path
    
    def get_data(self):
        csv_data= pd.read_csv(self.file_path)

        return csv_data
