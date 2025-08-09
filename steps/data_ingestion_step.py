import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:

    #Determine the file extension
    file_extension = ".zip"

    #Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    #Ingest the data and the load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    return df