import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy


@step
def clean_data(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, 'X_train'],
Annotated[pd.DataFrame, 'X_test'],
Annotated[pd.Series, 'y_train'],
Annotated[pd.Series, 'y_test'],]:
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divider = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divider)
        X_train, X_text, y_train, y_test = data_cleaning.handle_data()
        logging.info(f'Data cleaning complete')
        return X_train, X_text, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
