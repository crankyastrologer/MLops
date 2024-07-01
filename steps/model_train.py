import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from .config import ModelNameConfig
from src.model_dev import LinearRegressionModel
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                config: ModelNameConfig, ) -> RegressorMixin:
    try:
        model = None
        if config.model_name == "LinearRegressionModel":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model type not supported {config.model_name}")
    except Exception as e:
        logging.error(e)
        raise e
