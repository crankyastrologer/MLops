import os
import json

import numpy as np
import pandas as pd

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model,experiment_tracker
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
experiment_tracker = experiment_tracker
docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.92


@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
        min_accuracy: float = 0,
        workers: int = 1,
        timeout: float = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    data_path = "C:/Users/ansh0/PycharmProjects/pythonProject/data/olist_customers_dataset.csv"
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)

    model = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(rmse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
