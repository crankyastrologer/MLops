from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from typing import cast

from zenml.integrations.mlflow.services import MLFlowDeploymentService

from pipelines.deployment_pipeline import continuous_deployment_pipeline
import click
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

docker_settings = DockerSettings(required_integrations=[MLFLOW])
@click.command()
@click.option("--config","-c",type=click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT],
                                           ),default="deploy_and_predict",
              help="Choose deployment or predict mode",)
@click.option('--min-accuracy',default=0.92,type=float,help="min accuracy required to deply the model")
def run_deployment(config:str,min_accuracy:float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config==DEPLOY or config==DEPLOY_AND_PREDICT
    if DEPLOY:
        print("deploying model...")
        continuous_deployment_pipeline(min_accuracy=min_accuracy,
                                       workers=3,
                                       timeout=60,)
    else: print("unable to deploy")
    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    run_deployment()

