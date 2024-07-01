import logging

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Tuple, Annotated
from zenml import step
from src.evaluation import MSE, R2, RMSE


@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2_score"],
Annotated[float, "Rmse_score"],]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()

        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('mse',value=mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('r2',value=r2)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('rmse',value=rmse)

        return r2, rmse
    except Exception as e:
        logging.error(e)
        raise e
