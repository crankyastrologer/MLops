from zenml import pipeline

from steps.config import ModelNameConfig
from steps.ingest_data import ingest_df
from  steps.clean_data import clean_data
from steps.model_train import train_model,experiment_tracker
from steps.evaluation import evaluate_model
experiment_tracker = experiment_tracker
@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test = clean_data(df)

    model = train_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
    r2_score,rmse= evaluate_model(model,X_test,y_test)
