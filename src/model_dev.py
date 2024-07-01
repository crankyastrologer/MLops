import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):

    @abstractmethod
    def train(self, X_train:pd.DataFrame, y_train:pd.Series):
        pass

class LinearRegressionModel(Model):

    def train(self, X_train:pd.DataFrame, y_train:pd.Series,**kwargs)->RegressorMixin:

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error(e)
            raise e