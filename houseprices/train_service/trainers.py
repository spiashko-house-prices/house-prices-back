from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import linear_model


class Trainer(ABC):

    def __init__(self):
        self.model = None
        self.error = None

    @abstractmethod
    def _fit_model(self, x, y):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def train(self, df_train, df_test, features_full_list):
        """

        :rtype: Trainer
        :param df_train: train dataset
        :type df_train:  pd.DataFrame
        :param df_test: test dataset
        :type df_test:  pd.DataFrame
        :param features_full_list: full list of features
        :type features_full_list: list
        """
        x = df_train[features_full_list]
        y = df_train['SalePrice'].values

        test_x = df_test[features_full_list]
        test_y = df_test['SalePrice'].values

        self.model = self._fit_model(x, np.log1p(y))

        self.error = Trainer.calc_error(test_y, np.expm1(self.model.predict(test_x)))

    def predict(self, x):
        return np.expm1(self.model.predict(x))

    @staticmethod
    def calc_error(actual, predicted):
        """

        :rtype: float
        """
        actual = np.log(actual)
        predicted = np.log(predicted)
        return np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))


class GradientBoosting(Trainer):

    def get_name(self):
        return "gradientBoosting"

    def _fit_model(self, x, y):
        model_xgb = xgb.XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.1)
        model_xgb.fit(x, y)
        return model_xgb


class Linear(Trainer):

    def get_name(self):
        return "linear"

    def _fit_model(self, x, y):
        linear = linear_model.LinearRegression()
        linear.fit(x, y)
        return linear


class Ridge(Trainer):

    def get_name(self):
        return "ridge"

    def _fit_model(self, x, y):
        ridge = linear_model.RidgeCV(cv=10)
        ridge.fit(x, y)
        return ridge


class Lasso(Trainer):

    def get_name(self):
        return "lasso"

    def _fit_model(self, x, y):
        lasso = linear_model.LassoCV()
        lasso.fit(x, y)
        return lasso


class ElasticNet(Trainer):

    def get_name(self):
        return "elastic_net"

    def _fit_model(self, x, y):
        elastic_net = linear_model.ElasticNetCV(cv=10)
        elastic_net.fit(x, y)
        return elastic_net
