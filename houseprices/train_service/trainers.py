from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import linear_model as linear_model
from sklearn.model_selection import train_test_split


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

        preds = np.expm1(self.model.predict(test_x))
        self.error = Trainer.calc_error(test_y, preds)

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
        x_tr, x_val, y_tr, y_val = train_test_split(x, y, random_state=42, test_size=0.20)
        eval_set = [(x_val, y_val)]
        model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1)
        model_xgb.fit(x_tr, y_tr, eval_metric="rmse", early_stopping_rounds=500, eval_set=eval_set, verbose=False)
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
        ridge = linear_model.RidgeCV()
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
        elastic_net = linear_model.ElasticNetCV()
        elastic_net.fit(x, y)
        return elastic_net
