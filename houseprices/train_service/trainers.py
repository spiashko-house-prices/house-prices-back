import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import linear_model


class Trainer:
    def __init__(self, name, model, error):
        self.name = name
        self.model = model
        self.error = error


def calc_error(actual, predicted):
    """

    :rtype: float
    """
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))


def train_gb(df_train_cleaned, df_train, features_full_list):
    """

    :rtype: Trainer
    :param df_train_cleaned: cleaned dataset
    :type df_train_cleaned:  pd.DataFrame
    :param df_train: full dataset
    :type df_train:  pd.DataFrame
    :param features_full_list: full list of features
    :type features_full_list: list
    :return: logarithmic error
    """

    x = df_train_cleaned[features_full_list]
    y = df_train_cleaned['SalePrice'].values

    full_x = df_train[features_full_list]
    full_y = df_train['SalePrice'].values

    model_xgb = xgb.XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.1)
    model_xgb.fit(x, np.log1p(y))

    xgb_preds = np.expm1(model_xgb.predict(full_x))
    error = calc_error(full_y, xgb_preds)
    return Trainer("gradientBoosting", model_xgb, error)


def train_linear(df_train_cleaned, df_train, features_full_list):
    """

    :rtype: Trainer
    :param df_train_cleaned: cleaned dataset
    :type df_train_cleaned:  pd.DataFrame
    :param df_train: full dataset
    :type df_train:  pd.DataFrame
    :param features_full_list: full list of features
    :type features_full_list: list
    :return: logarithmic error
    """

    x = df_train_cleaned[features_full_list]
    y = df_train_cleaned['SalePrice'].values

    full_x = df_train[features_full_list]
    full_y = df_train['SalePrice'].values

    linear = linear_model.LinearRegression()
    linear.fit(x, np.log1p(y))

    preds = np.expm1(linear.predict(full_x))
    error = calc_error(full_y, preds)
    return Trainer("linear", linear, error)


def train_ridge(df_train_cleaned, df_train, features_full_list):
    """

    :rtype: Trainer
    :param df_train_cleaned: cleaned dataset
    :type df_train_cleaned:  pd.DataFrame
    :param df_train: full dataset
    :type df_train:  pd.DataFrame
    :param features_full_list: full list of features
    :type features_full_list: list
    :return: logarithmic error
    """

    x = df_train_cleaned[features_full_list]
    y = df_train_cleaned['SalePrice'].values

    full_x = df_train[features_full_list]
    full_y = df_train['SalePrice'].values

    ridge = linear_model.RidgeCV(cv=10)
    ridge.fit(x, np.log(y))

    preds = np.expm1(ridge.predict(full_x))
    error = calc_error(full_y, preds)
    return Trainer("ridge", ridge, error)


def train_lasso_lars(df_train_cleaned, df_train, features_full_list):
    """

    :rtype: Trainer
    :param df_train_cleaned: cleaned dataset
    :type df_train_cleaned:  pd.DataFrame
    :param df_train: full dataset
    :type df_train:  pd.DataFrame
    :param features_full_list: full list of features
    :type features_full_list: list
    :return: logarithmic error
    """

    x = df_train_cleaned[features_full_list]
    y = df_train_cleaned['SalePrice'].values

    full_x = df_train[features_full_list]
    full_y = df_train['SalePrice'].values

    lasso_lars = linear_model.LassoLarsCV(max_iter=10000)
    lasso_lars.fit(x, np.log(y))

    preds = np.expm1(lasso_lars.predict(full_x))
    error = calc_error(full_y, preds)
    return Trainer("lasso_lars", lasso_lars, error)


def train_elastic_net(df_train_cleaned, df_train, features_full_list):
    """

    :rtype: Trainer
    :param df_train_cleaned: cleaned dataset
    :type df_train_cleaned:  pd.DataFrame
    :param df_train: full dataset
    :type df_train:  pd.DataFrame
    :param features_full_list: full list of features
    :type features_full_list: list
    :return: logarithmic error
    """

    x = df_train_cleaned[features_full_list]
    y = df_train_cleaned['SalePrice'].values

    full_x = df_train[features_full_list]
    full_y = df_train['SalePrice'].values

    elastic_net = linear_model.ElasticNetCV(cv=10, random_state=42)
    elastic_net.fit(x, np.log(y))

    preds = np.expm1(elastic_net.predict(full_x))
    error = calc_error(full_y, preds)
    return Trainer("elastic_net", elastic_net, error)
