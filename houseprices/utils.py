import numpy as np
import pandas as pd

from houseprices import db


def get_train_data_as_df():
    data = db["train_data"].find({})

    df = pd.DataFrame(list(data))

    return df


def get_test_data_as_df():
    data = db["test_data"].find({})

    df = pd.DataFrame(list(data))

    return df


def dealing_with_missing_data(df_train, features):
    """

    :type features: list
    :param features:
    :rtype: list
    :type df_train: pd.DataFrame
    :param df_train:
    :return: list of applicable features
    """

    # missing data
    total = df_train[features].isnull().sum().sort_values(ascending=False)
    percent = (df_train[features].isnull().sum() / df_train[features].isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # remove features where data is missed more then 5 times
    features = list(set(features).difference(set((missing_data[missing_data['Total'] > 5]).index.tolist())))

    # remove left images where data is missed
    df_train.drop(df_train[features].loc[df_train[features].isnull().any(axis=1)].index, inplace=True)
    assert df_train[features].isnull().sum().max() == 0  # just checking that there's no missing data missing...

    return features


def perform_encoding(frame, features):
    """
    perform encoding of categorical features
    :type frame: pd.DataFrame
    :type features: list
    :rtype: dict
    :param frame:
    :param features:
    :return: dummies
    """
    dummies = {}
    for feature in features:
        ordering = pd.DataFrame()
        ordering['val'] = frame[feature].unique()
        ordering.index = ordering.val
        ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
        ordering['ordering'] = range(1, ordering.shape[0] + 1)
        ordering = ordering['ordering'].to_dict()

        for cat, o in ordering.items():
            frame.loc[frame[feature] == cat, feature] = o

        dummies[feature] = ordering
    return dummies


def prepare_model(frame, dummies, numerical_int, numerical_float, categorical, added_boolean_columns):
    """

    :type added_boolean_columns: list
    :type categorical: list
    :type numerical_float: list
    :type numerical_int: list
    :type dummies: dict
    :type frame: pd.DataFrame
    :rtype: dict
    :param frame:
    :param dummies:
    :param numerical_int:
    :param numerical_float:
    :param categorical:
    :param added_boolean_columns:
    :return: model_for_client
    """
    model_for_client = {}

    numerical_to_model = []
    for feature in numerical_int:
        numerical_to_model.append(
            {"featureName": feature, "type": "int", "min": int(frame[feature].min()),
             "max": int(frame[feature].max())})

    for feature in numerical_float:
        numerical_to_model.append(
            {"featureName": feature, "type": "float", "min": float(frame[feature].min()),
             "max": float(frame[feature].max())})

    categorical_to_model = []
    for feature in categorical:
        categorical_to_model.append({"featureName": feature, "values": list(dummies[feature].keys())})

    boolean_to_model = []
    for feature in added_boolean_columns:
        boolean_to_model.append({"featureName": feature})

    model_for_client['numerical'] = numerical_to_model
    model_for_client['categorical'] = categorical_to_model
    model_for_client['boolean'] = boolean_to_model
    model_for_client['total'] = len(numerical_int + numerical_float + categorical + added_boolean_columns)

    return model_for_client


def get_type_list(g, type_name):
    try:
        result = g[type_name].tolist()
    except KeyError:
        result = []
    return result


def get_features_by_type(df_train_cleaned, base_features):
    # get features by type
    """

    :param df_train_cleaned:
    :param base_features:
    :return: categorical, numerical_int, numerical_float
    :rtype: object
    """
    g = {k.name: v for k, v in df_train_cleaned[base_features].columns.to_series().groupby(
        df_train_cleaned[base_features].dtypes).groups.items()}

    categorical = get_type_list(g, "object")
    numerical_int = get_type_list(g, "int64")
    numerical_float = get_type_list(g, "float64")

    return categorical, numerical_int, numerical_float


def log_transformation(frame, feature):
    new_feature_name = new_log_feature_name(feature)
    frame[new_feature_name] = np.log1p(frame[feature].values)


def new_quadratic_feature_name(feature):
    return feature + '2'


def new_log_feature_name(feature):
    return feature + 'Log'


def quadratic(frame, feature):
    new_feature_name = new_quadratic_feature_name(feature)
    frame[new_feature_name] = frame[feature] ** 2


def boolean_transformation(frame, feature, new_feature_name, threshold):
    frame[new_feature_name] = frame[feature].apply(lambda x: 1 if x > threshold else 0)


def transform_before_learn(frame, to_log_transform, to_pow_transform, to_boolean_transform):
    for c in to_log_transform:
        log_transformation(frame, c)
    for c in to_pow_transform:
        quadratic(frame, c)
    for item in to_boolean_transform:
        boolean_transformation(frame, item['featureName'], item['newFeatureName'], item['threshold'])


def transform_before_predict(frame, to_log_transform, to_pow_transform):
    for c in to_log_transform:
        log_transformation(frame, c)
    for c in to_pow_transform:
        quadratic(frame, c)


def get_filtered_features_for_transform(filtered_base_features, filtered_features_to_boolean_transform, content):
    filtered_to_log_transform = [cat for cat in
                                 filter(lambda cat: cat in filtered_base_features, content['toLogTransform'])]
    filtered_to_pow_transform = [cat for cat in
                                 filter(lambda cat: cat in filtered_base_features, content['toPowTransform'])]
    filtered_to_boolean_transform = [o for o in
                                     filter(lambda o: o['featureName'] in filtered_features_to_boolean_transform,
                                            content['toBooleanTransform'])]
    return filtered_to_log_transform, filtered_to_pow_transform, filtered_to_boolean_transform


def get_added_columns(filtered_to_log_transform, filtered_to_pow_transform, filtered_features_to_boolean_transform):
    added_quadratic_columns = list(map(new_quadratic_feature_name, filtered_to_pow_transform))

    added_log_columns = list(map(new_log_feature_name, filtered_to_log_transform))

    added_boolean_columns = [o['newFeatureName'] for o in filtered_features_to_boolean_transform]

    return added_log_columns, added_quadratic_columns, added_boolean_columns


def encode_df(frame, dummies):
    for feature in dummies.keys():
        for cat, o in dummies[feature].items():
            if frame[feature].dtype != np.int64:  # it is not look like robust
                frame.loc[frame[feature] == cat, feature] = o


def prepare_predict_df(predict_df, dummies, to_log_transform, to_pow_transform):
    encode_df(predict_df, dummies)
    transform_before_predict(predict_df, to_log_transform, to_pow_transform)
