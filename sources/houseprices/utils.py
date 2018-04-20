import numpy as np
import pandas as pd


def get_dataset_as_df():
    return pd.read_csv('../input/train.csv')


def get_cleaned_dataset(df):
    """

    :rtype: pd.DataFrame
    :param df: data frame
    :type df: pd.DataFrame
    :return:
    """
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
    categorical = g["object"].tolist()
    numerical_int = g["int64"].tolist()
    numerical_float = g["float64"].tolist()

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


def encode_test_df(frame, dummies):
    for feature in dummies.keys():
        for cat, o in dummies[feature].items():
            if frame[feature].dtype != np.int64:
                frame.loc[frame[feature] == cat, feature] = o


def prepare_predict_df(predict_df, dummies, to_log_transform, to_pow_transform):
    encode_test_df(predict_df, dummies)
    transform_before_predict(predict_df, to_log_transform, to_pow_transform)


def fill_na(frame):
    # Alley : data description says NA means "no alley access"
    frame["Alley"].fillna("None", inplace=True)
    # BedroomAbvGr : NA most likely means 0
    frame["BedroomAbvGr"].fillna(0, inplace=True)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    frame["BsmtQual"].fillna("No", inplace=True)
    frame["BsmtCond"].fillna("No", inplace=True)
    frame["BsmtExposure"].fillna("No", inplace=True)
    frame["BsmtFinType1"].fillna("No", inplace=True)
    frame["BsmtFinType2"].fillna("No", inplace=True)
    frame["BsmtFullBath"].fillna(0, inplace=True)
    frame["BsmtHalfBath"].fillna(0, inplace=True)
    frame["BsmtUnfSF"].fillna(0, inplace=True)
    # CentralAir : NA most likely means No
    frame["CentralAir"].fillna("N", inplace=True)
    # Condition : NA most likely means Normal
    frame["Condition1"].fillna("Norm", inplace=True)
    frame["Condition2"].fillna("Norm", inplace=True)
    # EnclosedPorch : NA most likely means no enclosed porch
    frame["EnclosedPorch"].fillna(0, inplace=True)
    # External stuff : NA most likely means average
    frame["ExterCond"].fillna("TA", inplace=True)
    frame["ExterQual"].fillna("TA", inplace=True)
    # Fence : data description says NA means "no fence"
    frame["Fence"].fillna("No", inplace=True)
    # FireplaceQu : data description says NA means "no fireplace"
    frame["FireplaceQu"].fillna("No", inplace=True)
    frame["Fireplaces"].fillna(0, inplace=True)
    # Functional : data description says NA means typical
    frame["Functional"].fillna("Typ", inplace=True)
    # GarageType etc : data description says NA for garage features is "no garage"
    frame["GarageType"].fillna("No", inplace=True)
    frame["GarageFinish"].fillna("No", inplace=True)
    frame["GarageQual"].fillna("No", inplace=True)
    frame["GarageCond"].fillna("No", inplace=True)
    frame["GarageArea"].fillna(0, inplace=True)
    frame["GarageCars"].fillna(0, inplace=True)
    # HalfBath : NA most likely means no half baths above grade
    frame["HalfBath"].fillna(0, inplace=True)
    # HeatingQC : NA most likely means typical
    frame["HeatingQC"].fillna("TA", inplace=True)
    # KitchenAbvGr : NA most likely means 0
    frame["KitchenAbvGr"].fillna(0, inplace=True)
    # KitchenQual : NA most likely means typical
    frame["KitchenQual"].fillna("TA", inplace=True)
    # LotFrontage : NA most likely means no lot frontage
    frame["LotFrontage"].fillna(0, inplace=True)
    # LotShape : NA most likely means regular
    frame["LotShape"].fillna("Reg", inplace=True)
    # MasVnrType : NA most likely means no veneer
    frame["MasVnrType"].fillna("None", inplace=True)
    frame["MasVnrArea"].fillna(0, inplace=True)
    # MiscFeature : data description says NA means "no misc feature"
    frame["MiscFeature"].fillna("No", inplace=True)
    frame["MiscVal"].fillna(0, inplace=True)
    # OpenPorchSF : NA most likely means no open porch
    frame["OpenPorchSF"].fillna(0, inplace=True)
    # PavedDrive : NA most likely means not paved
    frame["PavedDrive"].fillna("N", inplace=True)
    # PoolQC : data description says NA means "no pool"
    frame["PoolQC"].fillna("No", inplace=True)
    frame["PoolArea"].fillna(0, inplace=True)
    # SaleCondition : NA most likely means normal sale
    frame["SaleCondition"].fillna("Normal", inplace=True)
    # ScreenPorch : NA most likely means no screen porch
    frame["ScreenPorch"].fillna(0, inplace=True)
    # TotRmsAbvGrd : NA most likely means 0
    frame["TotRmsAbvGrd"].fillna(0, inplace=True)
    # Utilities : NA most likely means all public utilities
    frame["Utilities"].fillna("AllPub", inplace=True)
    # WoodDeckSF : NA most likely means no wood deck
    frame["WoodDeckSF"].fillna(0, inplace=True)


def make_transform(frame):
    """
    Transform some numerical features to categorical and some categorical to numerical.
    :type frame: pd.DataFrame
    :param frame:
    """
    # Some numerical features are actually really categories
    frame.replace(
        {
            "MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                           50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                           80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                           150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
            "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        },
        inplace=True
    )
