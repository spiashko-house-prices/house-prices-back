import pickle

import numpy as np
import pandas as pd

from houseprices.constants import path_to_prediction_stuff, get_trainer_path
from houseprices.utils import prepare_predict_df


def predict_request_processor(content):
    predict_df = pd.DataFrame()
    for item in content:
        predict_df[item["featureName"]] = [item["value"]]

    prediction_stuff = pickle.load(open(path_to_prediction_stuff, "rb"))
    features_full_list = prediction_stuff["features_full_list"]
    dummies = prediction_stuff["dummies"]
    to_log_transform = prediction_stuff["to_log_transform"]
    to_pow_transform = prediction_stuff["to_pow_transform"]
    methods = prediction_stuff["methods"]
    prepare_predict_df(predict_df, dummies, to_log_transform, to_pow_transform)

    methods_df = pd.DataFrame()
    methods_df['name'] = [o['name'] for o in methods]
    methods_df['value'] = [o['value'] for o in methods]

    prediction = 0
    # check that values sum is 1
    values_sum = 0.
    for trainer in methods_df['name']:
        value = float(methods_df.loc[methods_df['name'] == trainer]['value'])
        values_sum += value
        trainer = pickle.load(open(get_trainer_path(trainer), "rb"))
        prediction = prediction + float(
            np.around(np.expm1(trainer.model.predict(predict_df[features_full_list])), decimals=0)) * value

    return prediction
