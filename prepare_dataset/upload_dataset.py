import pandas as pd
from pymongo import MongoClient


def upload(df_train, df_test, mongodb_url):
    """

    :type mongodb_url: str
    :param mongodb_url:
    :type df_train: pd.DataFrame
    :param df_train:
    :type df_test: pd.DataFrame
    :param df_test:
    :return:
    """
    client = MongoClient(mongodb_url)
    db = client.get_default_database()

    train_data = db["train_data"]

    train_data.remove({})
    train_data.insert_many(df_train.to_dict('records'))

    test_data = db["test_data"]

    test_data.remove({})
    test_data.insert_many(df_test.to_dict('records'))

