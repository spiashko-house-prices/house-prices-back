import pickle

from flask import jsonify
from flask import request

from houseprices import app
from houseprices.constants import path_to_client_model
from houseprices.predict_service.predict_service import predict_request_processor
from houseprices.train_service.train_service import train_request_preprocessor, train_request_processor, verify_request
from houseprices.utils import *


@app.route('/')
def index():
    return "Web server is running"


@app.route('/api/features', methods=['GET'])
def get_features():
    response_body = get_dataset_as_df().columns.tolist()
    response_body.remove('Id')
    response_body.remove('SalePrice')
    response_body.remove('_id')
    return jsonify(response_body)


@app.route('/api/methods', methods=['GET'])
def get_methods():
    response_body = [
        'gradientBoosting',
        'linear',
        'ridge',
        'lasso_lars',
        'elastic_net'
    ]
    return jsonify(response_body)


@app.route('/api/train', methods=['POST'])
def train():
    content = request.get_json()

    verify_request(content)

    df_train = get_dataset_as_df()
    df_train_cleaned = get_cleaned_dataset(df_train)

    features_full_list = train_request_preprocessor(df_train_cleaned, content)

    response_body = train_request_processor(df_train_cleaned, df_train, features_full_list, content)
    return jsonify(response_body)


@app.route('/api/model', methods=['GET'])
def get_model():
    instance_collection = db["instance"]
    model_for_client = pickle.loads(instance_collection.find_one({"objectName": "model_for_client"})["value"])
    return jsonify(model_for_client)


@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.get_json()

    prediction = predict_request_processor(content)

    response_body = {"value": prediction}
    return jsonify(response_body)
