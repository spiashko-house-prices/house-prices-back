import pickle

from flask import jsonify
from flask import request

from houseprices import app
from houseprices.constants import available_trainers
from houseprices.predict_service.predict_service import predict_request_processor
from houseprices.train_service.train_service import train_request_preprocessor, train_request_processor, verify_request, \
    save_admin_model
from houseprices.utils import *


@app.route('/')
def index():
    return "Web server is running"


@app.route('/api/features', methods=['GET'])
def get_features():
    response_body = get_train_data_as_df().columns.tolist()
    response_body.remove('SalePrice')
    response_body.remove('_id')
    return jsonify(response_body)


@app.route('/api/methods', methods=['GET'])
def get_methods():
    return jsonify(available_trainers)


@app.route('/api/train', methods=['POST'])
def train():
    content = request.get_json()

    verify_request(content)

    df_train = get_train_data_as_df()
    df_test = get_test_data_as_df()
    features_full_list = train_request_preprocessor(df_train, df_test, content)

    response_body = train_request_processor(df_train, df_train, features_full_list, content)

    save_admin_model(content)
    return jsonify(response_body)


@app.route('/api/admin_model', methods=['GET'])
def get_admin_model():
    instance_collection = db["instance"]
    admin_model = instance_collection.find_one({"objectName": "admin_model"})["value"]
    return jsonify(admin_model)


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
