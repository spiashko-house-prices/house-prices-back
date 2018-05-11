import os

from flask import jsonify
from flask import make_response
from flask import request

from houseprices import app, auth
from houseprices.constants import available_trainers
from houseprices.predict_service.predict_service import predict_request_processor
from houseprices.train_service.train_service import train_request_preprocessor, train_request_processor, verify_request, \
    save_admin_model
from houseprices.utils import *


@auth.verify_password
def verify_password(username, password):
    if (username == os.environ['ADMIN_USERNAME']) and (password == os.environ['ADMIN_PASSWORD']):
        return True
    return False


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)


@app.route('/')
def index():
    return "Web server is running"


@app.route('/api/features', methods=['GET'])
@auth.login_required
def get_features():
    response_body = get_features_from_db()
    return jsonify(response_body)


@app.route('/api/methods', methods=['GET'])
@auth.login_required
def get_methods():
    return jsonify(available_trainers)


@app.route('/api/train', methods=['POST'])
@auth.login_required
def train():
    content = request.get_json()

    verify_request(content)

    encoded_dataset = db["encoded_dataset"].find({})
    encoded_dataset = pd.DataFrame(list(encoded_dataset))
    encoded_dataset.drop(columns="_id", inplace=True)

    instance_collection = db["instance"]
    train_data_size = instance_collection.find_one({"objectName": "train_data_size"})["value"]

    features_full_list = train_request_preprocessor(encoded_dataset, content)

    df_train = encoded_dataset[:train_data_size]
    df_test = encoded_dataset[train_data_size:]

    response_body = train_request_processor(df_train, df_test, features_full_list, content)

    save_admin_model(content)
    return jsonify(response_body)


@app.route('/api/admin_model', methods=['GET'])
@auth.login_required
def get_admin_model():
    instance_collection = db["instance"]
    admin_model = instance_collection.find_one({"objectName": "admin_model"})["value"]
    return jsonify(admin_model)


@app.route('/api/model', methods=['GET'])
def get_model():
    instance_collection = db["instance"]
    model_for_client = instance_collection.find_one({"objectName": "model_for_client"})["value"]
    return jsonify(model_for_client)


@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.get_json()

    prediction = predict_request_processor(content)

    response_body = {"value": prediction}
    return jsonify(response_body)
