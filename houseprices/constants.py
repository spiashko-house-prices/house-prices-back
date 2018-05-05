import os

from houseprices import app

path_to_client_model = os.path.join(app.instance_path, 'model_for_client.pickle.bin')
path_to_prediction_stuff = os.path.join(app.instance_path, 'prediction_stuff.pickle.bin')

available_trainers = ["gradientBoosting", "linear", "ridge", "lasso", "elastic_net"]


def get_trainer_path(trainer_name):
    return os.path.join(app.instance_path, '{}.pickle.bin'.format(trainer_name))
