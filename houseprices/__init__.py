import os

from flask import Flask
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from pymongo import MongoClient

MONGODB_URL = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URL)
db = client.get_default_database()

app = Flask(__name__)
cors = CORS(app)
auth = HTTPBasicAuth()

import houseprices.controllers
