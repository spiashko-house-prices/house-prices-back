import os
from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient

MONGODB_URL = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URL)
# Issue the serverStatus command and print the results
db = client.get_default_database()

app = Flask(__name__)
cors = CORS(app)

import houseprices.controllers
