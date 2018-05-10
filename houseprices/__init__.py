import os
import numpy as np
import pandas as pd

from flask import Flask
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from pymongo import MongoClient
# import warnings
# warnings.filterwarnings('ignore')
from houseprices.utils import encode_on_startup

MONGODB_URL = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URL)
db = client.get_database()

encode_on_startup()

app = Flask(__name__)
cors = CORS(app)
auth = HTTPBasicAuth()

import houseprices.controllers
