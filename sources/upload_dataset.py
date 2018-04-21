import os
from pymongo import MongoClient
import numpy as np
import pandas as pd

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
MONGODB_URL = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URL)
# Issue the serverStatus command and print the results
db = client.get_default_database()

houses = db["houses"]

df = pd.read_csv('../input/train.csv')

houses.insert_many(df.to_dict('records'))