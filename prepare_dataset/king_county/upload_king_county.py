import os

import pandas as pd

from prepare_dataset.king_county.king_county_utils import parse_date, make_transform
from prepare_dataset.upload_dataset import upload

# source https://www.kaggle.com/harlfoxem/housesalesprediction/

df = pd.read_csv("kc_house_data.csv")

parse_date(df)
make_transform(df)

df.drop(columns=['id', "date"], inplace=True)

df.rename(index=str, columns={"price": "SalePrice"}, inplace=True)

train_data = df[:11000]
test_data = df[11000:]

mongodb_url = os.environ['MONGODB_URI']

upload(train_data, test_data, mongodb_url)

print("done.")
