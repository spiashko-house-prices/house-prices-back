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

to_log_transform = ["GrLivArea", "TotalBsmtSF", "MasVnrArea", "BsmtFinSF1"]

to_pow_transform = ["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "Neighborhood", "GrLivArea"]

to_boolean_transform = {
    "TotalBsmtSF": {"new_feature_name": "HasBasement", "threshold": 0},
    "GarageArea": {"new_feature_name": "HasGarage", "threshold": 0},
    "2ndFlrSF": {"new_feature_name": "Has2ndFloor", "threshold": 0},
    "MasVnrArea": {"new_feature_name": "HasMasVnr", "threshold": 0},
    "WoodDeckSF": {"new_feature_name": "HasWoodDeck", "threshold": 0},
    "OpenPorchSF": {"new_feature_name": "HasPorch", "threshold": 0},
    "PoolArea": {"new_feature_name": "HasPool", "threshold": 0},
    "YearBuilt": {"new_feature_name": "IsNew", "threshold": 2000},
}
