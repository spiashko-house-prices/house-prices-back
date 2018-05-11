import os

import pandas as pd

from prepare_dataset.ames.ames_utils import fill_na, make_transform, rename_columns, dealing_with_missing_data, \
    update_dots
from prepare_dataset.upload_dataset import upload

df = pd.read_csv('https://ww2.amstat.org/publications/jse/v19n3/Decock/AmesHousing.txt', sep='\t')
rename_columns(df)

fill_na(df)
make_transform(df)

df.drop(columns=['Order', 'PID'], inplace=True)

update_dots(df)
dealing_with_missing_data(df)

train_data = df[:1460]
test_data = df[1460:]

mongodb_url = os.environ['MONGODB_URI']

upload(train_data, test_data, mongodb_url)

print("done.")
