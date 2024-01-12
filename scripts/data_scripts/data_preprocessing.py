import sys
import os
import yaml

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 data_preprocessing.py *data_file.csv*\n")
    sys.exit(-1)

f_input = sys.argv[1]
f_train_output = os.path.join("data", "baselines", "train.csv")
f_test_output = os.path.join("data", "baselines", "test.csv")
os.makedirs(os.path.join("data", "baselines"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["split"]
split_ratio = params["split_ratio"]

# df = pd.read_csv("../../data/raw/housing_price_dataset.csv")
df = pd.read_csv(f_input)

encoder = LabelEncoder()
df['Neighborhood'] = encoder.fit_transform(df['Neighborhood'])
df.head()

df.drop(df[df.Price <= 0].index, axis = 0, inplace = True)

ds = df.drop(columns = ["YearBuilt"], axis = 1)
ds_train, ds_test = train_test_split(ds, test_size=split_ratio, random_state=42)

# ds_train.to_csv("../../data/baselines/train.csv")
# ds_test.to_csv("../../data/baselines/test.csv")

ds_train.to_csv(f_train_output)
ds_test.to_csv(f_test_output)