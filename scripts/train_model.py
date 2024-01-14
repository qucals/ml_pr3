import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


df = pd.read_csv("/home/ml-srv/ml_pr3/datasets/data_train.csv", header=None)

mapping = {df.columns[0]: "id", df.columns[1]: "counts"}
df = df.rename(columns=mapping)

df["counts"] = df["counts"].replace(to_replace=[np.nan], value=0, inplace=True)
df["counts"] = df["counts"].astype("Int64").fillna(0)

model = LinearRegression()
model.fit(df["id"].values.reshape(-1, 1), df["counts"])

with open("/home/ml-srv/ml_pr3/models/data.pickle", "wb") as f:
    pickle.dump(model, f)
