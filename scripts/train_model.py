import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression


df = pd.read_csv("/home/ml-srv/mp_pr3/datasets/data_train.csv", header=None)
df.colums = ["id", "counts"]

model = LinearRegression()
model.fit(df["id"].values.reshape(-1, 1), df["counts"])

with open("/home/ml-srv/ml_pr3/models/data.pickle", "wb") as f:
    pickle.dump(model, f)
