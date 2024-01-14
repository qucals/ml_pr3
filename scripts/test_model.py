import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression


df = pd.read_csv("/home/ml-srv/ml_pr3/datasets/data_test.csv", header=None)
mapping = {df.columns[0]: "id", df.columns[1]: "counts"}
df = df.rename(columns=mapping)

model = LinearRegression()

with open("/home/ml-srv/ml_pr3/models/data.pickle", "rb") as f:
    model = pickle.load(f)

score = model.score(df["id"].values.reshape(-1, 1), df["counts"])
print("score=", score)
