import pickle
import pandas as pd
import numpy as np
import mlflow

from sklearn.linear_model import LinearRegression
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

df = pd.read_csv("/home/ml-srv/ml_pr3/datasets/data_train.csv", header=None)

mapping = {df.columns[0]: "id", df.columns[1]: "counts"}
df = df.rename(columns=mapping)

df["counts"] = df["counts"].replace(to_replace=[np.nan], value=0, inplace=True)
df["counts"] = df["counts"].astype("Int64").fillna(0)

model = LinearRegression()

with mlflow.start_run():
    mlflow.sklearn.log_model(model, artifact_path="lr", registered_model_name="lr")
    mlflow.log_artifact(
        local_path="/home/ml-srv/ml_pr3/scripts/train_model.py",
        artifact_path="train_model code",
    )
    mlflow.end_run()

model.fit(df["id"].values.reshape(-1, 1), df["counts"])

with open("/home/ml-srv/ml_pr3/models/data.pickle", "wb") as f:
    pickle.dump(model, f)
