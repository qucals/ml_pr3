import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression


df = pd.read_csv('/home/ml-srv/ml_pr3/datasets/data_test.csv', header=None)
df.column = ['id', 'counts']