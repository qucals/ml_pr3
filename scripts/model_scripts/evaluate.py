import os
import sys
import pickle
import json

import pandas as pd

from sklearn.metrics import r2_score

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 evaluate.py data-file.csv model\n")
    sys.exit(-1)
    
df = pd.read_csv(sys.argv[1], header=0)
X = df.iloc[0:, [1, 2, 3, 4]]
y = df.iloc[0:, 5]

with open(sys.argv[2], "rb") as fd:
    model = pickle.load(fd)
    
y_pred = model.predict(X)
r2score = r2_score(y, y_pred)
    
prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": r2score}, fd)