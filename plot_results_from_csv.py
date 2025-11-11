import numpy as np
from plot_utils import plot_results
from sklearn.metrics import r2_score
import json
import pandas as pd

# ========== INVERSE MIN-MAX FUNCTION ==========
def invert_minmax(data, min_vals, max_vals):
    return data * (max_vals - min_vals + 1e-8) + min_vals

y_pred = np.genfromtxt('csv files/predictions.csv', delimiter=',', skip_header=0)
y_test = np.genfromtxt('csv files/real_values.csv', delimiter=',', skip_header=0)

# ========== LOAD SCALERS ==========
with open("files/minmax_scalers.json", "r") as json_file:
    scalers = json.load(json_file)

# The last house (test house) scaler is used for inversion
test_scaler = scalers[-1]   # <- house10 scaler
min_vals = np.array([test_scaler["min"][col] for col in ['st', 'wh', 'wm', 'fridge']])
max_vals = np.array([test_scaler["max"][col] for col in ['st', 'wh', 'wm', 'fridge']])

# ========== APPLY TO PREDICTIONS & REAL VALUES ==========
y_pred = invert_minmax(y_pred, min_vals, max_vals)
y_test = invert_minmax(y_test, min_vals, max_vals)

accuracy = r2_score(y_test, y_pred)

print(accuracy)
print(y_pred.shape, y_test.shape)
plot_results(y_test, y_pred)