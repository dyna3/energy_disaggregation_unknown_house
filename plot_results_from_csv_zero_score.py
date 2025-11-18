import numpy as np
from plot_utils import plot_results
from sklearn.metrics import r2_score
import json
import pandas as pd

# ========== INVERSE STANDARDIZATION FUNCTION ==========
def invert_standardization(data, mean_vals, std_vals):
    """Revert Z-score normalization."""
    return data * std_vals + mean_vals

# ========== LOAD PREDICTIONS AND GROUND TRUTH ==========
y_pred = np.genfromtxt('csv files/predictions_masked_ssl.csv', delimiter=',', skip_header=0)
y_test = np.genfromtxt('csv files/real_values_masked_ssl.csv', delimiter=',', skip_header=0)

"""
# ========== LOAD SCALERS (new Z-score format) ==========
with open("files/standard_scalers.json", "r") as json_file:
    scalers = json.load(json_file)

# Use last house (test house) scaler for inversion
test_scaler = scalers[-1]
mean_vals = np.array([test_scaler["mean"][col] for col in ['st', 'wh', 'wm', 'fridge']])
std_vals  = np.array([test_scaler["std"][col]  for col in ['st', 'wh', 'wm', 'fridge']])

# ========== INVERT NORMALIZATION ==========
y_pred = invert_standardization(y_pred, mean_vals, std_vals)
y_test = invert_standardization(y_test, mean_vals, std_vals)
"""
# ========== METRICS ==========
accuracy = r2_score(y_test, y_pred)
print("R² Score:", accuracy)
print("Shapes: y_pred =", y_pred.shape, ", y_test =", y_test.shape)

# ========== PLOT RESULTS ==========
plot_results(y_test, y_pred)
