import numpy as np
from plot_utils import plot_results
from sklearn.metrics import r2_score

y_pred = np.genfromtxt('csv files/predictions.csv', delimiter=',', skip_header=0)
y_test = np.genfromtxt('csv files/real_values.csv', delimiter=',', skip_header=0)
accuracy = r2_score(y_test, y_pred)

print(accuracy)
print(y_pred.shape, y_test.shape)
plot_results(y_test, y_pred)