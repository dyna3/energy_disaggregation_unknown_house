import pandas as pd
import numpy as np
import json
from plot_utils import plot_columns, plot_features
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.dtensor.python.numpy_util import to_numpy

# Load and preprocess
data_files = ['csv files/house2_mod.csv',
              'csv files/house3_mod.csv',
              'csv files/house10_mod.csv',
              ]

columns = ['agg', 'agg_act', 'st', 'wh', 'wm', 'fridge']
df_list = []
agg_means = []
days_to_keep = 8
rows_to_keep = days_to_keep * 86400  # 1 day = 86400 seconds

for f in data_files:
    df = pd.read_csv(f).head(rows_to_keep)
    df = df[[col for col in columns if col in df.columns]]
    df = df.reindex(columns=columns)  # pad missing with NaN
    mean_agg = df["agg"].mean()
    agg_means.append(mean_agg)
    df = df / mean_agg  # normalize all columns
    df_list.append(df)

#plot_features(df_list, days_to_keep)

# Save agg means to JSON
with open("files/agg_means.json", "w") as json_file:
    json.dump(agg_means, json_file)


# Combine data from different houses
data = pd.concat(df_list, axis=0, ignore_index=True).to_numpy()

# Extract features and targets
X = data[:, :2]  # agg, agg_act
y = data[:, 2:]  # st, wh, wm, fridge

# Split into house segments
house_segments = [(i * rows_to_keep, (i + 1) * rows_to_keep) for i in range(len(data_files))]
train_indices = np.concatenate([np.arange(*house_segments[i]) for i in range(len(data_files)-1)])
#eval_indices = np.arange(*house_segments[4])
# Test: select specific days from last house in data_files list
test_start = house_segments[2][0] + 1 * 86400 # Start of day 3
test_end = house_segments[2][0] + 4 * 86400    # End of day 4
test_indices = np.arange(test_start, test_end)

X_train_raw, y_train_raw = X[train_indices], y[train_indices]
#X_eval_raw, y_eval_raw = X[eval_indices], y[eval_indices]
X_test_raw, y_test_raw = X[test_indices], y[test_indices]

#plot_columns(X_test_raw)
#plot_columns(y_test_raw)


# Function to create sequences for LSTM
def create_sequences(X, y, seq_length=128):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 128  # seconds
X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
#X_eval, y_eval = create_sequences(X_eval_raw, y_eval_raw, sequence_length)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)

print(X_train.shape, y_train.shape)
input_shape = (sequence_length, X_train.shape[2])
target_shape = y_test.shape[2]

# Build LSTM model
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(target_shape, activation='relu'))
        ])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_split = 0.1, epochs=5, batch_size=128, callbacks=[early_stop])

# Evaluate
eval_loss, eval_mae = model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}")

# Predict on test data
y_pred = model.predict(X_test)


def revert_sequences(sequences, seq_length):
    """
    Revert the sequence creation process to reconstruct the original signal.

    Args:
        sequences (numpy array): Predicted or ground truth sequences,
                                 shape (num_samples, seq_length, num_features).
        seq_length (int): Length of each sequence window.

    Returns:
        numpy array: Reconstructed original signal, shape (original_length, num_features).
    """
    # Initialize an array to store the reconstructed signal
    num_samples, _, num_features = sequences.shape
    original_length = num_samples + seq_length - 1
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))  # Track the number of overlaps at each position

    for i in range(num_samples):
        reconstructed[i:i + seq_length] += sequences[i]
        counts[i:i + seq_length] += 1

    # Average overlapping regions
    reconstructed /= counts
    return reconstructed

y_test_reconstructed = revert_sequences(y_test, sequence_length)
y_pred_reconstructed = revert_sequences(y_pred, sequence_length)

accuracy = r2_score(y_test_reconstructed, y_pred_reconstructed)
print("accuracy:", accuracy)

# save files
np.savetxt('csv files/predictions.csv', y_pred_reconstructed, delimiter=',')
np.savetxt('csv files/real_values.csv', y_test_reconstructed, delimiter=',')
