import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load and preprocess
data_files = ['csv files/house2_mod.csv',
              'csv files/house3_mod.csv',
              'csv files/house10_mod.csv',
              'csv files/house12_mod.csv',
              'csv files/house15_mod.csv',
              'csv files/house16_mod.csv']

columns = ['agg', 'agg_act', 'st', 'wh', 'wm', 'fridge']
df_list = []
days_to_keep = 15
rows_to_keep = days_to_keep * 86400  # 1 day = 86400 seconds

for f in data_files:
    df = pd.read_csv(f).head(rows_to_keep)
    df = df[[col for col in columns if col in df.columns]]
    df = df.reindex(columns=columns)  # pad missing with NaN
    df_list.append(df)


# Combine and normalize
all_data = pd.concat(df_list, axis=0, ignore_index=True)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(all_data)

# Extract features and targets
X_all = normalized_data[:, :2]  # agg, agg_act
y_all = normalized_data[:, 2:]  # st, wh, wm, fridge

# Split into house segments
house_segments = [(i * rows_to_keep, (i + 1) * rows_to_keep) for i in range(len(data_files))]
train_indices = np.concatenate([np.arange(*house_segments[i]) for i in range(len(data_files)-2)])
eval_indices = np.arange(*house_segments[4])
test_indices = np.arange(house_segments[5][1] - 2 * 86400, house_segments[5][1])  # last 2 days of house16

X_train_raw, y_train_raw = X_all[train_indices], y_all[train_indices]
X_eval_raw, y_eval_raw = X_all[eval_indices], y_all[eval_indices]
X_test_raw, y_test_raw = X_all[test_indices], y_all[test_indices]


# Function to create sequences for LSTM
def create_sequences(X, y, seq_length=64):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 64  # 60 seconds
X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
X_eval, y_eval = create_sequences(X_eval_raw, y_eval_raw, sequence_length)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)

print(X_train.shape, y_train.shape)


# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])  # 4 outputs: st, wh, wm, fridge
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_split = 0.1,
          epochs=2, batch_size=128,
          callbacks=[early_stop])

# Evaluate
eval_loss, eval_mae = model.evaluate(X_eval, y_eval)
print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}")

# Predict on test data
y_pred = model.predict(X_test)
