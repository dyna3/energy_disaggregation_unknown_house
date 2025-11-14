import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# ===================== CONFIGURATION ===================== #
data_files = [
    'csv files/house2_mod.csv',
    'csv files/house10_mod.csv',
    'csv files/house4_mod.csv',
    #'csv files/house17_mod.csv',
    #'csv files/house18_mod.csv',
    'csv files/house20_mod.csv',
    'csv files/house12_mod.csv',
]

columns = ['agg', 'agg_act', 'st', 'wh', 'wm', 'fridge']
days_to_keep = 7
seconds_per_day = 86400
rows_to_keep = days_to_keep * seconds_per_day
sequence_length = 128  # LSTM sequence length

# ===================== FUNCTIONS ===================== #
def shuffle_days_across_houses(dfs, day_length=86400):
    """
    Shuffle full-day blocks across multiple houses.
    """
    all_days = []
    day_house_map = []

    # Extract days from all houses
    for house_idx, df in enumerate(dfs):
        num_days = len(df) // day_length
        days = [df.iloc[i*day_length:(i+1)*day_length] for i in range(num_days)]
        all_days.extend(days)
        day_house_map.extend([house_idx]*num_days)

    # Shuffle the days globally
    perm = np.random.permutation(len(all_days))
    shuffled_days = [all_days[i] for i in perm]
    shuffled_house_map = np.random.choice(range(len(dfs)), size=len(shuffled_days))

    # Recombine into new DataFrames for each house
    new_dfs = [pd.DataFrame(columns=dfs[0].columns) for _ in dfs]
    for day, house_idx in zip(shuffled_days, shuffled_house_map):
        new_dfs[house_idx] = pd.concat([new_dfs[house_idx], day], ignore_index=True)

    return new_dfs

def plot_house(df, house_name):
    plt.figure(figsize=(14, 5))
    for col in df.columns:
        plt.plot(df[col], label=col, linewidth=0.8)
    plt.title(f"House: {house_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def revert_sequences(sequences, seq_length):
    num_samples, _, num_features = sequences.shape
    original_length = num_samples + seq_length - 1
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))
    for i in range(num_samples):
        reconstructed[i:i+seq_length] += sequences[i]
        counts[i:i+seq_length] += 1
    reconstructed /= counts
    return reconstructed

# ===================== LOAD + PREPROCESS ===================== #
df_list = []
scalers = []   # store min/max values so we can revert predictions later

for f in data_files:
    df = pd.read_csv(f).head(rows_to_keep)
    df = df[[col for col in columns if col in df.columns]]
    df = df.reindex(columns=columns)

    # record min and max for each column
    min_vals = df.min()
    max_vals = df.max()
    scalers.append({"min": min_vals.to_dict(), "max": max_vals.to_dict()})

    # Min-Max Normalization: (x - min) / (max - min)
    df = (df - min_vals) / (max_vals - min_vals + 1e-8)

    df_list.append(df)

# Save scalers to restore original values later if needed
with open("files/minmax_scalers.json", "w") as json_file:
    json.dump(scalers, json_file)

# ===================== PLOT EACH HOUSE BEFORE SHUFFLING ===================== #
#for df, name in zip(df_list, data_files):
#    plot_house(df, name)

# ===================== SHUFFLE DAYS ACROSS TRAINING HOUSES ===================== #
# Exclude last house for testing
#df_list[:-1] = shuffle_days_across_houses(df_list[:-1], seconds_per_day)

# ===================== PLOT EACH HOUSE AFTER SHUFFLING ===================== #
"""
for df, name in zip(df_list[:-1], data_files[:-1]):
    plot_house(df, name + " (after shuffle)")
"""
# ===================== COMBINE DATA ===================== #
data = pd.concat(df_list, ignore_index=True).to_numpy()
X = data[:, :2]  # agg, agg_act
y = data[:, 2:]  # appliances

# ===================== SPLITTING ===================== #
house_segments = [(i * rows_to_keep, (i + 1) * rows_to_keep) for i in range(len(df_list))]

train_indices = np.concatenate([np.arange(*house_segments[i]) for i in range(len(df_list)-1)])

test_start = house_segments[-1][0] + 1 * seconds_per_day
test_end   = house_segments[-1][0] + 4 * seconds_per_day
test_indices = np.arange(test_start, test_end)

X_train_raw, y_train_raw = shuffle(X[train_indices], y[train_indices])
X_test_raw, y_test_raw = X[test_indices], y[test_indices]

# ===================== SEQUENCE CREATION ===================== #
X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)

input_shape = (sequence_length, X_train.shape[2])
target_shape = y_test.shape[2]

# ===================== MODEL ===================== #
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(target_shape, activation='relu'))
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=128, callbacks=[early_stop])

# ===================== EVALUATION ===================== #
eval_loss, eval_mae = model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}")

y_pred = model.predict(X_test)

y_test_reconstructed = revert_sequences(y_test, sequence_length)
y_pred_reconstructed = revert_sequences(y_pred, sequence_length)

accuracy = r2_score(y_test_reconstructed, y_pred_reconstructed)
print("R² Score Accuracy:", accuracy)

# ===================== SAVE RESULTS ===================== #
np.savetxt('csv files/predictions.csv', y_pred_reconstructed, delimiter=',')
np.savetxt('csv files/real_values.csv', y_test_reconstructed, delimiter=',')


