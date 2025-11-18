import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

# ===================== CONFIGURATION ===================== #
data_files = [
    'csv files/house2_mod.csv',
    'csv files/house12_mod.csv',
    'csv files/house4_mod.csv',
    'csv files/house20_mod.csv',
    'csv files/house15_mod.csv',
    'csv files/house10_mod.csv',
]

columns = ['agg', 'agg_act', 'st', 'wh', 'wm', 'fridge']
days_to_keep = 7
seconds_per_day = 86400
rows_to_keep = days_to_keep * seconds_per_day
sequence_length = 128  # LSTM sequence length

# SSL hyperparams
mask_prob = 0.15
ssl_epochs = 5
ssl_batch_size = 128

# Fine-tuning hyperparams
fine_tune_epochs = 5
ft_batch_size = 128

os.makedirs("files", exist_ok=True)

# ===================== UTIL FUNCTIONS ===================== #
def shuffle_days_across_houses(dfs, day_length=86400):
    """
    Shuffle whole days across houses and return list of new dfs (same length as input).
    """
    all_days = []
    house_maps = []
    for house_idx, df in enumerate(dfs):
        num_days = len(df) // day_length
        days = [df.iloc[i*day_length:(i+1)*day_length].reset_index(drop=True) for i in range(num_days)]
        all_days.extend(days)
        house_maps.extend([house_idx] * len(days))

    perm = np.random.permutation(len(all_days))
    shuffled_days = [all_days[i] for i in perm]
    shuffled_house_map = np.random.choice(range(len(dfs)), size=len(shuffled_days))

    new_dfs = [pd.DataFrame(columns=dfs[0].columns) for _ in dfs]
    for day, house_idx in zip(shuffled_days, shuffled_house_map):
        new_dfs[house_idx] = pd.concat([new_dfs[house_idx], day], ignore_index=True)
    return new_dfs

def create_sequences(X, y, seq_length):
    """
    Create overlapping sequences from continuous arrays X and y.
    Returns: X_seq: (n_samples, seq_length, n_features_X)
             y_seq: (n_samples, seq_length, n_features_y)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def revert_sequences(sequences, seq_length):
    """
    Reconstruct the original long sequence from overlapping sequences by averaging overlaps.
    sequences: (n_samples, seq_length, n_features)
    Returns: reconstructed array shape (original_length, n_features)
    """
    num_samples, seq_len, num_features = sequences.shape
    original_length = num_samples + seq_len
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))

    for i in range(num_samples):
        reconstructed[i:i+seq_len] += sequences[i]
        counts[i:i+seq_len] += 1

    # avoid division by zero
    counts[counts == 0] = 1
    reconstructed = reconstructed[:original_length- (seq_len - 1)] / counts[:original_length- (seq_len - 1)]
    return reconstructed

def make_masked_sequences(data, seq_len=128, mask_prob=0.15):
    """
    For contrastive/SSL masked reconstruction: create sequences and zero out random elements
    with probability mask_prob. Masking applied per-element (per timestep x per feature).
    data: numpy array shape (timesteps, features)
    """
    X_in, y_target = [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len].copy()
        target = data[i:i+seq_len].copy()
        mask = np.random.rand(*seq.shape) < mask_prob
        seq[mask] = 0.0
        X_in.append(seq)
        y_target.append(target)
    return np.array(X_in), np.array(y_target)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def inverse_transform_array(arr, scaler_dict, cols):
    """
    Inverse transform a 2D numpy arr (timesteps, n_features) using scaler_dict that contains
    mean/std for the columns (keys in 'cols' order). scaler_dict should have "mean" and "std"
    mapping column names to numbers.
    """
    mean = np.array([scaler_dict["mean"][c] for c in cols])
    std = np.array([scaler_dict["std"][c] for c in cols])
    return arr * (std + 1e-8) + mean

# ===================== LOAD + STANDARDIZE (Z-SCORE) ===================== #
df_list = []
scalers = []   # store mean/std values per file so we can revert predictions later

for f in data_files:
    df = pd.read_csv(f).head(rows_to_keep)
    # keep only columns that exist in file and align order
    df = df[[col for col in columns if col in df.columns]]
    df = df.reindex(columns=columns)  # ensure consistent order; missing columns will be NaN

    # compute mean and std per column (skip NaNs)
    mean_vals = df.mean()
    std_vals = df.std(ddof=0)  # population std; you can use ddof=1 if you prefer sample std
    scalers.append({"mean": mean_vals.fillna(0).to_dict(), "std": std_vals.fillna(1).to_dict()})

    # Z-score normalization: (x - mean) / std
    df = (df - mean_vals) / (std_vals + 1e-8)

    # If original data had NaNs (missing columns), fill with zeros (already standardized to 0)
    df = df.fillna(0.0)

    df_list.append(df)

# Save scalers to restore original values later if needed
save_json("files/standard_scalers.json", scalers)

# ===================== COMBINE DATA ===================== #
data = pd.concat(df_list, ignore_index=True).to_numpy()
# first two columns are aggregates (agg, agg_act)
X = data[:, :2]  # agg, agg_act
y = data[:, 2:]  # appliances (st, wh, wm, fridge)

# ===================== SPLITTING ===================== #
# compute house segments based on how many rows we loaded per house (rows_to_keep)
house_segments = [(i * rows_to_keep, (i + 1) * rows_to_keep) for i in range(len(df_list))]

# train on all houses except last one
train_indices = np.concatenate([np.arange(*house_segments[i]) for i in range(len(df_list)-1)])

# test (unseen) house = last house; pick days inside it for test/unlabeled
test_house_start, test_house_end = house_segments[-1]

# choose a chunk of days within last house for SSL + evaluation
# Here we use days 1..4 (adjust if rows_to_keep is different)
test_start = test_house_start + 1 * seconds_per_day
test_end   = test_house_start + 4 * seconds_per_day
test_indices = np.arange(test_start, test_end)

# Prepare train / test raw arrays
X_train_raw, y_train_raw = shuffle(X[train_indices], y[train_indices], random_state=42)
X_test_raw, y_test_raw = X[test_indices], y[test_indices]

# ===================== SSL DATA: masked reconstruction on UNLABELED target aggregates ===================== #
unlabeled_target = X_test_raw  # shape (timesteps, 2)
print("Unlabeled target shape (timesteps,features):", unlabeled_target.shape)

# Create masked reconstruction sequences
X_ssl, y_ssl = make_masked_sequences(unlabeled_target, seq_len=sequence_length, mask_prob=mask_prob)
print("SSL dataset shapes X:", X_ssl.shape, " y:", y_ssl.shape)

# ===================== CREATE SEQUENCES FOR SUPERVISED TRAINING ===================== #
X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)

# ensure numpy float32 for TF
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
X_ssl = X_ssl.astype(np.float32)
y_ssl = y_ssl.astype(np.float32)

input_shape = (sequence_length, X_train.shape[2])
target_shape = y_test.shape[2]

print("Supervised train shape:", X_train.shape, y_train.shape)
print("Supervised test shape:", X_test.shape, y_test.shape)
print("Input shape:", input_shape, "Target dim:", target_shape)

# ===================== DEFINE SSL MODEL (masked reconstruction) ===================== #
ssl_input = Input(shape=input_shape, name="ssl_input")

# Encoder
x = layers.Conv1D(128, 3, activation='relu', padding='same', name='enc_conv1')(ssl_input)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='enc_bilstm1')(x)
x = layers.Dropout(0.2, name='enc_dropout1')(x)
x = layers.LSTM(64, return_sequences=True, name='enc_lstm2')(x)
x = layers.Dropout(0.2, name='enc_dropout2')(x)
encoded = layers.LSTM(32, return_sequences=True, name='enc_lstm3')(x)  # keep sequence

# Decoder
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='dec_bilstm1')(encoded)
x = layers.Dropout(0.2, name='dec_dropout1')(x)
decoded = layers.TimeDistributed(layers.Dense(input_shape[1], activation='linear'), name='decoded')(x)

ssl_model = models.Model(ssl_input, decoded, name='ssl_model')
ssl_model.compile(optimizer='adam', loss='mse')
ssl_model.summary()

# ===================== RUN SSL PRETRAINING ===================== #
print("Starting SSL pretraining...")
ssl_history = ssl_model.fit(X_ssl, y_ssl, epochs=ssl_epochs, batch_size=ssl_batch_size, validation_split=0.1)

# Save encoder weights (we will copy them into NILM model)
encoder_model = models.Model(ssl_input, encoded)
encoder_weights_path = "files/encoder_masked_pretrained_weights.weights.h5"
encoder_model.save_weights(encoder_weights_path)
print(f"Saved encoder weights to {encoder_weights_path}")

# ===================== BUILD NILM MODEL (same-ish architecture) ===================== #
nilm_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape, name='enc_conv1_seq'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), name='enc_bilstm1_seq'),
    tf.keras.layers.Dropout(0.2, name='enc_dropout1_seq'),
    tf.keras.layers.LSTM(64, return_sequences=True, name='enc_lstm2_seq'),
    tf.keras.layers.Dropout(0.2, name='enc_dropout2_seq'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), name='enc_lstm3_seq'),
    tf.keras.layers.Dropout(0.2, name='nilm_dropout_after_enc'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(target_shape, activation='linear'), name='nilm_output')
])

nilm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
nilm_model.summary()

# ===================== COPY ENCODER WEIGHTS FROM ssl_model -> nilm_model ===================== #
# Best-effort copying by matching layer names where possible.
print("Copying encoder weights from SSL model to NILM model (best-effort)...")
copied = 0

# load encoder weights into a temporary model to inspect
# We saved encoder_model weights earlier; create a model with same architecture as encoder_model and load weights
tmp_encoder = models.Model(ssl_input, encoded)
tmp_encoder.load_weights(encoder_weights_path)

# Make a map of ssl encoder layers by name
ssl_enc_layers_by_name = {l.name: l for l in tmp_encoder.layers}

for i, nl in enumerate(nilm_model.layers):
    # try to find a ssl layer with a similar name
    possible_names = []
    # map common prefixes
    if nl.name.startswith('enc_conv1_seq'):
        possible_names = ['enc_conv1']
    elif nl.name.startswith('enc_bilstm1_seq'):
        possible_names = ['enc_bilstm1']
    elif nl.name.startswith('enc_dropout1_seq'):
        possible_names = ['enc_dropout1']
    elif nl.name.startswith('enc_lstm2_seq'):
        possible_names = ['enc_lstm2']
    elif nl.name.startswith('enc_dropout2_seq'):
        possible_names = ['enc_dropout2']
    elif nl.name.startswith('enc_lstm3_seq'):
        possible_names = ['enc_lstm3']
    else:
        possible_names = []

    for pname in possible_names:
        if pname in ssl_enc_layers_by_name:
            ssl_layer = ssl_enc_layers_by_name[pname]
            try:
                # attempt direct weights copy
                if len(ssl_layer.get_weights()) > 0 and len(nl.get_weights()) > 0:
                    nl.set_weights(ssl_layer.get_weights())
                    copied += 1
                    print(f"Copied weights: {pname} -> {nl.name}")
            except Exception as e:
                # handle Bidirectional wrappers specially
                try:
                    if isinstance(nl, tf.keras.layers.Bidirectional) and isinstance(ssl_layer, tf.keras.layers.Bidirectional):
                        nl.forward_layer.set_weights(ssl_layer.forward_layer.get_weights())
                        nl.backward_layer.set_weights(ssl_layer.backward_layer.get_weights())
                        copied += 1
                        print(f"Copied Bidirectional inner layers for {nl.name}")
                except Exception as e2:
                    print(f"Failed to copy weights from {pname} to {nl.name}: {e2}")

print(f"Total encoder layers copied (approx): {copied}")

# ===================== FINE-TUNE NILM MODEL ON LABELED SOURCE HOUSES ===================== #
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("Starting fine-tuning on labeled source houses...")
nilm_history = nilm_model.fit(X_train, y_train, validation_split=0.1, epochs=fine_tune_epochs, batch_size=ft_batch_size, callbacks=[early_stop])

# ===================== EVALUATION ===================== #
eval_loss, eval_mae = nilm_model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}")

y_pred = nilm_model.predict(X_test)

y_test_reconstructed = revert_sequences(y_test, sequence_length)
y_pred_reconstructed = revert_sequences(y_pred, sequence_length)

# ===================== INVERSE TRANSFORM BACK TO ORIGINAL UNITS ===================== #
# The test set came from the last house file (data_files[-1]) so use scalers[-1]
test_house_scaler = scalers[-1]  # contains "mean" and "std" dicts
target_cols = columns[2:]  # appliances columns order: ['st','wh','wm','fridge']

# y_test_reconstructed and y_pred_reconstructed have shape (timesteps, n_targets)
y_test_orig = inverse_transform_array(y_test_reconstructed, test_house_scaler, target_cols)
y_pred_orig = inverse_transform_array(y_pred_reconstructed, test_house_scaler, target_cols)

# compute R^2 on original units
accuracy = r2_score(y_test_orig, y_pred_orig)
print("R² Score (original units):", accuracy)

# ===================== SAVE RESULTS ===================== #
np.savetxt('csv files/predictions_masked_ssl.csv', y_pred_orig, delimiter=',')
np.savetxt('csv files/real_values_masked_ssl.csv', y_test_orig, delimiter=',')
print("Saved results to 'csv files/predictions_masked_ssl.csv' and 'csv files/real_values_masked_ssl.csv'")
