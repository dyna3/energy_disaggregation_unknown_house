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
    'csv files/house20_mod.csv',
    'csv files/house12_mod.csv',
]

columns = ['agg', 'agg_act', 'st', 'wh', 'wm', 'fridge']
days_to_keep = 7
seconds_per_day = 86400
rows_to_keep = days_to_keep * seconds_per_day
sequence_length = 128  # LSTM sequence length

# SSL hyperparams
mask_prob = 0.15
ssl_epochs = 2
ssl_batch_size = 128

# Fine-tuning hyperparams
fine_tune_epochs = 2
ft_batch_size = 128

# ===================== FUNCTIONS ===================== #
def shuffle_days_across_houses(dfs, day_length=86400):
    all_days = []
    for house_idx, df in enumerate(dfs):
        num_days = len(df) // day_length
        days = [df.iloc[i*day_length:(i+1)*day_length] for i in range(num_days)]
        all_days.extend(days)
    perm = np.random.permutation(len(all_days))
    shuffled_days = [all_days[i] for i in perm]
    shuffled_house_map = np.random.choice(range(len(dfs)), size=len(shuffled_days))
    new_dfs = [pd.DataFrame(columns=dfs[0].columns) for _ in dfs]
    for day, house_idx in zip(shuffled_days, shuffled_house_map):
        new_dfs[house_idx] = pd.concat([new_dfs[house_idx], day], ignore_index=True)
    return new_dfs

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

def make_masked_sequences(data, seq_len=128, mask_prob=0.15):
    X_in, y_target = [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len].copy()
        target = data[i:i+seq_len].copy()
        # mask per-element with mask_prob
        mask = np.random.rand(*seq.shape) < mask_prob
        seq[mask] = 0.0   # zero-out masked positions
        X_in.append(seq)
        y_target.append(target)
    return np.array(X_in), np.array(y_target)

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

# ===================== COMBINE DATA ===================== #
data = pd.concat(df_list, ignore_index=True).to_numpy()
X = data[:, :2]  # agg, agg_act
y = data[:, 2:]  # appliances

# ===================== SPLITTING ===================== #
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
X_train_raw, y_train_raw = shuffle(X[train_indices], y[train_indices])
X_test_raw, y_test_raw = X[test_indices], y[test_indices]

# ===================== SSL DATA: masked reconstruction on UNLABELED target aggregates ===================== #
# We will use the aggregate channels only for SSL (agg, agg_act)
unlabeled_target = X_test_raw  # shape (timesteps, 2)
print("Unlabeled target shape (timesteps,features):", unlabeled_target.shape)

# Create masked reconstruction sequences
X_ssl, y_ssl = make_masked_sequences(unlabeled_target, seq_len=sequence_length, mask_prob=mask_prob)
print("SSL dataset shapes X:", X_ssl.shape, " y:", y_ssl.shape)

# ===================== CREATE SEQUENCES FOR SUPERVISED TRAINING ===================== #
X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)

input_shape = (sequence_length, X_train.shape[2])
target_shape = y_test.shape[2]

print("Supervised train shape:", X_train.shape, y_train.shape)
print("Supervised test shape:", X_test.shape, y_test.shape)
print("Input shape:", input_shape, "Target dim:", target_shape)

# ===================== DEFINE SSL MODEL (masked reconstruction) ===================== #
from tensorflow.keras import layers, models, Input

ssl_input = Input(shape=input_shape, name="ssl_input")

# Encoder (kept temporal: return_sequences=True throughout so decoder can reconstruct sequence)
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
# Build a small encoder model that maps input -> encoded (same structure)
encoder_model = models.Model(ssl_input, encoded)
encoder_model.save_weights("encoder_masked_pretrained_weights.weights.h5")
print("Saved encoder weights to encoder_masked_pretrained_weights.h5")

# ===================== BUILD NILM MODEL (same architecture as your original) ===================== #
# We'll build a Sequential model similar to original and attempt to copy encoder weights
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
# We'll attempt to copy weights layer by layer for the encoder portion.
# We expect ordering:
# ssl_model.layers: [InputLayer, enc_conv1, enc_bilstm1, enc_dropout1, enc_lstm2, enc_dropout2, enc_lstm3, dec_bilstm1, ...]
# nilm_model.layers: [conv1, Bidirectional, Dropout, LSTM, Dropout, LSTM, Dropout, TimeDistributed]
# We'll copy weights for the first 7 layers where applicable.

print("Copying encoder weights from SSL model to NILM model (best-effort)...")
copied = 0
# collect ssl encoder layer objects (excluding InputLayer)
ssl_layers = [l for l in ssl_model.layers if l.name not in ('ssl_input', 'decoded', 'dec_bilstm1', 'dec_dropout1')]
# ssl_layers should include encoder layers; safer to pick by name prefix 'enc_'
ssl_enc_layers = [l for l in ssl_model.layers if l.name.startswith('enc_')]

# corresponding nilm layers indices to try to map: 0..5 cover encoder layers
for idx_ssl, ssl_layer in enumerate(ssl_enc_layers):
    try:
        nilm_layer = nilm_model.layers[idx_ssl]
        # only copy if shapes compatible and layer has weights
        ssl_w = ssl_layer.get_weights()
        if len(ssl_w) == 0:
            continue
        try:
            nilm_layer.set_weights(ssl_w)
            copied += 1
            print(f"Copied weights: SSL layer '{ssl_layer.name}' -> NILM layer '{nilm_layer.name}'")
        except Exception as e:
            # For Bidirectional wrappers, shapes may differ in naming; try to copy inner layer weights
            print(f"Couldn't directly copy to {nilm_layer.name}: {e}")
            # try to handle Bidirectional mapping
            if isinstance(nilm_layer, tf.keras.layers.Bidirectional):
                # tls: try to set inner forward/backward weights if possible
                try:
                    # ssl_layer may be a Bidirectional (functional) as well
                    if isinstance(ssl_layer, tf.keras.layers.Bidirectional):
                        nilm_layer.forward_layer.set_weights(ssl_layer.forward_layer.get_weights())
                        nilm_layer.backward_layer.set_weights(ssl_layer.backward_layer.get_weights())
                        copied += 1
                        print(f"Copied Bidirectional inner layers for {nilm_layer.name}")
                except Exception as e2:
                    print("Bidirectional copy failed:", e2)
            # otherwise just continue
    except IndexError:
        break

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

accuracy = r2_score(y_test_reconstructed, y_pred_reconstructed)
print("R² Score Accuracy:", accuracy)

# ===================== SAVE RESULTS ===================== #
np.savetxt('csv files/predictions.csv', y_pred_reconstructed, delimiter=',')
np.savetxt('csv files/real_values.csv', y_test_reconstructed, delimiter=',')
print("Saved results to csv files/predictions_masked_ssl.csv and real_values_masked_ssl.csv")
