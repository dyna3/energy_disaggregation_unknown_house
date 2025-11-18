import os
import json
import numpy as np
import pandas as pd
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

# Sequence / forecasting lengths
sequence_length = 128   # encoder input window
forecast_horizon = 100  # predict next 100 samples from previous 128

# SSL hyperparams
ssl_epochs = 2
ssl_batch_size = 128
FORCE_SSL = False  # set True to retrain SSL even if encoder weights exist

# Fine-tuning hyperparams
fine_tune_epochs = 2
ft_batch_size = 128

# Weight file paths (Keras 3 naming: *.weights.h5)
ENCODER_WEIGHTS_PATH = "weights/encoder_forecast_pretrained.weights.h5"

# Output files
SCALER_JSON = "files/minmax_scalers.json"
OUT_PRED_CSV = 'csv files/predictions_forecast_ssl.csv'
OUT_REAL_CSV = 'csv files/real_values_forecast_ssl.csv'

# ===================== HELPERS ===================== #
def create_supervised_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i:i+seq_len])
    return np.array(Xs), np.array(ys)

def create_forecast_sequences(series, seq_len, forecast_horizon):
    Xs, Ys = [], []
    total = len(series)
    for i in range(total - seq_len - forecast_horizon + 1):
        Xs.append(series[i:i+seq_len])
        Ys.append(series[i+seq_len:i+seq_len+forecast_horizon])
    return np.array(Xs), np.array(Ys)

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

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

# ===================== LOAD + PREPROCESS ===================== #
print("Loading CSVs and normalizing...")

df_list = []
scalers = []

for f in data_files:
    df = pd.read_csv(f).head(rows_to_keep)
    df = df[[col for col in columns if col in df.columns]]
    df = df.reindex(columns=columns)

    min_vals = df.min()
    max_vals = df.max()
    scalers.append({"min": min_vals.to_dict(), "max": max_vals.to_dict()})

    df = (df - min_vals) / (max_vals - min_vals + 1e-8)
    df_list.append(df)

ensure_dir_for_file(SCALER_JSON)
with open(SCALER_JSON, "w") as jf:
    json.dump(scalers, jf)

# combine into arrays
data = pd.concat(df_list, ignore_index=True).to_numpy()
X_all = data[:, :2]  # aggregate channels (agg, agg_act)
y_all = data[:, 2:]  # appliances

# make house segments (each file is one house, same sizing)
house_segments = [(i * rows_to_keep, (i + 1) * rows_to_keep) for i in range(len(df_list))]

# train on all houses except last one
train_indices = np.concatenate([np.arange(*house_segments[i]) for i in range(len(df_list) - 1)])

# for SSL & test pick blocks inside last (unseen) house
last_start, last_end = house_segments[-1]
# choose days 1..4 for SSL (as in your example)
ssl_start = last_start + 1 * seconds_per_day
ssl_end   = last_start + 4 * seconds_per_day
ssl_indices = np.arange(ssl_start, ssl_end)

# choose subsequent days for supervised test (adjust as needed)
test_start = last_start + 4 * seconds_per_day
test_end = min(last_end, last_start + 6 * seconds_per_day)
test_indices = np.arange(test_start, test_end)

# prepare training arrays
X_train_raw, y_train_raw = shuffle(X_all[train_indices], y_all[train_indices])
X_unlabeled_target = X_all[ssl_indices]   # continuous mains for SSL (timesteps, 2)
X_test_raw, y_test_raw = X_all[test_indices], y_all[test_indices]

print("Unlabeled target shape (timesteps,features):", X_unlabeled_target.shape)

# ===================== CREATE SSL FORECASTING DATASETS ===================== #
X_ssl, y_ssl = create_forecast_sequences(X_unlabeled_target, sequence_length, forecast_horizon)
print("SSL forecasting dataset shapes -> X:", X_ssl.shape, " y:", y_ssl.shape)

# ===================== CREATE SUPERVISED SEQUENCES FOR NILM ===================== #
X_train, y_train = create_supervised_sequences(X_train_raw, y_train_raw, sequence_length)
X_test, y_test = create_supervised_sequences(X_test_raw, y_test_raw, sequence_length)

input_shape = (sequence_length, X_train.shape[2])
target_shape = y_test.shape[2]

print("Supervised train shape:", X_train.shape, y_train.shape)
print("Supervised test shape:", X_test.shape, y_test.shape)
print("Input shape:", input_shape, "Target dim:", target_shape)

# ===================== BUILD SSL FORECASTING MODEL ===================== #
# Encoder (same style as your original)
from tensorflow.keras import layers, models, Input

ssl_input = Input(shape=input_shape, name="ssl_input")

# encoder: conv + BiLSTM + LSTM stack (returns sequences)
x = layers.Conv1D(128, 3, activation='relu', padding='same', name='enc_conv1')(ssl_input)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='enc_bilstm1')(x)
x = layers.LSTM(64, return_sequences=True, name='enc_lstm2')(x)
encoded_seq = layers.LSTM(32, return_sequences=True, name='enc_lstm3')(x)  # shape (seq_len, hidden)

# pool encoded sequence to a fixed vector for forecasting (global average)
pooled = layers.GlobalAveragePooling1D(name='global_pool')(encoded_seq)  # shape (hidden,)

# Forecast decoder: RepeatVector -> LSTM stack -> TimeDistributed Dense (predict next horizon)
y = layers.RepeatVector(forecast_horizon, name='repeat_vec')(pooled)
y = layers.LSTM(128, return_sequences=True, name='fdec_lstm1')(y)
y = layers.LSTM(64, return_sequences=True, name='fdec_lstm2')(y)
forecast_out = layers.TimeDistributed(layers.Dense(input_shape[1], activation='linear'), name='forecast_out')(y)

ssl_model = models.Model(ssl_input, forecast_out, name='ssl_forecast_model')
ssl_model.compile(optimizer='adam', loss='mse')
ssl_model.summary()

# ===================== RUN SSL FORECASTING PRETRAINING (or load weights) ===================== #
ensure_dir_for_file(ENCODER_WEIGHTS_PATH)

if (not FORCE_SSL) and os.path.exists(ENCODER_WEIGHTS_PATH):
    print("Found existing encoder weights, loading (skipping SSL training).")
    # Build encoder model same as in SSL to load weights
    # encoder_model maps ssl_input -> encoded_seq
    encoder_model = models.Model(ssl_input, encoded_seq)
    # need to build it fresh with same input shape before loading weights
    # (weights saved at end will match the model architecture)
    encoder_model.load_weights(ENCODER_WEIGHTS_PATH)
else:
    print("Starting SSL forecasting pretraining (this may take a while)...")
    # Train ssl_model: input = past 128 samples, target = next 100 samples of mains (2 features)
    ssl_history = ssl_model.fit(X_ssl, y_ssl, epochs=ssl_epochs, batch_size=ssl_batch_size, validation_split=0.1)
    # After training, extract encoder (input -> encoded_seq) and save encoder weights
    encoder_model = models.Model(ssl_input, encoded_seq)
    # Save encoder weights (Keras 3 naming)
    encoder_model.save_weights(ENCODER_WEIGHTS_PATH)
    print("Saved encoder weights to:", ENCODER_WEIGHTS_PATH)

# ===================== BUILD NILM MODEL (reuse encoder architecture + new appliance decoder) ===================== #
# Rebuild encoder (fresh model) and load saved weights to ensure independence from SSL graph
from tensorflow.keras import Input as KInput

encoder_input = KInput(shape=input_shape)
# Recreate encoder layers (same architecture) using functional API to get a model we can re-use
e = layers.Conv1D(128, 3, activation='relu', padding='same', name='enc_conv1')(encoder_input)
e = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='enc_bilstm1')(e)
e = layers.LSTM(64, return_sequences=True, name='enc_lstm2')(e)
e = layers.LSTM(32, return_sequences=True, name='enc_lstm3')(e)
encoder_for_nilm = models.Model(encoder_input, e, name='encoder_for_nilm')

# Load weights into this encoder model
encoder_for_nilm.load_weights(ENCODER_WEIGHTS_PATH)
print("Loaded encoder weights into NILM encoder.")

# Build appliance decoder: LSTM stack -> TimeDistributed Dense (appliance outputs)
decoder_input = layers.Input(shape=encoder_for_nilm.output_shape[1:], name='dec_in')
d = layers.LSTM(64, return_sequences=True, name='ap_dec_lstm1')(decoder_input)
d = layers.LSTM(32, return_sequences=True, name='ap_dec_lstm2')(d)
appliances_out = layers.TimeDistributed(layers.Dense(target_shape, activation='linear'), name='appliances_out')(d)
appliance_decoder = models.Model(decoder_input, appliances_out, name='appliance_decoder')

# Build full NILM model: encoder_for_nilm -> appliance_decoder
nilm_input = Input(shape=input_shape, name='nilm_input')
enc_seq = encoder_for_nilm(nilm_input)
nilm_out = appliance_decoder(enc_seq)
nilm_model = models.Model(nilm_input, nilm_out, name='nilm_model')

nilm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
nilm_model.summary()

# ===================== FINE-TUNE NILM MODEL ON LABELED SOURCE HOUSES ===================== #
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("Starting fine-tuning on labeled source houses...")
nilm_history = nilm_model.fit(X_train, y_train, validation_split=0.1, epochs=fine_tune_epochs, batch_size=ft_batch_size, callbacks=[early_stop])

# ===================== EVALUATION ===================== #
print("Evaluating on unseen house test block...")
eval_loss, eval_mae = nilm_model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}")

y_pred = nilm_model.predict(X_test)

y_test_reconstructed = revert_sequences(y_test, sequence_length)
y_pred_reconstructed = revert_sequences(y_pred, sequence_length)

accuracy = r2_score(y_test_reconstructed, y_pred_reconstructed)
print("R² Score Accuracy:", accuracy)

# ===================== SAVE RESULTS ===================== #
ensure_dir_for_file(OUT_PRED_CSV)
ensure_dir_for_file(OUT_REAL_CSV)
np.savetxt(OUT_PRED_CSV, y_pred_reconstructed, delimiter=',')
np.savetxt(OUT_REAL_CSV, y_test_reconstructed, delimiter=',')
print("Saved results to", OUT_PRED_CSV, "and", OUT_REAL_CSV)
