import numpy as np
import pywt

# 1. Wavelet Packet Transform with padding awareness
def wavelet_packet_transform(signal, wavelet='db1', maxlevel=3):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = wp.get_level(wp.maxlevel, order='natural')
    flattened_coeffs = np.concatenate([node.data for node in nodes])
    return wp, flattened_coeffs, len(signal)  # Return original length

# 2. Reconstruct signal using original length to remove padding
def reconstruct_signals(wp_list, coeffs_array, original_length):
    reconstructed_signals = []
    for i, wp in enumerate(wp_list):
        flattened_coeffs = coeffs_array[:, i]
        nodes = wp.get_level(wp.maxlevel, order='natural')
        start = 0
        for node in nodes:
            end = start + len(node.data)
            node.data = flattened_coeffs[start:end]
            start = end
        reconstructed_signal = wp.reconstruct(update=True)
        # Trim to original length in case of wavelet padding
        reconstructed_signals.append(reconstructed_signal[:original_length])
    return np.column_stack(reconstructed_signals)

# Create sequences
def create_sequences(X, y, seq_length, step):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_length + 1, step):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)


# Reconstruct original signal
def revert_sequences(sequences, seq_length, step):
    num_samples, _, num_features = sequences.shape
    # Calculate the length of the reconstructed signal based on step size
    original_length = (num_samples - 1) * step + seq_length
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))

    for i in range(num_samples):
        start = i * step
        end = start + seq_length
        reconstructed[start:end] += sequences[i]
        counts[start:end] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    reconstructed /= counts
    return reconstructed

