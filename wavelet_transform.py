import numpy as np
import pywt

def reconstruct_signals(wp_list, coeffs_array):
    """ Reconstruct signals from a list of WaveletPacket objects and flattened coefficients. """
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
        reconstructed_signals.append(reconstructed_signal)
    return np.column_stack(reconstructed_signals)

def wavelet_packet_transform(signal, wavelet='db1', maxlevel=3):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = wp.get_level(wp.maxlevel, order='natural')
    flattened_coeffs = np.concatenate([node.data for node in nodes])
    return wp, flattened_coeffs