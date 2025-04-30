import matplotlib.pyplot as plt
import numpy as np

def plot_features(df_list, num_days):
    """
    Plots all features in each DataFrame for the first `num_days` days,
    showing day numbers (1–N) on the x-axis.

    Assumes a sampling rate of 1 second.

    Parameters:
        df_list (list of pd.DataFrame): List of aligned DataFrames.
        num_days (int): Number of days to include in the plot.
    """
    seconds_per_day = 24 * 60 * 60
    num_samples = num_days * seconds_per_day

    for i, df in enumerate(df_list):
        df_subset = df.iloc[:num_samples]  # Select data for the specified number of days

        plt.figure(figsize=(12, 6))
        for col in df_subset.columns:
            plt.plot(df_subset.index, df_subset[col], label=col)

        # Set x-ticks at daily intervals and label them as Day 1 to N
        tick_positions = np.arange(0, num_samples + 1, seconds_per_day)
        tick_labels = [str(i) for i in range(1, len(tick_positions))]
        plt.xticks(tick_positions[:-1], tick_labels)

        plt.title(f"Feature Plot for DataFrame {i + 1} (First {num_days} Days)")
        plt.xlabel("Day")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_results(y_test, y_pred):
    """
    param y_test: a numpy array with shape (n_samples, 4) containing real energy consumption per device
    param y_pred: a numpy array with shape (n_samples, 4) containing predicted energy consumption per device
    return: plots comparing predictions vs. ground truth for each device
    """
    num_devices = y_test.shape[1]
    plt.figure(figsize=(12, 8))

    for i in range(num_devices):
        plt.subplot(2, 2, i + 1)
        plt.plot(y_test[:, i], label='Actual', linewidth=2)
        plt.plot(y_pred[:, i], label='Predicted', linestyle='--', linewidth=2)
        plt.title(f'Device {i+1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_columns(data, title_prefix="Column"):
    """
    Plots each column of a 2D NumPy array as a separate subplot.

    :param data: a 2D NumPy array of shape (n_samples, n_columns)
    :param title_prefix: optional prefix for subplot titles
    """
    num_columns = data.shape[1]
    num_rows = (num_columns + 1) // 2  # layout in 2 columns

    plt.figure(figsize=(12, 4 * num_rows))

    for i in range(num_columns):
        plt.subplot(num_rows, 2, i + 1)
        plt.plot(data[:, i], linewidth=2)
        plt.title(f'{title_prefix} {i+1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
