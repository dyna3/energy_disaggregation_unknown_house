import matplotlib.pyplot as plt

def plot_features(df_list):
    """
    Plots all features in each DataFrame in one plot per DataFrame.

    Parameters:
        df_list (list of pd.DataFrame): List of aligned DataFrames.
    """
    for i, df in enumerate(df_list):
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)

        plt.title(f"Feature Plot for DataFrame {i + 1}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()