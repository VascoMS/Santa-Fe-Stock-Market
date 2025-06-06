import pandas as pd
from scipy.stats import skew, kurtosis
import os
print("Current working directory:", os.getcwd())


def compute_table1_stats(price_series: pd.Series, volume_series: pd.Series) -> pd.Series:
    # Compute simple returns (you can switch to log returns if needed)
    returns = price_series.diff().dropna()

    stats = {
        "Mean": returns.mean(),
        "Std. Dev.": returns.std(ddof=1),
        "Skewness": skew(returns),
        "Kurtosis": kurtosis(returns, fisher=False),  # match normal dist = 3
        "Volume traded (per 10k periods)": volume_series.sum() / (len(volume_series) / 10_000)
    }

    return pd.Series(stats)

price_series = pd.read_csv('2025-06-06_18-08-58_SLOW/experiment_1/data/SLOW_250000_price_history.csv', header=None).squeeze("columns")[10000:]
volume_series = pd.read_csv('2025-06-06_18-08-58_SLOW/experiment_1/data/SLOW_250000_combined_volume.csv', header=None).squeeze("columns")[10000:]

stats = compute_table1_stats(price_series, volume_series)
print(stats)
