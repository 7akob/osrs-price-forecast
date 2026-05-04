import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

prices = pd.read_csv("data/prices.csv", index_col="timestamp", parse_dates=True)
items = ["Shark", "Nature rune", "Dragon bones"]

for item in items:
    series = prices[f"{item}_high"].dropna()

    adf_result = adfuller(series)
    print(f"\n{item} ADF test:")
    print(f"  ADF statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No — needs differencing'}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(series, lags=40, ax=axes[0], title=f"{item} — ACF")
    plot_pacf(series, lags=40, ax=axes[1], title=f"{item} — PACF")
    plt.tight_layout()
    plt.savefig(f"plots/acf_pacf_{item.replace(' ', '_')}.png", dpi=150)
    plt.show()