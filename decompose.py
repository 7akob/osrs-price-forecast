import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

prices = pd.read_csv("data/prices.csv", index_col="timestamp", parse_dates=True)

items = ["Shark", "Nature rune", "Dragon bones"]

for item in items:
    series = prices[f"{item}_high"].dropna()
    result = seasonal_decompose(series, model="additive", period=7)

    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"{item} — Seasonal Decomposition (period=7)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"plots/decomp_{item.replace(' ', '_')}.png", dpi=150)
    plt.show()