import pandas as pd
import matplotlib.pyplot as plt

prices = pd.read_csv("data/prices.csv", index_col="timestamp", parse_dates=True)

items = ["Shark", "Nature rune", "Dragon bones"]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, item in zip(axes, items):
    ax.plot(prices[f"{item}_high"], label="High", color="steelblue")
    ax.plot(prices[f"{item}_low"], label="Low", color="coral", alpha=0.7)
    ax.set_title(item)
    ax.legend(fontsize=8)
    ax.set_ylabel("Price (gp)")

plt.suptitle("OSRS Grand Exchange — Daily Prices (May 2025 to May 2026)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/raw_prices.png", dpi=150)
plt.show()