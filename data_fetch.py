import requests
import pandas as pd

# I chose three commonly traded items
ITEMS = {
    "Shark": 385,
    "Nature rune": 561,
    "Dragon bones": 536
}

def fetch_item(name, item_id):
    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id={item_id}"
    headers = {"User-Agent": "arcada-predictive-analytics-project"}
    r = requests.get(url, headers=headers)
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")
    df = df[["avgHighPrice", "avgLowPrice"]].rename(columns={
        "avgHighPrice": f"{name}_high",
        "avgLowPrice": f"{name}_low"
    })
    return df

dfs = [fetch_item(name, id) for name, id in ITEMS.items()]
prices = pd.concat(dfs, axis=1)
prices = prices.dropna()
prices.to_csv("data/prices.csv")
print(prices.head())
print(f"\nShape: {prices.shape}")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")