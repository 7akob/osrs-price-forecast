import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np

prices = pd.read_csv("data/prices.csv", index_col="timestamp", parse_dates=True)
items = ["Shark", "Nature rune", "Dragon bones"]

starting_params = {
    "Shark":        (2, 0, 1, 1, 0, 1, 7),
    "Nature rune":  (1, 1, 1, 1, 1, 1, 7),
    "Dragon bones": (1, 1, 1, 1, 1, 1, 7),
}

results_summary = []

for item in items:
    series = prices[f"{item}_high"].dropna()
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    p, d, q, P, D, Q, s = starting_params[item]

    print(f"\n{item}: running auto_arima...")
    auto = auto_arima(
        train, start_p=p, start_q=q, d=d,
        start_P=P, start_Q=Q, D=D, m=s,
        seasonal=True, stepwise=True,
        suppress_warnings=True, error_action="ignore"
    )
    print(f"  Best order: {auto.order}, seasonal: {auto.seasonal_order}")

    model = SARIMAX(train, order=auto.order, seasonal_order=auto.seasonal_order)
    fit = model.fit(disp=False)

    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    print(f"  MAE: {mae:.2f} gp")
    print(f"  MAPE: {mape:.2f}%")
    results_summary.append({"Item": item, "Order": auto.order,
                             "Seasonal": auto.seasonal_order,
                             "MAE": round(mae, 2), "MAPE%": round(mape, 2)})

    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label="Train", color="steelblue")
    test.plot(ax=ax, label="Actual", color="green")
    forecast.plot(ax=ax, label="Forecast", color="coral", linestyle="--")
    ax.set_title(f"{item} — SARIMA Forecast vs Actual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plots/forecast_{item.replace(' ', '_')}.png", dpi=150)
    plt.show()

print("\n=== Summary ===")
print(pd.DataFrame(results_summary).to_string(index=False))