import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from kafka import KafkaConsumer
import json
import threading
import time

@st.cache_resource
def get_live_store():
    return {"Shark": [], "Nature rune": [], "Dragon bones": []}

@st.cache_resource
def load_and_fit():
    prices = pd.read_csv("data/prices.csv", index_col="timestamp", parse_dates=True)
    models = {}
    items = ["Shark", "Nature rune", "Dragon bones"]
    params = {
        "Shark":        (2, 0, 1, 1, 0, 1, 7),
        "Nature rune":  (1, 1, 1, 1, 1, 1, 7),
        "Dragon bones": (1, 1, 1, 1, 1, 1, 7),
    }
    for item in items:
        series = prices[f"{item}_high"].dropna()
        split = int(len(series) * 0.8)
        train = series[:split]
        p, d, q, P, D, Q, s = params[item]
        auto = auto_arima(train, start_p=p, start_q=q, d=d,
                          start_P=P, start_Q=Q, D=D, m=s,
                          seasonal=True, stepwise=True,
                          suppress_warnings=True, error_action="ignore")
        fit = SARIMAX(train, order=auto.order,
                      seasonal_order=auto.seasonal_order).fit(disp=False)
        forecast_obj = fit.get_forecast(steps=len(series) - len(train))
        models[item] = {
            "series": series,
            "train": train,
            "test": series[split:],
            "forecast": forecast_obj.predicted_mean,
            "conf_int": forecast_obj.conf_int(),
            "order": auto.order,
            "seasonal": auto.seasonal_order
        }
    return models

@st.cache_resource
def start_kafka(store):
    def kafka_thread():
        while True:
            try:
                consumer = KafkaConsumer(
                    "osrs-prices",
                    bootstrap_servers="localhost:9092",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    auto_offset_reset="earliest",
                    # no consumer_timeout_ms — runs forever
                )
                for msg in consumer:
                    d = msg.value
                    if d["item"] in store:
                        store[d["item"]].append({
                            "timestamp": pd.to_datetime(d["timestamp"], unit="s"),
                            "high": d["high"]
                        })
            except Exception as e:
                print(f"Kafka error: {e}, restarting in 5s...")
                time.sleep(5)
    threading.Thread(target=kafka_thread, daemon=True).start()
    return True

# UI
st.set_page_config(page_title="OSRS GE Forecast", layout="wide")
st.title("OSRS Grand Exchange — SARIMA Price Forecasting")
st.caption("Live prices via Kafka | Historical SARIMA forecast with 95% confidence interval")

live_store = get_live_store()
start_kafka(live_store)

with st.spinner("Fitting SARIMA models..."):
    models = load_and_fit()

selected = st.selectbox("Select item", ["Shark", "Nature rune", "Dragon bones"])
m = models[selected]
live = live_store[selected]

fig = go.Figure()
fig.add_trace(go.Scatter(x=m["train"].index, y=m["train"].values,
                         name="Train", line=dict(color="steelblue")))
fig.add_trace(go.Scatter(x=m["test"].index, y=m["test"].values,
                         name="Actual", line=dict(color="green")))
fig.add_trace(go.Scatter(x=m["forecast"].index, y=m["forecast"].values,
                         name="Forecast", line=dict(color="coral", dash="dash")))
fig.add_trace(go.Scatter(
    x=list(m["conf_int"].index) + list(m["conf_int"].index[::-1]),
    y=list(m["conf_int"].iloc[:, 0]) + list(m["conf_int"].iloc[:, 1][::-1]),
    fill="toself", fillcolor="rgba(255,127,80,0.15)",
    line=dict(color="rgba(255,255,255,0)"), name="95% CI"
))

if live:
    ldf = pd.DataFrame(live)
    fig.add_trace(go.Scatter(x=ldf["timestamp"], y=ldf["high"],
                             name="Live (Kafka)", line=dict(color="yellow", width=2)))

fig.update_layout(title=f"{selected} — Forecast vs Actual",
                  xaxis_title="Date", yaxis_title="Price (gp)", height=500)
st.plotly_chart(fig, width="stretch")

col1, col2, col3 = st.columns(3)
col1.metric("SARIMA Order", str(m["order"]))
col2.metric("Seasonal Order", str(m["seasonal"]))
col3.metric("Live ticks received", len(live))

st.subheader("Latest live prices from Kafka")
if live:
    st.dataframe(pd.DataFrame(live).tail(10))
else:
    st.info("Waiting for live data — make sure kafka_producer.py is running.")

time.sleep(5)
st.rerun()