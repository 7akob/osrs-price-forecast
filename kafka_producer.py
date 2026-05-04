import requests
import json
import time
from kafka import KafkaProducer

ITEMS = {"Shark": 385, "Nature rune": 561, "Dragon bones": 536}
URL = "https://prices.runescape.wiki/api/v1/osrs/latest"
HEADERS = {"User-Agent": "arcada-predictive-analytics-project"}

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Producer running, sending every 30s. Ctrl+C to stop.")
while True:
    r = requests.get(URL, headers=HEADERS).json()["data"]
    for name, item_id in ITEMS.items():
        if str(item_id) in r:
            payload = {
                "item": name,
                "high": r[str(item_id)].get("high"),
                "low": r[str(item_id)].get("low"),
                "timestamp": int(time.time())
            }
            producer.send("osrs-prices", value=payload)
            print(f"Sent: {payload}")
    producer.flush()
    time.sleep(30)