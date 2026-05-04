import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "osrs-prices",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest"
)

print("Consumer listening...")
for message in consumer:
    data = message.value
    print(f"[{data['item']}] high={data['high']} low={data['low']} ts={data['timestamp']}")