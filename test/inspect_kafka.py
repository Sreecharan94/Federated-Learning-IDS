# check_topic.py
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'network_traffic',
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='earliest',  # ← Read from beginning
    enable_auto_commit=False,
    consumer_timeout_ms=5000  # Stop after 5 sec of no new messages
)

print("Reading up to 5 messages from 'network_traffic' (from beginning)...\n")

count = 0
for msg in consumer:
    try:
        data = json.loads(msg.value.decode('utf-8'))
        print(f"[{count+1}] Sample keys: {list(data.keys())[:4]}")
        print(f"    Example: Destination Port = {data.get(' Destination Port', data.get('Destination Port', 'N/A'))}")
        count += 1
        if count >= 10:
            break
    except Exception as e:
        print("Error decoding:", e)

print(f"\nTotal messages available (sampled): ~{count}")
consumer.close()