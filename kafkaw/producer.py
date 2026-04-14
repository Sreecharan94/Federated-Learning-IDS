# producer.py
import os
import sys
import yaml
import logging
import glob
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Reduce Kafka client log verbosity
logging.getLogger('kafka').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_kafka_producer(bootstrap_servers: list, api_version: tuple) -> KafkaProducer:
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            api_version=api_version,
            value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v,
            acks='all',
            retries=3,
            retry_backoff_ms=1000
        )
        logger.info(f"Kafka producer connected to {bootstrap_servers} (API version: {api_version})")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise


def send_csv_to_kafka(producer: KafkaProducer, csv_path: str, topic: str, batch_size: int = 1000):
    file_name = os.path.basename(csv_path)
    logger.info(f"Processing file: {csv_path}")

    try:
        # Read CSV in chunks
        chunk_iter = pd.read_csv(csv_path, chunksize=batch_size, low_memory=False)
        total_sent = 0

        for chunk in chunk_iter:
            # 🔥 CRITICAL FIX: Clean column names (strip whitespace)
            chunk.columns = chunk.columns.str.strip()

            for _, row in chunk.iterrows():
                try:
                    message = row.to_json()
                    producer.send(topic, value=message)
                except KafkaError as ke:
                    logger.error(f"Kafka send error in {file_name}: {ke}")
                except Exception as e:
                    logger.error(f"Serialization error in {file_name}: {e}")

            total_sent += len(chunk)
            if total_sent % batch_size == 0:
                logger.info(f"Sent {total_sent} rows to Kafka from {file_name}")

        logger.info(f"Completed sending {total_sent} rows from {file_name}")

    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--config":
        print("Usage: python producer.py --config ./configs/config.yaml")
        sys.exit(1)

    config_path = sys.argv[2]
    config = load_config(config_path)

    kafka_config = config.get('kafka', {})
    bootstrap_servers = kafka_config.get('bootstrap_servers', ['127.0.0.1:9092'])
    api_version = tuple(kafka_config.get('api_version', [2, 5, 0]))
    topic = config.get('topic', 'network_traffic')
    data_dir = config.get('data_dir', 'data/CICIDS2018')

    # Ensure IPv4
    bootstrap_servers = [s.replace('localhost', '127.0.0.1') for s in bootstrap_servers]

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV file(s)")
    for f in csv_files:
        logger.info(f"Found CSV file: {f}")

    producer = None
    try:
        producer = create_kafka_producer(bootstrap_servers, api_version)
        for csv_file in sorted(csv_files):
            send_csv_to_kafka(producer, csv_file, topic)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if producer:
            logger.info("Flushing and closing Kafka producer...")
            producer.flush(timeout=30)
            producer.close()
            logger.info("Producer closed successfully")


if __name__ == "__main__":
    main()