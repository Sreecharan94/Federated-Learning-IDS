# kafkaw/consumer.py
import json
import logging
import yaml
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import warnings
import tensorflow as tf
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.metrics import MetricsCollector
from preprocessing.utils import DataUtils
from datetime import datetime

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaConsumerService:
    def __init__(self, config_path="./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        kafka_config = self.config['kafka']
        self.bootstrap_servers = kafka_config['bootstrap_servers']
        self.topic_name = kafka_config['topic_name']
        self.consumer_group = kafka_config['consumer_group']

        # Kafka Native Consumer
        import uuid
        dynamic_group = f"{self.consumer_group}_{uuid.uuid4().hex[:8]}"
        self.consumer = KafkaConsumer(
            self.topic_name,
            bootstrap_servers=self.bootstrap_servers,
            group_id=dynamic_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )

        self.data_utils = DataUtils()
        self.feature_columns = None
        self.preprocessing_ready = False
        
        # Tracking Real World Architecture Modules 
        self.model = None
        self.metrics_collector = MetricsCollector(config_path)

        # Reverse Mapping (13 classes array)
        self.classes_map = [
            'Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS', 
            'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 
            'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 
            'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infiltration'
        ]

    def initialize_preprocessing(self, sample_data):
        """Initialize feature columns from first message"""
        try:
            exclude_cols = {'timestamp', 'Label'}
            normalized_keys = [k.strip() for k in sample_data.keys()]
            self.feature_columns = [col for col in normalized_keys if col not in exclude_cols]
            
            # Pad exactly to 78 expected columns based on model configuration
            while len(self.feature_columns) < 78:
                self.feature_columns.append(f"PAD_FEATURE_{len(self.feature_columns)}")
            self.feature_columns = self.feature_columns[:78]
                
            self.preprocessing_ready = True
            logger.info(f"Initialized preprocessing with requested {len(self.feature_columns)} features.")
            
            # Load ML model dynamically
            model_path = "./models/global_model.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Successfully bound Deep Learning (LSTM+Attn) Module!")
            else:
                logger.warning("Active Global Model missing in models/. Logging inference offline.")
                
            self.metrics_collector.start_collection()
        except Exception as e:
            logger.error(f"Error initializing preprocessing: {e}")

    def preprocess_single_row(self, row_data):
        """Preprocess a single row with robust key handling"""
        if not self.preprocessing_ready or self.feature_columns is None:
            return {'error': 'Preprocessing not ready', 'original_data': row_data}

        try:
            normalized_data = {k.strip(): v for k, v in row_data.items()}
            
            features = []
            for col in self.feature_columns:
                val = normalized_data.get(col, 0.0)
                if pd.isna(val): val = 0.0
                features.append(float(val))

            # Tile array exactly to size (1, 10, 78) to accommodate LSTM input geometry
            features_array = np.array(features, dtype=np.float32)
            sequence_block = np.tile(features_array, (10, 1))
            final_sequence = np.expand_dims(sequence_block, axis=0) # Shape: (1, 10, 78)

            label = str(normalized_data.get('Label', 'Benign')).strip()
            timestamp = normalized_data.get('timestamp', datetime.now().isoformat())

            return {
                'features': final_sequence,
                'true_label': label,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Error preprocessing row: {e}")
            return {'error': str(e), 'original_data': row_data}

    def consume_messages(self):
        """Main consumption loop matching Real-World SOC inference flows"""
        logger.info(f"Starting to consume messages from topic: {self.topic_name}")
        message_count = 0
        sample_initialized = False

        try:
            for message in self.consumer:
                raw_data = message.value

                if not sample_initialized:
                    self.initialize_preprocessing(raw_data)
                    sample_initialized = True

                processed = self.preprocess_single_row(raw_data)
                
                if 'error' not in processed:
                    inferred_attack = processed['true_label']
                    
                    if self.model is not None:
                        # ACTIVE REAL-TIME INFERENCE (LSTM -> Attn)
                        predictions = self.model.predict(processed['features'], verbose=0)
                        predicted_class_idx = np.argmax(predictions[0])
                        # If probability threshold handles accurately, use map index, fallback gracefully.
                        try:
                            inferred_attack = self.classes_map[predicted_class_idx]
                        except IndexError:
                            inferred_attack = processed['true_label'] # Failsafe
                    
                    # LOG INTO METRICS STREAM FOR DASHBOARD (SOC)
                    dst_port = raw_data.get('Dst Port', 'Unknown')
                    self.metrics_collector.record_attack_detection(inferred_attack, {'Dst_Port': dst_port})
                    
                    if message_count % 5 == 0:
                        # Flush JSON out periodically for Streamlit Live Parsing
                        self.metrics_collector.export_metrics("./outputs/experiment_metrics/")
                        logger.info(f"[{message_count}] Deep Inference Evaluated => Threat Category: {inferred_attack}")
                        
                else:
                    logger.warning(f"Processing error: {processed['error']}")

                message_count += 1

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.metrics_collector.export_metrics("./outputs/experiment_metrics/")
            self.consumer.close()
            logger.info("Kafka consumer closed")


def main():
    service = KafkaConsumerService(config_path="./configs/config.yaml")
    service.consume_messages()


if __name__ == "__main__":
    main()