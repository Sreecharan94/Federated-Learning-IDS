# monitoring/metrics.py
import psutil
import time
import threading
from collections import deque
import json
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize metric stores
        self.system_metrics = deque(maxlen=1000)
        self.kafka_metrics = deque(maxlen=1000)
        self.attack_metrics = deque(maxlen=1000)
        self.federated_metrics = deque(maxlen=1000)
        
        # REMOVED: Kafka monitoring variables
        # self.kafka_consumer = None
        # self.kafka_topic = self.config['kafka']['topic_name']
        # self.bootstrap_servers = self.config['kafka']['bootstrap_servers']
        
        # Collection flags
        self.collecting = False
        self.collection_thread = None
        
        # Attack type tracking
        self.attack_counts = {}
        self.latest_alerts = deque(maxlen=100)
        self.total_messages_processed = 0
        
        logger.info("Metrics collector initialized (Kafka monitoring disabled for training)")
    
    def start_collection(self):
        """Start collecting metrics in background"""
        if self.collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop collecting metrics"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        # REMOVED: Kafka consumer close
        # if self.kafka_consumer:
        #     self.kafka_consumer.close()
        
        logger.info("Stopped metrics collection")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.collecting:
            try:
                # Collect system metrics
                sys_metrics = self._collect_system_metrics()
                self.system_metrics.append(sys_metrics)
                
                # REMOVED: Kafka metrics collection
                # kafka_metrics = self._collect_kafka_metrics()
                # if kafka_metrics:
                #     self.kafka_metrics.append(kafka_metrics)
                
                # Collect attack metrics
                attack_metrics = self._collect_attack_metrics()
                if attack_metrics:
                    self.attack_metrics.append(attack_metrics)
                
                # Wait before next collection
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        timestamp = datetime.now().isoformat()
        
        # Get network IO counters safely
        network_io_counters = psutil.net_io_counters()
        network_io_dict = network_io_counters._asdict() if network_io_counters else {}

        return {
            'timestamp': timestamp,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'network_io': network_io_dict,
            'load_average': getattr(psutil, 'getloadavg', lambda: (0, 0, 0))()
        }
    
    # REMOVED: _collect_kafka_metrics method entirely
    
    def _collect_attack_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect attack detection metrics"""
        if not self.attack_counts:
            return None
        
        timestamp = datetime.now().isoformat()
        total_attacks = sum(self.attack_counts.values())
        
        return {
            'timestamp': timestamp,
            'total_attacks': total_attacks,
            'attack_types': dict(self.attack_counts),
            'benign_count': self.attack_counts.get('Benign', 0),
            'messages_processed': self.total_messages_processed
        }
    
    def record_attack_detection(self, attack_type: str, details: dict = None):
        """Record an attack detection"""
        self.attack_counts[attack_type] = self.attack_counts.get(attack_type, 0) + 1
        self.total_messages_processed += 1
        
        if attack_type != 'Benign':
            port_str = details.get('Dst_Port', 'Unknown') if details else 'Unknown'
            alert = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'type': attack_type,
                'target': f"Port {port_str}"
            }
            self.latest_alerts.appendleft(alert)
    
    def record_federated_round(self, round_num: int, accuracy: float, loss: float, 
                              active_clients: int, round_time: float):
        """Record federated learning round metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'round_number': round_num,
            'accuracy': accuracy,
            'loss': loss,
            'active_clients': active_clients,
            'round_time': round_time
        }
        self.federated_metrics.append(metrics)
    
    def get_recent_system_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent system metrics"""
        return list(self.system_metrics)[-limit:]
    
    def get_recent_kafka_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent Kafka metrics - returns empty list for training"""
        return []  # Return empty list instead of Kafka metrics
    
    def get_recent_attack_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent attack metrics"""
        return list(self.attack_metrics)[-limit:]
    
    def get_recent_federated_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent federated metrics"""
        return list(self.federated_metrics)[-limit:]
    
    def get_overall_attack_distribution(self) -> Dict[str, int]:
        """Get overall attack type distribution"""
        return dict(self.attack_counts)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get current system summary"""
        if not self.system_metrics:
            return {}

        recent_metrics = self.get_recent_system_metrics(10)
        if not recent_metrics:
            return {}

        # Calculate averages
        avg_cpu = np.mean([m['cpu_percent'] for m in recent_metrics])
        avg_memory = np.mean([m['memory_percent'] for m in recent_metrics])
        avg_disk = np.mean([m['disk_percent'] for m in recent_metrics])

        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_disk_percent': avg_disk,
            'current_process_count': recent_metrics[-1]['process_count'],
            'peak_memory_gb': max([m['memory_used_gb'] for m in recent_metrics])
        }
    
    def export_metrics(self, output_dir: str = "./outputs/experiment_metrics/"):
        """Export metrics to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export system metrics
        if self.system_metrics:
            sys_df = pd.DataFrame(list(self.system_metrics))
            sys_df.to_csv(f"{output_dir}/system_metrics.csv", index=False)
        
        # Export attack metrics
        if self.attack_metrics:
            attack_df = pd.DataFrame(list(self.attack_metrics))
            attack_df.to_csv(f"{output_dir}/attack_metrics.csv", index=False)
        
        # Export federated metrics
        if self.federated_metrics:
            fed_df = pd.DataFrame(list(self.federated_metrics))
            fed_df.to_csv(f"{output_dir}/federated_metrics.csv", index=False)
        
        # Export attack distribution
        attack_dist = self.get_overall_attack_distribution()
        with open(f"{output_dir}/attack_distribution.json", 'w') as f:
            json.dump({
                'distribution': attack_dist,
                'latest_alerts': list(self.latest_alerts)
            }, f, indent=2)
        
        logger.info(f"Metrics exported to {output_dir}")

# REMOVED: KafkaThroughputMonitor class entirely (not needed for training)

def test_metrics_collector():
    """Test the metrics collector"""
    collector = MetricsCollector()
    
    # Start collection
    collector.start_collection()
    
    # Simulate some activity
    for i in range(5):
        collector.record_attack_detection(f"Attack_{i%3}")
        time.sleep(0.5)
    
    # Get metrics
    sys_metrics = collector.get_recent_system_metrics(5)
    print(f"Recent system metrics: {len(sys_metrics)} entries")
    
    attack_dist = collector.get_overall_attack_distribution()
    print(f"Attack distribution: {attack_dist}")
    
    summary = collector.get_system_summary()
    print(f"System summary: {summary}")
    
    # Stop collection
    collector.stop_collection()
    
    print("Metrics collector test passed!")

if __name__ == "__main__":
    test_metrics_collector()