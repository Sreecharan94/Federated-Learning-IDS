# federated/server.py
import tensorflow as tf # Import tensorflow
import keras
import numpy as np
from typing import List, Dict, Any, Optional 
import logging
from .aggregator import FederatedAggregator
import copy
import yaml # Add yaml import
from .client import ClientManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(self, model_fn, config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.global_model = model_fn()
        self.model_fn = model_fn
        self.aggregator = FederatedAggregator()
        
        self.num_rounds = self.config['federated']['num_rounds']
        self.client_manager: Optional['ClientManager'] = None # Explicitly type hint as Optional
        
        # Training history
        self.training_history = {
            'rounds': [],
            'global_accuracy': [],
            'global_loss': [],
            'active_clients': [],
            'round_times': []
        }
        
        logger.info("Initialized federated server")
    
    def set_client_manager(self, client_manager):
        """Set the client manager for the server"""
        self.client_manager = client_manager
        logger.info("Client manager set for server")
    
    def get_global_model_weights(self) -> List[np.ndarray]:
        """Get current global model weights"""
        if self.global_model is None:
            raise ValueError("Global model is not initialized")
        return self.global_model.get_weights()
    
    def set_global_model_weights(self, weights: List[np.ndarray]):
        """Set global model weights"""
        if self.global_model is None:
            raise ValueError("Global model is not initialized")
        self.global_model.set_weights(weights)
    
    def aggregate_client_updates(self, active_clients: List, client_weights: List[List[np.ndarray]], 
                                client_samples: List[int]) -> List[np.ndarray]:
        """Aggregate updates from active clients"""
        aggregated_weights = self.aggregator.aggregate(
            client_weights, 
            client_samples
        )
        return aggregated_weights
    
    def broadcast_model_weights(self, active_clients: List):
        """Broadcast global model weights to active clients"""
        global_weights = self.get_global_model_weights()
        
        for client in active_clients:
            client.set_model_weights(global_weights)
        
        logger.info(f"Broadcast global model weights to {len(active_clients)} clients")
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the global model on test data"""
        if len(X_test) == 0:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        if self.global_model is None: # Check if model is loaded
             logger.error("Global model is not initialized for evaluation.")
             return {'loss': float('inf'), 'accuracy': 0.0} # Return error values

        eval_results = self.global_model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'global_loss': float(eval_results[0]),
            'global_accuracy': float(eval_results[1])
        }
        
        logger.info(f"Global model evaluation - Loss: {eval_results[0]:.4f}, Accuracy: {eval_results[1]:.4f}")
        
        return metrics
    
    def federated_training_round(self, round_num: int, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None):
        """Execute a single federated training round"""
        import time
        start_time = time.time()
        
        logger.info(f"Starting federated training round {round_num}")
        
        # Check if client_manager is set
        if self.client_manager is None:
             logger.error("Client manager is not set. Cannot proceed with training round.")
             return # Or raise an exception

        # Select active clients for this round
        active_clients = self.client_manager.select_active_clients(round_num) # Safe call now
        
        # Broadcast current global model weights to active clients
        self.broadcast_model_weights(active_clients)
        
        # Collect client updates
        client_weights = []
        client_samples = []
        client_metrics = []
        
        for client in active_clients:
            # Train client locally
            train_metrics = client.train_local_model()
            client_metrics.append(train_metrics)
            
            # Get updated weights and sample count
            updated_weights = client.get_model_weights()
            client_weights.append(updated_weights)
            client_samples.append(train_metrics['samples_trained'])
        
        # Aggregate client updates
        if client_weights:
            aggregated_weights = self.aggregate_client_updates(
                active_clients, client_weights, client_samples
            )
            
            # Update global model
            self.set_global_model_weights(aggregated_weights)
        
        # Evaluate global model
        if X_test is not None and y_test is not None:
            eval_metrics = self.evaluate_global_model(X_test, y_test)
        else:
            eval_metrics = {'global_loss': 0.0, 'global_accuracy': 0.0}
        
        # Record round metrics
        round_time = time.time() - start_time
        avg_train_loss = np.mean([m['loss'] for m in client_metrics]) if client_metrics else 0.0
        avg_train_acc = np.mean([m['accuracy'] for m in client_metrics]) if client_metrics else 0.0
        
        self.training_history['rounds'].append(round_num)
        self.training_history['global_accuracy'].append(eval_metrics['global_accuracy'])
        self.training_history['global_loss'].append(eval_metrics['global_loss'])
        self.training_history['active_clients'].append(len(active_clients))
        self.training_history['round_times'].append(round_time)
        
        logger.info(f"Round {round_num} completed - "
                   f"Global Acc: {eval_metrics['global_accuracy']:.4f}, "
                   f"Global Loss: {eval_metrics['global_loss']:.4f}, "
                   f"Avg Client Acc: {avg_train_acc:.4f}, "
                   f"Active Clients: {len(active_clients)}, "
                   f"Time: {round_time:.2f}s")
        
        return {
            'round_num': round_num,
            'global_accuracy': eval_metrics['global_accuracy'],
            'global_loss': eval_metrics['global_loss'],
            'avg_client_accuracy': avg_train_acc,
            'avg_client_loss': avg_train_loss,
            'active_clients': len(active_clients),
            'round_time': round_time
        }
    
    def federated_training(self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Execute complete federated training"""
        logger.info("Starting federated training...")
        
        training_results = []
        
        for round_num in range(1, self.num_rounds + 1):
            round_result = self.federated_training_round(round_num, X_test, y_test)
            if round_result: # Only append if result is not None due to error
                 training_results.append(round_result)
        
        logger.info("Federated training completed!")
        
        # Save global model
        self.save_global_model()
        
        return {
            'training_results': training_results,
            'training_history': self.training_history,
            'final_accuracy': self.training_history['global_accuracy'][-1] if self.training_history['global_accuracy'] else 0.0
        }
    
    def save_global_model(self, filepath: str = "./models/global_model.h5"):
        """Save the global model"""
        if self.global_model is None:
             logger.error("Cannot save global model: model is not initialized.")
             return
        try:
            self.global_model.save(filepath)
            logger.info(f"Global model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving global model: {e}")
    
    def load_global_model(self, filepath: str = "./models/global_model.h5"):
        """Load the global model"""
        try:
            self.global_model = keras.models.load_model(filepath) # keras.models should resolve
            logger.info(f"Global model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading global model: {e}")

# The test_server function remains similar, ensuring it doesn't rely on external keras imports directly.
def test_server():
    """Test the federated server functionality"""
    def create_test_model():
        inputs = keras.Input(shape=(10, 78))  # Also update shape to match your config
        x = keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam', 
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=[keras.metrics.Accuracy()]  # ← Use actual object
        )
        return model

    # Test server initialization
    server = FederatedServer(create_test_model)
    
    # Test basic operations
    weights = server.get_global_model_weights()
    print(f"Global model has {len(weights)} weight matrices")
    
    # Test evaluation with dummy data
    X_test = np.random.random((100, 50, 78)).astype(np.float32)
    y_test = np.random.randint(0, 10, 100).astype(np.int32)
    
    eval_metrics = server.evaluate_global_model(X_test, y_test)
    print(f"Initial evaluation - Loss: {eval_metrics['global_loss']:.4f}, Accuracy: {eval_metrics['global_accuracy']:.4f}")
    
    print("Server test passed!")

if __name__ == "__main__":
    test_server()