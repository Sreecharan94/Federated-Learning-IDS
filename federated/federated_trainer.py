# federated/federated_trainer.py
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from .server import FederatedServer
from .client import ClientManager
from model.architecture import LSTMMultiHeadAttentionModel
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedTrainer:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.model_builder = LSTMMultiHeadAttentionModel(config_path)
        self.server = None
        self.client_manager = None
        
        # Training metrics
        self.metrics = {
            'rounds': [],
            'global_accuracy': [],
            'global_loss': [],
            'avg_client_accuracy': [],
            'avg_client_loss': [],
            'active_clients': [],
            'round_times': [],
            'client_accuracies': [],
            'client_losses': []
        }
        
        logger.info("Initialized federated trainer")
    
    def setup_federated_system(self, num_classes: int):
        """Setup the federated learning system"""
        logger.info("Setting up federated learning system...")
        
        # Create model function
        model_fn = self.model_builder.create_federated_model_fn(num_classes)
        
        # Initialize server
        self.server = FederatedServer(model_fn, self.config_path)
        
        # Initialize client manager
        self.client_manager = ClientManager(self.config_path)
        
        # Create clients
        clients = self.client_manager.create_clients(model_fn)
        
        # Set client manager in server
        self.server.set_client_manager(self.client_manager)
        
        logger.info(f"Federated system setup complete with {len(clients)} clients")
    
    def train_federated_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None, 
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train federated model with numpy arrays (not dictionaries)
        This matches what your preprocessing pipeline returns
        """
        logger.info("Starting federated training...")

        if self.client_manager is None or self.server is None:
            logger.error("Client manager or Server is not set. Cannot start federated training.")
            return {'final_model': None, 'training_history': {}, 'final_accuracy': 0.0}

        # Use the correct method name - assign_data_to_clients
        # This method expects X_train and y_train as numpy arrays
        self.client_manager.assign_data_to_clients(X_train, y_train, non_iid=True)
        
        # Pass test data correctly (can be None)
        training_results = self.server.federated_training(X_test, y_test)
        
        self._extract_training_metrics(training_results)
        
        logger.info("Federated training completed!")
        
        return {
            'final_model': self.server.global_model,
            'training_history': self.metrics,
            'final_accuracy': self.metrics['global_accuracy'][-1] if self.metrics['global_accuracy'] else 0.0
        }
    
    def _extract_training_metrics(self, training_results: Dict[str, Any]):
        """Extract and store training metrics"""
        training_results_list = training_results.get('training_results', [])
        
        for result in training_results_list:
            self.metrics['rounds'].append(result.get('round_num', 0))
            self.metrics['global_accuracy'].append(result.get('global_accuracy', 0.0))
            self.metrics['global_loss'].append(result.get('global_loss', 0.0))
            self.metrics['avg_client_accuracy'].append(result.get('avg_client_accuracy', 0.0))
            self.metrics['avg_client_loss'].append(result.get('avg_client_loss', 0.0))
            self.metrics['active_clients'].append(result.get('active_clients', 0))
            self.metrics['round_times'].append(result.get('round_time', 0.0))
    
    def plot_training_progress(self, save_path: str = "./outputs/plots/federated_training.png"):
        """Plot federated training progress"""
        if not self.metrics['rounds']:
            logger.warning("No training metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Global accuracy
        axes[0, 0].plot(self.metrics['rounds'], self.metrics['global_accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_title('Global Model Accuracy Over Rounds')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Global loss
        axes[0, 1].plot(self.metrics['rounds'], self.metrics['global_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Global Model Loss Over Rounds')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Active clients
        axes[1, 0].plot(self.metrics['rounds'], self.metrics['active_clients'], 'g-', linewidth=2)
        axes[1, 0].set_title('Active Clients Per Round')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Number of Clients')
        axes[1, 0].grid(True)
        
        # Round times
        axes[1, 1].plot(self.metrics['rounds'], self.metrics['round_times'], 'm-', linewidth=2)
        axes[1, 1].set_title('Round Execution Time')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        logger.info(f"Training progress plot saved to {save_path}")
    
    def evaluate_federated_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the federated model"""
        if self.server is None:
            raise ValueError("Federated system not initialized")
        
        eval_results = self.server.evaluate_global_model(X_test, y_test)
        
        logger.info(f"Federated model evaluation: Loss={eval_results['global_loss']:.4f}, "
                   f"Accuracy={eval_results['global_accuracy']:.4f}")
        
        return eval_results
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get statistics about clients"""
        if self.client_manager is None:
            return {}
        
        clients = self.client_manager.get_all_clients()
        stats = {
            'total_clients': len(clients),
            'total_samples': sum(c.get_client_info()['data_samples'] for c in clients),
            'client_info': [c.get_client_info() for c in clients]
        }
        
        return stats

def test_federated_trainer():
    """Test the federated trainer"""
    # Create dummy data with sequence_length=10 (to match your config)
    X_train = np.random.random((1000, 10, 78)).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000).astype(np.int32)
    X_test = np.random.random((200, 10, 78)).astype(np.float32)
    y_test = np.random.randint(0, 10, 200).astype(np.int32)
    
    # Create trainer
    trainer = FederatedTrainer()
    
    # Setup federated system
    trainer.setup_federated_system(num_classes=10)
    
    # Train federated model - pass arrays directly
    results = trainer.train_federated_model(X_train, y_train, X_test, y_test)
    
    print(f"Training completed with final accuracy: {results['final_accuracy']:.4f}")
    
    # Print client statistics
    client_stats = trainer.get_client_statistics()
    print(f"Client statistics: {client_stats}")
    
    print("Federated trainer test passed!")

if __name__ == "__main__":
    test_federated_trainer()