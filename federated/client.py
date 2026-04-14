# federated/client.py
import tensorflow as tf
import keras
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import logging
from sklearn.model_selection import train_test_split
import copy
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id: int, model_fn: Callable[[], keras.Model], config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client_id = client_id
        self.model_fn = model_fn
        self.local_model = model_fn() # Create the model instance

        self.local_epochs = self.config['federated']['local_epochs']
        self.local_batch_size = self.config['federated']['local_batch_size']

        # Local data storage
        self.X_local: Optional[np.ndarray] = None
        self.y_local: Optional[np.ndarray] = None
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None

        logger.info(f"Initialized federated client {client_id}")

    def set_local_data(self, X: np.ndarray, y: np.ndarray):
        """Set local training data for the client"""
        # FIX: Check inputs
        if X is None or y is None:
            logger.error(f"Client {self.client_id}: Cannot set local data - X or y is None.")
            return

        self.X_local = X
        self.y_local = y

        # Split data into train and validation - ADD CHECKS FOR None
        if self.X_local is not None and self.y_local is not None and len(self.X_local) > 10:  # Only split if we have enough data and both are not None
            self.train_indices, self.val_indices = train_test_split(
                np.arange(len(self.X_local)), # Use len of X_local which is checked for None
                test_size=0.2,
                random_state=42 + self.client_id
            )
        else:
            if self.X_local is not None:
                self.train_indices = np.arange(len(self.X_local))
                self.val_indices = np.arange(len(self.X_local))
            else:
                 logger.warning(f"Client {self.client_id}: Cannot set local data - X is None.")
                 return # Early return if X is None

        logger.info(f"Client {self.client_id}: Set local data with {len(self.X_local) if self.X_local is not None else 0} samples")
        logger.info(f"Training samples: {len(self.train_indices) if self.train_indices is not None else 0}, Validation samples: {len(self.val_indices) if self.val_indices is not None else 0}")

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current model weights"""
        if self.local_model is None:
            raise ValueError("Local model is not initialized")
        return self.local_model.get_weights()

    def set_model_weights(self, weights: List[np.ndarray]):
        """Set model weights"""
        if self.local_model is None:
            raise ValueError("Local model is not initialized")
        self.local_model.set_weights(weights)

    def train_local_model(self) -> Dict[str, float]:
        """Train the local model"""
        # FIX: Check if data is set
        if self.X_local is None or self.y_local is None:
            raise ValueError("Local data not set for client")

        logger.info(f"Client {self.client_id}: Starting local training...")

        # Get local training data - Safe to use len now if set_local_data succeeded
        if self.train_indices is None: # Check if indices were set
             logger.error(f"Client {self.client_id}: Training indices not set. Did set_local_data run?")
             return {'loss': float('inf'), 'accuracy': 0.0, 'epochs_completed': 0, 'samples_trained': 0} # Return default if no data

        X_train_local = self.X_local[self.train_indices]
        y_train_local = self.y_local[self.train_indices]

        # Train the model locally
        history_obj = self.local_model.fit(
            X_train_local, y_train_local,
            epochs=self.local_epochs,
            batch_size=self.local_batch_size,
            verbose="0",
            validation_data=(
                self.X_local[self.val_indices], self.y_local[self.val_indices]
            ) if self.val_indices is not None and len(self.val_indices) > 0 else None
        )

        # FIX: Check if fit returned a history object
        if history_obj is None:
            logger.error(f"Client {self.client_id}: Model fit returned None. Cannot retrieve metrics.")
            return {'loss': float('inf'), 'accuracy': 0.0, 'epochs_completed': 0, 'samples_trained': len(X_train_local)}

        # Calculate training metrics - SAFE to access history_obj.history now
        final_loss = float(history_obj.history['loss'][-1]) if 'loss' in history_obj.history else 0.0
        final_accuracy = float(history_obj.history['accuracy'][-1]) if 'accuracy' in history_obj.history else 0.0

        metrics = {
            'loss': final_loss,
            'accuracy': final_accuracy,
            'epochs_completed': len(history_obj.history['loss']) if 'loss' in history_obj.history else 0,
            'samples_trained': len(X_train_local)
        }

        logger.info(f"Client {self.client_id}: Training completed. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

        return metrics

    def evaluate_local_model(self) -> Dict[str, float]:
        """Evaluate the local model"""
        if self.X_local is None or self.y_local is None:
            raise ValueError("Local data not set for client")

        # Evaluate on validation data
        if self.val_indices is None or len(self.val_indices) == 0:
             logger.warning(f"Client {self.client_id}: No validation indices set. Skipping evaluation.")
             return {'eval_loss': 0.0, 'eval_accuracy': 0.0}

        X_val_local = self.X_local[self.val_indices]
        y_val_local = self.y_local[self.val_indices]

        # Evaluate
        eval_results = self.local_model.evaluate(X_val_local, y_val_local, verbose="0")

        metrics = {
            'eval_loss': float(eval_results[0]),
            'eval_accuracy': float(eval_results[1])
        }

        logger.info(f"Client {self.client_id}: Evaluation - Loss: {eval_results[0]:.4f}, Accuracy: {eval_results[1]:.4f}")

        return metrics

    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the client"""
        return {
            'client_id': self.client_id,
            'data_samples': len(self.X_local) if self.X_local is not None else 0,
            'train_samples': len(self.train_indices) if self.train_indices is not None else 0,
            'val_samples': len(self.val_indices) if self.val_indices is not None else 0
        }

class ClientManager:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_clients = self.config['federated']['num_clients']
        self.clients: Dict[int, FederatedClient] = {}
        self.participation_rate = self.config['federated']['client_participation_rate']
        self.config_path = config_path # Store config path

    def create_clients(self, model_fn: Callable[[], keras.Model]) -> List[FederatedClient]:
        """Create federated clients"""
        clients = []

        for client_id in range(self.num_clients):
            client = FederatedClient(client_id, model_fn, self.config_path) # Pass the stored config path
            clients.append(client)
            self.clients[client_id] = client

        logger.info(f"Created {len(clients)} federated clients")
        return clients

    def assign_data_to_clients(self, X: np.ndarray, y: np.ndarray, non_iid: bool = True):
        """Assign data to clients (with optional non-IID distribution)"""
        num_samples = len(X)
        samples_per_client = num_samples // self.num_clients

        if non_iid:
            # Non-IID: Each client gets data from specific classes
            unique_labels = np.unique(y)
            num_labels = len(unique_labels)

            for client_id in range(self.num_clients):
                # Assign specific label ranges to clients
                start_label = (client_id * num_labels) // self.num_clients
                end_label = ((client_id + 1) * num_labels) // self.num_clients

                # Get indices for this client's labels
                client_label_mask = np.isin(y, unique_labels[start_label:end_label])
                client_indices = np.where(client_label_mask)[0]

                # Limit number of samples per client
                if len(client_indices) > samples_per_client:
                    client_indices = np.random.choice(client_indices, samples_per_client, replace=False)

                client_X = X[client_indices]
                client_y = y[client_indices]

                self.clients[client_id].set_local_data(client_X, client_y)

        else:
            # IID: Randomly distribute data
            shuffled_indices = np.random.permutation(num_samples)

            for client_id in range(self.num_clients):
                start_idx = client_id * samples_per_client
                end_idx = min((client_id + 1) * samples_per_client, num_samples)

                client_indices = shuffled_indices[start_idx:end_idx]

                client_X = X[client_indices]
                client_y = y[client_indices]

                self.clients[client_id].set_local_data(client_X, client_y)

        logger.info("Assigned data to clients successfully")

    def select_active_clients(self, round_num: int) -> List[FederatedClient]:
        """Select active clients for the current round"""
        num_active = max(1, int(self.num_clients * self.participation_rate))

        # Use round number as seed for consistent selection
        np.random.seed(round_num)
        active_client_ids = np.random.choice(
            list(self.clients.keys()),
            size=num_active,
            replace=False
        )

        active_clients = [self.clients[client_id] for client_id in active_client_ids]

        logger.info(f"Round {round_num}: Selected {len(active_clients)} active clients")

        return active_clients

    def get_all_clients(self) -> List[FederatedClient]:
        """Get all clients"""
        return list(self.clients.values())

def test_client():
    """Test the federated client functionality"""
    # Create a simple model function for testing
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

    # Create dummy data
    X_dummy = np.random.random((1000, 50, 78)).astype(np.float32)
    y_dummy = np.random.randint(0, 10, 1000).astype(np.int32)

    # Test client creation and training
    client = FederatedClient(0, create_test_model)
    client.set_local_data(X_dummy[:200], y_dummy[:200])

    # Test initial evaluation
    eval_before = client.evaluate_local_model()
    print(f"Before training - Loss: {eval_before['eval_loss']:.4f}, Accuracy: {eval_before['eval_accuracy']:.4f}")

    # Train the client
    train_metrics = client.train_local_model()
    print(f"Training - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")

    # Test evaluation after training
    eval_after = client.evaluate_local_model()
    print(f"After training - Loss: {eval_after['eval_loss']:.4f}, Accuracy: {eval_after['eval_accuracy']:.4f}")

    print("Client test passed!")

if __name__ == "__main__":
    test_client()