import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedAggregator:
    def __init__(self):
        self.method = 'fedavg'
        logger.info("Initialized federated aggregator")
    
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                 client_samples: List[int], method: str = 'fedavg') -> List[np.ndarray]:
        """
        Aggregate client weights using specified method
        
        Args:
            client_weights: List of weight lists from each client
            client_samples: Number of samples each client trained on
            method: Aggregation method ('fedavg', 'uniform', etc.)
        
        Returns:
            Aggregated weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        if method.lower() == 'fedavg':
            return self._fedavg_aggregate(client_weights, client_samples)
        elif method.lower() == 'uniform':
            return self._uniform_aggregate(client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _fedavg_aggregate(self, client_weights: List[List[np.ndarray]], 
                         client_samples: List[int]) -> List[np.ndarray]:
        """
        Federated Averaging (FedAvg) aggregation
        Weighted average based on number of samples per client
        """
        if len(client_weights) != len(client_samples):
            raise ValueError("Number of weight sets must match number of sample counts")
        
        # Calculate total samples
        total_samples = sum(client_samples)
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
        
        # Calculate weights for each client
        client_weights_normalized = [samples / total_samples for samples in client_samples]
        
        # Aggregate each layer separately
        num_layers = len(client_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Stack weights for this layer from all clients
            layer_weights = []
            layer_weights_normalized = []
            
            for client_idx, weights in enumerate(client_weights):
                layer_weights.append(weights[layer_idx])
                layer_weights_normalized.append(client_weights_normalized[client_idx])
            
            # Perform weighted average
            weighted_sum = np.zeros_like(layer_weights[0])
            for w, weight in zip(layer_weights, layer_weights_normalized):
                weighted_sum += w * weight
            
            aggregated_weights.append(weighted_sum)
        
        logger.info(f"Aggregated weights from {len(client_weights)} clients using FedAvg")
        return aggregated_weights
    
    def _uniform_aggregate(self, client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Uniform averaging (simple mean)
        """
        num_clients = len(client_weights)
        if num_clients == 0:
            raise ValueError("No clients to aggregate")
        
        num_layers = len(client_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Average weights for this layer across all clients
            layer_stack = np.stack([weights[layer_idx] for weights in client_weights])
            averaged_layer = np.mean(layer_stack, axis=0)
            aggregated_weights.append(averaged_layer)
        
        logger.info(f"Aggregated weights from {len(client_weights)} clients using uniform averaging")
        return aggregated_weights
    
    def secure_aggregate(self, client_weights: List[List[np.ndarray]], 
                        client_samples: List[int], noise_scale: float = 0.01) -> List[np.ndarray]:
        """
        Secure aggregation with differential privacy (simplified version)
        Adds noise to protect privacy
        """
        # First perform regular FedAvg aggregation
        aggregated_weights = self._fedavg_aggregate(client_weights, client_samples)
        
        # Add Gaussian noise for differential privacy
        noised_weights = []
        for weights in aggregated_weights:
            noise = np.random.normal(0, noise_scale, weights.shape)
            noised_weights.append(weights + noise)
        
        logger.info(f"Applied secure aggregation with noise scale {noise_scale}")
        return noised_weights

class AdvancedAggregator(FederatedAggregator):
    """Extended aggregator with additional methods"""
    
    def trimmed_mean_aggregate(self, client_weights: List[List[np.ndarray]], 
                              trim_ratio: float = 0.1) -> List[np.ndarray]:
        """
        Trimmed mean aggregation to handle Byzantine failures
        Removes extreme values before averaging
        """
        if not client_weights:
            raise ValueError("No client weights provided")
        
        num_layers = len(client_weights[0])
        num_clients = len(client_weights)
        
        # Calculate number of clients to trim from each side
        trim_count = int(trim_ratio * num_clients)
        
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Get weights for this layer from all clients
            layer_weights = [weights[layer_idx] for weights in client_weights]
            
            # For each parameter in the layer, calculate trimmed mean
            if len(layer_weights[0].shape) == 0:  # Scalar
                params = [float(w) for w in layer_weights]
                sorted_params = sorted(params)
                trimmed_params = sorted_params[trim_count:num_clients-trim_count]
                aggregated_param = np.mean(trimmed_params)
                aggregated_weights.append(np.array(aggregated_param))
            else:  # Array
                # Stack along new dimension and compute trimmed mean
                stacked = np.stack(layer_weights, axis=-1)  # Shape: [..., num_clients]
                
                # Sort along the last dimension
                sorted_stacked = np.sort(stacked, axis=-1)
                
                # Take middle values (after trimming)
                trimmed = sorted_stacked[..., trim_count:num_clients-trim_count]
                
                # Compute mean along the last dimension
                aggregated_layer = np.mean(trimmed, axis=-1)
                aggregated_weights.append(aggregated_layer)
        
        logger.info(f"Applied trimmed mean aggregation with trim ratio {trim_ratio}")
        return aggregated_weights
    
    def krum_aggregate(self, client_weights: List[List[np.ndarray]], 
                      num_selected: int = 1) -> List[np.ndarray]:
        """
        Krum aggregation algorithm for Byzantine robustness
        Selects the client update closest to its neighbors
        """
        if not client_weights:
            raise ValueError("No client weights provided")
        
        num_clients = len(client_weights)
        if num_selected >= num_clients:
            raise ValueError("num_selected must be less than number of clients")
        
        # Calculate distances between all pairs of clients
        distances = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                # Calculate squared Euclidean distance between client updates
                diff = [np.sum((w1 - w2)**2) for w1, w2 in zip(client_weights[i], client_weights[j])]
                dist = sum(diff)  # Sum across all layers
                distances[i][j] = distances[j][i] = dist
        
        # For each client, find sum of distances to k closest neighbors
        k = num_clients - 2  # As per original Krum paper
        scores = []
        
        for i in range(num_clients):
            neighbor_distances = distances[i].copy()
            neighbor_distances[i] = float('inf')  # Exclude self
            sorted_distances = np.sort(neighbor_distances)
            score = np.sum(sorted_distances[:k])
            scores.append(score)
        
        # Select clients with lowest scores
        selected_indices = np.argsort(scores)[:num_selected]
        
        # Average the selected client updates
        selected_weights = [client_weights[i] for i in selected_indices]
        
        # Simple average of selected weights
        num_layers = len(selected_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_stack = np.stack([weights[layer_idx] for weights in selected_weights])
            averaged_layer = np.mean(layer_stack, axis=0)
            aggregated_weights.append(averaged_layer)
        
        logger.info(f"Applied Krum aggregation, selected {num_selected} out of {num_clients} clients")
        return aggregated_weights

def test_aggregator():
    """Test the aggregator functionality"""
    import numpy as np
    
    # Create dummy client weights (simulating 3 clients, each with 2 layers)
    client_weights = []
    for i in range(3):
        # Layer 1: shape (10, 5)
        layer1 = np.random.random((10, 5)) + i * 0.1
        # Layer 2: shape (5,)
        layer2 = np.random.random(5) + i * 0.1
        client_weights.append([layer1, layer2])
    
    client_samples = [100, 150, 200]  # Different sample sizes
    
    # Test FedAvg
    aggregator = FederatedAggregator()
    fedavg_result = aggregator.aggregate(client_weights, client_samples, 'fedavg')
    print(f"FedAvg result shapes: {[w.shape for w in fedavg_result]}")
    
    # Test uniform aggregation
    uniform_result = aggregator.aggregate(client_weights, client_samples, 'uniform')
    print(f"Uniform result shapes: {[w.shape for w in uniform_result]}")
    
    # Test advanced aggregators
    advanced_agg = AdvancedAggregator()
    trimmed_result = advanced_agg.trimmed_mean_aggregate(client_weights, trim_ratio=0.2)
    print(f"Trimmed mean result shapes: {[w.shape for w in trimmed_result]}")
    
    krum_result = advanced_agg.krum_aggregate(client_weights, num_selected=2)
    print(f"Krum result shapes: {[w.shape for w in krum_result]}")
    
    print("Aggregator tests passed!")

if __name__ == "__main__":
    test_aggregator()