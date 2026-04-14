# model/architecture.py
import tensorflow as tf # Main import
import keras  # Correct import path
# Access submodules via tf.keras.*
from keras.layers import (
    Input, LSTM, Dense, Dropout, MultiHeadAttention, 
    LayerNormalization, Add, GlobalAveragePooling1D
)
from keras.models import Model # Correct import path
from keras.optimizers import Adam # Correct import path
import logging
import yaml # Add yaml import for config loading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMMultiHeadAttentionModel:
    def __init__(self, config_path="./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lstm_units = self.config['model']['lstm_units']
        self.attention_heads = self.config['model']['attention_heads']
        self.dense_units = self.config['model']['dense_units']
        self.dropout_rate = self.config['model']['dropout_rate']
        self.learning_rate = self.config['model']['learning_rate']
        self.input_shape = tuple(self.config['model']['input_shape'])
        self.num_classes = None  # Will be set during model creation
        
    def build_lstm_attention_layer(self, inputs, name_prefix=""):
        """Build LSTM layer with multi-head attention"""
        # LSTM layer
        lstm_out = LSTM(
            self.lstm_units, 
            return_sequences=True, 
            name=f"{name_prefix}lstm"
        )(inputs)
        
        # Multi-head attention
        attention_out = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.lstm_units // self.attention_heads, 
            name=f"{name_prefix}multihead_attention"
        )(lstm_out, lstm_out)
        
        # Add & Norm
        attention_out = Add(name=f"{name_prefix}add")([lstm_out, attention_out])
        attention_out = LayerNormalization(name=f"{name_prefix}layer_norm")(attention_out)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D(name=f"{name_prefix}global_avg_pool")(attention_out)
        
        return pooled
    
    def build_model(self, num_classes: int):
        """Build the complete model"""
        self.num_classes = num_classes
        
        # Input layer
        inputs = Input(shape=self.input_shape, name="input_layer")
        
        # LSTM + Attention layers
        features = self.build_lstm_attention_layer(inputs, name_prefix="")
        
        # Dense layers
        x = Dense(self.dense_units, activation='relu', name="dense_1")(features)
        x = Dropout(self.dropout_rate, name="dropout_1")(x)
        
        x = Dense(self.dense_units // 2, activation='relu', name="dense_2")(x) # Ensure integer division
        x = Dropout(self.dropout_rate, name="dropout_2")(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name="output_layer")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_IDS_Model")
        
        # Compile model - USE CORRECT METRIC FOR SPARSE LABELS
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'] 
        )
        
        logger.info(f"Model built successfully with {num_classes} classes")
        logger.info(f"Model summary:")
        model.summary()
        
        return model
    
    def create_federated_model_fn(self, num_classes: int):
        """Create a function that returns a compiled model for federated learning"""
        def create_model():
            return self.build_model(num_classes)
        return create_model

# Example usage and testing remains largely the same, ensuring tf.keras is used consistently
def test_model():
    """Test the model architecture"""
    import numpy as np
    
    # Create dummy config
    config = {
        'model': {
            'lstm_units': 64,
            'attention_heads': 8,
            'dense_units': 32,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'input_shape': [50, 78]
        }
    }
    
    # Save temp config
    import yaml
    with open('./temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Test model creation
    model_builder = LSTMMultiHeadAttentionModel('./temp_config.yaml')
    model = model_builder.build_model(num_classes=10)
    
    # Test with dummy data
    dummy_input = np.random.random((1, 50, 78))
    prediction = model.predict(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {prediction.shape}")
    print("Model test passed!")
    
    import os
    os.remove('./temp_config.yaml')

if __name__ == "__main__":
    test_model()