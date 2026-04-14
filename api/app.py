# api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import logging
from datetime import datetime
import psutil
import os
from typing import Optional
import yaml
import pickle
import keras

# Import your models (adjust path if needed)
from .models import (
    PredictionRequest, BatchPredictionRequest, 
    PredictionResponse, BatchPredictionResponse,
    ModelInfoResponse, HealthCheckResponse,
    create_prediction_response, create_batch_prediction_response
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
global_model: Optional[keras.Model] = None
label_mapping: dict = {}
feature_columns: list = []
model_loaded: bool = False
config: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global global_model, label_mapping, feature_columns, model_loaded, config
    
    # Startup
    logger.info("Starting FL-IDS API...")
    
    # Load configuration
    try:
        with open("./configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
    
    # Load the global model
    model_path = "./models/global_model.h5"
    if os.path.exists(model_path):
        try:
            global_model = keras.models.load_model(model_path)
            logger.info(f"Global model loaded from {model_path}")
            
            # Load preprocessing objects
            preprocessing_path = "./models/preprocessing.pkl"
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    preprocessing_objects = pickle.load(f)
                    
                    label_encoder = preprocessing_objects['label_encoder']
                    label_mapping = {i: str(cls) for i, cls in enumerate(label_encoder.classes_)}
                    
                    feature_columns = preprocessing_objects['feature_columns']
                    logger.info(f"Loaded label mapping with {len(label_mapping)} classes")
            
            model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    else:
        logger.warning(f"Model file not found at {model_path}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FL-IDS API...")

app = FastAPI(
    title="FL-IDS Enterprise API",
    description="Federated Learning based Intrusion Detection System API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "FL-IDS Enterprise API", "status": "running"}

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        kafka_connected=True,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    if not model_loaded or global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_shape = [int(s) for s in global_model.input_shape[1:]]
    return ModelInfoResponse(
        model_name="LSTM_Attention_IDS_Model",
        model_version="1.0.0",
        num_classes=len(label_mapping),
        input_shape=input_shape,
        status="loaded"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_loaded or global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        features = convert_flow_to_features(request.flow_data, feature_columns)
        
        # CRITICAL FIX: Create sequence of length 10 by repeating the feature vector
        # Model expects shape (1, 10, 78), not (1, 1, 78)
        features_sequence = np.tile(features, (10, 1))  # Repeat 10 times: (10, 78)
        features_sequence = features_sequence.reshape(1, 10, -1)  # Reshape to (1, 10, 78)
        
        prediction_probs = global_model.predict(features_sequence, verbose="0")
        predicted_class_idx = int(np.argmax(prediction_probs[0]))
        confidence = float(np.max(prediction_probs[0]))
        prediction_score = float(prediction_probs[0][predicted_class_idx])
        
        attack_type = label_mapping.get(predicted_class_idx, "Unknown")
        response = create_prediction_response(attack_type, confidence, prediction_score)
        
        logger.info(f"Prediction: {attack_type} with confidence {confidence:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model_loaded or global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        for flow_data in request.flows:
            features = convert_flow_to_features(flow_data, feature_columns)
            
            # CRITICAL FIX: Create sequence of length 10 for each sample
            features_sequence = np.tile(features, (10, 1))
            features_sequence = features_sequence.reshape(1, 10, -1)
            
            prediction_probs = global_model.predict(features_sequence, verbose="0")
            predicted_class_idx = int(np.argmax(prediction_probs[0]))
            confidence = float(np.max(prediction_probs[0]))
            prediction_score = float(prediction_probs[0][predicted_class_idx])
            
            attack_type = label_mapping.get(predicted_class_idx, "Unknown")
            pred_response = create_prediction_response(attack_type, confidence, prediction_score)
            predictions.append(pred_response)
        
        batch_response = create_batch_prediction_response(predictions)
        logger.info(f"Batch prediction: {len(predictions)} flows processed")
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    # Handle Windows drive letter
    disk_usage = psutil.disk_usage('C:\\' if os.name == 'nt' else '/').percent
    process_count = len(psutil.pids())
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_usage_percent": disk_usage,
        "process_count": process_count,
        "timestamp": datetime.now().isoformat()
    }

def convert_flow_to_features(flow_data, expected_features):
    """Convert flow data to feature array using dictionary access"""
    # Convert Pydantic model to dictionary
    flow_dict = flow_data.dict() if hasattr(flow_data, 'dict') else vars(flow_data)
    
    features = []
    
    # Handle the first 5 known features with flexible key matching
    known_features_map = {
        'Dst Port': ['dst_port', 'dstport', 'destination_port', 'dst_port'],
        'Protocol': ['protocol', 'proto'],
        'Flow Duration': ['flow_duration', 'duration'],
        'Tot Fwd Pkts': ['tot_fwd_pkts', 'total_forward_packets', 'fwd_pkts'],
        'Tot Bwd Pkts': ['tot_bwd_pkts', 'total_backward_packets', 'bwd_pkts']
    }
    
    for i, col in enumerate(expected_features):
        if i < 5:
            # Try multiple possible keys for known features
            found_value = 0.0
            for possible_key in known_features_map.get(col, [col.lower().replace(' ', '_'), col]):
                # Try exact match first
                if col in flow_dict:
                    found_value = flow_dict[col]
                    break
                # Try cleaned key
                elif possible_key in flow_dict:
                    found_value = flow_dict[possible_key]
                    break
                # Try case-insensitive match
                elif any(k.lower() == possible_key.lower() for k in flow_dict.keys()):
                    key_match = next(k for k in flow_dict.keys() if k.lower() == possible_key.lower())
                    found_value = flow_dict[key_match]
                    break
            
            features.append(float(found_value) if found_value is not None else 0.0)
        else:
            # For remaining features, try direct key access
            col_clean = col.lower().replace(' ', '_').replace('-', '_')
            
            # Try multiple variations
            value = 0.0
            if col in flow_dict:
                value = flow_dict[col]
            elif col_clean in flow_dict:
                value = flow_dict[col_clean]
            elif col.replace(' ', '') in flow_dict:
                value = flow_dict[col.replace(' ', '')]
            else:
                # Log missing feature for debugging
                logger.debug(f"Feature '{col}' not found in flow data. Available keys: {list(flow_dict.keys())}")
            
            features.append(float(value) if value is not None else 0.0)
    
    # Ensure we have exactly 78 features
    if len(features) != 78:
        logger.warning(f"Feature count mismatch: expected 78, got {len(features)}")
        # Pad or truncate to 78 features
        if len(features) < 78:
            features.extend([0.0] * (78 - len(features)))
        else:
            features = features[:78]
    
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)