from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NetworkFlow(BaseModel):
    """Single network flow data with all 78 CICIDS2018 features"""
    # All 78 CICIDS2018 features as snake_case fields
    dst_port: Optional[float] = None
    protocol: Optional[float] = None
    flow_duration: Optional[float] = None
    tot_fwd_pkts: Optional[float] = None
    tot_bwd_pkts: Optional[float] = None
    totlen_fwd_pkts: Optional[float] = None
    totlen_bwd_pkts: Optional[float] = None
    fwd_pkt_len_max: Optional[float] = None
    fwd_pkt_len_min: Optional[float] = None
    fwd_pkt_len_mean: Optional[float] = None
    fwd_pkt_len_std: Optional[float] = None
    bwd_pkt_len_max: Optional[float] = None
    bwd_pkt_len_min: Optional[float] = None
    bwd_pkt_len_mean: Optional[float] = None
    bwd_pkt_len_std: Optional[float] = None
    flow_byts_s: Optional[float] = None
    flow_pkts_s: Optional[float] = None
    flow_iat_mean: Optional[float] = None
    flow_iat_std: Optional[float] = None
    flow_iat_max: Optional[float] = None
    flow_iat_min: Optional[float] = None
    fwd_iat_tot: Optional[float] = None
    fwd_iat_mean: Optional[float] = None
    fwd_iat_std: Optional[float] = None
    fwd_iat_max: Optional[float] = None
    fwd_iat_min: Optional[float] = None
    bwd_iat_tot: Optional[float] = None
    bwd_iat_mean: Optional[float] = None
    bwd_iat_std: Optional[float] = None
    bwd_iat_max: Optional[float] = None
    bwd_iat_min: Optional[float] = None
    fwd_psh_flags: Optional[float] = None
    bwd_psh_flags: Optional[float] = None
    fwd_urg_flags: Optional[float] = None
    bwd_urg_flags: Optional[float] = None
    fwd_header_len: Optional[float] = None
    bwd_header_len: Optional[float] = None
    fwd_pkts_s: Optional[float] = None
    bwd_pkts_s: Optional[float] = None
    pkt_len_min: Optional[float] = None
    pkt_len_max: Optional[float] = None
    pkt_len_mean: Optional[float] = None
    pkt_len_std: Optional[float] = None
    pkt_len_var: Optional[float] = None
    fin_flag_cnt: Optional[float] = None
    syn_flag_cnt: Optional[float] = None
    rst_flag_cnt: Optional[float] = None
    psh_flag_cnt: Optional[float] = None
    ack_flag_cnt: Optional[float] = None
    urg_flag_cnt: Optional[float] = None
    cwe_flag_count: Optional[float] = None
    ece_flag_cnt: Optional[float] = None
    down_up_ratio: Optional[float] = None
    pkt_size_avg: Optional[float] = None
    fwd_seg_size_avg: Optional[float] = None
    bwd_seg_size_avg: Optional[float] = None
    fwd_byts_b_avg: Optional[float] = None
    fwd_pkts_b_avg: Optional[float] = None
    fwd_blk_rate_avg: Optional[float] = None
    bwd_byts_b_avg: Optional[float] = None
    bwd_pkts_b_avg: Optional[float] = None
    bwd_blk_rate_avg: Optional[float] = None
    subflow_fwd_pkts: Optional[float] = None
    subflow_fwd_byts: Optional[float] = None
    subflow_bwd_pkts: Optional[float] = None
    subflow_bwd_byts: Optional[float] = None
    init_fwd_win_byts: Optional[float] = None
    init_bwd_win_byts: Optional[float] = None
    fwd_act_data_pkts: Optional[float] = None
    fwd_seg_size_min: Optional[float] = None
    active_mean: Optional[float] = None
    active_std: Optional[float] = None
    active_max: Optional[float] = None
    active_min: Optional[float] = None
    idle_mean: Optional[float] = None
    idle_std: Optional[float] = None
    idle_max: Optional[float] = None
    idle_min: Optional[float] = None
    
    timestamp: Optional[str] = None

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    flow_data: NetworkFlow  # FIXED: Added colon and proper type hint

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    flows: List[NetworkFlow]

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    attack_type: str
    confidence: float
    prediction_score: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]
    total_flows: int
    benign_count: int
    attack_count: int

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    model_version: str
    num_classes: int
    input_shape: List[int]
    status: str

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    kafka_connected: bool
    version: str

# Feature mapping from CICIDS2018 column names to Pydantic field names
FEATURE_MAPPING = {
    'Dst Port': 'dst_port',
    'Protocol': 'protocol',
    'Flow Duration': 'flow_duration',
    'Tot Fwd Pkts': 'tot_fwd_pkts',
    'Tot Bwd Pkts': 'tot_bwd_pkts',
    'TotLen Fwd Pkts': 'totlen_fwd_pkts',
    'TotLen Bwd Pkts': 'totlen_bwd_pkts',
    'Fwd Pkt Len Max': 'fwd_pkt_len_max',
    'Fwd Pkt Len Min': 'fwd_pkt_len_min',
    'Fwd Pkt Len Mean': 'fwd_pkt_len_mean',
    'Fwd Pkt Len Std': 'fwd_pkt_len_std',
    'Bwd Pkt Len Max': 'bwd_pkt_len_max',
    'Bwd Pkt Len Min': 'bwd_pkt_len_min',
    'Bwd Pkt Len Mean': 'bwd_pkt_len_mean',
    'Bwd Pkt Len Std': 'bwd_pkt_len_std',
    'Flow Byts/s': 'flow_byts_s',
    'Flow Pkts/s': 'flow_pkts_s',
    'Flow IAT Mean': 'flow_iat_mean',
    'Flow IAT Std': 'flow_iat_std',
    'Flow IAT Max': 'flow_iat_max',
    'Flow IAT Min': 'flow_iat_min',
    'Fwd IAT Tot': 'fwd_iat_tot',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Fwd IAT Std': 'fwd_iat_std',
    'Fwd IAT Max': 'fwd_iat_max',
    'Fwd IAT Min': 'fwd_iat_min',
    'Bwd IAT Tot': 'bwd_iat_tot',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Bwd IAT Std': 'bwd_iat_std',
    'Bwd IAT Max': 'bwd_iat_max',
    'Bwd IAT Min': 'bwd_iat_min',
    'Fwd PSH Flags': 'fwd_psh_flags',
    'Bwd PSH Flags': 'bwd_psh_flags',
    'Fwd URG Flags': 'fwd_urg_flags',
    'Bwd URG Flags': 'bwd_urg_flags',
    'Fwd Header Len': 'fwd_header_len',
    'Bwd Header Len': 'bwd_header_len',
    'Fwd Pkts/s': 'fwd_pkts_s',
    'Bwd Pkts/s': 'bwd_pkts_s',
    'Pkt Len Min': 'pkt_len_min',
    'Pkt Len Max': 'pkt_len_max',
    'Pkt Len Mean': 'pkt_len_mean',
    'Pkt Len Std': 'pkt_len_std',
    'Pkt Len Var': 'pkt_len_var',
    'FIN Flag Cnt': 'fin_flag_cnt',
    'SYN Flag Cnt': 'syn_flag_cnt',
    'RST Flag Cnt': 'rst_flag_cnt',
    'PSH Flag Cnt': 'psh_flag_cnt',
    'ACK Flag Cnt': 'ack_flag_cnt',
    'URG Flag Cnt': 'urg_flag_cnt',
    'CWE Flag Count': 'cwe_flag_count',
    'ECE Flag Cnt': 'ece_flag_cnt',
    'Down/Up Ratio': 'down_up_ratio',
    'Pkt Size Avg': 'pkt_size_avg',
    'Fwd Seg Size Avg': 'fwd_seg_size_avg',
    'Bwd Seg Size Avg': 'bwd_seg_size_avg',
    'Fwd Byts/b Avg': 'fwd_byts_b_avg',
    'Fwd Pkts/b Avg': 'fwd_pkts_b_avg',
    'Fwd Blk Rate Avg': 'fwd_blk_rate_avg',
    'Bwd Byts/b Avg': 'bwd_byts_b_avg',
    'Bwd Pkts/b Avg': 'bwd_pkts_b_avg',
    'Bwd Blk Rate Avg': 'bwd_blk_rate_avg',
    'Subflow Fwd Pkts': 'subflow_fwd_pkts',
    'Subflow Fwd Byts': 'subflow_fwd_byts',
    'Subflow Bwd Pkts': 'subflow_bwd_pkts',
    'Subflow Bwd Byts': 'subflow_bwd_byts',
    'Init Fwd Win Byts': 'init_fwd_win_byts',
    'Init Bwd Win Byts': 'init_bwd_win_byts',
    'Fwd Act Data Pkts': 'fwd_act_data_pkts',
    'Fwd Seg Size Min': 'fwd_seg_size_min',
    'Active Mean': 'active_mean',
    'Active Std': 'active_std',
    'Active Max': 'active_max',
    'Active Min': 'active_min',
    'Idle Mean': 'idle_mean',
    'Idle Std': 'idle_std',
    'Idle Max': 'idle_max',
    'Idle Min': 'idle_min'
}

def convert_flow_to_features(flow_data: NetworkFlow, expected_features: List[str]) -> np.ndarray:
    """
    Convert NetworkFlow object to feature array matching preprocessing format
    Uses the complete FEATURE_MAPPING for all 78 CICIDS2018 features
    """
    features = []
    
    for col in expected_features:
        if col in FEATURE_MAPPING:
            field_name = FEATURE_MAPPING[col]
            val = getattr(flow_data, field_name, 0.0)
        else:
            # Fallback: try direct attribute access
            clean_col = col.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
            val = getattr(flow_data, clean_col, 0.0)
        
        # Handle None values
        if val is None:
            val = 0.0
            
        features.append(float(val))
    
    # Ensure we have exactly the expected number of features
    if len(features) != len(expected_features):
        logger.warning(f"Feature count mismatch: expected {len(expected_features)}, got {len(features)}")
        if len(features) < len(expected_features):
            features.extend([0.0] * (len(expected_features) - len(features)))
        else:
            features = features[:len(expected_features)]
    
    return np.array(features, dtype=np.float32)

def create_prediction_response(attack_type: str, confidence: float, 
                            prediction_score: float) -> PredictionResponse:
    """Create a prediction response object"""
    from datetime import datetime
    
    return PredictionResponse(
        attack_type=attack_type,
        confidence=confidence,
        prediction_score=prediction_score,
        timestamp=datetime.now().isoformat()
    )

def create_batch_prediction_response(predictions: List[PredictionResponse]) -> BatchPredictionResponse:
    """Create a batch prediction response object"""
    attack_count = sum(1 for p in predictions if p.attack_type.lower() != 'benign')
    benign_count = len(predictions) - attack_count
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_flows=len(predictions),
        benign_count=benign_count,
        attack_count=attack_count
    )

# Example usage validation
def validate_models():
    """Validate the Pydantic models with sample data"""
    # Test single prediction request with sample data
    flow = NetworkFlow(
        dst_port=22.0,
        protocol=6.0,
        flow_duration=1000000.0,
        tot_fwd_pkts=500.0,
        tot_bwd_pkts=10.0,
        totlen_fwd_pkts=50000.0,
        totlen_bwd_pkts=1000.0,
        fwd_pkt_len_max=1000.0,
        fwd_pkt_len_min=100.0,
        fwd_pkt_len_mean=500.0,
        fwd_pkt_len_std=50.0,
        # ... you can add more fields as needed
        timestamp="2023-01-01T00:00:00Z"
    )
    
    req = PredictionRequest(flow_data=flow)
    print(f"Single prediction request created successfully")
    
    # Test response
    resp = create_prediction_response("Bot", 0.95, 0.87)
    print(f"Prediction response created successfully")
    
    print("Model validation passed!")

if __name__ == "__main__":
    validate_models()