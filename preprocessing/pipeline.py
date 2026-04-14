# preprocessing/pipeline.py
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from .utils import DataUtils
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # CRITICAL FIX: Disable Kafka during training
        self.data_utils = DataUtils()
        self.sequence_length = self.config['data']['sequence_length']
        self.chunk_size = self.config.get('data', {}).get('chunk_size', 10000)
        self.processed_data_dir = self.config.get('data', {}).get('processed_data_dir', './data/processed_chunks/')

    def get_class_weights(self, y: np.ndarray) -> dict:
        """Calculate class weights for imbalanced dataset"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        logger.info(f"Class weights calculated: {class_weights}")
        return class_weights

    def _get_csv_file_paths(self) -> List[str]:
        """Get full paths to all CSV files"""
        dataset_path = Path(self.config['data']['dataset_path'])
        csv_files = self.config['data']['csv_files']
        
        csv_paths = []
        for filename in csv_files:
            if '*' in filename:
                csv_paths.extend([str(f) for f in dataset_path.glob(filename)])
            else:
                full_path = dataset_path / filename
                if full_path.exists():
                    csv_paths.append(str(full_path))
                else:
                    logger.warning(f"CSV file not found: {full_path}")
        
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {dataset_path} matching patterns: {csv_files}")
            
        logger.info(f"Found {len(csv_paths)} CSV files to process.")
        return csv_paths

    def preprocess_data(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Complete preprocessing pipeline using chunked processing"""
        logger.info("Starting chunked preprocessing pipeline...")
        
        dataset_path = Path(self.config['data']['dataset_path'])
        csv_files = self.config['data']['csv_files']
        
        csv_paths = [str(dataset_path / filename) for filename in csv_files if (dataset_path / filename).exists()]
        
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        logger.info(f"Found {len(csv_paths)} CSV files to process.")
        
        # Prepare data in chunks and save to disk
        metadata = self.data_utils.prepare_data_for_training_chunked(
            csv_paths,
            self.config['data']['target_column'],
            self.sequence_length,
            self.chunk_size,
            self.processed_data_dir
        )
        
        # NOW LOAD SEQUENCES FROM CHUNKS (this should work with 4.4M samples)
        train_data, val_data, test_data = self.data_utils.load_and_prepare_sequences_from_chunks(
            metadata,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # Calculate class weights based on training data
        class_weights = self.get_class_weights(train_data['y'])
        metadata['class_weights'] = class_weights
        
        logger.info("Chunked preprocessing completed successfully!")
        logger.info(f"Train set: {train_data['X'].shape}")
        logger.info(f"Validation set: {val_data['X'].shape}")
        logger.info(f"Test set: {test_data['X'].shape}")
        
        return train_data, val_data, test_data, metadata

    def _estimate_class_weights_from_chunks(self, metadata: Dict) -> dict:
        """Estimate class weights by sampling a few chunks"""
        logger.info("Estimating class weights from sample chunks...")
        
        # Sample first few chunks to estimate class distribution
        chunk_files = metadata['chunk_files'][:10]  # Sample first 10 chunks
        all_labels = []
        
        for _, y_file in chunk_files:
            try:
                y_chunk = np.load(y_file)
                all_labels.extend(y_chunk.tolist())
            except Exception as e:
                logger.warning(f"Error loading chunk for class weights: {e}")
                continue
        
        if not all_labels:
            # Fallback to equal weights
            num_classes = metadata.get('num_classes', 13)
            return {i: 1.0 for i in range(num_classes)}
        
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(all_labels)
        weights = compute_class_weight('balanced', classes=classes, y=np.array(all_labels))
        class_weights = dict(zip(classes, weights))
        logger.info(f"Estimated class weights: {class_weights}")
        return class_weights

    def save_preprocessing_artifacts(self, filepath: str = "./models/preprocessing.pkl"):
        """Save preprocessing objects"""
        self.data_utils.save_preprocessing_objects(filepath)
        logger.info(f"Preprocessing artifacts saved to {filepath}")

    def load_preprocessing_artifacts(self, filepath: str = "./models/preprocessing.pkl"):
        """Load preprocessing objects"""
        self.data_utils.load_preprocessing_objects(filepath)
        logger.info(f"Preprocessing artifacts loaded from {filepath}")