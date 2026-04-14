# preprocessing/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List, Any, Optional, Union
import pickle
import joblib
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class DataUtils:
    def __init__(self, use_kafka: bool = False):
        self.use_kafka = use_kafka
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: Optional[List[str]] = None
        self.feature_columns_cleaned: Optional[List[str]] = None
        self.model = None
        
        # Incremental stats for chunked fitting
        self._partial_sums: Optional[np.ndarray] = None
        self._partial_sq_sums: Optional[np.ndarray] = None
        self._count = 0
        self._fitted_scaler = False
        self._fitted_encoder = False

    def _handle_missing_values(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Handle missing and infinite values in a chunk"""
        df_chunk = df_chunk.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median_val = df_chunk[col].median()
            df_chunk[col].fillna(median_val, inplace=True)

        categorical_cols = df_chunk.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            mode_series = df_chunk[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else 'Unknown'
            df_chunk[col].fillna(mode_val, inplace=True)

        return df_chunk

    def _update_scaling_stats(self, df_chunk: pd.DataFrame):
        """Update partial sums for StandardScaler calculation"""
        numeric_chunk = df_chunk.select_dtypes(include=[np.number])
        numeric_values = numeric_chunk.values.astype(np.float64)

        if self._partial_sums is None:
            self._partial_sums = np.nansum(numeric_values, axis=0)
            self._partial_sq_sums = np.nansum(numeric_values**2, axis=0)
            self._count = len(numeric_chunk)
        else:
            self._partial_sums += np.nansum(numeric_values, axis=0)
            self._partial_sq_sums += np.nansum(numeric_values**2, axis=0)
            self._count += len(numeric_chunk)

    def _fit_scaler_from_stats(self):
        """Fit the StandardScaler using calculated statistics"""
        if self._partial_sums is not None and self._partial_sq_sums is not None:
            mean = self._partial_sums / self._count
            variance = (self._partial_sq_sums / self._count) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 1e-8))

            self.scaler.mean_ = mean.astype(np.float64)
            self.scaler.scale_ = std.astype(np.float64)
            self.scaler.n_features_in_ = len(mean)
            self.scaler.n_samples_seen_ = self._count
            self._fitted_scaler = True
            logger.info(f"Fitted StandardScaler on {self._count} samples.")
        else:
            logger.warning("No data found to fit scaler.")

    def _fit_label_encoder_incrementally(self, y_chunks: List[pd.Series]):
        """Fit the LabelEncoder on all label chunks"""
        if not y_chunks:
            logger.error("No label chunks provided.")
            return

        all_labels = pd.concat(y_chunks, ignore_index=True)
        self.label_encoder.fit(all_labels)
        self._fitted_encoder = True
        logger.info(f"Fitted LabelEncoder with {len(self.label_encoder.classes_)} classes.")

    def _transform_chunk(self, df_chunk: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a single chunk after scalers are fitted"""
        if not self._fitted_scaler or not self._fitted_encoder:
            raise RuntimeError("Scaler and/or Encoder must be fitted before transforming chunks.")

        X_chunk = df_chunk.drop(columns=[target_column])
        y_chunk = df_chunk[target_column]

        X_numeric = X_chunk.select_dtypes(include=[np.number])
        
        # Handle feature columns safely
        if self.feature_columns_cleaned is not None:
            X_numeric = X_numeric.reindex(
                columns=[col for col in self.feature_columns_cleaned if col in X_numeric.columns],
                fill_value=0.0
            )
        elif self.feature_columns is not None:
            # Fallback (shouldn't normally happen)
            cleaned_orig = [str(col).strip().replace(' ', '_').replace('-', '_') 
                           for col in self.feature_columns]
            X_numeric = X_numeric.reindex(
                columns=[col for col in cleaned_orig if col in X_numeric.columns],
                fill_value=0.0
            )

        X_values = X_numeric.values.astype(np.float64)
        X_scaled_chunk = self.scaler.transform(X_values)
        y_encoded_chunk = self.label_encoder.transform(y_chunk)

        return X_scaled_chunk, np.asarray(y_encoded_chunk, dtype=np.int32)

    def prepare_data_for_training_chunked(
        self,
        csv_paths: List[str],
        target_column: str,
        sequence_length: int = 50,
        chunk_size: int = 10000,
        output_dir: str = "./data/processed_chunks/",
        use_kafka: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare data by processing CSVs in chunks for TRAINING ONLY.
        NO KAFKA INVOLVED IN THIS METHOD.
        Samples rows in pattern: 1-5, 11-15, 21-25, etc. (every 10 rows, take 5 consecutive)
        """
        logger.info("Starting chunked data preparation...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize variables for incremental stats
        all_label_chunks = []
        chunk_files = []
        total_samples_processed = 0
        first_chunk_original_cols = None
        
        def should_skip_row(index):
            """Skip rows not in the pattern 1-5, 11-15, 21-25, etc."""
            if index == 0:  # Keep header
                return False
            # Convert to 0-based index for calculation (row 1 = index 0 after header)
            row_num = index - 1  # Adjust for header
            block = row_num // 10  # Which 10-row block (0, 1, 2, ...)
            position_in_block = row_num % 10  # Position within block (0-9)
            # Keep positions 0-4 (which correspond to rows 1-5, 11-15, 21-25, etc.)
            return position_in_block >= 5
        
        # Process each CSV file
        for csv_path in csv_paths:
            logger.info(f"Processing file: {csv_path}")
            try:
                # Read CSV with row sampling pattern
                df_sample = pd.read_csv(csv_path, skiprows=should_skip_row)
                
                # Create chunk iterator from sampled dataframe
                chunk_iter = [df_sample[i:i+chunk_size] for i in range(0, len(df_sample), chunk_size)]
                
                for chunk_idx, chunk in enumerate(chunk_iter):
                    if chunk.empty:
                        continue
                        
                    # Store original column names from first chunk
                    if first_chunk_original_cols is None:
                        first_chunk_original_cols = [
                            col for col in chunk.columns 
                            if col != target_column
                        ]
                    
                    # Clean column names (strip spaces)
                    chunk.columns = chunk.columns.str.strip()
                    
                    # Handle missing values
                    chunk = self._handle_missing_values(chunk)
                    
                    # Store labels for encoder fitting
                    all_label_chunks.append(chunk[target_column].copy())
                    
                    # Update scaling stats (only numeric columns)
                    features_chunk = chunk.drop(columns=[target_column])
                    self._update_scaling_stats(features_chunk)
                    
                    total_samples_processed += len(chunk)
                    
                    if total_samples_processed % 100000 == 0:
                        logger.info(f"Processed {total_samples_processed} samples...")

            except Exception as e:
                logger.error(f"Error processing file {csv_path}: {e}")
                continue

        if not all_label_chunks:
            raise ValueError("No data loaded from CSV files.")

        # Fit encoders
        self._fit_label_encoder_incrementally(all_label_chunks)
        self._fit_scaler_from_stats()

        # Store both original and cleaned feature columns
        if first_chunk_original_cols:
            self.feature_columns = first_chunk_original_cols
            self.feature_columns_cleaned = [
                str(col).strip().replace(' ', '_').replace('-', '_')
                for col in first_chunk_original_cols
            ]
            logger.info(f"Stored {len(self.feature_columns)} ORIGINAL feature columns for inference.")
        else:
            raise ValueError("Could not determine feature columns.")

        # Second pass: Transform and save chunks
        chunk_idx_global = 0
        last_X_scaled_shape = None
        
        # Reuse the same sampling function for consistency
        for csv_path in csv_paths:
            logger.info(f"Transforming: {csv_path}")
            try:
                # Apply same sampling pattern
                df_sample = pd.read_csv(csv_path, skiprows=should_skip_row)
                chunk_iter = [df_sample[i:i+chunk_size] for i in range(0, len(df_sample), chunk_size)]
                
                for chunk in chunk_iter:
                    if chunk.empty:
                        continue
                        
                    chunk_clean = chunk.copy()
                    chunk_clean.columns = (
                        chunk_clean.columns
                        .str.strip()
                        .str.replace(' ', '_')
                        .str.replace('-', '_')
                    )
                    chunk_clean = self._handle_missing_values(chunk_clean)
                    X_scaled, y_encoded = self._transform_chunk(chunk_clean, target_column)

                    # Save chunks
                    X_file = f"{output_dir}/X_chunk_{chunk_idx_global}.npy"
                    y_file = f"{output_dir}/y_chunk_{chunk_idx_global}.npy"
                    np.save(X_file, X_scaled)
                    np.save(y_file, y_encoded)
                    chunk_files.append((X_file, y_file))
                    last_X_scaled_shape = X_scaled.shape
                    chunk_idx_global += 1

            except Exception as e:
                logger.error(f"Error transforming {csv_path}: {e}")
                continue

        # Create label mapping with safe conversion
        label_mapping = {}
        if self._fitted_encoder and hasattr(self.label_encoder, 'classes_'):
            encoded_classes = self.label_encoder.transform(self.label_encoder.classes_)
            # Safe conversion to list
            encoded_array = np.asarray(encoded_classes)
            encoded_list = encoded_array.tolist()
            label_mapping = dict(zip(self.label_encoder.classes_, encoded_list))

        metadata = {
            'feature_columns': self.feature_columns,
            'feature_columns_cleaned': self.feature_columns_cleaned,
            'label_mapping': label_mapping,
            'input_shape': (sequence_length, last_X_scaled_shape[1]) if last_X_scaled_shape is not None else (sequence_length, 0),
            'num_classes': len(label_mapping),
            'sequence_length': sequence_length,
            'chunk_files': chunk_files,
            'total_samples': total_samples_processed,
            'fitted_scaler': self._fitted_scaler,
            'fitted_encoder': self._fitted_encoder
        }

        logger.info(f"Preparation completed. Saved {len(chunk_files)} chunks.")
        return metadata

    def load_and_prepare_sequences_from_chunks(
        self,
        metadata: Dict[str, Any],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Load preprocessed chunks and create sequences (standard approach)
        This works with sequence_length=10 which uses ~7MB per array
        """
        logger.info("Loading sequences from chunks...")
        chunk_files = metadata['chunk_files']
        sequence_length = metadata['sequence_length']

        all_X_sequences = []
        all_y_sequences = []

        # Process chunks in smaller batches to manage memory
        batch_size = 100  # Process 100 chunks at a time
        for i in range(0, len(chunk_files), batch_size):
            batch_files = chunk_files[i:i+batch_size]
            batch_X = []
            batch_y = []
            
            for X_file, y_file in batch_files:
                try:
                    X_chunk = np.load(X_file)
                    y_chunk = np.load(y_file)
                    X_seq, y_seq = self._create_sequences(X_chunk, y_chunk, sequence_length)
                    
                    if len(X_seq) > 0:
                        batch_X.append(X_seq)
                        batch_y.append(y_seq)
                        
                except Exception as e:
                    logger.error(f"Error loading chunk {X_file}: {e}")
                    continue
            
            if batch_X:
                all_X_sequences.extend(batch_X)
                all_y_sequences.extend(batch_y)
        
        if not all_X_sequences:
            raise ValueError("No sequences were created from chunks.")

        # Concatenate all sequences
        X_all_seq = np.concatenate(all_X_sequences, axis=0)
        y_all_seq = np.concatenate(all_y_sequences, axis=0)

        logger.info(f"Total sequences created: {X_all_seq.shape[0]}")

        # Shuffle and split
        total_samples = len(X_all_seq)
        indices = np.random.permutation(total_samples)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_data = {'X': X_all_seq[train_idx], 'y': y_all_seq[train_idx]}
        val_data = {'X': X_all_seq[val_idx], 'y': y_all_seq[val_idx]}
        test_data = {'X': X_all_seq[test_idx], 'y': y_all_seq[test_idx]}

        logger.info(f"Data split: Train={train_data['X'].shape}, Val={val_data['X'].shape}, Test={test_data['X'].shape}")
        return train_data, val_data, test_data
    
    def _create_sequences(self, data: np.ndarray, labels: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        if len(data) <= sequence_length:
            return np.array([]), np.array([])
            
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(labels[i + sequence_length])
    
        # Use float32 to save memory
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def save_preprocessing_objects(self, filepath: str):
        """Save scaler, encoder, and feature columns"""
        if not (self._fitted_scaler and self._fitted_encoder):
            raise RuntimeError("Objects not fitted. Cannot save.")
        
        objects = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'feature_columns_cleaned': self.feature_columns_cleaned,
            '_fitted_scaler': self._fitted_scaler,
            '_fitted_encoder': self._fitted_encoder
        }
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        logger.info(f"Preprocessing objects saved to {filepath}")

    def load_preprocessing_objects(self, filepath: str):
        """Load scaler, encoder, and feature columns"""
        try:
            with open(filepath, 'rb') as f:
                objects = pickle.load(f)
            self.scaler = objects['scaler']
            self.label_encoder = objects['label_encoder']
            self.feature_columns = objects.get('feature_columns')
            self.feature_columns_cleaned = objects.get('feature_columns_cleaned', self.feature_columns)
            self._fitted_scaler = objects.get('_fitted_scaler', True)
            self._fitted_encoder = objects.get('_fitted_encoder', True)
            logger.info(f"Preprocessing objects loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading preprocessing objects: {e}")
            raise

    # === REAL-TIME INFERENCE METHODS ===

    def prepare_single_sample(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """
        Prepare a single sample for prediction.
        Converts raw features (with spaces) to cleaned format and scales.
        """
        if not self._fitted_scaler:
            raise RuntimeError("Scaler not fitted. Load preprocessing objects first.")
        
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not initialized. Call prepare_data_for_training_chunked() first.")

        # Convert raw keys to cleaned format (match training)
        cleaned_features = {}
        for k, v in features_dict.items():
            clean_key = str(k).strip().replace(' ', '_').replace('-', '_')
            cleaned_features[clean_key] = v
        
        # Build feature vector in training order
        feature_vector = []
        for orig_col in self.feature_columns:
            clean_col = str(orig_col).strip().replace(' ', '_').replace('-', '_')
            val = cleaned_features.get(clean_col, 0.0)
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            feature_vector.append(float(val))
        
        # Scale and return
        X = np.array(feature_vector).reshape(1, -1)
        return self.scaler.transform(X)

    def predict(self, features_dict: Dict[str, Any]) -> str:
        """
        Predict attack type for a single sample.
        Returns human-readable label (e.g., 'Bot', 'DDoS').
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        X_scaled = self.prepare_single_sample(features_dict)
        encoded_pred = self.model.predict(X_scaled)[0]
        
        if self._fitted_encoder:
            return str(self.label_encoder.inverse_transform([encoded_pred])[0])
        return str(encoded_pred)

    def predict_proba(self, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction probabilities for all classes"""
        if self.model is None:
            raise RuntimeError("No model loaded.")
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError("Model doesn't support probability predictions.")
        
        X_scaled = self.prepare_single_sample(features_dict)
        proba = self.model.predict_proba(X_scaled)[0]
        
        if self._fitted_encoder:
            classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
            return {str(cls): float(p) for cls, p in zip(classes, proba)}
        return {f"class_{i}": float(p) for i, p in enumerate(proba)}

    def load_model(self, model_path: str):
        """Load a trained scikit-learn/XGBoost model"""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")