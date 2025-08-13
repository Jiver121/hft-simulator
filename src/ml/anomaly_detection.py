"""
Anomaly Detection for Financial Markets

This module provides anomaly detection capabilities including market regime
change detection, outlier identification, and GPU-accelerated computations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Comprehensive anomaly detection for financial time series.
    
    Features:
    - Multiple detection algorithms
    - Market regime change detection
    - Real-time anomaly scoring
    - GPU acceleration support
    """
    
    def __init__(self,
                 methods: List[str] = None,
                 contamination: float = 0.1,
                 window_size: int = 100,
                 use_gpu: bool = True):
        """
        Initialize anomaly detector.
        
        Args:
            methods: List of detection methods to use
            contamination: Expected proportion of anomalies
            window_size: Window size for rolling detection
            use_gpu: Whether to use GPU acceleration
        """
        self.methods = methods or ['isolation_forest', 'local_outlier', 'statistical']
        self.contamination = contamination
        self.window_size = window_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
        # GPU device
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        logger.info(f"Initialized AnomalyDetector with methods: {self.methods}")
        logger.info(f"Using device: {self.device}")
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit anomaly detection models.
        
        Args:
            X: Training data
            feature_names: Names of features
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Fit scaler
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Fit different detection models
        for method in self.methods:
            if method == 'isolation_forest':
                self.models['isolation_forest'] = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1
                )
                self.models['isolation_forest'].fit(X_scaled)
                
            elif method == 'local_outlier':
                # Use DBSCAN for local outlier detection
                self.models['local_outlier'] = DBSCAN(
                    eps=0.5,
                    min_samples=5
                )
                labels = self.models['local_outlier'].fit_predict(X_scaled)
                # Count outliers (label -1)
                outlier_ratio = np.sum(labels == -1) / len(labels)
                logger.info(f"Local outlier detection found {outlier_ratio:.2%} outliers")
                
            elif method == 'statistical':
                # Robust covariance-based detection
                self.models['statistical'] = EllipticEnvelope(
                    contamination=self.contamination,
                    random_state=42
                )
                self.models['statistical'].fit(X_scaled)
                
            elif method == 'autoencoder':
                # Neural network-based detection
                self.models['autoencoder'] = AnomalyAutoencoder(
                    input_dim=X.shape[1],
                    encoding_dim=X.shape[1] // 2,
                    device=self.device
                )
                self.models['autoencoder'].fit(X_scaled)
        
        logger.info("Anomaly detection models fitted successfully")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict anomalies using fitted models.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of anomaly predictions by method
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")
        
        # Scale data
        X_scaled = self.scalers['main'].transform(X)
        
        predictions = {}
        
        for method, model in self.models.items():
            try:
                if method == 'isolation_forest':
                    scores = model.decision_function(X_scaled)
                    predictions[method] = {
                        'anomaly_labels': model.predict(X_scaled),
                        'anomaly_scores': scores
                    }
                    
                elif method == 'local_outlier':
                    labels = model.fit_predict(X_scaled)
                    predictions[method] = {
                        'anomaly_labels': labels,
                        'anomaly_scores': (labels == -1).astype(int)
                    }
                    
                elif method == 'statistical':
                    scores = model.decision_function(X_scaled)
                    predictions[method] = {
                        'anomaly_labels': model.predict(X_scaled),
                        'anomaly_scores': scores
                    }
                    
                elif method == 'autoencoder':
                    scores = model.predict(X_scaled)
                    threshold = np.percentile(scores, (1 - self.contamination) * 100)
                    predictions[method] = {
                        'anomaly_labels': (scores > threshold).astype(int) * 2 - 1,  # Convert to -1/1
                        'anomaly_scores': scores
                    }
                    
            except Exception as e:
                logger.error(f"Error in {method} prediction: {e}")
                predictions[method] = {
                    'anomaly_labels': np.zeros(len(X)),
                    'anomaly_scores': np.zeros(len(X))
                }
        
        return predictions
    
    def detect_regime_changes(self, 
                            X: np.ndarray,
                            window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect market regime changes.
        
        Args:
            X: Time series data
            window_size: Window size for rolling detection
            
        Returns:
            Dictionary with regime change information
        """
        window_size = window_size or self.window_size
        
        regime_changes = []
        regime_scores = []
        
        # Rolling anomaly detection
        for i in range(window_size, len(X)):
            window_data = X[i-window_size:i]
            
            # Detect anomalies in current window
            predictions = self.predict(window_data)
            
            # Aggregate anomaly scores across methods
            avg_score = np.mean([
                np.mean(np.abs(pred['anomaly_scores']))
                for pred in predictions.values()
            ])
            
            regime_scores.append(avg_score)
            
            # Detect significant changes
            if len(regime_scores) > 10:
                recent_avg = np.mean(regime_scores[-10:])
                historical_avg = np.mean(regime_scores[:-10])
                
                if recent_avg > historical_avg * 2:  # 2x threshold
                    regime_changes.append({
                        'timestamp_index': i,
                        'score': avg_score,
                        'change_magnitude': recent_avg / historical_avg
                    })
        
        return {
            'regime_changes': regime_changes,
            'regime_scores': regime_scores,
            'change_points': [change['timestamp_index'] for change in regime_changes]
        }
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance for anomaly detection.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of feature importance scores
        """
        if 'isolation_forest' not in self.models:
            logger.warning("Isolation Forest not available for feature importance")
            return {}
        
        # Use permutation importance
        base_predictions = self.predict(X)
        base_score = np.mean(np.abs(base_predictions['isolation_forest']['anomaly_scores']))
        
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Shuffle feature i
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get new predictions
            perm_predictions = self.predict(X_permuted)
            perm_score = np.mean(np.abs(perm_predictions['isolation_forest']['anomaly_scores']))
            
            # Calculate importance as score difference
            importance = abs(perm_score - base_score)
            feature_importance[feature_name] = importance
        
        return feature_importance


class AnomalyAutoencoder(nn.Module):
    """GPU-accelerated autoencoder for anomaly detection."""
    
    def __init__(self,
                 input_dim: int,
                 encoding_dim: int = 64,
                 device: Optional[torch.device] = None):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension
            device: Device to run on
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.device = device or torch.device('cpu')
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim * 2, input_dim)
        )
        
        self.to(self.device)
        self.is_fitted = False
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self,
            X: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001):
        """
        Train the autoencoder.
        
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch, in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.forward(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        logger.info("Autoencoder training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict reconstruction errors (anomaly scores).
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (reconstruction errors)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.forward(X_tensor)
            
            # Calculate reconstruction error
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            
        return errors.cpu().numpy()


class GPUAccelerator:
    """Utilities for GPU acceleration of ML computations."""
    
    def __init__(self):
        """Initialize GPU accelerator."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU available: {self.gpu_name}")
            logger.info(f"GPU memory: {self.gpu_memory / 1e9:.1f} GB")
        else:
            logger.info("GPU not available, using CPU")
    
    def optimize_model_for_gpu(self, model: nn.Module) -> nn.Module:
        """
        Optimize PyTorch model for GPU training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        if not self.gpu_available:
            return model
        
        # Move to GPU
        model = model.to(self.device)
        
        # Enable mixed precision if supported
        if hasattr(torch.cuda, 'amp'):
            model = model.half()  # Convert to half precision
        
        # Optimize for inference
        if hasattr(torch.jit, 'script'):
            try:
                model = torch.jit.script(model)
                logger.info("Model optimized with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        return model
    
    def accelerated_matrix_operations(self, 
                                    X: np.ndarray,
                                    operation: str = 'svd') -> np.ndarray:
        """
        Perform matrix operations with GPU acceleration.
        
        Args:
            X: Input matrix
            operation: Operation type ('svd', 'eigen', 'cholesky')
            
        Returns:
            Operation result
        """
        if not self.gpu_available:
            # Fallback to CPU operations
            if operation == 'svd':
                return np.linalg.svd(X)
            elif operation == 'eigen':
                return np.linalg.eigh(X)
            elif operation == 'cholesky':
                return np.linalg.cholesky(X)
        
        # GPU operations
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if operation == 'svd':
            U, S, V = torch.svd(X_tensor)
            return U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()
        
        elif operation == 'eigen':
            eigenvals, eigenvecs = torch.symeig(X_tensor, eigenvectors=True)
            return eigenvals.cpu().numpy(), eigenvecs.cpu().numpy()
        
        elif operation == 'cholesky':
            L = torch.cholesky(X_tensor)
            return L.cpu().numpy()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def batch_process_features(self,
                             feature_data: np.ndarray,
                             batch_size: int = 1000) -> List[np.ndarray]:
        """
        Process large feature matrices in batches for memory efficiency.
        
        Args:
            feature_data: Large feature matrix
            batch_size: Processing batch size
            
        Returns:
            List of processed batches
        """
        results = []
        n_samples = len(feature_data)
        
        for i in range(0, n_samples, batch_size):
            batch = feature_data[i:i + batch_size]
            
            if self.gpu_available:
                # Process on GPU
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # Example processing: normalize
                batch_normalized = torch.nn.functional.normalize(batch_tensor, dim=1)
                batch_result = batch_normalized.cpu().numpy()
            else:
                # Process on CPU
                batch_result = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            
            results.append(batch_result)
        
        return results
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        return {
            'gpu_available': True,
            'memory_allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'memory_reserved': torch.cuda.memory_reserved() / 1e9,    # GB
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1e9,  # GB
        }
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
