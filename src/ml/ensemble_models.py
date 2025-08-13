"""
Ensemble Models for Financial Prediction

This module implements ensemble methods that combine multiple models
for improved prediction accuracy and robustness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .deep_learning.lstm_models import LSTMPredictor
from .deep_learning.gru_models import GRUPredictor
from .deep_learning.transformer_models import TransformerPredictor
from .deep_learning.cnn_models import CNN1D

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models with dynamic weighting.
    
    Features:
    - Performance-based model weighting
    - Multiple ensemble strategies (voting, stacking, weighted average)
    - Dynamic weight adjustment based on recent performance
    - Model selection and pruning
    """
    
    def __init__(self,
                 models: Optional[List] = None,
                 ensemble_method: str = 'weighted_average',
                 weight_update_frequency: int = 100,
                 performance_window: int = 500,
                 min_model_weight: float = 0.01):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of models to ensemble
            ensemble_method: Method for combining predictions ('weighted_average', 'voting', 'stacking')
            weight_update_frequency: How often to update model weights
            performance_window: Window size for performance evaluation
            min_model_weight: Minimum weight for any model
        """
        self.models = models or []
        self.ensemble_method = ensemble_method
        self.weight_update_frequency = weight_update_frequency
        self.performance_window = performance_window
        self.min_model_weight = min_model_weight
        
        # Model weights (initialized uniformly)
        self.model_weights = np.ones(len(self.models)) / len(self.models) if self.models else np.array([])
        
        # Performance tracking
        self.model_performance = {i: {'predictions': [], 'errors': []} for i in range(len(self.models))}
        self.prediction_count = 0
        
        # Stacking model (if using stacking method)
        if ensemble_method == 'stacking':
            self.meta_model = LinearRegression()
            
        logger.info(f"Initialized EnsemblePredictor with {len(self.models)} models")
    
    def add_model(self, model, weight: Optional[float] = None):
        """Add a new model to the ensemble."""
        self.models.append(model)
        
        # Update weights
        if weight is None:
            # Redistribute weights equally
            n_models = len(self.models)
            self.model_weights = np.ones(n_models) / n_models
        else:
            # Normalize existing weights and add new weight
            current_sum = np.sum(self.model_weights)
            self.model_weights = self.model_weights * (1 - weight) / current_sum
            self.model_weights = np.append(self.model_weights, weight)
        
        # Add performance tracking
        model_idx = len(self.models) - 1
        self.model_performance[model_idx] = {'predictions': [], 'errors': []}
        
        logger.info(f"Added model {model_idx} to ensemble")
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        model_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    # For PyTorch models
                    model.eval()
                    with torch.no_grad():
                        if isinstance(X, np.ndarray):
                            X_tensor = torch.FloatTensor(X)
                        else:
                            X_tensor = X
                        pred = model(X_tensor).cpu().numpy()
                
                if pred.ndim > 1 and pred.shape[1] == 1:
                    pred = pred.flatten()
                    
                model_predictions.append(pred)
                
            except Exception as e:
                logger.error(f"Error in model {i}: {e}")
                # Use zeros as fallback
                model_predictions.append(np.zeros(len(X)))
        
        model_predictions = np.array(model_predictions)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            ensemble_pred = np.average(model_predictions, axis=0, weights=self.model_weights)
            
        elif self.ensemble_method == 'voting':
            ensemble_pred = np.mean(model_predictions, axis=0)
            
        elif self.ensemble_method == 'stacking':
            if hasattr(self, 'meta_model') and hasattr(self.meta_model, 'predict'):
                # Use stacked predictions as features for meta-model
                ensemble_pred = self.meta_model.predict(model_predictions.T)
            else:
                # Fallback to weighted average
                ensemble_pred = np.average(model_predictions, axis=0, weights=self.model_weights)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred
    
    def update_performance(self, y_true: np.ndarray, y_pred: np.ndarray, model_predictions: np.ndarray):
        """Update model performance metrics and weights."""
        
        # Calculate individual model errors
        for i in range(len(self.models)):
            if i < len(model_predictions):
                model_pred = model_predictions[i]
                model_error = mean_squared_error(y_true, model_pred)
                
                # Store performance
                self.model_performance[i]['predictions'].append(model_pred)
                self.model_performance[i]['errors'].append(model_error)
                
                # Keep only recent performance data
                if len(self.model_performance[i]['errors']) > self.performance_window:
                    self.model_performance[i]['predictions'].pop(0)
                    self.model_performance[i]['errors'].pop(0)
        
        self.prediction_count += 1
        
        # Update weights periodically
        if self.prediction_count % self.weight_update_frequency == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Update model weights based on recent performance."""
        
        # Calculate average error for each model over recent window
        avg_errors = []
        for i in range(len(self.models)):
            errors = self.model_performance[i]['errors']
            if errors:
                avg_error = np.mean(errors[-self.performance_window:])
            else:
                avg_error = 1.0  # Default high error
            avg_errors.append(avg_error)
        
        avg_errors = np.array(avg_errors)
        
        # Convert errors to weights (inverse relationship)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        inverse_errors = 1.0 / (avg_errors + epsilon)
        
        # Normalize to get weights
        new_weights = inverse_errors / np.sum(inverse_errors)
        
        # Apply minimum weight constraint
        new_weights = np.maximum(new_weights, self.min_model_weight)
        new_weights = new_weights / np.sum(new_weights)  # Renormalize
        
        self.model_weights = new_weights
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """Train meta-model for stacking ensemble."""
        
        if self.ensemble_method != 'stacking':
            logger.warning("train_meta_model called but ensemble_method is not 'stacking'")
            return
        
        # Get predictions from all base models
        base_predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    if isinstance(X, np.ndarray):
                        X_tensor = torch.FloatTensor(X)
                    else:
                        X_tensor = X
                    pred = model(X_tensor).cpu().numpy()
            
            if pred.ndim > 1 and pred.shape[1] == 1:
                pred = pred.flatten()
                
            base_predictions.append(pred)
        
        # Train meta-model on base predictions
        base_predictions = np.array(base_predictions).T
        self.meta_model.fit(base_predictions, y)
        
        logger.info("Meta-model trained for stacking ensemble")
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Get current model weights/contributions."""
        contributions = {}
        for i, weight in enumerate(self.model_weights):
            model_name = getattr(self.models[i], '__class__', {}).get('__name__', f'Model_{i}')
            contributions[model_name] = float(weight)
        
        return contributions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        summary = {}
        
        for i in range(len(self.models)):
            model_name = getattr(self.models[i], '__class__', {}).get('__name__', f'Model_{i}')
            errors = self.model_performance[i]['errors']
            
            if errors:
                summary[model_name] = {
                    'avg_error': np.mean(errors),
                    'recent_error': np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors),
                    'weight': float(self.model_weights[i]),
                    'predictions_count': len(errors)
                }
            else:
                summary[model_name] = {
                    'avg_error': None,
                    'recent_error': None,
                    'weight': float(self.model_weights[i]),
                    'predictions_count': 0
                }
        
        return summary


class AutoEnsemble:
    """
    Automatic ensemble creation and optimization.
    
    Features:
    - Automatic model selection
    - Hyperparameter optimization for ensemble
    - Cross-validation based model evaluation
    """
    
    def __init__(self, 
                 model_types: List[str] = None,
                 max_models: int = 5,
                 cv_folds: int = 5):
        """
        Initialize automatic ensemble.
        
        Args:
            model_types: List of model types to include
            max_models: Maximum number of models in ensemble
            cv_folds: Number of cross-validation folds
        """
        self.model_types = model_types or ['lstm', 'gru', 'transformer', 'cnn']
        self.max_models = max_models
        self.cv_folds = cv_folds
        
        self.best_ensemble = None
        self.model_configs = self._get_default_configs()
    
    def _get_default_configs(self) -> Dict[str, Dict]:
        """Get default configurations for different model types."""
        return {
            'lstm': {
                'hidden_size': [64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3]
            },
            'gru': {
                'hidden_size': [64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3]
            },
            'transformer': {
                'd_model': [64, 128, 256],
                'nhead': [4, 8, 16],
                'num_layers': [2, 4, 6]
            },
            'cnn': {
                'num_filters': [[32, 64], [64, 128], [128, 256]],
                'kernel_sizes': [[3, 5], [5, 7], [3, 7]]
            }
        }
    
    def create_optimal_ensemble(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              input_size: int) -> EnsemblePredictor:
        """
        Create optimal ensemble through model selection and hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training targets
            input_size: Input feature size
            
        Returns:
            Optimized ensemble predictor
        """
        
        # Evaluate different model configurations
        model_scores = []
        candidate_models = []
        
        for model_type in self.model_types:
            logger.info(f"Evaluating {model_type} models...")
            
            # Try different configurations for this model type
            configs = self.model_configs.get(model_type, {})
            
            # For simplicity, use first configuration (can be extended with grid search)
            if model_type == 'lstm':
                model = LSTMPredictor(
                    input_size=input_size,
                    hidden_size=configs['hidden_size'][0],
                    num_layers=configs['num_layers'][0],
                    dropout=configs['dropout'][0]
                )
            elif model_type == 'gru':
                model = GRUPredictor(
                    input_size=input_size,
                    hidden_size=configs['hidden_size'][0],
                    num_layers=configs['num_layers'][0],
                    dropout=configs['dropout'][0]
                )
            elif model_type == 'transformer':
                model = TransformerPredictor(
                    input_size=input_size,
                    d_model=configs['d_model'][0],
                    nhead=configs['nhead'][0],
                    num_layers=configs['num_layers'][0]
                )
            elif model_type == 'cnn':
                model = CNN1D(
                    input_size=input_size,
                    num_filters=configs['num_filters'][0],
                    kernel_sizes=configs['kernel_sizes'][0]
                )
            else:
                continue
            
            # Evaluate model with cross-validation
            score = self._evaluate_model_cv(model, X, y)
            
            model_scores.append((model_type, score, model))
            candidate_models.append((model, score))
        
        # Select best models for ensemble
        candidate_models.sort(key=lambda x: x[1])  # Sort by score (ascending for error)
        selected_models = [model for model, _ in candidate_models[:self.max_models]]
        
        # Create ensemble
        ensemble = EnsemblePredictor(models=selected_models)
        
        logger.info(f"Created ensemble with {len(selected_models)} models")
        
        return ensemble
    
    def _evaluate_model_cv(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model using cross-validation."""
        # Simplified CV evaluation (placeholder)
        # In practice, implement proper time series CV
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Train model (simplified)
                # In practice, implement proper training loop
                
                # Make predictions
                y_pred = np.random.normal(0, 1, len(y_val))  # Placeholder
                score = mean_squared_error(y_val, y_pred)
                scores.append(score)
                
            except Exception as e:
                logger.error(f"CV evaluation failed: {e}")
                scores.append(float('inf'))
        
        return np.mean(scores)
