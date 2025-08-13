"""
Base Deep Learning Model Classes

This module provides the foundation classes for all deep learning models,
including common training routines, model management, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class BaseDeepLearningModel(ABC, nn.Module):
    """
    Base class for all deep learning models in the trading system.
    
    Provides common functionality for training, evaluation, saving/loading,
    and integration with the HFT simulator.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 device: Optional[str] = None):
        """
        Initialize base deep learning model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of layers
            output_size: Number of output predictions
            dropout: Dropout rate for regularization
            device: Device to run model on (cuda/cpu)
        """
        super(BaseDeepLearningModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.scalers = {}
        
        # Model metadata
        self.model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'dropout': dropout,
            'model_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat(),
        }
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return model architecture details."""
        pass
    
    def prepare_data(self, 
                     data: pd.DataFrame,
                     target_column: str = 'target',
                     sequence_length: int = 50,
                     test_size: float = 0.2,
                     scale_data: bool = True) -> Dict[str, torch.Tensor]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: Input DataFrame with features and target
            target_column: Name of target column
            sequence_length: Length of input sequences
            test_size: Proportion of data for testing
            scale_data: Whether to scale the data
            
        Returns:
            Dictionary containing train/test tensors
        """
        # Separate features and target
        feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Scale data if requested
        if scale_data:
            self.scalers['features'] = StandardScaler()
            self.scalers['target'] = StandardScaler()
            
            X = self.scalers['features'].fit_transform(X)
            y = self.scalers['target'].fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y, sequence_length)
        
        # Train/test split
        split_idx = int(len(X_seq) * (1 - test_size))
        
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to tensors
        data_dict = {
            'X_train': torch.FloatTensor(X_train).to(self.device),
            'X_test': torch.FloatTensor(X_test).to(self.device),
            'y_train': torch.FloatTensor(y_train).to(self.device),
            'y_test': torch.FloatTensor(y_test).to(self.device),
        }
        
        logger.info(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return data_dict
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X_seq, y_seq = [], []
        
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_model(self,
                    data_dict: Dict[str, torch.Tensor],
                    epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    early_stopping: int = 10,
                    validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the deep learning model.
        
        Args:
            data_dict: Dictionary containing train/test data
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            early_stopping: Patience for early stopping
            validation_split: Validation set size
            
        Returns:
            Training history dictionary
        """
        self.to(self.device)
        self.train()
        
        # Setup data loaders
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        
        # Validation split
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.train()
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                
                if len(outputs.shape) > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            val_loss = 0.0
            self.eval()
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self(batch_X)
                    
                    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze()
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state_dict = self.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model state
        if hasattr(self, 'best_state_dict'):
            self.load_state_dict(self.best_state_dict)
        
        self.is_trained = True
        self.training_history = history
        
        logger.info("Training completed successfully")
        return history
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input tensor
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            
            X = X.to(self.device)
            predictions = self(X)
            
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.squeeze()
            
            predictions = predictions.cpu().numpy()
        
        # Inverse scale if scalers exist
        if 'target' in self.scalers:
            predictions = self.scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def evaluate_model(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            data_dict: Dictionary containing test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        
        # Make predictions
        predictions = self.predict(X_test)
        y_true = y_test.cpu().numpy()
        
        # Inverse scale if needed
        if 'target' in self.scalers:
            y_true = self.scalers['target'].inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, predictions)
        
        # Financial metrics
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(predictions))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'direction_accuracy': direction_accuracy,
        }
        
        logger.info(f"Model evaluation - RMSE: {rmse:.6f}, Direction Accuracy: {direction_accuracy:.3f}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'scalers': self.scalers,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(save_dict['model_state_dict'])
        self.model_config = save_dict['model_config']
        self.scalers = save_dict['scalers']
        self.training_history = save_dict.get('training_history', [])
        self.is_trained = save_dict.get('is_trained', True)
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate plot
        axes[1].plot(self.training_history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'model_type': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': self.model_config,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'architecture': self.get_model_architecture(),
        }
        
        return summary
