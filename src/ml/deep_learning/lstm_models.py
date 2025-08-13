"""
LSTM Models for Financial Time Series Prediction

This module implements Long Short-Term Memory (LSTM) neural networks optimized
for financial market prediction, including standard LSTM, bidirectional LSTM,
and stacked variants with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

from .base_models import BaseDeepLearningModel

logger = logging.getLogger(__name__)

class LSTMPredictor(BaseDeepLearningModel):
    """
    Standard LSTM model for price prediction.
    
    Features:
    - Multi-layer LSTM architecture
    - Dropout for regularization
    - Flexible output layers
    - Batch normalization option
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 batch_first: bool = True,
                 use_batch_norm: bool = False,
                 device: Optional[str] = None):
        """
        Initialize LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            batch_first: Whether batch dimension is first
            use_batch_norm: Whether to use batch normalization
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.batch_first = batch_first
        self.use_batch_norm = use_batch_norm
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=False
        )
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized LSTM with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        if self.batch_first:
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        else:
            last_output = lstm_out[-1, :, :]  # (batch_size, hidden_size)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            last_output = self.batch_norm(last_output)
        
        # Apply dropout
        output = self.dropout(last_output)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return LSTM architecture details."""
        return {
            'type': 'LSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'batch_first': self.batch_first,
            'use_batch_norm': self.use_batch_norm,
            'bidirectional': False,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMPredictor(BaseDeepLearningModel):
    """
    Bidirectional LSTM model for enhanced pattern recognition.
    
    Features:
    - Bidirectional processing for both past and future context
    - Attention mechanism for important feature selection
    - Multiple prediction heads
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 device: Optional[str] = None):
        """
        Initialize Bidirectional LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        
        # Adjust for bidirectional output
        lstm_output_size = hidden_size * 2
        
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized BiLSTM with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Bidirectional LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # BiLSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        if self.use_attention:
            # Apply attention mechanism
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Use mean of attended sequence
            sequence_output = attended_out.mean(dim=1)  # (batch_size, hidden_size * 2)
        else:
            # Use the last output
            sequence_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        output = self.dropout(sequence_output)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        if not self.use_attention:
            return None
        
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            return attention_weights.cpu().numpy()
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return BiLSTM architecture details."""
        return {
            'type': 'BiLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'use_attention': self.use_attention,
            'bidirectional': True,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StackedLSTMPredictor(BaseDeepLearningModel):
    """
    Stacked LSTM with residual connections and layer normalization.
    
    Features:
    - Multiple LSTM layers with residual connections
    - Layer normalization for stable training
    - Progressive feature reduction
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list = [128, 64, 32],
                 output_size: int = 1,
                 dropout: float = 0.2,
                 use_residual: bool = True,
                 device: Optional[str] = None):
        """
        Initialize Stacked LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden sizes for each LSTM layer
            output_size: Number of prediction outputs
            dropout: Dropout rate
            use_residual: Whether to use residual connections
            device: Device to run on
        """
        super().__init__(input_size, hidden_sizes[0], len(hidden_sizes), output_size, dropout, device)
        
        self.hidden_sizes = hidden_sizes
        self.use_residual = use_residual
        
        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        current_input_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # LSTM layer
            lstm_layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0  # We'll handle dropout manually
            )
            self.lstm_layers.append(lstm_layer)
            
            # Layer normalization
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
            
            current_input_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )
        
        # Residual projection layers (if dimensions don't match)
        if use_residual:
            self.residual_projections = nn.ModuleList()
            for i in range(len(hidden_sizes)):
                if i == 0:
                    proj_input_size = input_size
                else:
                    proj_input_size = hidden_sizes[i-1]
                
                if proj_input_size != hidden_sizes[i]:
                    self.residual_projections.append(
                        nn.Linear(proj_input_size, hidden_sizes[i])
                    )
                else:
                    self.residual_projections.append(None)
        
        self.to(self.device)
        logger.info(f"Initialized Stacked LSTM with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Stacked LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        current_input = x
        
        for i, (lstm_layer, layer_norm, dropout) in enumerate(
            zip(self.lstm_layers, self.layer_norms, self.dropouts)
        ):
            # LSTM forward pass
            lstm_out, _ = lstm_layer(current_input)
            
            # Apply layer normalization to the last timestep
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            normalized_output = layer_norm(last_output)
            
            # Residual connection
            if self.use_residual and i > 0:
                # Get residual from previous layer's last output
                residual = current_input[:, -1, :]  # (batch_size, prev_hidden_size)
                
                # Project if dimensions don't match
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                
                normalized_output = normalized_output + residual
            
            # Apply dropout
            output = dropout(normalized_output)
            
            # Prepare for next layer (expand back to sequence format)
            if i < len(self.lstm_layers) - 1:
                current_input = output.unsqueeze(1).repeat(1, x.size(1), 1)
            else:
                final_output = output
        
        # Final prediction
        predictions = self.output_layer(final_output)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return Stacked LSTM architecture details."""
        return {
            'type': 'StackedLSTM',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'num_layers': len(self.hidden_sizes),
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
