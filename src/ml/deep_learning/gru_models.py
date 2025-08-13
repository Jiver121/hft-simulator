"""
GRU Models for Financial Time Series Prediction

This module implements Gated Recurrent Unit (GRU) neural networks optimized
for financial market prediction with efficient memory usage and fast training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

from .base_models import BaseDeepLearningModel

logger = logging.getLogger(__name__)

class GRUPredictor(BaseDeepLearningModel):
    """
    Standard GRU model for price prediction.
    
    Features:
    - Multi-layer GRU architecture
    - Efficient memory usage compared to LSTM
    - Dropout regularization
    - Optional layer normalization
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 batch_first: bool = True,
                 use_layer_norm: bool = False,
                 device: Optional[str] = None):
        """
        Initialize GRU predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden layer size
            num_layers: Number of GRU layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            batch_first: Whether batch dimension is first
            use_layer_norm: Whether to use layer normalization
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.batch_first = batch_first
        self.use_layer_norm = use_layer_norm
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=False
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized GRU with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use the last output
        if self.batch_first:
            last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        else:
            last_output = gru_out[-1, :, :]  # (batch_size, hidden_size)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            last_output = self.layer_norm(last_output)
        
        # Apply dropout
        output = self.dropout(last_output)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return GRU architecture details."""
        return {
            'type': 'GRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'batch_first': self.batch_first,
            'use_layer_norm': self.use_layer_norm,
            'bidirectional': False,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionGRU(BaseDeepLearningModel):
    """
    GRU with attention mechanism for focusing on important timesteps.
    
    Features:
    - Self-attention mechanism
    - Multiple attention heads
    - Positional encoding
    - Residual connections
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 num_heads: int = 8,
                 use_positional_encoding: bool = True,
                 device: Optional[str] = None):
        """
        Initialize Attention GRU predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden layer size
            num_layers: Number of GRU layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            num_heads: Number of attention heads
            use_positional_encoding: Whether to use positional encoding
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_size, dropout, max_len=1000)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized Attention GRU with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, seq_len, hidden_size)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            gru_out = self.positional_encoding(gru_out)
        
        # Self-attention
        attended_out, attention_weights = self.attention(gru_out, gru_out, gru_out)
        
        # Residual connection and layer norm
        attended_out = self.norm1(attended_out + gru_out)
        
        # Feedforward network
        ffn_out = self.ffn(attended_out)
        
        # Second residual connection and layer norm
        output = self.norm2(ffn_out + attended_out)
        
        # Global average pooling over sequence dimension
        pooled_output = output.mean(dim=1)  # (batch_size, hidden_size)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Final prediction
        predictions = self.output_layer(pooled_output)
        
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights as numpy array
        """
        self.eval()
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            
            if self.use_positional_encoding:
                gru_out = self.positional_encoding(gru_out)
            
            _, attention_weights = self.attention(gru_out, gru_out, gru_out)
            return attention_weights.cpu().numpy()
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return Attention GRU architecture details."""
        return {
            'type': 'AttentionGRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'use_positional_encoding': self.use_positional_encoding,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiGRUPredictor(BaseDeepLearningModel):
    """
    Bidirectional GRU model for enhanced context understanding.
    
    Features:
    - Bidirectional processing
    - Attention mechanism
    - Feature fusion from both directions
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
        Initialize Bidirectional GRU predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden layer size
            num_layers: Number of GRU layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.use_attention = use_attention
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Adjust for bidirectional output
        gru_output_size = hidden_size * 2
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized BiGRU with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Bidirectional GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # BiGRU forward pass
        gru_out, hidden = self.gru(x)
        # gru_out shape: (batch_size, seq_len, hidden_size * 2)
        
        if self.use_attention:
            # Apply attention mechanism
            attended_out, attention_weights = self.attention(gru_out, gru_out, gru_out)
            # Use mean of attended sequence
            sequence_output = attended_out.mean(dim=1)  # (batch_size, hidden_size * 2)
        else:
            # Use the last output
            sequence_output = gru_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Feature fusion
        fused_features = self.feature_fusion(sequence_output)
        
        # Apply dropout
        output = self.dropout(fused_features)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return BiGRU architecture details."""
        return {
            'type': 'BiGRU',
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


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
