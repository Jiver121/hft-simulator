"""
Transformer Models for Financial Time Series Prediction

This module implements Transformer architectures optimized for financial market
prediction, including standard Transformers and specialized financial variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, Any
import logging

from .base_models import BaseDeepLearningModel

logger = logging.getLogger(__name__)

class TransformerPredictor(BaseDeepLearningModel):
    """
    Standard Transformer model for price prediction.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Feedforward networks
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 dim_feedforward: int = 512,
                 max_seq_length: int = 1000,
                 device: Optional[str] = None):
        """
        Initialize Transformer predictor.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            dim_feedforward: Feedforward network dimension
            max_seq_length: Maximum sequence length for positional encoding
            device: Device to run on
        """
        super().__init__(input_size, d_model, num_layers, output_size, dropout, device)
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-layer norm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized Transformer with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)
        # encoded shape: (batch_size, seq_len, d_model)
        
        # Global average pooling
        pooled_output = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        output = self.dropout(pooled_output)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from all layers for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of attention weights by layer
        """
        attention_weights = {}
        
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        for i, layer in enumerate(self.transformer_encoder.layers):
            # Forward through transformer layer and capture attention weights
            x, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights[f'layer_{i}'] = attn_weights.detach().cpu()
            
            # Continue forward pass
            x = layer.norm1(x)
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(x2)
            x = layer.norm2(x)
        
        return attention_weights
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return Transformer architecture details."""
        return {
            'type': 'Transformer',
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'dim_feedforward': self.dim_feedforward,
            'max_seq_length': self.max_seq_length,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FinancialTransformer(BaseDeepLearningModel):
    """
    Specialized Transformer for financial time series with financial-specific features.
    
    Features:
    - Multi-scale temporal attention
    - Price-aware attention mechanisms
    - Volatility-based position encoding
    - Financial embedding layers
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 use_financial_embedding: bool = True,
                 use_volatility_encoding: bool = True,
                 device: Optional[str] = None):
        """
        Initialize Financial Transformer predictor.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            output_size: Number of prediction outputs
            dropout: Dropout rate
            use_financial_embedding: Whether to use financial embedding layers
            use_volatility_encoding: Whether to use volatility-based encoding
            device: Device to run on
        """
        super().__init__(input_size, d_model, num_layers, output_size, dropout, device)
        
        self.d_model = d_model
        self.nhead = nhead
        self.use_financial_embedding = use_financial_embedding
        self.use_volatility_encoding = use_volatility_encoding
        
        # Input processing
        if use_financial_embedding:
            self.price_embedding = FinancialEmbedding(1, d_model // 4)  # For price
            self.volume_embedding = FinancialEmbedding(1, d_model // 4)  # For volume
            self.feature_embedding = nn.Linear(input_size - 2, d_model // 2)  # Other features
            self.input_projection = nn.Linear(d_model, d_model)
        else:
            self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        if use_volatility_encoding:
            self.pos_encoding = VolatilityAwarePositionalEncoding(d_model, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Multi-scale attention layers
        self.multi_scale_attention = MultiScaleAttention(d_model, nhead, dropout)
        
        # Standard transformer layers
        encoder_layer = FinancialTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        
        self.transformer_layers = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output layers with financial-specific processing
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head output for different prediction horizons
        self.output_heads = nn.ModuleDict({
            'short_term': nn.Linear(d_model, output_size),
            'medium_term': nn.Linear(d_model, output_size),
            'long_term': nn.Linear(d_model, output_size),
        })
        
        # Final aggregation
        self.output_aggregation = nn.Linear(output_size * 3, output_size)
        
        self.to(self.device)
        logger.info(f"Initialized Financial Transformer with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Financial Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        if self.use_financial_embedding:
            # Separate financial features
            prices = x[:, :, 0:1]  # Assume first column is price
            volumes = x[:, :, 1:2]  # Assume second column is volume
            other_features = x[:, :, 2:]  # Remaining features
            
            # Apply embeddings
            price_emb = self.price_embedding(prices)
            volume_emb = self.volume_embedding(volumes)
            feature_emb = self.feature_embedding(other_features)
            
            # Concatenate embeddings
            x = torch.cat([price_emb, volume_emb, feature_emb], dim=-1)
            x = self.input_projection(x)
        else:
            x = self.input_projection(x)
        
        # Add positional encoding
        if self.use_volatility_encoding:
            # Extract volatility proxy from features (assuming it's available)
            volatility = x.std(dim=-1, keepdim=True)  # Simple volatility proxy
            x = self.pos_encoding(x, volatility)
        else:
            x = self.pos_encoding(x)
        
        # Multi-scale attention
        x = self.multi_scale_attention(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Global average pooling
        pooled_output = x.mean(dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        output = self.dropout(pooled_output)
        
        # Multi-head predictions
        short_pred = self.output_heads['short_term'](output)
        medium_pred = self.output_heads['medium_term'](output)
        long_pred = self.output_heads['long_term'](output)
        
        # Aggregate predictions
        combined = torch.cat([short_pred, medium_pred, long_pred], dim=-1)
        final_prediction = self.output_aggregation(combined)
        
        return final_prediction
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return Financial Transformer architecture details."""
        return {
            'type': 'FinancialTransformer',
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'use_financial_embedding': self.use_financial_embedding,
            'use_volatility_encoding': self.use_volatility_encoding,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x + pe
        return self.dropout(x)


class VolatilityAwarePositionalEncoding(nn.Module):
    """Volatility-aware positional encoding for financial data."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable volatility scaling
        self.volatility_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """Add volatility-aware positional encoding."""
        batch_size, seq_len, d_model = x.size()
        
        # Generate position indices
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(0).unsqueeze(-1)
        
        # Create sinusoidal encodings
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(batch_size, seq_len, d_model, device=x.device)
        
        # Scale position by volatility
        scaled_position = position * (1 + self.volatility_scale * volatility)
        
        pe[:, :, 0::2] = torch.sin(scaled_position * div_term)
        pe[:, :, 1::2] = torch.cos(scaled_position * div_term)
        
        x = x + pe
        return self.dropout(x)


class FinancialEmbedding(nn.Module):
    """Financial-aware embedding layer for price and volume data."""
    
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply financial embedding."""
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for capturing patterns at different time scales."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        # Different scale attention modules
        self.short_attention = nn.MultiheadAttention(d_model, nhead // 2, dropout, batch_first=True)
        self.long_attention = nn.MultiheadAttention(d_model, nhead // 2, dropout, batch_first=True)
        
        # Scale fusion
        self.scale_fusion = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale attention."""
        # Short-term attention (local patterns)
        short_out, _ = self.short_attention(x, x, x)
        
        # Long-term attention with downsampling
        # Simple downsampling by taking every other timestep
        x_downsampled = x[:, ::2, :]  # Downsample by factor of 2
        long_out, _ = self.long_attention(x_downsampled, x_downsampled, x_downsampled)
        
        # Upsample back to original length
        long_out_upsampled = F.interpolate(
            long_out.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False
        ).transpose(1, 2)
        
        # Concatenate and fuse
        combined = torch.cat([short_out, long_out_upsampled], dim=-1)
        fused = self.scale_fusion(combined)
        
        # Residual connection and layer norm
        output = self.layer_norm(fused + x)
        
        return output


class FinancialTransformerLayer(nn.Module):
    """Custom transformer layer with financial-specific modifications."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        
        # Enhanced feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through financial transformer layer."""
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
