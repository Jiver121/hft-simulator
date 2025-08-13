"""
Multi-Scale Neural Networks for Financial Time Series

This module implements multi-scale architectures that can capture patterns
at different time horizons and frequencies in financial data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import logging

from .base_models import BaseDeepLearningModel

logger = logging.getLogger(__name__)

class MultiScalePredictor(BaseDeepLearningModel):
    """
    Multi-scale neural network for capturing patterns at different time horizons.
    
    Features:
    - Multiple parallel branches for different time scales
    - Hierarchical feature fusion
    - Multi-resolution attention
    """
    
    def __init__(self,
                 input_size: int,
                 scales: List[int] = [1, 3, 7, 15],  # Different time scales
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 fusion_method: str = 'attention',
                 device: Optional[str] = None):
        """
        Initialize Multi-Scale predictor.
        
        Args:
            input_size: Number of input features
            scales: List of different time scales to capture
            hidden_size: Hidden layer size for each scale
            num_layers: Number of layers per scale
            output_size: Number of prediction outputs
            dropout: Dropout rate
            fusion_method: Method to fuse multi-scale features ('attention', 'concat', 'weighted')
            device: Device to run on
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.scales = scales
        self.fusion_method = fusion_method
        self.num_scales = len(scales)
        
        # Create branches for different scales
        self.scale_branches = nn.ModuleDict()
        
        for scale in scales:
            branch_name = f'scale_{scale}'
            
            # Each branch is a combination of CNN and LSTM
            branch = MultiScaleBranch(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                scale=scale,
                dropout=dropout
            )
            
            self.scale_branches[branch_name] = branch
        
        # Feature fusion layer
        if fusion_method == 'attention':
            self.fusion_layer = AttentionFusion(hidden_size, self.num_scales)
            fusion_output_size = hidden_size
        elif fusion_method == 'concat':
            self.fusion_layer = nn.Identity()
            fusion_output_size = hidden_size * self.num_scales
        elif fusion_method == 'weighted':
            self.fusion_layer = WeightedFusion(hidden_size, self.num_scales)
            fusion_output_size = hidden_size
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_output_size, fusion_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized Multi-Scale Predictor with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Multi-Scale predictor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # Process through each scale branch
        scale_features = []
        
        for scale in self.scales:
            branch_name = f'scale_{scale}'
            branch = self.scale_branches[branch_name]
            
            # Process input through this scale
            scale_feature = branch(x)
            scale_features.append(scale_feature)
        
        # Fuse multi-scale features
        if self.fusion_method == 'concat':
            fused_features = torch.cat(scale_features, dim=-1)
        else:
            # Stack features for attention or weighted fusion
            stacked_features = torch.stack(scale_features, dim=1)  # (batch, num_scales, hidden_size)
            fused_features = self.fusion_layer(stacked_features)
        
        # Final prediction
        output = self.dropout(fused_features)
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_scale_contributions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get individual scale contributions for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of scale contributions
        """
        scale_outputs = {}
        
        for scale in self.scales:
            branch_name = f'scale_{scale}'
            branch = self.scale_branches[branch_name]
            
            scale_output = branch(x)
            scale_outputs[f'scale_{scale}'] = scale_output.detach().cpu()
        
        return scale_outputs
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Return Multi-Scale architecture details."""
        return {
            'type': 'MultiScalePredictor',
            'input_size': self.input_size,
            'scales': self.scales,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'fusion_method': self.fusion_method,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleBranch(nn.Module):
    """Individual branch for processing one time scale."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 scale: int,
                 dropout: float = 0.2):
        """
        Initialize branch for specific time scale.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of layers
            scale: Time scale for this branch
            dropout: Dropout rate
        """
        super().__init__()
        
        self.scale = scale
        
        # Scale-specific preprocessing
        if scale > 1:
            # Use convolution to capture patterns at this scale
            self.conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size // 2,
                kernel_size=scale,
                stride=1,
                padding=scale // 2
            )
            conv_output_size = hidden_size // 2
        else:
            # For scale 1, use linear projection
            self.conv = nn.Linear(input_size, hidden_size // 2)
            conv_output_size = hidden_size // 2
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=conv_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through scale branch.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Scale features of shape (batch_size, hidden_size)
        """
        if self.scale > 1:
            # Apply convolution
            # Transpose for Conv1d: (batch, features, seq_len)
            x_transposed = x.transpose(1, 2)
            conv_out = self.conv(x_transposed)
            # Transpose back: (batch, seq_len, features)
            x_processed = conv_out.transpose(1, 2)
        else:
            # Apply linear transformation
            x_processed = self.conv(x)
        
        # Apply activation
        x_processed = F.relu(x_processed)
        
        # GRU processing
        gru_out, hidden = self.gru(x_processed)
        
        # Use last output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Layer normalization
        normalized = self.layer_norm(last_output)
        
        # Dropout
        output = self.dropout(normalized)
        
        return output


class AttentionFusion(nn.Module):
    """Attention-based fusion of multi-scale features."""
    
    def __init__(self, hidden_size: int, num_scales: int):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention fusion.
        
        Args:
            x: Stacked features of shape (batch_size, num_scales, hidden_size)
            
        Returns:
            Fused features of shape (batch_size, hidden_size)
        """
        # Self-attention across scales
        attended, _ = self.attention(x, x, x)
        
        # Global average pooling across scales
        pooled = attended.mean(dim=1)
        
        # Layer normalization
        output = self.layer_norm(pooled)
        
        return output


class WeightedFusion(nn.Module):
    """Learnable weighted fusion of multi-scale features."""
    
    def __init__(self, hidden_size: int, num_scales: int):
        super().__init__()
        
        # Learnable weights for each scale
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Feature transformation
        self.transform = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply weighted fusion.
        
        Args:
            x: Stacked features of shape (batch_size, num_scales, hidden_size)
            
        Returns:
            Fused features of shape (batch_size, hidden_size)
        """
        # Normalize weights
        normalized_weights = F.softmax(self.scale_weights, dim=0)
        
        # Weighted combination
        weighted = torch.einsum('bsh,s->bh', x, normalized_weights)
        
        # Feature transformation
        transformed = self.transform(weighted)
        
        # Layer normalization
        output = self.layer_norm(transformed)
        
        return output
