"""
CNN Models for Financial Pattern Recognition

This module implements Convolutional Neural Networks optimized for financial 
time series pattern recognition and hybrid CNN-RNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

from .base_models import BaseDeepLearningModel

logger = logging.getLogger(__name__)

class CNN1D(BaseDeepLearningModel):
    """1D CNN for financial time series pattern recognition."""
    
    def __init__(self,
                 input_size: int,
                 num_filters: list = [64, 128, 256],
                 kernel_sizes: list = [3, 5, 7],
                 output_size: int = 1,
                 dropout: float = 0.2,
                 device: Optional[str] = None):
        
        super().__init__(input_size, num_filters[0], len(num_filters), output_size, dropout, device)
        
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        in_channels = input_size
        for i, (num_filter, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            # Conv layer
            conv = nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
            
            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(num_filter))
            
            # Pooling
            self.pooling_layers.append(nn.MaxPool1d(2))
            
            in_channels = num_filter
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(num_filters[-1], num_filters[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters[-1] // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized CNN1D with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pooling_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        
        # Output
        x = self.dropout(x)
        predictions = self.output_layer(x)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        return {
            'type': 'CNN1D',
            'input_size': self.input_size,
            'num_filters': self.num_filters,
            'kernel_sizes': self.kernel_sizes,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNLSTMHybrid(BaseDeepLearningModel):
    """Hybrid CNN-LSTM model combining pattern recognition with sequence modeling."""
    
    def __init__(self,
                 input_size: int,
                 cnn_filters: list = [64, 128],
                 cnn_kernels: list = [3, 5],
                 lstm_hidden_size: int = 128,
                 lstm_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 device: Optional[str] = None):
        
        super().__init__(input_size, lstm_hidden_size, lstm_layers, output_size, dropout, device)
        
        self.cnn_filters = cnn_filters
        self.cnn_kernels = cnn_kernels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        # CNN feature extractor
        self.cnn_layers = nn.ModuleList()
        self.cnn_bns = nn.ModuleList()
        
        in_channels = input_size
        for filters, kernel in zip(cnn_filters, cnn_kernels):
            self.cnn_layers.append(nn.Conv1d(in_channels, filters, kernel, padding=kernel//2))
            self.cnn_bns.append(nn.BatchNorm1d(filters))
            in_channels = filters
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
        
        self.to(self.device)
        logger.info(f"Initialized CNN-LSTM with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        # Transpose for Conv1d: (batch, features, seq_len)
        cnn_input = x.transpose(1, 2)
        
        for conv, bn in zip(self.cnn_layers, self.cnn_bns):
            cnn_input = conv(cnn_input)
            cnn_input = bn(cnn_input)
            cnn_input = F.relu(cnn_input)
        
        # Transpose back for LSTM: (batch, seq_len, features)
        lstm_input = cnn_input.transpose(1, 2)
        
        # LSTM sequence modeling
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Output prediction
        output = self.dropout(last_output)
        predictions = self.output_layer(output)
        
        return predictions
    
    def get_model_architecture(self) -> Dict[str, Any]:
        return {
            'type': 'CNNLSTMHybrid',
            'input_size': self.input_size,
            'cnn_filters': self.cnn_filters,
            'cnn_kernels': self.cnn_kernels,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_layers': self.lstm_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
