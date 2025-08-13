"""
Deep Learning Models for HFT Price Prediction

This module implements state-of-the-art deep learning architectures optimized for
financial time series prediction and trading signal generation.

Models:
- LSTM: Long Short-Term Memory networks for sequential pattern learning
- GRU: Gated Recurrent Units for efficient sequence modeling
- Transformer: Attention-based models for complex pattern recognition
- CNN-LSTM: Hybrid convolutional-recurrent networks
- Multi-scale Neural Networks: Different time horizon predictions
"""

from .lstm_models import LSTMPredictor, BiLSTMPredictor
from .gru_models import GRUPredictor, AttentionGRU
from .transformer_models import TransformerPredictor, FinancialTransformer
from .cnn_models import CNN1D, CNNLSTMHybrid
from .multiscale_models import MultiScalePredictor
from .base_models import BaseDeepLearningModel

__all__ = [
    'LSTMPredictor',
    'BiLSTMPredictor', 
    'GRUPredictor',
    'AttentionGRU',
    'TransformerPredictor',
    'FinancialTransformer',
    'CNN1D',
    'CNNLSTMHybrid',
    'MultiScalePredictor',
    'BaseDeepLearningModel',
]
