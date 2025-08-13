"""
Real-time Feature Store for HFT Trading

This module provides a comprehensive feature engineering system with 500+ 
financial features optimized for high-frequency trading applications.

Components:
- FeatureStore: Main feature store class with real-time computation
- FeatureEngineers: Specialized feature engineering modules
- FeatureTransformers: Data transformation and normalization
- FeatureValidators: Feature quality and consistency checks
- FeatureCache: High-performance caching system
"""

from .feature_store import FeatureStore
from .technical_features import TechnicalFeatureEngineer
from .microstructure_features import MicrostructureFeatureEngineer
from .statistical_features import StatisticalFeatureEngineer
from .sentiment_features import SentimentFeatureEngineer
from .market_regime_features import MarketRegimeFeatureEngineer
from .cross_asset_features import CrossAssetFeatureEngineer
from .feature_transformers import FeatureTransformer, FeatureScaler
from .feature_validators import FeatureValidator
from .feature_cache import FeatureCache

__all__ = [
    'FeatureStore',
    'TechnicalFeatureEngineer',
    'MicrostructureFeatureEngineer', 
    'StatisticalFeatureEngineer',
    'SentimentFeatureEngineer',
    'MarketRegimeFeatureEngineer',
    'CrossAssetFeatureEngineer',
    'FeatureTransformer',
    'FeatureScaler',
    'FeatureValidator',
    'FeatureCache',
]
