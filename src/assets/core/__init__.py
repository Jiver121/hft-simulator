"""
Core Multi-Asset Trading Infrastructure

This package provides the foundational components for multi-asset trading:
- Base asset classes and interfaces
- Unified order book implementation
- Risk management and Greeks calculations
- Cross-asset correlation analysis
- Portfolio optimization engines
"""

from .base_asset import BaseAsset, AssetType, AssetInfo, EquityAsset
from .order_book import UnifiedOrderBook, OrderBookLevel
from .risk_models import RiskModel, GreeksCalculator
# from .correlation import CrossAssetCorrelationAnalyzer
# from .portfolio import MultiAssetPortfolioOptimizer

__all__ = [
    'BaseAsset', 'AssetType', 'AssetInfo', 'EquityAsset',
    'UnifiedOrderBook', 'OrderBookLevel',
    'RiskModel', 'GreeksCalculator',
    # 'CrossAssetCorrelationAnalyzer',
    # 'MultiAssetPortfolioOptimizer'
]
