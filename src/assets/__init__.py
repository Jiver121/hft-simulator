"""
Multi-Asset Class Trading Infrastructure

This package provides comprehensive trading support for multiple asset classes
including equities, cryptocurrencies, options, futures, FX, and fixed income.

Key Features:
- Unified order book abstraction for all asset classes
- Asset-specific pricing models and risk calculations
- Cross-asset correlation analysis and portfolio optimization
- Support for synthetic instruments and complex spreads
"""

from .core.base_asset import BaseAsset, AssetType
from .core.order_book import UnifiedOrderBook
from .core.risk_models import RiskModel, GreeksCalculator
# from .core.correlation import CrossAssetCorrelationAnalyzer
# from .core.portfolio import MultiAssetPortfolioOptimizer

# Asset classes
# from .crypto.crypto_asset import CryptoAsset
# from .options.options_asset import OptionsAsset
# from .futures.futures_asset import FuturesAsset
# from .fx.fx_asset import FXAsset
# from .fixed_income.bond_asset import BondAsset

__all__ = [
    # Core infrastructure
    'BaseAsset', 'AssetType', 'UnifiedOrderBook', 
    'RiskModel', 'GreeksCalculator',
    # 'CrossAssetCorrelationAnalyzer', 'MultiAssetPortfolioOptimizer',
    
    # Asset classes
    # 'CryptoAsset', 'OptionsAsset', 'FuturesAsset', 'FXAsset', 'BondAsset'
]
