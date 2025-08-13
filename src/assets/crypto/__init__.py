"""
Cryptocurrency Trading Module

This module provides comprehensive cryptocurrency trading support including:
- Spot and futures cryptocurrency trading
- DEX integration for decentralized trading
- Cross-exchange arbitrage opportunities
- DeFi protocol integration
- Staking and yield farming capabilities
"""

from .crypto_asset import CryptoAsset
from .dex_integration import DEXIntegrator
from .defi_protocols import DeFiProtocol

__all__ = ['CryptoAsset', 'DEXIntegrator', 'DeFiProtocol']
