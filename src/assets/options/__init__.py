"""
Options Trading Module

This module provides comprehensive options trading support including:
- Black-Scholes and binomial pricing models
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Volatility modeling and implied volatility calculations
- Multi-leg strategy support (spreads, straddles, etc.)
- American and European option styles
"""

from .options_asset import OptionsAsset
from .pricing_models import BlackScholesModel, BinomialModel
from .greeks_calculator import GreeksCalculator
from .volatility_models import VolatilityModel

__all__ = ['OptionsAsset', 'BlackScholesModel', 'BinomialModel', 'GreeksCalculator', 'VolatilityModel']
