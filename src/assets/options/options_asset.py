"""
Options Asset Implementation

This module provides comprehensive options trading functionality including
pricing models, Greeks calculations, and multi-leg strategy support.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.engine.order_types import Order, Trade
from src.utils.constants import OrderSide, OrderType
from ..core.base_asset import BaseAsset, AssetInfo, AssetType
from .pricing_models import BlackScholesModel, BinomialModel, MonteCarloModel


class OptionStyle(Enum):
    """Option exercise style"""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


class OptionType(Enum):
    """Option type"""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionsAssetInfo(AssetInfo):
    """Options-specific asset information"""
    underlying_symbol: str = ""
    strike_price: float = 0.0
    expiry_date: Optional[pd.Timestamp] = None
    option_type: OptionType = OptionType.CALL
    option_style: OptionStyle = OptionStyle.AMERICAN
    
    # Contract specifications
    contract_multiplier: int = 100  # Standard options contract
    settlement_style: str = "physical"  # physical or cash
    
    # Greeks and pricing parameters
    implied_volatility: float = 0.2
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    
    # Market microstructure
    min_price_increment: float = 0.01
    max_position_limit: int = 1000
    
    # Strategy information
    strategy_legs: List[Dict[str, Any]] = field(default_factory=list)


class OptionsAsset(BaseAsset):
    """
    Options asset with comprehensive pricing and risk management
    
    Features:
    - Multiple pricing models (Black-Scholes, Binomial, Monte Carlo)
    - Real-time Greeks calculations
    - Multi-leg strategy support
    - Volatility surface modeling
    - Early exercise optimization for American options
    """
    
    def __init__(self, options_info: OptionsAssetInfo):
        super().__init__(options_info)
        self.options_info = options_info
        
        # Initialize pricing models
        self.bs_model = BlackScholesModel(dividend_yield=options_info.dividend_yield)
        self.binomial_model = BinomialModel(steps=100, dividend_yield=options_info.dividend_yield)
        self.mc_model = MonteCarloModel(dividend_yield=options_info.dividend_yield)
        
        # Current market data
        self._underlying_price: Optional[float] = None
        self._implied_volatility = options_info.implied_volatility
        self._risk_free_rate = options_info.risk_free_rate
        
        # Greeks cache
        self._greeks_cache: Dict[str, float] = {}
        self._greeks_timestamp: Optional[pd.Timestamp] = None
        self._cache_duration = pd.Timedelta(minutes=1)  # Cache for 1 minute
        
        # Strategy tracking
        self._strategy_positions: Dict[str, int] = {}  # leg_id -> position
        self._strategy_pnl = 0.0
    
    @property
    def underlying_price(self) -> Optional[float]:
        """Current underlying asset price"""
        return self._underlying_price
    
    @property
    def strike_price(self) -> float:
        """Strike price of the option"""
        return self.options_info.strike_price
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years"""
        if self.options_info.expiry_date is None:
            return 0.0
        
        now = pd.Timestamp.now()
        if now >= self.options_info.expiry_date:
            return 0.0
        
        time_diff = self.options_info.expiry_date - now
        return time_diff.total_seconds() / (365.25 * 24 * 3600)
    
    @property
    def is_expired(self) -> bool:
        """Check if option is expired"""
        return self.time_to_expiry <= 0
    
    @property
    def moneyness(self) -> Optional[float]:
        """Calculate moneyness (S/K for calls, K/S for puts)"""
        if self._underlying_price is None or self.strike_price == 0:
            return None
        
        if self.options_info.option_type == OptionType.CALL:
            return self._underlying_price / self.strike_price
        else:
            return self.strike_price / self._underlying_price
    
    def calculate_fair_value(self, model: str = "black_scholes", **kwargs) -> float:
        """
        Calculate option fair value using specified model
        
        Args:
            model: Pricing model ("black_scholes", "binomial", "monte_carlo")
            **kwargs: Additional model parameters
            
        Returns:
            Fair value of the option
        """
        if self.is_expired:
            return self.calculate_intrinsic_value()
        
        if self._underlying_price is None:
            raise ValueError(f"No underlying price available for {self.symbol}")
        
        # Get model parameters
        spot = self._underlying_price
        strike = self.strike_price
        time_to_expiry = self.time_to_expiry
        risk_free_rate = self._risk_free_rate
        volatility = self._implied_volatility
        option_type = self.options_info.option_type.value
        
        # Override with any provided kwargs
        spot = kwargs.get('spot', spot)
        strike = kwargs.get('strike', strike)
        time_to_expiry = kwargs.get('time_to_expiry', time_to_expiry)
        risk_free_rate = kwargs.get('risk_free_rate', risk_free_rate)
        volatility = kwargs.get('volatility', volatility)
        
        if model.lower() == "black_scholes":
            return self.bs_model.calculate_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )
        elif model.lower() == "binomial":
            exercise_style = self.options_info.option_style.value
            return self.binomial_model.calculate_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, exercise_style
            )
        elif model.lower() == "monte_carlo":
            return self.mc_model.calculate_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )
        else:
            raise ValueError(f"Unknown pricing model: {model}")
    
    def get_risk_metrics(self, position_size: int = 0) -> Dict[str, float]:
        """Calculate options-specific risk metrics including Greeks"""
        if self.is_expired:
            return self._get_expired_risk_metrics(position_size)
        
        # Get current Greeks
        greeks = self.get_greeks()
        
        # Calculate position-level risk metrics
        position_delta = greeks['delta'] * position_size * self.options_info.contract_multiplier
        position_gamma = greeks['gamma'] * position_size * self.options_info.contract_multiplier
        position_theta = greeks['theta'] * position_size * self.options_info.contract_multiplier
        position_vega = greeks['vega'] * position_size * self.options_info.contract_multiplier
        position_rho = greeks['rho'] * position_size * self.options_info.contract_multiplier
        
        # Calculate other risk metrics
        option_value = self.calculate_fair_value()
        position_value = option_value * position_size * self.options_info.contract_multiplier
        
        risk_metrics = {
            # Greeks
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'rho': greeks['rho'],
            
            # Position-level Greeks
            'position_delta': position_delta,
            'position_gamma': position_gamma,
            'position_theta': position_theta,
            'position_vega': position_vega,
            'position_rho': position_rho,
            
            # Standard metrics
            'position_value': position_value,
            'intrinsic_value': self.calculate_intrinsic_value(),
            'time_value': option_value - self.calculate_intrinsic_value(),
            'moneyness': self.moneyness or 0.0,
            'time_to_expiry': self.time_to_expiry,
            
            # Risk measures
            'pin_risk': self._calculate_pin_risk(),
            'early_exercise_risk': self._calculate_early_exercise_risk(),
            'volatility_risk': abs(position_vega) * 0.01  # 1% vol move risk
        }
        
        return risk_metrics
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate options order"""
        # Basic validation
        is_valid, error_msg = super().validate_order(order)
        if not is_valid:
            return False, error_msg
        
        # Check if option is expired
        if self.is_expired:
            return False, f"Option {self.symbol} is expired"
        
        # Check position limits
        if (abs(order.volume) > self.options_info.max_position_limit):
            return False, f"Order volume exceeds position limit"
        
        # Check price increments
        if order.price is not None:
            min_increment = self.options_info.min_price_increment
            if not self._is_valid_price_increment(order.price, min_increment):
                return False, f"Price must be in increments of {min_increment}"
        
        # Check if option has sufficient time value for meaningful trading
        if self.time_to_expiry < 1/365:  # Less than 1 day
            return False, "Insufficient time to expiry for trading"
        
        return True, "Options order validated successfully"
    
    def get_greeks(self, model: str = "black_scholes", force_recalc: bool = False) -> Dict[str, float]:
        """
        Get option Greeks with caching
        
        Args:
            model: Pricing model to use
            force_recalc: Force recalculation even if cached
            
        Returns:
            Dictionary of Greeks
        """
        # Check cache
        now = pd.Timestamp.now()
        if (not force_recalc and self._greeks_cache and self._greeks_timestamp and
            now - self._greeks_timestamp < self._cache_duration):
            return self._greeks_cache.copy()
        
        if self.is_expired or self._underlying_price is None:
            return self._zero_greeks()
        
        # Calculate Greeks
        spot = self._underlying_price
        strike = self.strike_price
        time_to_expiry = self.time_to_expiry
        risk_free_rate = self._risk_free_rate
        volatility = self._implied_volatility
        option_type = self.options_info.option_type.value
        
        if model.lower() == "black_scholes":
            greeks = self.bs_model.calculate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )
        elif model.lower() == "binomial":
            exercise_style = self.options_info.option_style.value
            greeks = self.binomial_model.calculate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, exercise_style
            )
        else:
            raise ValueError(f"Greeks calculation not supported for model: {model}")
        
        # Cache results
        self._greeks_cache = greeks
        self._greeks_timestamp = now
        
        return greeks
    
    def calculate_intrinsic_value(self) -> float:
        """Calculate intrinsic value of the option"""
        if self._underlying_price is None:
            return 0.0
        
        if self.options_info.option_type == OptionType.CALL:
            return max(0.0, self._underlying_price - self.strike_price)
        else:  # PUT
            return max(0.0, self.strike_price - self._underlying_price)
    
    def calculate_implied_volatility(self, market_price: float) -> Optional[float]:
        """Calculate implied volatility from market price"""
        if self.is_expired or self._underlying_price is None:
            return None
        
        return self.bs_model.calculate_implied_volatility(
            market_price, self._underlying_price, self.strike_price,
            self.time_to_expiry, self._risk_free_rate, 
            self.options_info.option_type.value
        )
    
    def update_underlying_price(self, price: float) -> None:
        """Update underlying asset price and invalidate Greeks cache"""
        self._underlying_price = price
        self._greeks_cache.clear()
        self._greeks_timestamp = None
    
    def update_implied_volatility(self, volatility: float) -> None:
        """Update implied volatility and invalidate Greeks cache"""
        self._implied_volatility = volatility
        self._greeks_cache.clear()
        self._greeks_timestamp = None
    
    def update_risk_free_rate(self, rate: float) -> None:
        """Update risk-free rate and invalidate Greeks cache"""
        self._risk_free_rate = rate
        self._greeks_cache.clear()
        self._greeks_timestamp = None
    
    def should_exercise_early(self) -> bool:
        """
        Determine if American option should be exercised early
        
        Uses comparison between intrinsic value and option value
        """
        if (self.options_info.option_style != OptionStyle.AMERICAN or 
            self.is_expired or self._underlying_price is None):
            return False
        
        intrinsic_value = self.calculate_intrinsic_value()
        option_value = self.calculate_fair_value(model="binomial")
        
        # Exercise if intrinsic value is very close to option value
        # (considering transaction costs and bid-ask spread)
        exercise_threshold = 0.01  # $0.01 threshold
        
        return intrinsic_value > 0 and (option_value - intrinsic_value) < exercise_threshold
    
    def create_strategy_leg(self, leg_id: str, position_size: int, 
                           strategy_type: str = "single") -> Dict[str, Any]:
        """
        Create a strategy leg for multi-leg strategies
        
        Args:
            leg_id: Unique identifier for the leg
            position_size: Number of contracts (positive for long, negative for short)
            strategy_type: Type of strategy this leg belongs to
            
        Returns:
            Strategy leg information
        """
        leg_info = {
            'leg_id': leg_id,
            'symbol': self.symbol,
            'position_size': position_size,
            'strike': self.strike_price,
            'expiry': self.options_info.expiry_date,
            'option_type': self.options_info.option_type.value,
            'strategy_type': strategy_type,
            'entry_price': self._current_price,
            'entry_time': pd.Timestamp.now()
        }
        
        self._strategy_positions[leg_id] = position_size
        return leg_info
    
    def calculate_strategy_pnl(self) -> float:
        """Calculate P&L for multi-leg strategy positions"""
        total_pnl = 0.0
        current_price = self.calculate_fair_value()
        
        for leg_id, position_size in self._strategy_positions.items():
            # This is simplified - would need entry prices for each leg
            leg_pnl = position_size * current_price * self.options_info.contract_multiplier
            total_pnl += leg_pnl
        
        return total_pnl
    
    # Private helper methods
    
    def _get_expired_risk_metrics(self, position_size: int) -> Dict[str, float]:
        """Get risk metrics for expired options"""
        intrinsic_value = self.calculate_intrinsic_value()
        position_value = intrinsic_value * position_size * self.options_info.contract_multiplier
        
        return {
            'delta': 1.0 if intrinsic_value > 0 else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'position_value': position_value,
            'intrinsic_value': intrinsic_value,
            'time_value': 0.0,
            'moneyness': self.moneyness or 0.0,
            'time_to_expiry': 0.0,
            'pin_risk': 0.0,
            'early_exercise_risk': 0.0,
            'volatility_risk': 0.0
        }
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    def _calculate_pin_risk(self) -> float:
        """
        Calculate pin risk (risk of finishing at-the-money at expiry)
        
        Pin risk is highest when the option is near-the-money and close to expiry
        """
        if self.is_expired or self._underlying_price is None:
            return 0.0
        
        # Distance from strike as percentage of underlying price
        distance_from_strike = abs(self._underlying_price - self.strike_price) / self._underlying_price
        
        # Time factor (higher risk closer to expiry)
        time_factor = max(0, 1 - self.time_to_expiry * 365)  # 0 to 1, higher closer to expiry
        
        # Pin risk is highest when at-the-money and close to expiry
        pin_risk = (1 - distance_from_strike) * time_factor
        
        return min(1.0, pin_risk)
    
    def _calculate_early_exercise_risk(self) -> float:
        """Calculate risk of early exercise for American options"""
        if (self.options_info.option_style != OptionStyle.AMERICAN or 
            self.is_expired or self._underlying_price is None):
            return 0.0
        
        intrinsic_value = self.calculate_intrinsic_value()
        if intrinsic_value <= 0:
            return 0.0
        
        # Calculate time value
        option_value = self.calculate_fair_value(model="binomial")
        time_value = option_value - intrinsic_value
        
        # Higher risk when time value is low relative to intrinsic value
        if intrinsic_value > 0:
            risk_ratio = 1 - (time_value / intrinsic_value)
            return max(0.0, min(1.0, risk_ratio))
        
        return 0.0
    
    def _is_valid_price_increment(self, price: float, increment: float) -> bool:
        """Check if price is a valid increment"""
        remainder = price % increment
        return remainder < 1e-6 or (increment - remainder) < 1e-6
    
    def __str__(self) -> str:
        option_type = self.options_info.option_type.value.upper()
        return f"OptionsAsset({self.symbol}, {option_type}, K={self.strike_price})"
