"""
Fill Models for HFT Simulator

This module implements various fill probability models that simulate realistic
order execution behavior in different market conditions.

Educational Notes:
- Fill models determine whether and how orders get executed
- Real markets have partial fills, rejections, and latency
- Different models simulate various market conditions and execution venues
- Realistic modeling is crucial for accurate strategy backtesting
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot


@dataclass
class FillResult:
    """
    Result of a fill attempt
    
    Contains information about whether an order was filled,
    the fill price, volume, and any associated costs.
    """
    
    filled: bool
    fill_volume: int = 0
    fill_price: Optional[float] = None
    remaining_volume: int = 0
    rejection_reason: Optional[str] = None
    latency_microseconds: int = 0
    slippage: float = 0.0
    market_impact: float = 0.0
    
    @property
    def fill_ratio(self) -> float:
        """Calculate the ratio of volume filled"""
        original_volume = self.fill_volume + self.remaining_volume
        return self.fill_volume / original_volume if original_volume > 0 else 0.0
    
    @property
    def is_partial_fill(self) -> bool:
        """Check if this was a partial fill"""
        return self.filled and self.remaining_volume > 0
    
    @property
    def is_complete_fill(self) -> bool:
        """Check if this was a complete fill"""
        return self.filled and self.remaining_volume == 0


class FillModel(ABC):
    """
    Abstract base class for order fill models
    
    Fill models determine the probability and characteristics of order execution
    based on market conditions, order type, and other factors.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def calculate_fill_probability(self, order: Order, snapshot: BookSnapshot) -> float:
        """
        Calculate the probability that an order will be filled
        
        Args:
            order: Order to evaluate
            snapshot: Current market snapshot
            
        Returns:
            Fill probability between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def simulate_fill(self, order: Order, snapshot: BookSnapshot) -> FillResult:
        """
        Simulate the execution of an order
        
        Args:
            order: Order to execute
            snapshot: Current market snapshot
            
        Returns:
            FillResult with execution details
        """
        pass
    
    def calculate_slippage(self, order: Order, execution_price: float) -> float:
        """
        Calculate slippage for an executed order
        
        Args:
            order: The executed order
            execution_price: Actual execution price
            
        Returns:
            Slippage in price units (positive = worse than expected)
        """
        if order.is_market_order():
            # For market orders, slippage is relative to mid-price
            return 0.0  # Calculated elsewhere
        else:
            # For limit orders, slippage is relative to limit price
            if order.is_buy():
                return max(0, execution_price - order.price)
            else:
                return max(0, order.price - execution_price)
    
    def calculate_market_impact(self, order: Order, snapshot: BookSnapshot) -> float:
        """
        Calculate market impact of an order
        
        Args:
            order: Order to evaluate
            snapshot: Current market snapshot
            
        Returns:
            Estimated market impact in price units
        """
        if not snapshot.mid_price:
            return 0.0
        
        # Simple square-root impact model
        # Impact = volatility * sqrt(volume / average_volume) * direction
        
        # Estimate volatility from spread
        volatility = (snapshot.spread or 0.01) / snapshot.mid_price
        
        # Estimate average volume (simplified)
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
        avg_volume = max(total_volume / 10, 1000)  # Rough estimate
        
        # Calculate impact
        volume_ratio = order.volume / avg_volume
        impact_factor = volatility * np.sqrt(volume_ratio)
        
        # Direction: positive for buy orders, negative for sell orders
        direction = 1 if order.is_buy() else -1
        
        return direction * impact_factor * snapshot.mid_price


class PerfectFillModel(FillModel):
    """
    Perfect fill model - all orders execute immediately at expected prices
    
    This model is useful for:
    - Initial strategy development
    - Theoretical performance analysis
    - Comparing against realistic models
    
    Educational Notes:
    - Market orders always fill at mid-price
    - Limit orders fill if they cross the spread
    - No latency, slippage, or rejections
    """
    
    def __init__(self):
        super().__init__("PerfectFill")
    
    def calculate_fill_probability(self, order: Order, snapshot: BookSnapshot) -> float:
        """Perfect model - orders always fill if they can match"""
        if order.is_market_order():
            return 1.0
        
        # Limit orders fill if they cross the spread
        if order.is_buy() and snapshot.best_ask:
            return 1.0 if order.price >= snapshot.best_ask else 0.0
        elif order.is_sell() and snapshot.best_bid:
            return 1.0 if order.price <= snapshot.best_bid else 0.0
        
        return 0.0
    
    def simulate_fill(self, order: Order, snapshot: BookSnapshot) -> FillResult:
        """Simulate perfect execution"""
        fill_prob = self.calculate_fill_probability(order, snapshot)
        
        if fill_prob == 0.0:
            return FillResult(
                filled=False,
                remaining_volume=order.volume,
                rejection_reason="No matching liquidity"
            )
        
        # Determine execution price
        if order.is_market_order():
            # Market orders execute at mid-price (perfect world)
            fill_price = snapshot.mid_price
        else:
            # Limit orders execute at their limit price
            fill_price = order.price
        
        if fill_price is None:
            return FillResult(
                filled=False,
                remaining_volume=order.volume,
                rejection_reason="No market price available"
            )
        
        return FillResult(
            filled=True,
            fill_volume=order.volume,
            fill_price=fill_price,
            remaining_volume=0,
            latency_microseconds=0,
            slippage=0.0,
            market_impact=0.0
        )


class RealisticFillModel(FillModel):
    """
    Realistic fill model with latency, slippage, and partial fills
    
    This model simulates real-world execution characteristics:
    - Variable latency based on market conditions
    - Slippage due to market movement and impact
    - Partial fills for large orders
    - Order rejections under certain conditions
    
    Educational Notes:
    - Fill probability depends on order size relative to available liquidity
    - Latency varies based on market volatility and order type
    - Slippage increases with order size and market impact
    """
    
    def __init__(self, 
                 base_latency_us: int = 50,
                 latency_variance_us: int = 30,
                 base_fill_probability: float = 0.95,
                 slippage_factor: float = 0.0001,
                 partial_fill_threshold: float = 0.1):
        """
        Initialize realistic fill model
        
        Args:
            base_latency_us: Base latency in microseconds
            latency_variance_us: Latency variance
            base_fill_probability: Base probability of order execution
            slippage_factor: Factor for slippage calculation
            partial_fill_threshold: Threshold for partial fills
        """
        super().__init__("RealisticFill")
        
        self.base_latency_us = base_latency_us
        self.latency_variance_us = latency_variance_us
        self.base_fill_probability = base_fill_probability
        self.slippage_factor = slippage_factor
        self.partial_fill_threshold = partial_fill_threshold
    
    def calculate_fill_probability(self, order: Order, snapshot: BookSnapshot) -> float:
        """Calculate realistic fill probability"""
        base_prob = self.base_fill_probability
        
        # Market orders have higher fill probability
        if order.is_market_order():
            # Check if there's sufficient liquidity
            if order.is_buy() and snapshot.total_ask_volume > 0:
                liquidity_ratio = min(1.0, order.volume / snapshot.total_ask_volume)
                return base_prob * (1.0 - 0.3 * liquidity_ratio)  # Reduce prob for large orders
            elif order.is_sell() and snapshot.total_bid_volume > 0:
                liquidity_ratio = min(1.0, order.volume / snapshot.total_bid_volume)
                return base_prob * (1.0 - 0.3 * liquidity_ratio)
            else:
                return 0.1  # Low probability if no liquidity
        
        # Limit orders
        else:
            # Check if order crosses the spread
            if order.is_buy() and snapshot.best_ask:
                if order.price >= snapshot.best_ask:
                    # Aggressive limit order
                    return base_prob * 0.9
                else:
                    # Passive limit order - lower probability
                    return base_prob * 0.3
            elif order.is_sell() and snapshot.best_bid:
                if order.price <= snapshot.best_bid:
                    # Aggressive limit order
                    return base_prob * 0.9
                else:
                    # Passive limit order - lower probability
                    return base_prob * 0.3
        
        return 0.1  # Default low probability
    
    def simulate_fill(self, order: Order, snapshot: BookSnapshot) -> FillResult:
        """Simulate realistic order execution"""
        fill_prob = self.calculate_fill_probability(order, snapshot)
        
        # Determine if order fills
        if random.random() > fill_prob:
            return FillResult(
                filled=False,
                remaining_volume=order.volume,
                rejection_reason="Order not filled due to market conditions",
                latency_microseconds=self._simulate_latency(order, snapshot)
            )
        
        # Calculate execution details
        latency = self._simulate_latency(order, snapshot)
        fill_price = self._calculate_execution_price(order, snapshot)
        fill_volume = self._calculate_fill_volume(order, snapshot)
        
        if fill_price is None:
            return FillResult(
                filled=False,
                remaining_volume=order.volume,
                rejection_reason="No executable price available",
                latency_microseconds=latency
            )
        
        # Calculate slippage and market impact
        slippage = self._calculate_realistic_slippage(order, fill_price, snapshot)
        market_impact = self.calculate_market_impact(order, snapshot)
        
        return FillResult(
            filled=True,
            fill_volume=fill_volume,
            fill_price=fill_price,
            remaining_volume=order.volume - fill_volume,
            latency_microseconds=latency,
            slippage=slippage,
            market_impact=market_impact
        )
    
    def _simulate_latency(self, order: Order, snapshot: BookSnapshot) -> int:
        """Simulate execution latency"""
        base_latency = self.base_latency_us
        
        # Market orders typically have lower latency
        if order.is_market_order():
            base_latency *= 0.8
        
        # Add volatility-based latency
        if snapshot.spread and snapshot.mid_price:
            volatility = snapshot.spread / snapshot.mid_price
            volatility_factor = min(2.0, 1.0 + volatility * 10)
            base_latency *= volatility_factor
        
        # Add random variance
        variance = random.gauss(0, self.latency_variance_us)
        total_latency = max(10, int(base_latency + variance))  # Minimum 10 microseconds
        
        return total_latency
    
    def _calculate_execution_price(self, order: Order, snapshot: BookSnapshot) -> Optional[float]:
        """Calculate realistic execution price"""
        if order.is_market_order():
            # Market orders execute at best available price plus slippage
            if order.is_buy() and snapshot.best_ask:
                base_price = snapshot.best_ask
            elif order.is_sell() and snapshot.best_bid:
                base_price = snapshot.best_bid
            else:
                return None
            
            # Add market impact
            impact = self.calculate_market_impact(order, snapshot)
            return base_price + impact
        
        else:
            # Limit orders execute at limit price or better
            if order.is_buy() and snapshot.best_ask:
                if order.price >= snapshot.best_ask:
                    # Can execute at better price
                    return min(order.price, snapshot.best_ask)
            elif order.is_sell() and snapshot.best_bid:
                if order.price <= snapshot.best_bid:
                    # Can execute at better price
                    return max(order.price, snapshot.best_bid)
            
            return order.price
    
    def _calculate_fill_volume(self, order: Order, snapshot: BookSnapshot) -> int:
        """Calculate how much of the order gets filled"""
        # Check available liquidity
        if order.is_buy():
            available_liquidity = snapshot.total_ask_volume
        else:
            available_liquidity = snapshot.total_bid_volume
        
        if available_liquidity == 0:
            return 0
        
        # For small orders relative to liquidity, fill completely
        liquidity_ratio = order.volume / available_liquidity
        
        if liquidity_ratio <= self.partial_fill_threshold:
            return order.volume
        
        # For large orders, partial fill based on available liquidity
        fill_ratio = min(1.0, available_liquidity / order.volume)
        
        # Add some randomness to partial fills
        fill_ratio *= random.uniform(0.7, 1.0)
        
        return max(1, int(order.volume * fill_ratio))
    
    def _calculate_realistic_slippage(self, order: Order, execution_price: float, 
                                    snapshot: BookSnapshot) -> float:
        """Calculate realistic slippage"""
        if not snapshot.mid_price:
            return 0.0
        
        if order.is_market_order():
            # Slippage relative to mid-price
            expected_price = snapshot.mid_price
            slippage = abs(execution_price - expected_price)
        else:
            # Slippage relative to limit price
            slippage = self.calculate_slippage(order, execution_price)
        
        # Add volume-based slippage
        volume_factor = np.sqrt(order.volume / 1000)  # Normalize to 1000 shares
        volume_slippage = self.slippage_factor * snapshot.mid_price * volume_factor
        
        return slippage + volume_slippage


class AdaptiveFillModel(FillModel):
    """
    Adaptive fill model that adjusts behavior based on market conditions
    
    This model changes its parameters based on:
    - Market volatility
    - Liquidity levels
    - Time of day
    - Recent market activity
    
    Educational Notes:
    - Real markets have time-varying execution characteristics
    - Volatility affects fill rates and slippage
    - Liquidity varies throughout the trading day
    """
    
    def __init__(self):
        super().__init__("AdaptiveFill")
        
        # Base models for different market regimes
        self.calm_market_model = RealisticFillModel(
            base_latency_us=30,
            base_fill_probability=0.98,
            slippage_factor=0.00005
        )
        
        self.volatile_market_model = RealisticFillModel(
            base_latency_us=80,
            base_fill_probability=0.85,
            slippage_factor=0.0003
        )
        
        self.illiquid_market_model = RealisticFillModel(
            base_latency_us=100,
            base_fill_probability=0.70,
            slippage_factor=0.0005
        )
    
    def calculate_fill_probability(self, order: Order, snapshot: BookSnapshot) -> float:
        """Calculate fill probability based on market regime"""
        model = self._select_model(snapshot)
        return model.calculate_fill_probability(order, snapshot)
    
    def simulate_fill(self, order: Order, snapshot: BookSnapshot) -> FillResult:
        """Simulate fill using appropriate model for current market conditions"""
        model = self._select_model(snapshot)
        return model.simulate_fill(order, snapshot)
    
    def _select_model(self, snapshot: BookSnapshot) -> RealisticFillModel:
        """Select appropriate model based on market conditions"""
        # Calculate market regime indicators
        volatility_score = self._calculate_volatility_score(snapshot)
        liquidity_score = self._calculate_liquidity_score(snapshot)
        
        # Select model based on scores
        if volatility_score > 0.7:
            return self.volatile_market_model
        elif liquidity_score < 0.3:
            return self.illiquid_market_model
        else:
            return self.calm_market_model
    
    def _calculate_volatility_score(self, snapshot: BookSnapshot) -> float:
        """Calculate volatility score (0 = calm, 1 = very volatile)"""
        if not snapshot.spread or not snapshot.mid_price:
            return 0.5  # Default moderate volatility
        
        # Use spread as volatility proxy
        spread_pct = snapshot.spread / snapshot.mid_price
        
        # Normalize to 0-1 scale (assuming 0.5% spread = high volatility)
        volatility_score = min(1.0, spread_pct / 0.005)
        
        return volatility_score
    
    def _calculate_liquidity_score(self, snapshot: BookSnapshot) -> float:
        """Calculate liquidity score (0 = illiquid, 1 = very liquid)"""
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
        
        if total_volume == 0:
            return 0.0
        
        # Normalize based on typical volumes (assuming 10,000 = good liquidity)
        liquidity_score = min(1.0, total_volume / 10000)
        
        return liquidity_score


# Factory function for creating fill models
def create_fill_model(model_type: str, **kwargs) -> FillModel:
    """
    Factory function to create fill models
    
    Args:
        model_type: Type of model ('perfect', 'realistic', 'adaptive')
        **kwargs: Model-specific parameters
        
    Returns:
        Configured fill model
    """
    if model_type.lower() == 'perfect':
        return PerfectFillModel()
    elif model_type.lower() == 'realistic':
        return RealisticFillModel(**kwargs)
    elif model_type.lower() == 'adaptive':
        return AdaptiveFillModel()
    else:
        raise ValueError(f"Unknown fill model type: {model_type}")


# Utility functions for fill analysis
def analyze_fill_performance(fill_results: list[FillResult]) -> Dict[str, Any]:
    """
    Analyze the performance of fill results
    
    Args:
        fill_results: List of FillResult objects
        
    Returns:
        Dictionary with performance statistics
    """
    if not fill_results:
        return {}
    
    filled_results = [r for r in fill_results if r.filled]
    
    stats = {
        'total_orders': len(fill_results),
        'filled_orders': len(filled_results),
        'fill_rate': len(filled_results) / len(fill_results),
        'average_latency_us': np.mean([r.latency_microseconds for r in fill_results]),
        'average_slippage': np.mean([r.slippage for r in filled_results]) if filled_results else 0,
        'average_market_impact': np.mean([r.market_impact for r in filled_results]) if filled_results else 0,
        'partial_fill_rate': sum(1 for r in filled_results if r.is_partial_fill) / len(filled_results) if filled_results else 0,
        'rejection_reasons': {}
    }
    
    # Count rejection reasons
    for result in fill_results:
        if not result.filled and result.rejection_reason:
            reason = result.rejection_reason
            stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
    
    return stats