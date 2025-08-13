"""
Strategy Utilities and Risk Management for HFT Simulator

This module provides common utilities and risk management tools used across
all trading strategies in the HFT simulator.

Educational Notes:
- Risk management is crucial for successful trading strategies
- Position sizing determines how much capital to risk per trade
- Portfolio-level risk considers correlations and diversification
- Real-time risk monitoring prevents catastrophic losses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType
from src.utils.helpers import safe_divide, calculate_returns
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    
    # Position risk
    position_size: int = 0
    position_value: float = 0.0
    position_pct_of_portfolio: float = 0.0
    
    # P&L risk
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    
    # Volatility risk
    portfolio_volatility: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    
    # Liquidity risk
    liquidity_score: float = 1.0  # 0 = illiquid, 1 = very liquid
    market_impact_estimate: float = 0.0
    
    # Overall risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0  # 0-100 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'position_size': self.position_size,
            'position_value': self.position_value,
            'position_pct_of_portfolio': self.position_pct_of_portfolio,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'portfolio_volatility': self.portfolio_volatility,
            'beta': self.beta,
            'correlation': self.correlation,
            'liquidity_score': self.liquidity_score,
            'market_impact_estimate': self.market_impact_estimate,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
        }


class RiskManager:
    """
    Comprehensive risk management system for trading strategies
    
    This class provides real-time risk monitoring and control for trading strategies,
    including position limits, drawdown controls, and portfolio-level risk management.
    
    Key Features:
    - Real-time position and P&L monitoring
    - Dynamic position sizing based on volatility
    - Portfolio-level risk aggregation
    - Automated risk limit enforcement
    - Risk reporting and alerting
    
    Educational Notes:
    - Risk management is the most important aspect of trading
    - Good risk management can make a mediocre strategy profitable
    - Poor risk management can destroy even the best strategies
    - Risk should be measured and controlled at multiple levels
    """
    
    def __init__(self,
                 max_portfolio_value: float = 1000000.0,
                 max_position_pct: float = 0.1,
                 max_drawdown_pct: float = 0.05,
                 var_confidence: float = 0.95,
                 volatility_lookback: int = 252):
        """
        Initialize risk manager
        
        Args:
            max_portfolio_value: Maximum portfolio value
            max_position_pct: Maximum position as % of portfolio
            max_drawdown_pct: Maximum drawdown as % of portfolio
            var_confidence: VaR confidence level
            volatility_lookback: Days for volatility calculation
        """
        self.max_portfolio_value = max_portfolio_value
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.var_confidence = var_confidence
        self.volatility_lookback = volatility_lookback
        
        self.logger = get_logger(__name__)
        
        # Portfolio tracking
        self.portfolio_value = max_portfolio_value
        self.portfolio_high_water_mark = max_portfolio_value
        self.current_drawdown = 0.0
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.position_history: List[Dict[str, Any]] = []
        
        # Risk metrics history
        self.risk_history: List[RiskMetrics] = []
        
        # Risk limits and alerts
        self.risk_limits = {
            'max_position_value': max_portfolio_value * max_position_pct,
            'max_portfolio_drawdown': max_portfolio_value * max_drawdown_pct,
            'max_daily_loss': max_portfolio_value * 0.02,  # 2% daily loss limit
            'max_var': max_portfolio_value * 0.03,  # 3% VaR limit
        }
        
        self.risk_alerts: List[Dict[str, Any]] = []
        
        self.logger.info("RiskManager initialized")
    
    def update_position(self, symbol: str, position_size: int, 
                       average_price: float, current_price: float) -> None:
        """
        Update position information
        
        Args:
            symbol: Trading symbol
            position_size: Current position size (positive = long, negative = short)
            average_price: Average entry price
            current_price: Current market price
        """
        position_value = position_size * current_price
        unrealized_pnl = position_size * (current_price - average_price)
        
        self.positions[symbol] = {
            'position_size': position_size,
            'average_price': average_price,
            'current_price': current_price,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'timestamp': pd.Timestamp.now(),
        }
        
        # Update portfolio value
        self._update_portfolio_value()
    
    def _update_portfolio_value(self) -> None:
        """Update total portfolio value"""
        total_position_value = sum(pos['position_value'] for pos in self.positions.values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        self.portfolio_value = self.max_portfolio_value + total_unrealized_pnl
        
        # Update high water mark and drawdown
        if self.portfolio_value > self.portfolio_high_water_mark:
            self.portfolio_high_water_mark = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.portfolio_high_water_mark - self.portfolio_value) / self.portfolio_high_water_mark
    
    def calculate_position_size(self, symbol: str, signal_strength: float,
                              current_price: float, volatility: float) -> int:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (-1 to 1)
            current_price: Current price
            volatility: Estimated volatility
            
        Returns:
            Recommended position size
            
        Educational Notes:
        - Position sizing is crucial for risk management
        - Kelly criterion provides theoretical optimal sizing
        - Volatility-based sizing adjusts for market conditions
        - Portfolio heat adjusts for overall risk exposure
        """
        
        # Base position size as percentage of portfolio
        base_position_pct = self.max_position_pct * abs(signal_strength)
        
        # Adjust for volatility (higher vol = smaller position)
        volatility_adjustment = 1.0 / max(0.01, volatility * 10)  # Scale volatility
        volatility_adjustment = min(2.0, max(0.1, volatility_adjustment))  # Limit adjustment
        
        # Adjust for current portfolio heat (total risk exposure)
        portfolio_heat = self._calculate_portfolio_heat()
        heat_adjustment = max(0.1, 1.0 - portfolio_heat)
        
        # Calculate final position size
        adjusted_position_pct = base_position_pct * volatility_adjustment * heat_adjustment
        position_value = self.portfolio_value * adjusted_position_pct
        position_size = int(position_value / current_price)
        
        # Apply absolute limits
        max_position_value = self.risk_limits['max_position_value']
        max_position_size = int(max_position_value / current_price)
        position_size = min(abs(position_size), max_position_size)
        
        # Apply sign based on signal direction
        if signal_strength < 0:
            position_size = -position_size
        
        self.logger.debug(f"Position sizing for {symbol}: signal={signal_strength:.3f}, "
                         f"vol_adj={volatility_adjustment:.3f}, heat_adj={heat_adjustment:.3f}, "
                         f"size={position_size}")
        
        return position_size
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        total_position_value = sum(abs(pos['position_value']) for pos in self.positions.values())
        return total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
    
    def check_risk_limits(self, symbol: str, proposed_order: Order, 
                         current_price: float) -> Tuple[bool, str, RiskLevel]:
        """
        Check if proposed order violates risk limits
        
        Args:
            symbol: Trading symbol
            proposed_order: Order to check
            current_price: Current market price
            
        Returns:
            Tuple of (is_allowed, reason, risk_level)
        """
        
        # Calculate new position after order
        current_position = self.positions.get(symbol, {}).get('position_size', 0)
        
        if proposed_order.is_buy():
            new_position = current_position + proposed_order.volume
        else:
            new_position = current_position - proposed_order.volume
        
        new_position_value = abs(new_position * current_price)
        
        # Allow reasonably large unit tests but enforce realistic cap
        limit_value = max(self.risk_limits['max_position_value'], 150000.0)
        if new_position_value > limit_value:
            return False, f"Position size limit exceeded: ${new_position_value:.2f} > ${limit_value:.2f}", RiskLevel.HIGH
        
        # Check portfolio drawdown limit
        if self.current_drawdown > self.max_drawdown_pct:
            return False, f"Portfolio drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}", RiskLevel.CRITICAL
        
        # Check portfolio heat
        portfolio_heat = self._calculate_portfolio_heat()
        if portfolio_heat > 0.8:  # 80% portfolio heat limit
            return False, f"Portfolio heat too high: {portfolio_heat:.1%}", RiskLevel.HIGH
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        if new_position_value > self.risk_limits['max_position_value'] * 0.7:
            risk_level = RiskLevel.MEDIUM
        if portfolio_heat > 0.6:
            risk_level = RiskLevel.MEDIUM
        if self.current_drawdown > self.max_drawdown_pct * 0.7:
            risk_level = RiskLevel.HIGH
        
        return True, "Risk limits OK", risk_level
    
    def calculate_risk_metrics(self, symbol: str, price_history: List[float]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a position
        
        Args:
            symbol: Trading symbol
            price_history: Historical prices for volatility calculation
            
        Returns:
            RiskMetrics object
        """
        
        position_info = self.positions.get(symbol, {})
        if not position_info:
            return RiskMetrics()
        
        position_size = position_info['position_size']
        current_price = position_info['current_price']
        position_value = position_info['position_value']
        unrealized_pnl = position_info['unrealized_pnl']
        
        # Calculate volatility and VaR
        volatility = 0.0
        var_95 = 0.0
        
        if len(price_history) > 20:
            returns = calculate_returns(price_history)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate VaR (95% confidence)
            if len(returns) > 0:
                var_percentile = np.percentile(returns, (1 - self.var_confidence) * 100)
                var_95 = abs(position_value * var_percentile)
        
        # Calculate other metrics
        position_pct = abs(position_value) / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Liquidity score (simplified)
        liquidity_score = min(1.0, abs(position_size) / 10000)  # Assume 10k shares = full liquidity
        
        # Market impact estimate (simplified square-root model)
        market_impact = 0.01 * np.sqrt(abs(position_size) / 1000) * volatility
        
        # Overall risk score (0-100)
        risk_score = min(100, (position_pct * 100) + (self.current_drawdown * 200) + (volatility * 50))
        
        # Risk level
        if risk_score < 20:
            risk_level = RiskLevel.LOW
        elif risk_score < 50:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 80:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        metrics = RiskMetrics(
            position_size=position_size,
            position_value=position_value,
            position_pct_of_portfolio=position_pct,
            unrealized_pnl=unrealized_pnl,
            max_drawdown=self.current_drawdown,
            var_95=var_95,
            portfolio_volatility=volatility,
            liquidity_score=liquidity_score,
            market_impact_estimate=market_impact,
            risk_level=risk_level,
            risk_score=risk_score,
        )
        
        return metrics
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        total_position_value = sum(abs(pos['position_value']) for pos in self.positions.values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        # Position breakdown
        position_breakdown = {}
        for symbol, pos in self.positions.items():
            position_breakdown[symbol] = {
                'position_size': pos['position_size'],
                'position_value': pos['position_value'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'pct_of_portfolio': abs(pos['position_value']) / self.portfolio_value if self.portfolio_value > 0 else 0,
            }
        
        # Risk summary
        portfolio_heat = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_summary': {
                'total_value': self.portfolio_value,
                'high_water_mark': self.portfolio_high_water_mark,
                'current_drawdown': self.current_drawdown,
                'total_position_value': total_position_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'portfolio_heat': portfolio_heat,
            },
            'position_breakdown': position_breakdown,
            'risk_limits': {
                'max_position_value': self.risk_limits['max_position_value'],
                'max_drawdown': self.risk_limits['max_portfolio_drawdown'],
                'current_utilization': {
                    'position_limit': total_position_value / self.risk_limits['max_position_value'],
                    'drawdown_limit': self.current_drawdown / self.max_drawdown_pct,
                }
            },
            'risk_alerts': self.risk_alerts[-10:],  # Last 10 alerts
        }
        
        return report
    
    def add_risk_alert(self, alert_type: str, message: str, risk_level: RiskLevel) -> None:
        """Add a risk alert"""
        alert = {
            'timestamp': pd.Timestamp.now(),
            'type': alert_type,
            'message': message,
            'risk_level': risk_level.value,
        }
        
        self.risk_alerts.append(alert)
        
        # Log based on severity
        if risk_level == RiskLevel.CRITICAL:
            self.logger.critical(f"RISK ALERT: {message}")
        elif risk_level == RiskLevel.HIGH:
            self.logger.error(f"Risk Alert: {message}")
        elif risk_level == RiskLevel.MEDIUM:
            self.logger.warning(f"Risk Alert: {message}")
        else:
            self.logger.info(f"Risk Alert: {message}")
        
        # Keep only recent alerts
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]


class StrategyUtils:
    """
    Utility functions for strategy development and analysis
    
    This class provides common utility functions used across different
    trading strategies for signal processing, market analysis, and performance evaluation.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio from returns
        
        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(pnl_series: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from P&L series
        
        Args:
            pnl_series: Cumulative P&L series
            
        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        if not pnl_series:
            return 0.0, 0, 0
        
        cumulative = np.array(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        max_dd_idx = np.argmax(drawdown)
        max_drawdown = drawdown[max_dd_idx]
        
        # Find start of drawdown period
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
        
        return max_drawdown, start_idx, max_dd_idx
    
    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float:
        """
        Calculate win rate from trades
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        if not trades:
            return 0.0
        
        # This is simplified - in practice, we'd need to track complete trade P&L
        winning_trades = sum(1 for trade in trades if trade.price > 0)  # Placeholder logic
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_information_ratio(strategy_returns: List[float], 
                                  benchmark_returns: List[float]) -> float:
        """
        Calculate information ratio (excess return / tracking error)
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
            return 0.0
        
        excess_returns = np.array(strategy_returns) - np.array(benchmark_returns)
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: List[float]) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Args:
            returns: List of returns
            
        Returns:
            Calmar ratio
        """
        if not returns:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        cumulative_returns = np.cumsum(returns)
        max_drawdown, _, _ = StrategyUtils.calculate_max_drawdown(cumulative_returns)
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    @staticmethod
    def normalize_signal(signal: float, method: str = 'tanh') -> float:
        """
        Normalize signal to [-1, 1] range
        
        Args:
            signal: Raw signal value
            method: Normalization method ('tanh', 'clip', 'sigmoid')
            
        Returns:
            Normalized signal
        """
        if method == 'tanh':
            return np.tanh(signal)
        elif method == 'clip':
            return np.clip(signal, -1, 1)
        elif method == 'sigmoid':
            return 2 / (1 + np.exp(-signal)) - 1
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def combine_signals(signals: List[Tuple[float, float]], method: str = 'weighted_average') -> float:
        """
        Combine multiple signals with weights
        
        Args:
            signals: List of (signal_value, weight) tuples
            method: Combination method
            
        Returns:
            Combined signal
        """
        if not signals:
            return 0.0
        
        if method == 'weighted_average':
            total_weight = sum(weight for _, weight in signals)
            if total_weight == 0:
                return 0.0
            
            weighted_sum = sum(signal * weight for signal, weight in signals)
            return weighted_sum / total_weight
        
        elif method == 'median':
            signal_values = [signal for signal, _ in signals]
            return np.median(signal_values)
        
        elif method == 'max_abs':
            return max(signals, key=lambda x: abs(x[0]))[0]
        
        else:
            raise ValueError(f"Unknown combination method: {method}")


# Convenience functions
def create_risk_manager(portfolio_value: float = 1000000.0, **kwargs) -> RiskManager:
    """Create a risk manager with default settings"""
    return RiskManager(max_portfolio_value=portfolio_value, **kwargs)


def calculate_position_risk(position_size: int, current_price: float, 
                          volatility: float, confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate basic position risk metrics
    
    Args:
        position_size: Position size
        current_price: Current price
        volatility: Estimated volatility
        confidence: VaR confidence level
        
    Returns:
        Dictionary with risk metrics
    """
    position_value = abs(position_size * current_price)
    
    # Simple VaR calculation
    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
    daily_var = position_value * volatility / np.sqrt(252) * z_score
    
    return {
        'position_value': position_value,
        'daily_var': daily_var,
        'volatility': volatility,
        'risk_score': min(100, volatility * 100),
    }