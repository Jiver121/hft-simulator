"""
Risk Models and Greeks Calculator

This module provides comprehensive risk management tools for multi-asset trading:
- Asset-specific risk models
- Greeks calculations for derivatives
- Portfolio-level risk aggregation
- Stress testing and scenario analysis
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from datetime import datetime, timedelta

from .base_asset import BaseAsset, AssetType


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    GREEKS = "greeks"


@dataclass
class RiskScenario:
    """Risk scenario for stress testing"""
    name: str
    description: str
    market_shocks: Dict[str, float]  # asset -> shock percentage
    probability: float = 0.0
    time_horizon: int = 1  # days


class RiskModel(ABC):
    """Base class for risk models"""
    
    @abstractmethod
    def calculate_var(self, positions: Dict[str, float], 
                     confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, positions: Dict[str, float],
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        pass
    
    @abstractmethod
    def stress_test(self, positions: Dict[str, float],
                   scenarios: List[RiskScenario]) -> Dict[str, float]:
        """Perform stress testing under various scenarios"""
        pass


class ParametricRiskModel(RiskModel):
    """
    Parametric risk model using covariance matrices
    
    Assumes multivariate normal distribution of returns
    """
    
    def __init__(self, assets: List[BaseAsset], lookback_days: int = 252):
        self.assets = {asset.symbol: asset for asset in assets}
        self.lookback_days = lookback_days
        self.covariance_matrix: Optional[np.ndarray] = None
        self.mean_returns: Optional[np.ndarray] = None
        self.asset_symbols = list(self.assets.keys())
        self._last_update: Optional[pd.Timestamp] = None
    
    def update_parameters(self, returns_data: pd.DataFrame) -> None:
        """
        Update risk model parameters from historical returns
        
        Args:
            returns_data: DataFrame with columns as asset symbols and returns
        """
        # Filter for our assets
        asset_returns = returns_data[self.asset_symbols].tail(self.lookback_days)
        
        # Calculate covariance matrix and mean returns
        self.covariance_matrix = asset_returns.cov().values
        self.mean_returns = asset_returns.mean().values
        self._last_update = pd.Timestamp.now()
    
    def calculate_var(self, positions: Dict[str, float], 
                     confidence_level: float = 0.95) -> float:
        """Calculate portfolio VaR using parametric method"""
        if self.covariance_matrix is None:
            raise ValueError("Risk model not initialized. Call update_parameters first.")
        
        # Convert positions to array
        position_vector = np.array([positions.get(symbol, 0.0) for symbol in self.asset_symbols])
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(position_vector, np.dot(self.covariance_matrix, position_vector))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -z_score * portfolio_std
        
        return var
    
    def calculate_expected_shortfall(self, positions: Dict[str, float],
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if self.covariance_matrix is None:
            raise ValueError("Risk model not initialized. Call update_parameters first.")
        
        # Calculate portfolio standard deviation
        position_vector = np.array([positions.get(symbol, 0.0) for symbol in self.asset_symbols])
        portfolio_variance = np.dot(position_vector, np.dot(self.covariance_matrix, position_vector))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate Expected Shortfall
        z_score = stats.norm.ppf(1 - confidence_level)
        expected_shortfall = portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        return expected_shortfall
    
    def stress_test(self, positions: Dict[str, float],
                   scenarios: List[RiskScenario]) -> Dict[str, float]:
        """Perform stress testing under various scenarios"""
        stress_results = {}
        
        for scenario in scenarios:
            scenario_pnl = 0.0
            
            for asset_symbol, position in positions.items():
                if asset_symbol in scenario.market_shocks:
                    shock = scenario.market_shocks[asset_symbol]
                    # Simplified: assume linear relationship
                    asset_pnl = position * shock
                    scenario_pnl += asset_pnl
            
            stress_results[scenario.name] = scenario_pnl
        
        return stress_results
    
    def calculate_portfolio_beta(self, positions: Dict[str, float],
                               market_symbol: str = "SPY") -> float:
        """Calculate portfolio beta relative to market"""
        if market_symbol not in self.asset_symbols:
            raise ValueError(f"Market symbol {market_symbol} not in risk model")
        
        market_idx = self.asset_symbols.index(market_symbol)
        market_variance = self.covariance_matrix[market_idx, market_idx]
        
        portfolio_beta = 0.0
        total_position = sum(abs(pos) for pos in positions.values())
        
        if total_position > 0:
            for symbol, position in positions.items():
                if symbol in self.asset_symbols:
                    asset_idx = self.asset_symbols.index(symbol)
                    asset_market_cov = self.covariance_matrix[asset_idx, market_idx]
                    asset_beta = asset_market_cov / market_variance
                    weight = abs(position) / total_position
                    portfolio_beta += weight * asset_beta
        
        return portfolio_beta


class MonteCarloRiskModel(RiskModel):
    """
    Monte Carlo risk model for non-parametric risk assessment
    
    Uses simulation to handle non-normal distributions and complex dependencies
    """
    
    def __init__(self, assets: List[BaseAsset], num_simulations: int = 10000,
                 time_horizon: int = 1):
        self.assets = {asset.symbol: asset for asset in assets}
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.asset_symbols = list(self.assets.keys())
        self.simulation_results: Optional[np.ndarray] = None
    
    def run_simulations(self, returns_data: pd.DataFrame) -> None:
        """
        Run Monte Carlo simulations
        
        Args:
            returns_data: Historical returns data
        """
        asset_returns = returns_data[self.asset_symbols].dropna()
        
        # Bootstrap simulation
        simulation_results = []
        
        for _ in range(self.num_simulations):
            # Sample random period
            sampled_returns = asset_returns.sample(n=self.time_horizon, replace=True)
            cumulative_return = sampled_returns.sum()
            simulation_results.append(cumulative_return.values)
        
        self.simulation_results = np.array(simulation_results)
    
    def calculate_var(self, positions: Dict[str, float], 
                     confidence_level: float = 0.95) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if self.simulation_results is None:
            raise ValueError("Run simulations first")
        
        # Convert positions to array
        position_vector = np.array([positions.get(symbol, 0.0) for symbol in self.asset_symbols])
        
        # Calculate P&L for each simulation
        portfolio_pnl = np.dot(self.simulation_results, position_vector)
        
        # Calculate VaR as percentile
        var = -np.percentile(portfolio_pnl, (1 - confidence_level) * 100)
        
        return var
    
    def calculate_expected_shortfall(self, positions: Dict[str, float],
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall using Monte Carlo"""
        if self.simulation_results is None:
            raise ValueError("Run simulations first")
        
        position_vector = np.array([positions.get(symbol, 0.0) for symbol in self.asset_symbols])
        portfolio_pnl = np.dot(self.simulation_results, position_vector)
        
        # Calculate ES as mean of worst cases
        var_threshold = -np.percentile(portfolio_pnl, (1 - confidence_level) * 100)
        worst_cases = portfolio_pnl[portfolio_pnl <= -var_threshold]
        
        if len(worst_cases) > 0:
            return -np.mean(worst_cases)
        else:
            return 0.0
    
    def stress_test(self, positions: Dict[str, float],
                   scenarios: List[RiskScenario]) -> Dict[str, float]:
        """Perform stress testing"""
        stress_results = {}
        
        for scenario in scenarios:
            scenario_pnl = 0.0
            
            for asset_symbol, position in positions.items():
                if asset_symbol in scenario.market_shocks:
                    shock = scenario.market_shocks[asset_symbol]
                    asset_pnl = position * shock
                    scenario_pnl += asset_pnl
            
            stress_results[scenario.name] = scenario_pnl
        
        return stress_results


class GreeksCalculator:
    """
    Calculator for option Greeks across different asset types
    
    Provides unified interface for Greeks calculations regardless
    of underlying asset type or option structure
    """
    
    def __init__(self):
        self.supported_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    
    def calculate_portfolio_greeks(self, positions: Dict[str, Tuple[BaseAsset, int]]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks
        
        Args:
            positions: Dict of symbol -> (asset, position_size)
            
        Returns:
            Portfolio Greeks
        """
        portfolio_greeks = {greek: 0.0 for greek in self.supported_greeks}
        
        for symbol, (asset, position_size) in positions.items():
            if asset.asset_type in [AssetType.OPTIONS]:
                # Asset should have get_risk_metrics method with Greeks
                risk_metrics = asset.get_risk_metrics(position_size)
                
                for greek in self.supported_greeks:
                    if f'position_{greek}' in risk_metrics:
                        portfolio_greeks[greek] += risk_metrics[f'position_{greek}']
        
        return portfolio_greeks
    
    def calculate_hedge_ratio(self, option_asset: BaseAsset, hedge_asset: BaseAsset,
                            option_position: int) -> float:
        """
        Calculate hedge ratio for delta hedging
        
        Args:
            option_asset: Options position to hedge
            hedge_asset: Underlying asset for hedging
            option_position: Size of option position
            
        Returns:
            Hedge ratio (number of hedge assets needed)
        """
        if option_asset.asset_type != AssetType.OPTIONS:
            raise ValueError("Option asset must be of type OPTIONS")
        
        # Get option delta
        risk_metrics = option_asset.get_risk_metrics(option_position)
        delta = risk_metrics.get('position_delta', 0.0)
        
        # For simple delta hedging, hedge ratio = -delta
        return -delta
    
    def calculate_gamma_scalping_pnl(self, option_asset: BaseAsset, 
                                   underlying_price_moves: List[float],
                                   position_size: int = 1) -> float:
        """
        Calculate P&L from gamma scalping
        
        Args:
            option_asset: Options position
            underlying_price_moves: List of underlying price movements
            position_size: Size of option position
            
        Returns:
            Cumulative gamma scalping P&L
        """
        if option_asset.asset_type != AssetType.OPTIONS:
            raise ValueError("Asset must be of type OPTIONS")
        
        risk_metrics = option_asset.get_risk_metrics(position_size)
        gamma = risk_metrics.get('gamma', 0.0)
        
        # Simplified gamma P&L calculation
        total_pnl = 0.0
        for move in underlying_price_moves:
            # P&L = 0.5 * Gamma * (Delta S)^2
            gamma_pnl = 0.5 * gamma * (move ** 2) * position_size
            total_pnl += gamma_pnl
        
        return total_pnl
    
    def create_delta_neutral_portfolio(self, option_positions: Dict[str, Tuple[BaseAsset, int]],
                                     hedge_assets: List[BaseAsset]) -> Dict[str, float]:
        """
        Create delta-neutral portfolio
        
        Args:
            option_positions: Dict of option positions
            hedge_assets: Available assets for hedging
            
        Returns:
            Hedge quantities needed for each hedge asset
        """
        # Calculate total portfolio delta
        total_delta = 0.0
        for symbol, (asset, position_size) in option_positions.items():
            if asset.asset_type == AssetType.OPTIONS:
                risk_metrics = asset.get_risk_metrics(position_size)
                total_delta += risk_metrics.get('position_delta', 0.0)
        
        # Simple hedging: use first hedge asset to neutralize delta
        hedge_quantities = {}
        if hedge_assets and total_delta != 0:
            primary_hedge = hedge_assets[0]
            hedge_quantities[primary_hedge.symbol] = -total_delta
        
        return hedge_quantities
    
    def calculate_implied_correlation(self, multi_asset_option: BaseAsset,
                                    underlying_correlations: Dict[Tuple[str, str], float]) -> float:
        """
        Calculate implied correlation for multi-asset options
        
        This is a simplified implementation - real calculation would be more complex
        """
        # Placeholder for complex multi-asset correlation calculation
        return 0.0


class AdvancedRiskMetrics:
    """
    Advanced risk metrics calculation
    
    Includes sophisticated risk measures beyond basic VaR/ES
    """
    
    @staticmethod
    def calculate_maximum_drawdown(pnl_series: pd.Series) -> float:
        """Calculate maximum drawdown from P&L series"""
        cumulative = pnl_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if returns.std() == 0:
            return 0.0
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        annual_return = returns.mean() * 252
        max_dd = AdvancedRiskMetrics.calculate_maximum_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.95) -> float:
        """Calculate tail ratio (right tail / left tail)"""
        right_tail = returns.quantile(percentile)
        left_tail = returns.quantile(1 - percentile)
        
        if left_tail == 0:
            return float('inf')
        
        return abs(right_tail) / abs(left_tail)
