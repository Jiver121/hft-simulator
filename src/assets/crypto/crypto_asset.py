"""
Cryptocurrency Asset Implementation

This module provides cryptocurrency-specific trading functionality including
DEX integration, staking capabilities, and cross-chain bridge support.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from decimal import Decimal
import json
import requests
from datetime import datetime, timedelta

from src.engine.order_types import Order, Trade
from src.utils.constants import OrderSide, OrderType
from ..core.base_asset import BaseAsset, AssetInfo, AssetType


@dataclass
class CryptoAssetInfo(AssetInfo):
    """Cryptocurrency-specific asset information"""
    blockchain: str = "ethereum"
    contract_address: Optional[str] = None
    decimal_places: int = 8
    max_supply: Optional[float] = None
    circulating_supply: Optional[float] = None
    
    # DEX information
    dex_pairs: Dict[str, str] = field(default_factory=dict)  # dex_name -> pair_address
    liquidity_pools: Dict[str, float] = field(default_factory=dict)  # pool_id -> liquidity
    
    # Staking information
    staking_enabled: bool = False
    staking_apy: float = 0.0
    lock_period: int = 0  # days
    
    # Cross-chain support
    supported_chains: List[str] = field(default_factory=list)
    bridge_contracts: Dict[str, str] = field(default_factory=dict)


class CryptoAsset(BaseAsset):
    """
    Cryptocurrency asset with DEX and DeFi integration
    
    Features:
    - Real-time price feeds from multiple exchanges
    - DEX liquidity pool integration
    - Staking and yield farming
    - Cross-chain bridge support
    - Gas fee optimization
    """
    
    def __init__(self, crypto_info: CryptoAssetInfo):
        super().__init__(crypto_info)
        self.crypto_info = crypto_info
        self._gas_price: Optional[float] = None
        self._network_congestion: float = 0.0
        self._liquidity_data = {}
        self._staking_rewards = 0.0
        self._cross_chain_balances = {}
        
        # Initialize DEX connections
        self._dex_prices = {}
        self._arbitrage_opportunities = []
        
    @property
    def gas_price(self) -> Optional[float]:
        """Current gas price in Gwei"""
        return self._gas_price
    
    @property
    def network_congestion(self) -> float:
        """Network congestion level (0-1)"""
        return self._network_congestion
    
    def calculate_fair_value(self, **kwargs) -> float:
        """
        Calculate fair value using multiple price sources and liquidity
        
        Uses weighted average of:
        - CEX prices (centralized exchanges)
        - DEX prices (decentralized exchanges)
        - Liquidity pool ratios
        """
        if self._current_price is None:
            raise ValueError(f"No current price available for {self.symbol}")
        
        # Base price from current market
        base_price = self._current_price
        
        # Adjust for liquidity and DEX prices
        if self._dex_prices:
            dex_weights = self._calculate_dex_weights()
            dex_fair_value = sum(
                price * weight for price, weight in 
                zip(self._dex_prices.values(), dex_weights.values())
            )
            
            # Weighted average (70% CEX, 30% DEX)
            fair_value = 0.7 * base_price + 0.3 * dex_fair_value
        else:
            fair_value = base_price
        
        # Adjust for network congestion and gas costs
        if self.crypto_info.blockchain in ["ethereum", "polygon"]:
            gas_adjustment = self._calculate_gas_impact()
            fair_value *= (1 - gas_adjustment)
        
        return fair_value
    
    def get_risk_metrics(self, position_size: int = 0) -> Dict[str, float]:
        """Calculate crypto-specific risk metrics"""
        if self._current_price is None:
            return {}
        
        position_value = self.calculate_position_value(position_size)
        
        # Base risk metrics
        risk_metrics = {
            'position_value': position_value,
            'margin_requirement': self.calculate_margin_requirement(position_size),
            'gas_cost_risk': self._calculate_gas_cost_risk(position_size),
            'liquidity_risk': self._calculate_liquidity_risk(),
            'volatility': self._calculate_crypto_volatility(),
            'impermanent_loss_risk': self._calculate_impermanent_loss_risk()
        }
        
        # Add staking rewards if applicable
        if self.crypto_info.staking_enabled and position_size > 0:
            risk_metrics['staking_yield'] = self._calculate_staking_yield(position_size)
        
        return risk_metrics
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate cryptocurrency order with gas and liquidity checks"""
        # Basic validation
        is_valid, error_msg = super().validate_order(order)
        if not is_valid:
            return False, error_msg
        
        # Check network capacity
        if self._network_congestion > 0.9:
            return False, "Network congestion too high for order execution"
        
        # Check gas requirements
        if not self._validate_gas_requirements(order):
            return False, "Insufficient gas for order execution"
        
        # Check liquidity for large orders
        if order.volume > 1000:  # Large order threshold
            if not self._validate_liquidity(order):
                return False, "Insufficient liquidity for large order"
        
        return True, "Crypto order validated successfully"
    
    def update_dex_prices(self, dex_prices: Dict[str, float]) -> None:
        """Update prices from DEX sources"""
        self._dex_prices.update(dex_prices)
        self._detect_arbitrage_opportunities()
    
    def update_gas_price(self, gas_price_gwei: float) -> None:
        """Update current gas price"""
        self._gas_price = gas_price_gwei
        
    def update_network_congestion(self, congestion_level: float) -> None:
        """Update network congestion level"""
        self._network_congestion = max(0.0, min(1.0, congestion_level))
    
    def get_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Get current arbitrage opportunities"""
        return self._arbitrage_opportunities.copy()
    
    def calculate_optimal_execution_route(self, order_volume: int, 
                                        max_slippage: float = 0.01) -> Dict[str, Any]:
        """
        Calculate optimal execution route across CEX and DEX
        
        Args:
            order_volume: Size of order to execute
            max_slippage: Maximum acceptable slippage
            
        Returns:
            Optimal execution plan with routes and expected costs
        """
        routes = []
        
        # CEX route
        cex_route = {
            'exchange_type': 'CEX',
            'volume': order_volume,
            'expected_price': self._current_price,
            'slippage': self._estimate_cex_slippage(order_volume),
            'gas_cost': 0,  # No gas for CEX
            'execution_time': 1  # seconds
        }
        routes.append(cex_route)
        
        # DEX routes
        for dex_name, price in self._dex_prices.items():
            if dex_name in self.crypto_info.liquidity_pools:
                liquidity = self.crypto_info.liquidity_pools[dex_name]
                slippage = self._estimate_dex_slippage(order_volume, liquidity)
                
                if slippage <= max_slippage:
                    dex_route = {
                        'exchange_type': 'DEX',
                        'dex_name': dex_name,
                        'volume': order_volume,
                        'expected_price': price,
                        'slippage': slippage,
                        'gas_cost': self._estimate_gas_cost(),
                        'execution_time': 15  # seconds for blockchain confirmation
                    }
                    routes.append(dex_route)
        
        # Find optimal route
        best_route = min(routes, key=lambda r: r['expected_price'] * (1 + r['slippage']) + r['gas_cost'])
        
        return {
            'optimal_route': best_route,
            'all_routes': routes,
            'savings_vs_worst': max(routes, key=lambda r: r['expected_price'])['expected_price'] - best_route['expected_price']
        }
    
    def stake_tokens(self, amount: float, duration_days: int = None) -> Dict[str, Any]:
        """
        Stake tokens for yield
        
        Args:
            amount: Amount to stake
            duration_days: Staking duration (None for flexible)
            
        Returns:
            Staking information and expected rewards
        """
        if not self.crypto_info.staking_enabled:
            raise ValueError(f"Staking not supported for {self.symbol}")
        
        duration = duration_days or self.crypto_info.lock_period
        expected_rewards = amount * (self.crypto_info.staking_apy / 365) * duration
        
        staking_info = {
            'amount': amount,
            'duration_days': duration,
            'apy': self.crypto_info.staking_apy,
            'expected_rewards': expected_rewards,
            'lock_period': self.crypto_info.lock_period,
            'start_date': pd.Timestamp.now(),
            'end_date': pd.Timestamp.now() + pd.Timedelta(days=duration)
        }
        
        self._staking_rewards += expected_rewards
        return staking_info
    
    def estimate_bridge_cost(self, target_chain: str, amount: float) -> Dict[str, Any]:
        """Estimate cost of bridging tokens to another chain"""
        if target_chain not in self.crypto_info.supported_chains:
            raise ValueError(f"Bridge to {target_chain} not supported")
        
        # Simplified bridge cost calculation
        base_fee = 0.001  # 0.1% base fee
        gas_cost = self._estimate_gas_cost() * 2  # Bridge requires 2 transactions
        
        return {
            'target_chain': target_chain,
            'amount': amount,
            'bridge_fee': amount * base_fee,
            'gas_cost': gas_cost,
            'total_cost': amount * base_fee + gas_cost,
            'estimated_time': 600  # 10 minutes
        }
    
    # Private helper methods
    
    def _calculate_dex_weights(self) -> Dict[str, float]:
        """Calculate weights for DEX prices based on liquidity"""
        total_liquidity = sum(self.crypto_info.liquidity_pools.values())
        if total_liquidity == 0:
            return {dex: 1/len(self._dex_prices) for dex in self._dex_prices}
        
        return {
            dex: self.crypto_info.liquidity_pools.get(dex, 0) / total_liquidity 
            for dex in self._dex_prices
        }
    
    def _calculate_gas_impact(self) -> float:
        """Calculate impact of gas costs on fair value"""
        if self._gas_price is None:
            return 0.0
        
        # Simple model: higher gas = lower effective value for small trades
        gas_impact = min(0.05, self._gas_price / 1000)  # Max 5% impact
        return gas_impact * self._network_congestion
    
    def _calculate_gas_cost_risk(self, position_size: int) -> float:
        """Calculate gas cost risk for position"""
        if self._gas_price is None:
            return 0.0
        
        estimated_gas = 50000  # Base gas for transfer
        gas_cost_eth = estimated_gas * self._gas_price * 1e-9  # Convert Gwei to ETH
        
        # Convert to USD (simplified - would use real ETH/USD rate)
        gas_cost_usd = gas_cost_eth * 2000  # Assume ETH = $2000
        
        return gas_cost_usd
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk score"""
        if not self.crypto_info.liquidity_pools:
            return 1.0  # High risk if no liquidity data
        
        total_liquidity = sum(self.crypto_info.liquidity_pools.values())
        
        # Risk decreases with liquidity (logarithmic scale)
        if total_liquidity > 0:
            risk_score = 1.0 / (1.0 + np.log10(total_liquidity))
        else:
            risk_score = 1.0
        
        return min(1.0, risk_score)
    
    def _calculate_crypto_volatility(self) -> float:
        """Estimate volatility based on asset characteristics"""
        # Simplified volatility model
        base_volatility = 0.6  # 60% annual volatility for crypto
        
        # Adjust based on market cap (larger = less volatile)
        if self.crypto_info.circulating_supply and self._current_price:
            market_cap = self.crypto_info.circulating_supply * self._current_price
            if market_cap > 1e10:  # >$10B
                volatility_multiplier = 0.7
            elif market_cap > 1e9:  # >$1B
                volatility_multiplier = 0.85
            else:
                volatility_multiplier = 1.2
        else:
            volatility_multiplier = 1.0
        
        return base_volatility * volatility_multiplier
    
    def _calculate_impermanent_loss_risk(self) -> float:
        """Calculate impermanent loss risk for LP positions"""
        if not self.crypto_info.liquidity_pools:
            return 0.0
        
        # Simplified impermanent loss calculation
        volatility = self._calculate_crypto_volatility()
        
        # IL risk increases with volatility
        il_risk = min(0.3, volatility * 0.5)  # Max 30% IL risk
        
        return il_risk
    
    def _calculate_staking_yield(self, position_size: int) -> float:
        """Calculate expected staking yield"""
        position_value = self.calculate_position_value(position_size)
        annual_yield = position_value * self.crypto_info.staking_apy
        
        return annual_yield
    
    def _validate_gas_requirements(self, order: Order) -> bool:
        """Validate gas requirements for order"""
        if self._gas_price is None:
            return True  # Can't validate without gas price
        
        # Simple validation: reject if gas price too high
        max_acceptable_gas = 100  # 100 Gwei
        return self._gas_price <= max_acceptable_gas
    
    def _validate_liquidity(self, order: Order) -> bool:
        """Validate liquidity for large orders"""
        if not self.crypto_info.liquidity_pools:
            return True  # Can't validate without liquidity data
        
        total_liquidity = sum(self.crypto_info.liquidity_pools.values())
        order_value = order.volume * (order.price or self._current_price or 0)
        
        # Order should be less than 10% of total liquidity
        return order_value < total_liquidity * 0.1
    
    def _estimate_cex_slippage(self, volume: int) -> float:
        """Estimate slippage on centralized exchange"""
        # Simple model: slippage increases with volume
        base_slippage = 0.001  # 0.1%
        volume_impact = min(0.01, volume / 100000)  # Max 1% additional slippage
        
        return base_slippage + volume_impact
    
    def _estimate_dex_slippage(self, volume: int, liquidity: float) -> float:
        """Estimate slippage on DEX"""
        if liquidity <= 0:
            return 1.0  # 100% slippage if no liquidity
        
        # AMM slippage model: slippage = volume / (2 * liquidity)
        order_value = volume * (self._current_price or 100)
        slippage = order_value / (2 * liquidity)
        
        return min(0.5, slippage)  # Cap at 50%
    
    def _estimate_gas_cost(self) -> float:
        """Estimate gas cost in USD"""
        if self._gas_price is None:
            return 5.0  # Default $5 gas cost
        
        gas_limit = 100000  # Standard gas limit for DEX trade
        gas_cost_eth = gas_limit * self._gas_price * 1e-9
        gas_cost_usd = gas_cost_eth * 2000  # Assume ETH = $2000
        
        return gas_cost_usd
    
    def _detect_arbitrage_opportunities(self) -> None:
        """Detect arbitrage opportunities between exchanges"""
        self._arbitrage_opportunities.clear()
        
        if not self._dex_prices or self._current_price is None:
            return
        
        cex_price = self._current_price
        
        for dex_name, dex_price in self._dex_prices.items():
            price_diff = abs(dex_price - cex_price)
            spread_pct = price_diff / cex_price
            
            # Arbitrage opportunity if spread > 0.5%
            if spread_pct > 0.005:
                opportunity = {
                    'buy_exchange': 'CEX' if dex_price > cex_price else dex_name,
                    'sell_exchange': dex_name if dex_price > cex_price else 'CEX',
                    'buy_price': min(cex_price, dex_price),
                    'sell_price': max(cex_price, dex_price),
                    'spread_pct': spread_pct,
                    'profit_potential': spread_pct * 10000,  # Profit per $10k trade
                    'timestamp': pd.Timestamp.now()
                }
                self._arbitrage_opportunities.append(opportunity)
    
    def __str__(self) -> str:
        return f"CryptoAsset({self.symbol}, {self.crypto_info.blockchain})"
