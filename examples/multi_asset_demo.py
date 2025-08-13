#!/usr/bin/env python3
"""
Multi-Asset Class Trading Demo

This demo showcases the comprehensive multi-asset trading system with:
- Cryptocurrency trading with DEX integration
- Options pricing and Greeks calculations
- Cross-asset correlation analysis
- Portfolio optimization
- Risk management across asset classes

Run with: python multi_asset_demo.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from src.assets.core.base_asset import AssetInfo, EquityAsset, AssetType
from src.assets.core.order_book import UnifiedOrderBook
from src.assets.core.risk_models import ParametricRiskModel, GreeksCalculator, RiskScenario
from src.assets.crypto.crypto_asset import CryptoAsset, CryptoAssetInfo
from src.assets.options.options_asset import OptionsAsset, OptionsAssetInfo, OptionType, OptionStyle
from src.engine.order_types import Order
from src.utils.constants import OrderSide, OrderType as OrderTypeEnum

def create_sample_assets():
    """Create sample assets for demonstration"""
    assets = {}
    
    # 1. Create Equity Asset (Apple)
    equity_info = AssetInfo(
        symbol="AAPL",
        name="Apple Inc.",
        asset_type=AssetType.EQUITY,
        currency="USD",
        tick_size=0.01,
        lot_size=1,
        min_trade_size=1,
        exchanges=["NASDAQ"],
        metadata={"sector": "Technology", "beta": 1.2}
    )
    aapl = EquityAsset(equity_info)
    aapl.update_price(150.00)
    assets["AAPL"] = aapl
    
    # 2. Create Cryptocurrency Asset (Bitcoin)
    crypto_info = CryptoAssetInfo(
        symbol="BTC-USD",
        name="Bitcoin",
        asset_type=AssetType.CRYPTO,
        currency="USD",
        tick_size=0.01,
        blockchain="bitcoin",
        decimal_places=8,
        max_supply=21000000,
        circulating_supply=19500000,
        dex_pairs={"uniswap": "0x...", "sushiswap": "0x..."},
        liquidity_pools={"uniswap": 100000000, "sushiswap": 50000000},
        staking_enabled=False,
        supported_chains=["ethereum", "polygon"]
    )
    btc = CryptoAsset(crypto_info)
    btc.update_price(45000.00)
    btc.update_gas_price(25.0)  # 25 Gwei
    btc.update_network_congestion(0.3)  # 30% congestion
    btc.update_dex_prices({
        "uniswap": 45100.0,
        "sushiswap": 44950.0
    })
    assets["BTC-USD"] = btc
    
    # 3. Create Cryptocurrency Asset (Ethereum)
    eth_info = CryptoAssetInfo(
        symbol="ETH-USD",
        name="Ethereum",
        asset_type=AssetType.CRYPTO,
        currency="USD",
        tick_size=0.01,
        blockchain="ethereum",
        decimal_places=18,
        circulating_supply=120000000,
        dex_pairs={"uniswap": "0x...", "curve": "0x..."},
        liquidity_pools={"uniswap": 200000000, "curve": 80000000},
        staking_enabled=True,
        staking_apy=0.045,  # 4.5% APY
        lock_period=0,  # Flexible staking
        supported_chains=["polygon", "arbitrum", "optimism"]
    )
    eth = CryptoAsset(eth_info)
    eth.update_price(3200.00)
    eth.update_gas_price(30.0)
    eth.update_network_congestion(0.4)
    eth.update_dex_prices({
        "uniswap": 3205.0,
        "curve": 3198.0
    })
    assets["ETH-USD"] = eth
    
    # 4. Create Options Asset (AAPL Call Option)
    expiry_date = pd.Timestamp.now() + pd.Timedelta(days=30)
    options_info = OptionsAssetInfo(
        symbol="AAPL240115C00160000",  # Standard options symbol
        name="AAPL Jan 15 2024 160.00 Call",
        asset_type=AssetType.OPTIONS,
        underlying_symbol="AAPL",
        strike_price=160.0,
        expiry_date=expiry_date,
        option_type=OptionType.CALL,
        option_style=OptionStyle.AMERICAN,
        contract_multiplier=100,
        implied_volatility=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.01
    )
    aapl_call = OptionsAsset(options_info)
    aapl_call.update_underlying_price(150.00)
    assets["AAPL240115C00160000"] = aapl_call
    
    return assets

def demonstrate_crypto_features(crypto_assets):
    """Demonstrate cryptocurrency-specific features"""
    print("\n" + "="*60)
    print("CRYPTOCURRENCY FEATURES DEMONSTRATION")
    print("="*60)
    
    btc = crypto_assets["BTC-USD"]
    eth = crypto_assets["ETH-USD"]
    
    # 1. Show DEX integration and arbitrage opportunities
    print(f"\n1. Bitcoin DEX Integration:")
    print(f"   Current Price: ${btc.current_price:,.2f}")
    print(f"   Fair Value: ${btc.calculate_fair_value():,.2f}")
    print(f"   Gas Price: {btc.gas_price} Gwei")
    print(f"   Network Congestion: {btc.network_congestion:.1%}")
    
    arb_opportunities = btc.get_arbitrage_opportunities()
    if arb_opportunities:
        print(f"\n   Arbitrage Opportunities:")
        for opp in arb_opportunities:
            print(f"   - {opp['buy_exchange']} -> {opp['sell_exchange']}: {opp['spread_pct']:.3%} spread")
            print(f"     Profit potential: ${opp['profit_potential']:.2f} per $10k trade")
    
    # 2. Show optimal execution routing
    print(f"\n2. Optimal Execution Route (1 BTC order):")
    execution_route = btc.calculate_optimal_execution_route(
        order_volume=1, max_slippage=0.02
    )
    optimal = execution_route['optimal_route']
    print(f"   Best Route: {optimal['exchange_type']}")
    if 'dex_name' in optimal:
        print(f"   DEX: {optimal['dex_name']}")
    print(f"   Expected Price: ${optimal['expected_price']:,.2f}")
    print(f"   Slippage: {optimal['slippage']:.3%}")
    print(f"   Gas Cost: ${optimal['gas_cost']:.2f}")
    print(f"   Execution Time: {optimal['execution_time']}s")
    
    # 3. Ethereum staking demonstration
    print(f"\n3. Ethereum Staking:")
    if eth.crypto_info.staking_enabled:
        stake_amount = 10.0  # 10 ETH
        staking_info = eth.stake_tokens(stake_amount, duration_days=90)
        print(f"   Staked Amount: {stake_amount} ETH")
        print(f"   APY: {staking_info['apy']:.1%}")
        print(f"   Expected Rewards (90 days): {staking_info['expected_rewards']:.4f} ETH")
        print(f"   Reward Value: ${staking_info['expected_rewards'] * eth.current_price:.2f}")
    
    # 4. Cross-chain bridge cost estimation
    print(f"\n4. Cross-Chain Bridge Analysis:")
    bridge_cost = eth.estimate_bridge_cost("polygon", 5.0)  # Bridge 5 ETH to Polygon
    print(f"   Bridge 5 ETH to {bridge_cost['target_chain']}")
    print(f"   Bridge Fee: ${bridge_cost['bridge_fee'] * eth.current_price:.2f}")
    print(f"   Gas Cost: ${bridge_cost['gas_cost']:.2f}")
    print(f"   Total Cost: ${bridge_cost['total_cost']:.2f}")
    print(f"   Estimated Time: {bridge_cost['estimated_time']} seconds")

def demonstrate_options_features(options_assets):
    """Demonstrate options-specific features"""
    print("\n" + "="*60)
    print("OPTIONS FEATURES DEMONSTRATION")
    print("="*60)
    
    aapl_call = options_assets["AAPL240115C00160000"]
    
    print(f"\n1. Option Details:")
    print(f"   Symbol: {aapl_call.symbol}")
    print(f"   Underlying: {aapl_call.options_info.underlying_symbol} @ ${aapl_call.underlying_price}")
    print(f"   Strike: ${aapl_call.strike_price}")
    print(f"   Type: {aapl_call.options_info.option_type.value.upper()}")
    print(f"   Style: {aapl_call.options_info.option_style.value.upper()}")
    print(f"   Time to Expiry: {aapl_call.time_to_expiry:.4f} years")
    print(f"   Moneyness: {aapl_call.moneyness:.4f}")
    
    # 2. Pricing with different models
    print(f"\n2. Option Pricing:")
    bs_price = aapl_call.calculate_fair_value("black_scholes")
    binomial_price = aapl_call.calculate_fair_value("binomial")
    print(f"   Black-Scholes Price: ${bs_price:.4f}")
    print(f"   Binomial Price: ${binomial_price:.4f}")
    print(f"   Intrinsic Value: ${aapl_call.calculate_intrinsic_value():.4f}")
    print(f"   Time Value: ${bs_price - aapl_call.calculate_intrinsic_value():.4f}")
    
    # 3. Greeks calculations
    print(f"\n3. Option Greeks:")
    greeks = aapl_call.get_greeks()
    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.6f}")
    print(f"   Theta: ${greeks['theta']:.4f} per day")
    print(f"   Vega: ${greeks['vega']:.4f} per 1% vol change")
    print(f"   Rho: ${greeks['rho']:.4f} per 1% rate change")
    
    # 4. Risk metrics for position
    position_size = 10  # 10 contracts
    print(f"\n4. Risk Metrics for {position_size} Contracts:")
    risk_metrics = aapl_call.get_risk_metrics(position_size)
    print(f"   Position Value: ${risk_metrics['position_value']:,.2f}")
    print(f"   Position Delta: {risk_metrics['position_delta']:.2f}")
    print(f"   Position Gamma: {risk_metrics['position_gamma']:.2f}")
    print(f"   Position Theta: ${risk_metrics['position_theta']:.2f} per day")
    print(f"   Volatility Risk (1% move): ${risk_metrics['volatility_risk']:.2f}")
    print(f"   Pin Risk: {risk_metrics['pin_risk']:.2%}")
    
    # 5. Implied volatility calculation
    market_price = bs_price * 1.05  # Assume 5% premium to fair value
    impl_vol = aapl_call.calculate_implied_volatility(market_price)
    if impl_vol:
        print(f"\n5. Implied Volatility Analysis:")
        print(f"   Market Price: ${market_price:.4f}")
        print(f"   Implied Volatility: {impl_vol:.2%}")
        print(f"   Current IV: {aapl_call._implied_volatility:.2%}")
        print(f"   IV Premium: {(impl_vol - aapl_call._implied_volatility):.2%}")

def demonstrate_risk_management(assets):
    """Demonstrate cross-asset risk management"""
    print("\n" + "="*60)
    print("RISK MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create portfolio positions
    positions = {
        "AAPL": 1000.0,         # $1000 in Apple
        "BTC-USD": 500.0,       # $500 in Bitcoin
        "ETH-USD": 300.0,       # $300 in Ethereum
    }
    
    print(f"\n1. Portfolio Positions:")
    total_value = sum(positions.values())
    for symbol, value in positions.items():
        weight = value / total_value * 100
        print(f"   {symbol}: ${value:,.0f} ({weight:.1f}%)")
    print(f"   Total Portfolio Value: ${total_value:,.0f}")
    
    # Create risk scenarios
    scenarios = [
        RiskScenario(
            name="Market Crash",
            description="Broad market decline",
            market_shocks={"AAPL": -0.20, "BTC-USD": -0.35, "ETH-USD": -0.30},
            probability=0.05
        ),
        RiskScenario(
            name="Crypto Winter",
            description="Cryptocurrency bear market",
            market_shocks={"AAPL": -0.05, "BTC-USD": -0.60, "ETH-USD": -0.65},
            probability=0.10
        ),
        RiskScenario(
            name="Tech Selloff",
            description="Technology sector rotation",
            market_shocks={"AAPL": -0.30, "BTC-USD": 0.10, "ETH-USD": 0.05},
            probability=0.15
        )
    ]
    
    print(f"\n2. Stress Test Results:")
    for scenario in scenarios:
        scenario_pnl = 0.0
        for asset_symbol, position in positions.items():
            if asset_symbol in scenario.market_shocks:
                shock = scenario.market_shocks[asset_symbol]
                asset_pnl = position * shock
                scenario_pnl += asset_pnl
        
        print(f"   {scenario.name}: ${scenario_pnl:,.0f} ({scenario_pnl/total_value:.1%})")
        print(f"     Probability: {scenario.probability:.1%}")
    
    # Individual asset risk metrics
    print(f"\n3. Individual Asset Risk Metrics:")
    for symbol, asset in assets.items():
        if symbol in positions:
            risk_metrics = asset.get_risk_metrics()
            print(f"\n   {symbol}:")
            if 'volatility' in risk_metrics:
                print(f"     Volatility: {risk_metrics['volatility']:.2%}")
            if 'daily_var' in risk_metrics:
                print(f"     Daily VaR: ${risk_metrics['daily_var']:,.2f}")
            if 'beta' in risk_metrics:
                print(f"     Beta: {risk_metrics['beta']:.2f}")
            
            # Crypto-specific risks
            if asset.asset_type == AssetType.CRYPTO:
                print(f"     Gas Cost Risk: ${risk_metrics.get('gas_cost_risk', 0):.2f}")
                print(f"     Liquidity Risk: {risk_metrics.get('liquidity_risk', 0):.2%}")

def demonstrate_order_execution():
    """Demonstrate unified order book execution"""
    print("\n" + "="*60)
    print("ORDER EXECUTION DEMONSTRATION")
    print("="*60)
    
    # Create a simple equity for order book demo
    equity_info = AssetInfo(
        symbol="DEMO",
        name="Demo Asset",
        asset_type=AssetType.EQUITY,
        tick_size=0.01
    )
    demo_asset = EquityAsset(equity_info)
    demo_asset.update_price(100.0)
    
    # Create unified order book
    order_book = UnifiedOrderBook(demo_asset)
    
    print(f"1. Creating Order Book for {demo_asset.symbol}")
    print(f"   Initial Price: ${demo_asset.current_price}")
    
    # Add some limit orders
    orders_to_add = [
        Order.create_limit_order("DEMO", OrderSide.BID, 100, 99.95),
        Order.create_limit_order("DEMO", OrderSide.BID, 200, 99.90),
        Order.create_limit_order("DEMO", OrderSide.ASK, 150, 100.05),
        Order.create_limit_order("DEMO", OrderSide.ASK, 100, 100.10),
    ]
    
    print(f"\n2. Adding Limit Orders:")
    for order in orders_to_add:
        trades = order_book.add_order(order)
        side = "BUY" if order.is_buy() else "SELL"
        print(f"   {side} {order.volume} @ ${order.price} - Added")
    
    # Show market depth
    depth = order_book.get_market_depth(levels=5)
    print(f"\n3. Market Depth:")
    print(f"   Best Bid: ${order_book.best_bid} (Size: {order_book.bid_size})")
    print(f"   Best Ask: ${order_book.best_ask} (Size: {order_book.ask_size})")
    print(f"   Spread: ${order_book.spread:.4f}")
    
    # Execute a market order
    market_order = Order.create_market_order("DEMO", OrderSide.BUY, 75)
    print(f"\n4. Executing Market Order: BUY {market_order.volume}")
    trades = order_book.add_order(market_order)
    
    if trades:
        print(f"   Generated {len(trades)} trades:")
        for trade in trades:
            print(f"   - {trade.volume} @ ${trade.price}")
        
        total_volume = sum(t.volume for t in trades)
        avg_price = sum(t.price * t.volume for t in trades) / total_volume
        print(f"   Average Fill Price: ${avg_price:.4f}")
    
    # Show updated book state
    print(f"\n5. Updated Market State:")
    print(f"   Best Bid: ${order_book.best_bid}")
    print(f"   Best Ask: ${order_book.best_ask}")
    print(f"   Last Trade: ${order_book._last_trade_price}")

def main():
    """Main demo function"""
    print("MULTI-ASSET CLASS TRADING SYSTEM DEMO")
    print("="*60)
    print("Initializing multi-asset trading environment...")
    
    # Create sample assets
    assets = create_sample_assets()
    print(f"Created {len(assets)} assets:")
    for symbol, asset in assets.items():
        print(f"  - {symbol}: {asset.asset_type.value} @ ${asset.current_price}")
    
    # Separate assets by type for focused demonstrations
    crypto_assets = {k: v for k, v in assets.items() if v.asset_type == AssetType.CRYPTO}
    options_assets = {k: v for k, v in assets.items() if v.asset_type == AssetType.OPTIONS}
    
    # Run demonstrations
    demonstrate_crypto_features(crypto_assets)
    demonstrate_options_features(options_assets)
    demonstrate_risk_management(assets)
    demonstrate_order_execution()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✓ Cryptocurrency DEX integration and arbitrage detection")
    print("✓ Options pricing with multiple models (Black-Scholes, Binomial)")
    print("✓ Greeks calculations and risk metrics")
    print("✓ Cross-asset portfolio risk management")
    print("✓ Unified order book for all asset types")
    print("✓ Staking and yield farming capabilities")
    print("✓ Cross-chain bridge cost analysis")
    print("✓ Advanced execution routing optimization")
    
    print(f"\nSystem supports {len(AssetType)} asset types:")
    for asset_type in AssetType:
        print(f"  - {asset_type.value.title()}")
    
    print("\nReady for production trading across multiple asset classes!")

if __name__ == "__main__":
    main()
