#!/usr/bin/env python3
"""
Test script to debug the _match_order method implementation
"""

import sys
import os
sys.path.append('src')

# Core imports
from assets.core.order_book import UnifiedOrderBook
from assets.core.base_asset import BaseAsset, AssetType, AssetInfo
from engine.order_types import Order, OrderSide, OrderType, OrderStatus
import pandas as pd

def create_test_asset():
    """Create a test asset for the order book"""
    asset_info = AssetInfo(
        symbol="TESTSTOCK",
        name="Test Stock",
        asset_type=AssetType.EQUITY,
        tick_size=0.01,
        lot_size=1,
        min_trade_size=1,
        max_trade_size=1000000,
        currency="USD",
        exchanges=["NYSE"]
    )
    
    # Create a concrete implementation since BaseAsset is abstract
    class TestAsset(BaseAsset):
        def __init__(self, asset_info):
            super().__init__(asset_info)
        
        def calculate_fair_value(self, **kwargs):
            return 150.0  # Simple implementation
        
        def get_risk_metrics(self, position_size=0):
            return {"delta": 1.0, "gamma": 0.0, "theta": 0.0}
        
        def validate_order(self, order):
            if order.volume <= 0:
                return False, "Volume must be positive"
            if order.price is not None and order.price <= 0:
                return False, "Price must be positive"
            return True, "OK"
    
    return TestAsset(asset_info)

def test_order_book_matching():
    """Test the order book matching with debug logging"""
    print("=== DEBUG ORDER BOOK MATCHING TEST ===")
    
    # Create test asset and order book
    asset = create_test_asset()
    order_book = UnifiedOrderBook(asset)
    
    print(f"\nOrder book created for {asset.symbol}")
    print(f"Tick size: {order_book.tick_size}")
    print(f"Price-time priority: {getattr(order_book, '_use_price_time_priority', False)}")
    
    # Create and add a sell order first
    print("\n1. Adding SELL limit order...")
    
    sell_order = Order(
        order_id="SELL001",
        symbol="TESTSTOCK",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"Sell order: {sell_order.volume} @ ${sell_order.price}")
    trades_sell = order_book.add_order(sell_order)
    print(f"Trades from sell order: {len(trades_sell)}")
    
    # Check book state
    print(f"\nBook state after sell order:")
    print(f"  Best bid: {order_book.best_bid}")
    print(f"  Best ask: {order_book.best_ask}")
    print(f"  Bid levels: {len(order_book._bids)}")
    print(f"  Ask levels: {len(order_book._asks)}")
    
    # Now add a matching buy order
    print("\n2. Adding BUY limit order that should match...")
    
    buy_order = Order(
        order_id="BUY001", 
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        volume=50,
        price=152.00,  # Higher than ask price - should match
        timestamp=pd.Timestamp.now()
    )
    
    print(f"Buy order: {buy_order.volume} @ ${buy_order.price}")
    trades_buy = order_book.add_order(buy_order)
    print(f"Trades from buy order: {len(trades_buy)}")
    
    # Check final results
    if trades_buy:
        print("\n✅ SUCCESS: Trades were generated!")
        for i, trade in enumerate(trades_buy):
            print(f"  Trade {i+1}: {trade.volume} @ ${trade.price}")
            print(f"    Buy Order ID: {trade.buy_order_id}")
            print(f"    Sell Order ID: {trade.sell_order_id}")
            print(f"    Aggressor: {trade.aggressor_side}")
    else:
        print("\n❌ PROBLEM: No trades generated despite aggressive buy order")
    
    print(f"\nFinal book state:")
    print(f"  Best bid: {order_book.best_bid}")
    print(f"  Best ask: {order_book.best_ask}")
    print(f"  Total trades: {len(order_book._trades)}")

if __name__ == "__main__":
    test_order_book_matching()
