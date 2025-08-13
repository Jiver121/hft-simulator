#!/usr/bin/env python3
"""
Simplified debug script to test order book matching without validation issues
"""

import sys
import os
sys.path.append('src')

# Core imports
from engine.order_book import OrderBook
from engine.order_types import Order, OrderSide, OrderType
import pandas as pd
import logging

# Disable debug logging to reduce noise 
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(name)s - %(message)s')

def test_simple_matching():
    """Test simple order matching without circuit breakers"""
    print("=== SIMPLE ORDER MATCHING TEST ===")
    
    # Create order book
    book = OrderBook('TESTSTOCK')
    
    # Disable circuit breaker by setting high threshold
    book.circuit_breaker.threshold_pct = 100  # 100% threshold
    book.circuit_breaker.tripped = False
    
    print("\n1. Adding ASK order (limit sell)...")
    
    # Add ask order - this should work if circuit breaker is disabled
    ask_order = Order(
        order_id="ASK001",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    # First, let's add some initial liquidity to avoid circuit breaker
    # We'll patch the get_total_volume method temporarily
    original_get_total_volume = book.get_total_volume
    def mock_get_total_volume(side):
        return 1000  # Return fake liquidity
    book.get_total_volume = mock_get_total_volume
    
    print(f"Adding ASK: {ask_order.volume} @ ${ask_order.price}")
    trades_ask = book.add_order(ask_order)
    print(f"ASK trades: {len(trades_ask)}")
    
    # Restore original method
    book.get_total_volume = original_get_total_volume
    
    # Check book state
    print(f"Best ASK after adding: {book.get_best_ask()}")
    print(f"Best ASK Volume: {book.get_best_ask_volume()}")
    print(f"Total ASK levels: {len(book.asks)}")
    
    if len(book.asks) > 0:
        print("✓ ASK order was successfully added to book")
        
        print("\n2. Adding BUY limit order that should match...")
        
        # Add aggressive buy limit order that should match
        buy_order = Order(
            order_id="BUY001",
            symbol="TESTSTOCK",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=50,
            price=152.00,  # Higher than ask price - should match
            timestamp=pd.Timestamp.now()
        )
        
        print(f"Adding BUY: {buy_order.volume} @ ${buy_order.price}")
        trades_buy = book.add_order(buy_order)
        print(f"BUY trades: {len(trades_buy)}")
        
        if trades_buy:
            print("✓ SUCCESS: Trades were generated!")
            for i, trade in enumerate(trades_buy):
                print(f"  Trade {i+1}: {trade.volume} @ ${trade.price}")
        else:
            print("✗ PROBLEM: No trades generated despite aggressive buy order")
            
            # Debug why matching failed
            print(f"\nDEBUG INFO:")
            print(f"  Buy order side: {buy_order.side}")
            print(f"  Buy order is_buy(): {buy_order.is_buy()}")
            print(f"  Buy order price: {buy_order.price}")
            print(f"  Ask book state: {book.asks}")
            print(f"  Ask prices: {book.ask_prices}")
            
    else:
        print("✗ PROBLEM: ASK order was not added to book (still blocked)")

def test_manual_matching():
    """Test the matching logic manually"""
    print("\n\n=== MANUAL MATCHING TEST ===")
    
    # Create order book
    book = OrderBook('TESTSTOCK')
    
    # Add ask order directly to book bypassing validation
    ask_order = Order(
        order_id="ASK002",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    print("Manually adding ASK to book...")
    
    # Add directly to book structure
    from engine.order_types import PriceLevel
    price = 151.00
    book.asks[price] = PriceLevel(price)
    book.asks[price].add_order(ask_order)
    book.ask_prices.append(price)
    book.ask_prices.sort()  # Keep ascending order
    book.orders[ask_order.order_id] = ask_order
    
    print(f"Manual ASK add complete. Book state:")
    print(f"  ASK levels: {len(book.asks)}")
    print(f"  ASK prices: {book.ask_prices}")
    print(f"  Best ASK: {book.get_best_ask()}")
    
    # Now test matching
    buy_order = Order(
        order_id="BUY002",
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        volume=50,
        price=152.00,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"\nTesting matching with manual method...")
    print(f"BUY order: {buy_order.volume} @ ${buy_order.price}")
    
    # Test the matching logic directly
    trades = book._match_limit_order(buy_order)
    print(f"Direct matching result: {len(trades)} trades")
    
    if trades:
        for i, trade in enumerate(trades):
            print(f"  Trade {i+1}: {trade.volume} @ ${trade.price}")
    else:
        print("No trades from direct matching")
        
        # Debug matching conditions
        print(f"\nDEBUG matching conditions:")
        print(f"  Buy order is_buy(): {buy_order.is_buy()}")
        print(f"  Buy order price: {buy_order.price}")
        print(f"  Available ask prices: {book.ask_prices}")
        print(f"  Eligible prices (≤ {buy_order.price}): {[p for p in book.ask_prices if p <= buy_order.price]}")

if __name__ == "__main__":
    print("Starting simplified matching debug...")
    
    try:
        test_simple_matching()
        test_manual_matching()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebug complete.")
