#!/usr/bin/env python3
"""
Debug script to identify why order book matching is not generating trades
"""

import sys
import os
sys.path.append('src')

# Core imports
from engine.order_book import OrderBook
from engine.order_types import Order, OrderSide, OrderType
from execution.matching_engine import MatchingEngine
import pandas as pd
import logging

# Set up logging to show debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

def test_basic_matching():
    """Test basic order matching behavior"""
    print("=== DEBUG: Basic Order Matching Test ===")
    
    # Create order book
    book = OrderBook('TESTSTOCK')
    
    print("\n1. Creating initial orders...")
    
    # Add ask order first
    ask_order = Order(
        order_id="ASK001",
        symbol="TESTSTOCK", 
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"Adding ASK order: {ask_order.volume} @ ${ask_order.price}")
    trades_ask = book.add_order(ask_order)
    print(f"Trades from ASK: {len(trades_ask)}")
    
    # Check book state
    print(f"\nBook state after ASK:")
    print(f"  Best ASK: {book.get_best_ask()}")
    print(f"  Best ASK Volume: {book.get_best_ask_volume()}")
    print(f"  ASK levels: {len(book.asks)}")
    
    # Add matching buy market order
    buy_order = Order(
        order_id="BUY001",
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=50,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"\nAdding BUY market order: {buy_order.volume}")
    trades_buy = book.add_order(buy_order)
    print(f"Trades from BUY: {len(trades_buy)}")
    
    if trades_buy:
        for i, trade in enumerate(trades_buy):
            print(f"  Trade {i+1}: {trade.volume} @ ${trade.price}")
    else:
        print("  NO TRADES GENERATED!")
        print("  This is the problem we need to investigate")
    
    print(f"\nFinal book state:")
    print(f"  Best ASK: {book.get_best_ask()}")
    print(f"  Best ASK Volume: {book.get_best_ask_volume()}")

def test_matching_engine():
    """Test matching engine behavior"""
    print("\n\n=== DEBUG: Matching Engine Test ===")
    
    engine = MatchingEngine()
    
    # Add ask order
    ask_order = Order(
        order_id="ASK002",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"Submitting ASK to engine: {ask_order.volume} @ ${ask_order.price}")
    result_ask = engine.submit_order(ask_order)
    print(f"Engine result for ASK: {result_ask}")
    
    # Check if trades were generated
    if result_ask and hasattr(result_ask, 'trades'):
        print(f"Trades in ASK result: {len(result_ask.trades)}")
    
    # Add matching buy market order
    buy_order = Order(
        order_id="BUY002", 
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=50,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"\nSubmitting BUY market to engine: {buy_order.volume}")
    result_buy = engine.submit_order(buy_order)
    print(f"Engine result for BUY: {result_buy}")
    
    if result_buy and hasattr(result_buy, 'trades'):
        print(f"Trades in BUY result: {len(result_buy.trades)}")
        for i, trade in enumerate(result_buy.trades):
            print(f"  Trade {i+1}: {trade.volume} @ ${trade.price}")
    else:
        print("  NO TRADES IN ENGINE RESULT!")

def test_order_properties():
    """Test order property methods"""
    print("\n\n=== DEBUG: Order Properties Test ===")
    
    # Create test orders
    market_order = Order(
        order_id="MKT001",
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=100,
        timestamp=pd.Timestamp.now()
    )
    
    limit_order = Order(
        order_id="LMT001", 
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        volume=100,
        price=150.00,
        timestamp=pd.Timestamp.now()
    )
    
    print(f"Market order properties:")
    print(f"  is_market_order(): {market_order.is_market_order()}")
    print(f"  is_limit_order(): {market_order.is_limit_order()}")
    print(f"  is_buy(): {market_order.is_buy()}")
    print(f"  price: {market_order.price}")
    
    print(f"\nLimit order properties:")
    print(f"  is_market_order(): {limit_order.is_market_order()}")
    print(f"  is_limit_order(): {limit_order.is_limit_order()}")
    print(f"  is_buy(): {limit_order.is_buy()}")
    print(f"  price: {limit_order.price}")

if __name__ == "__main__":
    print("Starting order book matching debug...")
    
    # Run tests
    try:
        test_order_properties()
        test_basic_matching()
        test_matching_engine()
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebug complete.")
