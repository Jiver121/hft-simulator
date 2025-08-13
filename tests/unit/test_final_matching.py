#!/usr/bin/env python3
"""
Final test to confirm the matching engine fixes are working correctly
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

# Minimal logging to see results clearly
logging.basicConfig(level=logging.ERROR, format='%(message)s')

def test_guaranteed_matching():
    """Test with orders that are guaranteed to match"""
    print("=== FINAL MATCHING TEST ===")
    print("Testing market orders that should definitely match...\n")
    
    # Create matching engine
    engine = MatchingEngine("TESTSTOCK")
    
    print("1. Adding initial ASK order (sell limit)...")
    # Add ask order first to establish liquidity
    ask_order = Order(
        order_id="ASK001",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    trades_ask, order_update_ask = engine.process_order(ask_order)
    print(f"   ASK result: {len(trades_ask)} trades, order status: {order_update_ask.update_type}")
    
    # Check order book state
    book = engine.order_book
    print(f"   Order book: Best ASK = ${book.get_best_ask()}, Volume = {book.get_best_ask_volume()}")
    
    print("\n2. Adding BUY market order (should match immediately)...")
    # Add market buy order that should match against the ask
    buy_order = Order(
        order_id="BUY001",
        symbol="TESTSTOCK",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=50,
        timestamp=pd.Timestamp.now()
    )
    
    trades_buy, order_update_buy = engine.process_order(buy_order)
    print(f"   BUY result: {len(trades_buy)} trades, order status: {order_update_buy.update_type}")
    
    if trades_buy:
        print("\n✅ SUCCESS! Trades were generated:")
        for i, trade in enumerate(trades_buy):
            print(f"   Trade {i+1}: {trade.volume} shares @ ${trade.price}")
            print(f"              Buy: {trade.buy_order_id}, Sell: {trade.sell_order_id}")
    else:
        print("\n❌ PROBLEM: No trades generated")
        print("   This indicates the matching logic still has issues")
    
    # Check final book state
    print(f"\n   Final order book: Best ASK = ${book.get_best_ask()}, Volume = {book.get_best_ask_volume()}")
    print(f"   Total orders processed: {book.stats['orders_added']}")
    print(f"   Total trades executed: {book.stats['trades_executed']}")
    
def test_limit_order_crossing():
    """Test with limit orders that should cross and match"""
    print("\n\n=== LIMIT ORDER CROSSING TEST ===")
    print("Testing limit orders that cross the spread...\n")
    
    # Create new matching engine
    engine = MatchingEngine("TESTSTOCK2")
    
    print("1. Adding initial BID order (buy limit)...")
    # Add bid order first
    bid_order = Order(
        order_id="BID001",
        symbol="TESTSTOCK2",
        side=OrderSide.BID,
        order_type=OrderType.LIMIT,
        volume=100,
        price=150.00,
        timestamp=pd.Timestamp.now()
    )
    
    trades_bid, order_update_bid = engine.process_order(bid_order)
    print(f"   BID result: {len(trades_bid)} trades")
    
    print("\n2. Adding ASK order that crosses (sell limit below bid)...")
    # Add ask order at lower price - should match immediately
    ask_order = Order(
        order_id="ASK001", 
        symbol="TESTSTOCK2",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=75,
        price=149.50,  # Lower than bid - should match
        timestamp=pd.Timestamp.now()
    )
    
    trades_ask2, order_update_ask2 = engine.process_order(ask_order)
    print(f"   ASK result: {len(trades_ask2)} trades")
    
    if trades_ask2:
        print("\n✅ SUCCESS! Crossing trades generated:")
        for i, trade in enumerate(trades_ask2):
            print(f"   Trade {i+1}: {trade.volume} shares @ ${trade.price}")
    else:
        print("\n❌ PROBLEM: No crossing trades generated")
    
    # Check final state
    book = engine.order_book
    print(f"\n   Final state:")
    print(f"   Best BID = ${book.get_best_bid()}, Volume = {book.get_best_bid_volume()}")
    print(f"   Best ASK = ${book.get_best_ask()}, Volume = {book.get_best_ask_volume()}")

if __name__ == "__main__":
    print("Testing fixed matching engine behavior...\n")
    
    try:
        test_guaranteed_matching()
        test_limit_order_crossing()
        
        print("\n" + "="*50)
        print("CONCLUSION:")
        print("If trades were generated above, the core matching logic is FIXED!")
        print("If no trades, there may be additional issues to investigate.")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
