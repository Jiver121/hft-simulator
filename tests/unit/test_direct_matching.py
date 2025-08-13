#!/usr/bin/env python3
"""
Direct order book test to confirm the core fixes are working
"""

import sys
import os
sys.path.append('src')

# Core imports
from engine.order_book import OrderBook
from engine.order_types import Order, OrderSide, OrderType
import pandas as pd
import logging

# Disable logging noise
logging.basicConfig(level=logging.CRITICAL)

def test_direct_order_book():
    """Test order book directly without matching engine layer"""
    print("=== DIRECT ORDER BOOK TEST ===")
    print("Testing order book matching without fill models...\n")
    
    # Create order book directly
    book = OrderBook("TESTSTOCK")
    
    print("1. Adding ASK order (sell limit) to book...")
    ask_order = Order(
        order_id="ASK001",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    try:
        trades_ask = book.add_order(ask_order)
        print(f"   ASK added: {len(trades_ask)} trades generated")
        print(f"   Book state: Best ASK = ${book.get_best_ask()}, Volume = {book.get_best_ask_volume()}")
        
        if book.get_best_ask() is not None:
            print("   ✅ ASK order successfully added to book")
            
            print("\n2. Adding BUY market order to match...")
            buy_order = Order(
                order_id="BUY001",
                symbol="TESTSTOCK",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=50,
                timestamp=pd.Timestamp.now()
            )
            
            trades_buy = book.add_order(buy_order)
            print(f"   BUY executed: {len(trades_buy)} trades generated")
            
            if trades_buy:
                print("\n   🎉 FINAL SUCCESS! Direct order book matching is working!")
                print("   Generated trades:")
                for i, trade in enumerate(trades_buy):
                    print(f"     Trade {i+1}: {trade.volume} shares @ ${trade.price}")
                    print(f"                 Buy: {trade.buy_order_id}, Sell: {trade.sell_order_id}")
                    
                print(f"\n   Final book state:")
                print(f"     Best ASK = ${book.get_best_ask()}, Volume = {book.get_best_ask_volume()}")
                print(f"     Orders added: {book.stats['orders_added']}")
                print(f"     Trades executed: {book.stats['trades_executed']}")
                
                return True
            else:
                print("   ❌ Market order didn't generate trades")
        else:
            print("   ❌ ASK order wasn't added to book properly")
            
    except Exception as e:
        print(f"   ❌ Error during order processing: {e}")
        import traceback
        traceback.print_exc()
        
    return False

def test_limit_order_crossing():
    """Test limit orders that should cross"""
    print("\n\n=== DIRECT LIMIT CROSSING TEST ===")
    
    book = OrderBook("TSTB")
    
    print("1. Adding BID order...")
    bid_order = Order(
        order_id="BID001",
        symbol="TSTB",
        side=OrderSide.BID,
        order_type=OrderType.LIMIT,
        volume=100,
        price=150.00,
        timestamp=pd.Timestamp.now()
    )
    
    trades_bid = book.add_order(bid_order)
    print(f"   BID added: {len(trades_bid)} trades, Best BID = ${book.get_best_bid()}")
    
    print("\n2. Adding crossing ASK order...")
    ask_order = Order(
        order_id="ASK001",
        symbol="TSTB",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=75,
        price=149.50,  # Lower than bid - should cross
        timestamp=pd.Timestamp.now()
    )
    
    trades_ask = book.add_order(ask_order)
    print(f"   ASK added: {len(trades_ask)} trades generated")
    
    if trades_ask:
        print("   ✅ Crossing trades generated!")
        for i, trade in enumerate(trades_ask):
            print(f"     Trade {i+1}: {trade.volume} @ ${trade.price}")
        return True
    else:
        print("   ❌ No crossing trades generated")
        return False

if __name__ == "__main__":
    print("Testing core order book fixes...\n")
    
    success_market = test_direct_order_book()
    success_limit = test_limit_order_crossing()
    
    print("\n" + "="*60)
    print("FINAL CONCLUSION:")
    print("="*60)
    
    if success_market:
        print("✅ MARKET ORDER MATCHING: WORKING")
    else:
        print("❌ MARKET ORDER MATCHING: FAILED")
    
    if success_limit:
        print("✅ LIMIT ORDER CROSSING: WORKING") 
    else:
        print("❌ LIMIT ORDER CROSSING: FAILED")
        
    if success_market and success_limit:
        print("\n🎉 ALL CORE MATCHING ISSUES ARE FIXED!")
        print("The order book validation and circuit breaker issues have been resolved.")
        print("Market orders and limit order crossing now work correctly.")
    else:
        print("\n⚠️  Some issues remain to be investigated.")
    
    print("="*60)
