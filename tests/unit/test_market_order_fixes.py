#!/usr/bin/env python3
"""
Comprehensive test to verify market order matching fixes

This test verifies that:
1. Market orders check against the opposite side of the book (buy orders check asks, sell orders check bids)
2. Market orders match at the best available price regardless of their price attribute
3. Order type checks (is_market_order() method usage) work correctly
4. Market orders are processed immediately and not added to the book if unfilled
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

def test_market_order_opposite_side_matching():
    """Test that market orders check against the opposite side of the book"""
    print("=== TEST: Market Orders Match Against Opposite Side ===")
    
    book = OrderBook("TESTSTOCK")
    
    # Setup: Add liquidity to both sides
    ask_order = Order(
        order_id="ASK001",
        symbol="TESTSTOCK",
        side=OrderSide.ASK,
        order_type=OrderType.LIMIT,
        volume=100,
        price=151.00,
        timestamp=pd.Timestamp.now()
    )
    
    bid_order = Order(
        order_id="BID001", 
        symbol="TESTSTOCK",
        side=OrderSide.BID,
        order_type=OrderType.LIMIT,
        volume=100,
        price=149.00,
        timestamp=pd.Timestamp.now()
    )
    
    book.add_order(ask_order)
    book.add_order(bid_order)
    
    print(f"Initial book: Best BID = ${book.get_best_bid()}, Best ASK = ${book.get_best_ask()}")
    
    # Test 1: Market BUY should match against ASK side
    market_buy = Order(
        order_id="MKT_BUY001",
        symbol="TESTSTOCK", 
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=50,
        timestamp=pd.Timestamp.now()
    )
    
    trades_buy = book.add_order(market_buy)
    
    if trades_buy and trades_buy[0].price == 151.00:
        print("✅ Market BUY correctly matched against ASK at $151.00")
    else:
        print("❌ Market BUY did not match correctly against ASK")
        return False
    
    # Test 2: Market SELL should match against BID side
    market_sell = Order(
        order_id="MKT_SELL001",
        symbol="TESTSTOCK",
        side=OrderSide.SELL, 
        order_type=OrderType.MARKET,
        volume=50,
        timestamp=pd.Timestamp.now()
    )
    
    trades_sell = book.add_order(market_sell)
    
    if trades_sell and trades_sell[0].price == 149.00:
        print("✅ Market SELL correctly matched against BID at $149.00")
        return True
    else:
        print("❌ Market SELL did not match correctly against BID")
        return False

def test_market_order_ignores_price_attribute():
    """Test that market orders match regardless of their price attribute"""
    print("\n=== TEST: Market Orders Ignore Price Attribute ===")
    
    book = OrderBook("TESTSTOCK2")
    
    # Add some liquidity
    ask1 = Order("ASK001", "TESTSTOCK2", OrderSide.ASK, OrderType.LIMIT, volume=50, price=150.00)
    ask2 = Order("ASK002", "TESTSTOCK2", OrderSide.ASK, OrderType.LIMIT, volume=50, price=150.50)
    
    book.add_order(ask1)
    book.add_order(ask2)
    
    print(f"Book has asks at $150.00 (50 shares) and $150.50 (50 shares)")
    
    # Create market order with artificially high price - should still match at best ask
    market_buy = Order(
        order_id="MKT001",
        symbol="TESTSTOCK2",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=75,
        price=200.00,  # High price that should be ignored
        timestamp=pd.Timestamp.now()
    )
    
    trades = book.add_order(market_buy)
    
    # Should generate two trades: 50 @ $150.00 and 25 @ $150.50
    if len(trades) == 2:
        if trades[0].price == 150.00 and trades[0].volume == 50:
            if trades[1].price == 150.50 and trades[1].volume == 25:
                print("✅ Market order correctly matched at best available prices, ignoring order price attribute")
                return True
    
    print("❌ Market order did not match correctly at best prices")
    print(f"   Expected: 2 trades (50@150.00, 25@150.50)")
    print(f"   Got: {len(trades)} trades: {[(t.volume, t.price) for t in trades]}")
    return False

def test_is_market_order_method():
    """Test that is_market_order() method works correctly"""
    print("\n=== TEST: is_market_order() Method Usage ===")
    
    # Create different order types
    market_order = Order("MKT001", "TEST", OrderSide.BUY, OrderType.MARKET, volume=100)
    limit_order = Order("LMT001", "TEST", OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
    
    # Test method calls
    market_is_market = market_order.is_market_order()
    market_is_limit = market_order.is_limit_order()
    limit_is_market = limit_order.is_market_order()
    limit_is_limit = limit_order.is_limit_order()
    
    if market_is_market and not market_is_limit and not limit_is_market and limit_is_limit:
        print("✅ is_market_order() and is_limit_order() methods work correctly")
        return True
    else:
        print("❌ Order type methods not working correctly:")
        print(f"   Market order: is_market={market_is_market}, is_limit={market_is_limit}")
        print(f"   Limit order: is_market={limit_is_market}, is_limit={limit_is_limit}")
        return False

def test_market_orders_not_added_to_book():
    """Test that market orders are not added to the book if unfilled"""
    print("\n=== TEST: Market Orders Not Added to Book ===")
    
    book = OrderBook("TESTSTOCK3")
    
    # Test 1: Market order with no liquidity should not be added to book
    market_order_no_liquidity = Order(
        order_id="MKT_NO_LIQ",
        symbol="TESTSTOCK3",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=100,
        timestamp=pd.Timestamp.now()
    )
    
    trades = book.add_order(market_order_no_liquidity)
    
    # Should have no trades and order should not be in active book
    if len(trades) == 0:
        active_orders = book.get_order_count()
        if active_orders == 0:
            print("✅ Market order with no liquidity correctly rejected and not added to book")
        else:
            print("❌ Market order with no liquidity was added to book")
            return False
    else:
        print("❌ Market order with no liquidity somehow generated trades")
        return False
    
    # Test 2: Partially filled market order should not have remainder in book
    # Add some liquidity
    ask_order = Order("ASK001", "TESTSTOCK3", OrderSide.ASK, OrderType.LIMIT, volume=50, price=150.00)
    book.add_order(ask_order)
    
    # Market order for more than available
    large_market_order = Order(
        order_id="MKT_LARGE",
        symbol="TESTSTOCK3", 
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=100,  # More than the 50 available
        timestamp=pd.Timestamp.now()
    )
    
    trades = book.add_order(large_market_order)
    
    # Should have one trade for 50 shares, and no remaining volume in book
    if len(trades) == 1 and trades[0].volume == 50:
        active_orders_after = book.get_order_count() 
        # Should only have the unfilled portion of the ask order (if any)
        print(f"   Trade: {trades[0].volume} @ ${trades[0].price}")
        print(f"   Active orders after: {active_orders_after}")
        print(f"   Orders in book: {list(book.orders.keys())}")
        print(f"   Ask levels: {len(book.asks)}, Bid levels: {len(book.bids)}")
        # Check status of each order
        for order_id, order in book.orders.items():
            print(f"     Order {order_id}: status={order.status}, is_active={order.is_active()}, remaining_vol={order.remaining_volume}")
        if active_orders_after == 0:  # Ask order should be fully consumed
            print("✅ Partially filled market order correctly processed, remainder not added to book")
            return True
        else:
            print("❌ Partially filled market order left remnants in book")
            return False
    else:
        print("❌ Large market order did not execute correctly")
        print(f"   Expected: 1 trade of 50 shares")
        print(f"   Got: {len(trades)} trades: {[(t.volume, t.price) for t in trades]}")
        return False

def test_comprehensive_market_order_scenarios():
    """Test various comprehensive market order scenarios"""
    print("\n=== TEST: Comprehensive Market Order Scenarios ===")
    
    book = OrderBook("COMPREHENSIVE")
    
    # Setup a multi-level book
    orders = [
        Order("ASK1", "COMPREHENSIVE", OrderSide.ASK, OrderType.LIMIT, volume=100, price=151.00),
        Order("ASK2", "COMPREHENSIVE", OrderSide.ASK, OrderType.LIMIT, volume=200, price=151.50), 
        Order("ASK3", "COMPREHENSIVE", OrderSide.ASK, OrderType.LIMIT, volume=150, price=152.00),
        Order("BID1", "COMPREHENSIVE", OrderSide.BID, OrderType.LIMIT, volume=100, price=149.00),
        Order("BID2", "COMPREHENSIVE", OrderSide.BID, OrderType.LIMIT, volume=200, price=148.50),
        Order("BID3", "COMPREHENSIVE", OrderSide.BID, OrderType.LIMIT, volume=150, price=148.00),
    ]
    
    for order in orders:
        book.add_order(order)
    
    print(f"Multi-level book setup complete")
    print(f"ASKs: 100@151.00, 200@151.50, 150@152.00")
    print(f"BIDs: 100@149.00, 200@148.50, 150@148.00")
    
    # Test: Large market buy that consumes multiple levels
    large_market_buy = Order(
        order_id="LARGE_MKT_BUY",
        symbol="COMPREHENSIVE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=350,  # Should consume all of ASK1 and ASK2, plus 50 from ASK3
        timestamp=pd.Timestamp.now()
    )
    
    trades = book.add_order(large_market_buy)
    
    # Verify trades
    expected_trades = [
        (100, 151.00),  # All of ASK1
        (200, 151.50),  # All of ASK2  
        (50, 152.00)    # Partial ASK3
    ]
    
    if len(trades) == 3:
        all_correct = True
        for i, (expected_vol, expected_price) in enumerate(expected_trades):
            if trades[i].volume != expected_vol or trades[i].price != expected_price:
                all_correct = False
                break
        
        if all_correct:
            print("✅ Large market order correctly consumed multiple price levels")
            
            # Check that order is not in book and remaining ask volume is correct
            remaining_ask_vol = book.get_best_ask_volume()
            if remaining_ask_vol == 100:  # 150 - 50 from ASK3
                print("✅ Remaining liquidity correctly updated")
                return True
            else:
                print(f"❌ Remaining ask volume incorrect: expected 100, got {remaining_ask_vol}")
        else:
            print("❌ Large market order trades were incorrect")
    else:
        print(f"❌ Expected 3 trades, got {len(trades)}")
    
    return False

def run_all_tests():
    """Run all market order tests"""
    print("Starting comprehensive market order matching tests...\n")
    
    tests = [
        test_market_order_opposite_side_matching,
        test_market_order_ignores_price_attribute,
        test_is_market_order_method,
        test_market_orders_not_added_to_book,
        test_comprehensive_market_order_scenarios
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("MARKET ORDER MATCHING TEST RESULTS")
    print("="*60)
    
    test_names = [
        "Market orders match opposite side",
        "Market orders ignore price attribute", 
        "is_market_order() method works",
        "Market orders not added to book",
        "Comprehensive scenarios work"
    ]
    
    all_passed = True
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL MARKET ORDER TESTS PASSED!")
        print("Market order handling is working correctly according to requirements.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Market order handling needs additional fixes.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
