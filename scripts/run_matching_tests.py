#!/usr/bin/env python3
"""
Test Runner for Order Matching Scenarios

This script runs the comprehensive order matching tests to verify
that the fix for order matching issues is working correctly.

Usage:
    python run_matching_tests.py

The tests cover:
- Market buy order matching with existing ask limit order
- Market sell order matching with existing bid limit order
- Limit order crossing scenarios
- Partial fill scenarios
- Trade object creation with correct price, volume, and timestamps
"""

import sys
import os
import unittest
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the test module
try:
    from tests.unit.test_order_matching_scenarios import (
        TestOrderMatchingScenarios, 
        TestOrderMatchingPerformance
    )
    print("✓ Successfully imported test modules")
except ImportError as e:
    print(f"✗ Failed to import test modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def run_focused_tests():
    """Run focused tests for order matching scenarios"""
    print("="*80)
    print("ORDER MATCHING SCENARIO TESTS")
    print("="*80)
    print("Testing the core order matching functionality after bug fixes...")
    print()
    
    # Create test suite with specific tests
    suite = unittest.TestSuite()
    
    # Add core matching scenario tests
    core_tests = [
        'test_market_buy_matches_existing_ask_limit',
        'test_market_sell_matches_existing_bid_limit', 
        'test_limit_order_crossing_scenarios',
        'test_partial_fill_scenarios',
        'test_trade_object_creation_accuracy',
    ]
    
    print("Adding core matching tests:")
    for test_name in core_tests:
        suite.addTest(TestOrderMatchingScenarios(test_name))
        print(f"  ✓ {test_name}")
    
    # Add additional validation tests
    validation_tests = [
        'test_price_time_priority_enforcement',
        'test_edge_cases_and_error_conditions',
        'test_multiple_symbol_isolation',
    ]
    
    print("\nAdding validation tests:")
    for test_name in validation_tests:
        suite.addTest(TestOrderMatchingScenarios(test_name))
        print(f"  ✓ {test_name}")
    
    print(f"\nTotal tests to run: {suite.countTestCases()}")
    print("-" * 80)
    
    # Run the tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        stream=sys.stdout,
        descriptions=True
    )
    
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_count = total_tests - failures - errors
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Tests executed: {total_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        print("The order matching functionality is working correctly.")
        print("✅ Market orders match with existing limit orders")
        print("✅ Limit orders cross spreads correctly")
        print("✅ Partial fills are handled properly")
        print("✅ Trade objects are created with accurate data")
        print("✅ Price-time priority is enforced")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        
        if result.failures:
            print(f"\nFAILED TESTS ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"- {test}")
                # Print first few lines of traceback for context
                lines = traceback.split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"  {line}")
        
        if result.errors:
            print(f"\nERROR TESTS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"- {test}")
                # Print error summary
                lines = traceback.split('\n')
                for line in lines[-2:]:
                    if line.strip():
                        print(f"  {line}")
    
    print("="*80)
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*80)
    print("PERFORMANCE TESTS")
    print("="*80)
    
    # Create performance test suite
    perf_suite = unittest.TestSuite()
    perf_suite.addTest(TestOrderMatchingPerformance('test_high_volume_matching_performance'))
    perf_suite.addTest(TestOrderMatchingPerformance('test_memory_usage_stability'))
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    perf_result = runner.run(perf_suite)
    
    if perf_result.wasSuccessful():
        print("✅ Performance tests passed - engine handles high volume efficiently")
    else:
        print("⚠️  Performance tests had issues")
    
    return perf_result.wasSuccessful()


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("Running quick smoke test...")
    
    try:
        from src.execution.matching_engine import MatchingEngine
        from src.engine.order_types import Order
        from src.utils.constants import OrderSide, OrderType
        import pandas as pd
        
        # Create a simple test scenario
        engine = MatchingEngine("SMOKE_TEST")
        
        # Add ask order
        ask = Order(
            order_id="SMOKE_ASK",
            symbol="SMOKE_TEST",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=150.0,
            timestamp=pd.Timestamp.now()
        )
        
        trades1, _ = engine.process_order(ask)
        
        # Add market buy that should match
        buy = Order(
            order_id="SMOKE_BUY",
            symbol="SMOKE_TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=50,
            timestamp=pd.Timestamp.now()
        )
        
        trades2, _ = engine.process_order(buy)
        
        if len(trades2) == 1 and trades2[0].volume == 50 and trades2[0].price == 150.0:
            print("✅ Smoke test PASSED - basic matching is working")
            return True
        else:
            print(f"❌ Smoke test FAILED - expected 1 trade with volume 50 at price 150.0")
            print(f"   Got: {len(trades2)} trades")
            if trades2:
                print(f"   Trade details: volume={trades2[0].volume}, price={trades2[0].price}")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Starting Order Matching Test Suite...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Run smoke test first
    smoke_success = run_quick_smoke_test()
    
    if not smoke_success:
        print("\n❌ Smoke test failed - there may be fundamental issues")
        print("Please check the matching engine implementation before running full tests")
        sys.exit(1)
    
    # Run focused tests
    main_success = run_focused_tests()
    
    # Optionally run performance tests
    run_perf = input("\nRun performance tests? (y/N): ").lower().startswith('y')
    perf_success = True
    
    if run_perf:
        perf_success = run_performance_tests()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if main_success and perf_success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nThe order matching engine fix has been verified:")
        print("- Market orders correctly match with existing limit orders")
        print("- Limit orders properly cross spreads and generate trades")
        print("- Partial fills are handled accurately")
        print("- Trade objects contain correct price, volume, and timestamps")
        print("- Price-time priority is enforced correctly")
        
        if run_perf:
            print("- Performance benchmarks are met")
        
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nThe order matching engine may still have issues that need investigation.")
        exit_code = 1
    
    print("="*80)
    sys.exit(exit_code)
