#!/usr/bin/env python3
"""
Performance Benchmarks Demonstration Script

This script runs comprehensive performance benchmarks to demonstrate
the effectiveness of the implemented optimizations.

Usage:
    python scripts/run_performance_benchmarks.py

The script will:
1. Run analytics performance benchmarks 
2. Test data loading cache performance
3. Benchmark order book operations
4. Show memory optimization results
5. Generate a performance report
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our optimized components
from src.performance.analytics import (
    PerformanceAnalytics, 
    benchmark_analytics_performance,
    calculate_vwap,
    calculate_returns
)
from src.data.data_loader import CachedDataLoader
from src.engine.order_book import OrderBook
from src.engine.order_types import Order
from src.utils.constants import OrderSide, OrderType


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n📊 {title}")
    print("-" * 60)


def run_analytics_benchmarks():
    """Run analytics performance benchmarks"""
    print_header("ANALYTICS PERFORMANCE BENCHMARKS")
    
    # Test different data sizes
    sizes = [10000, 50000, 100000, 500000]
    results = {}
    
    for size in sizes:
        print_subheader(f"Testing with {size:,} data points")
        
        # Generate test data
        np.random.seed(42)
        prices = 100.0 + np.random.randn(size) * 0.5
        volumes = np.random.randint(100, 5000, size)
        
        analytics = PerformanceAnalytics()
        
        # Benchmark VWAP calculation
        start_time = time.perf_counter()
        vwap_optimized = analytics.calculate_vwap_vectorized(prices, volumes)
        optimized_time = time.perf_counter() - start_time
        
        # Benchmark naive implementation
        start_time = time.perf_counter() 
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        vwap_naive = total_value / total_volume
        naive_time = time.perf_counter() - start_time
        
        speedup = naive_time / optimized_time
        
        print(f"  Optimized VWAP: {optimized_time*1000:.2f}ms")
        print(f"  Naive VWAP:     {naive_time*1000:.2f}ms")
        print(f"  Speedup:        {speedup:.1f}x")
        print(f"  Accuracy:       ✅ {abs(vwap_optimized - vwap_naive) < 1e-10}")
        
        results[size] = {
            'optimized_time_ms': optimized_time * 1000,
            'naive_time_ms': naive_time * 1000,
            'speedup': speedup
        }
        
        # Test returns calculation
        returns_optimized = analytics.calculate_returns_vectorized(prices)
        returns_naive = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        
        print(f"  Returns calc:   ✅ {len(returns_optimized)} values computed")
    
    # Summary
    print_subheader("Performance Summary")
    avg_speedup = np.mean([r['speedup'] for r in results.values()])
    max_speedup = max([r['speedup'] for r in results.values()])
    
    print(f"  Average speedup across all sizes: {avg_speedup:.1f}x")
    print(f"  Maximum speedup achieved:         {max_speedup:.1f}x")
    print(f"  Performance target (>10x):        {'✅ ACHIEVED' if avg_speedup > 10 else '❌ NOT MET'}")
    
    return results


def run_caching_benchmarks():
    """Run data loading cache performance benchmarks"""
    print_header("DATA LOADING CACHE BENCHMARKS")
    
    # Create test data file
    print_subheader("Creating test dataset")
    
    np.random.seed(42)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50000, freq='1s'),
        'price': 100.0 + np.random.randn(50000) * 0.1,
        'volume': np.random.randint(100, 1000, 50000),
        'side': np.random.choice(['buy', 'sell'], 50000)
    })
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    test_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        print(f"  Created test file: {Path(temp_file.name).name}")
        print(f"  Data size:         {len(test_data):,} rows")
        print(f"  File size:         {os.path.getsize(temp_file.name) / 1024 / 1024:.1f}MB")
        
        # Test caching performance
        print_subheader("Cache Performance Test")
        
        loader = CachedDataLoader(cache_size=10, max_memory_mb=512)
        
        # First load (cold cache)
        print("  First load (cold cache)...")
        start_time = time.perf_counter()
        data1 = loader.load_data(temp_file.name)
        first_load_time = time.perf_counter() - start_time
        
        # Second load (warm cache)
        print("  Second load (warm cache)...")
        start_time = time.perf_counter()
        data2 = loader.load_data(temp_file.name)
        cached_load_time = time.perf_counter() - start_time
        
        # Verify data integrity
        data_match = data1.equals(data2)
        cache_speedup = first_load_time / cached_load_time
        
        print(f"  Cold load time:    {first_load_time*1000:.2f}ms")
        print(f"  Cached load time:  {cached_load_time*1000:.2f}ms")
        print(f"  Cache speedup:     {cache_speedup:.1f}x")
        print(f"  Data integrity:    {'✅ VERIFIED' if data_match else '❌ FAILED'}")
        
        # Cache statistics
        stats = loader.get_cache_stats()
        print(f"  Cache hit rate:    {stats['cache_efficiency']['memory_hit_rate']*100:.1f}%")
        print(f"  Cache target (>50x): {'✅ ACHIEVED' if cache_speedup > 50 else '❌ NOT MET'}")
        
        # Test query caching
        print_subheader("Query Cache Test")
        
        query = "price > 100.0 and volume > 500"
        
        # First query (uncached)
        start_time = time.perf_counter()
        result1 = loader.query_data(data1, query)
        first_query_time = time.perf_counter() - start_time
        
        # Second query (cached)
        start_time = time.perf_counter()
        result2 = loader.query_data(data1, query)
        cached_query_time = time.perf_counter() - start_time
        
        query_speedup = first_query_time / cached_query_time
        
        print(f"  First query:       {first_query_time*1000:.2f}ms")
        print(f"  Cached query:      {cached_query_time*1000:.2f}ms") 
        print(f"  Query speedup:     {query_speedup:.1f}x")
        print(f"  Results match:     {'✅ YES' if result1.equals(result2) else '❌ NO'}")
        
        return {
            'cache_speedup': cache_speedup,
            'query_speedup': query_speedup,
            'file_size_mb': os.path.getsize(temp_file.name) / 1024 / 1024
        }
    
    finally:
        # Cleanup
        os.unlink(temp_file.name)


def run_order_book_benchmarks():
    """Run order book performance benchmarks"""
    print_header("ORDER BOOK PERFORMANCE BENCHMARKS")
    
    print_subheader("Generating test orders")
    
    # Generate test orders
    np.random.seed(42)
    orders = []
    
    for i in range(20000):
        order = Order.create_limit_order(
            symbol="TEST",
            side=OrderSide.BID if np.random.random() > 0.5 else OrderSide.ASK,
            volume=int(np.random.randint(100, 1000)),
            price=100.0 + np.random.uniform(-5.0, 5.0)
        )
        orders.append(order)
    
    print(f"  Generated:         {len(orders):,} test orders")
    
    # Test order processing performance
    print_subheader("Order Processing Test")
    
    book = OrderBook("TEST")
    processed_orders = 0
    total_trades = 0
    
    start_time = time.perf_counter()
    
    for order in orders:
        try:
            trades = book.add_order(order)
            processed_orders += 1
            total_trades += len(trades)
        except Exception as e:
            # Skip invalid orders
            pass
    
    processing_time = time.perf_counter() - start_time
    orders_per_second = processed_orders / processing_time
    
    print(f"  Orders processed:  {processed_orders:,}")
    print(f"  Trades generated:  {total_trades:,}")
    print(f"  Processing time:   {processing_time*1000:.2f}ms")
    print(f"  Throughput:        {orders_per_second:.0f} orders/sec")
    print(f"  Target (>10k/sec): {'✅ ACHIEVED' if orders_per_second > 10000 else '❌ NOT MET'}")
    
    # Test snapshot generation
    print_subheader("Market Data Snapshot Test")
    
    snapshot_times = []
    for _ in range(1000):
        start_time = time.perf_counter()
        snapshot = book.get_snapshot()
        snapshot_times.append(time.perf_counter() - start_time)
    
    avg_snapshot_time = np.mean(snapshot_times)
    snapshots_per_second = 1.0 / avg_snapshot_time
    
    print(f"  Avg snapshot time: {avg_snapshot_time*1000:.2f}ms")
    print(f"  Snapshot rate:     {snapshots_per_second:.0f} snapshots/sec")
    print(f"  Target (>1k/sec):  {'✅ ACHIEVED' if snapshots_per_second > 1000 else '❌ NOT MET'}")
    
    # Order book statistics
    stats = book.get_statistics()
    print(f"  Final book state:  {stats['bid_levels']} bid levels, {stats['ask_levels']} ask levels")
    
    return {
        'orders_per_second': orders_per_second,
        'snapshots_per_second': snapshots_per_second,
        'trades_generated': total_trades
    }


def run_memory_benchmarks():
    """Run memory optimization benchmarks"""
    print_header("MEMORY OPTIMIZATION BENCHMARKS")
    
    try:
        import psutil
        process = psutil.Process()
        
        print_subheader("Memory Efficiency Test")
        
        # Measure baseline memory
        import gc
        gc.collect()
        memory_baseline = process.memory_info().rss / 1024 / 1024
        
        print(f"  Baseline memory:   {memory_baseline:.1f}MB")
        
        # Create large dataset
        print("  Creating large dataset...")
        np.random.seed(42)
        large_prices = 100.0 + np.random.randn(1000000) * 0.5
        large_volumes = np.random.randint(100, 5000, 1000000)
        
        memory_after_creation = process.memory_info().rss / 1024 / 1024
        creation_increase = memory_after_creation - memory_baseline
        
        print(f"  After data creation: {memory_after_creation:.1f}MB (+{creation_increase:.1f}MB)")
        
        # Perform analytics operations
        analytics = PerformanceAnalytics()
        
        vwap = analytics.calculate_vwap_vectorized(large_prices, large_volumes)
        returns = analytics.calculate_returns_vectorized(large_prices)
        
        memory_after_analytics = process.memory_info().rss / 1024 / 1024
        analytics_increase = memory_after_analytics - memory_after_creation
        
        print(f"  After analytics:   {memory_after_analytics:.1f}MB (+{analytics_increase:.1f}MB)")
        print(f"  VWAP result:       {vwap:.4f}")
        print(f"  Returns computed:  {len(returns):,}")
        
        # Cleanup
        del large_prices, large_volumes, vwap, returns
        gc.collect()
        
        memory_after_cleanup = process.memory_info().rss / 1024 / 1024
        print(f"  After cleanup:     {memory_after_cleanup:.1f}MB")
        
        total_increase = memory_after_cleanup - memory_baseline
        
        print(f"  Net memory impact: {total_increase:.1f}MB")
        print(f"  Memory efficiency: {'✅ EXCELLENT' if total_increase < 50 else '⚠️ ACCEPTABLE' if total_increase < 100 else '❌ POOR'}")
        
        return {
            'baseline_mb': memory_baseline,
            'peak_mb': memory_after_analytics,
            'final_mb': memory_after_cleanup,
            'net_increase_mb': total_increase
        }
        
    except ImportError:
        print("  ⚠️ psutil not available - skipping memory benchmarks")
        return {}


def generate_performance_report(analytics_results, cache_results, orderbook_results, memory_results):
    """Generate comprehensive performance report"""
    print_header("PERFORMANCE OPTIMIZATION REPORT")
    
    print_subheader("Executive Summary")
    
    # Calculate scores
    scores = []
    
    # Analytics score (based on average speedup)
    if analytics_results:
        avg_speedup = np.mean([r['speedup'] for r in analytics_results.values()])
        analytics_score = min(100, avg_speedup * 5)  # Max 100 for 20x speedup
        scores.append(analytics_score)
        print(f"  📊 Analytics Performance:  {analytics_score:.0f}/100 ({avg_speedup:.1f}x speedup)")
    
    # Cache score (based on cache speedup)
    if cache_results and 'cache_speedup' in cache_results:
        cache_speedup = cache_results['cache_speedup']
        cache_score = min(100, cache_speedup * 1.5)  # Max 100 for 67x speedup
        scores.append(cache_score)
        print(f"  💾 Caching Performance:    {cache_score:.0f}/100 ({cache_speedup:.1f}x speedup)")
    
    # Order book score (based on throughput)
    if orderbook_results and 'orders_per_second' in orderbook_results:
        throughput = orderbook_results['orders_per_second']
        orderbook_score = min(100, throughput / 500)  # Max 100 for 50k orders/sec
        scores.append(orderbook_score)
        print(f"  📈 Order Book Performance: {orderbook_score:.0f}/100 ({throughput:.0f} orders/sec)")
    
    # Memory score (inverse of memory usage increase)
    if memory_results and 'net_increase_mb' in memory_results:
        memory_increase = memory_results['net_increase_mb']
        memory_score = max(0, 100 - memory_increase)  # 100 for 0MB increase
        scores.append(memory_score)
        print(f"  🧠 Memory Efficiency:      {memory_score:.0f}/100 ({memory_increase:.1f}MB increase)")
    
    # Overall score
    if scores:
        overall_score = np.mean(scores)
        print(f"\n  🎯 OVERALL PERFORMANCE:    {overall_score:.0f}/100")
        
        if overall_score >= 90:
            grade = "A+"
            emoji = "🏆"
        elif overall_score >= 80:
            grade = "A"
            emoji = "🥇"
        elif overall_score >= 70:
            grade = "B+"
            emoji = "🥈"
        elif overall_score >= 60:
            grade = "B"
            emoji = "🥉"
        else:
            grade = "C"
            emoji = "📈"
        
        print(f"  {emoji} PERFORMANCE GRADE:      {grade}")
    
    print_subheader("Key Achievements")
    
    achievements = []
    
    if analytics_results:
        avg_speedup = np.mean([r['speedup'] for r in analytics_results.values()])
        if avg_speedup > 10:
            achievements.append(f"✅ Analytics operations are {avg_speedup:.1f}x faster through vectorization")
    
    if cache_results and cache_results.get('cache_speedup', 0) > 50:
        achievements.append(f"✅ Data loading is {cache_results['cache_speedup']:.1f}x faster with LRU caching")
    
    if orderbook_results and orderbook_results.get('orders_per_second', 0) > 10000:
        achievements.append(f"✅ Order processing throughput exceeds 10,000 orders/second")
    
    if memory_results and memory_results.get('net_increase_mb', 100) < 50:
        achievements.append("✅ Memory usage optimized with minimal overhead")
    
    for achievement in achievements:
        print(f"  {achievement}")
    
    print_subheader("Performance Targets Status")
    
    targets = [
        ("Numpy Vectorization (>10x speedup)", analytics_results and np.mean([r['speedup'] for r in analytics_results.values()]) > 10),
        ("LRU Caching (>50x speedup)", cache_results and cache_results.get('cache_speedup', 0) > 50),
        ("Order Processing (>10k/sec)", orderbook_results and orderbook_results.get('orders_per_second', 0) > 10000),
        ("Memory Efficiency (<100MB)", memory_results and memory_results.get('net_increase_mb', 100) < 100)
    ]
    
    for target, achieved in targets:
        status = "✅ ACHIEVED" if achieved else "❌ NOT MET"
        print(f"  {target}: {status}")
    
    print_subheader("Recommendations")
    
    recommendations = []
    
    if not (analytics_results and np.mean([r['speedup'] for r in analytics_results.values()]) > 10):
        recommendations.append("📌 Consider using Numba JIT compilation for additional analytics speedup")
    
    if not (cache_results and cache_results.get('cache_speedup', 0) > 50):
        recommendations.append("📌 Optimize cache eviction strategy and increase cache memory allocation")
    
    if not (orderbook_results and orderbook_results.get('orders_per_second', 0) > 10000):
        recommendations.append("📌 Consider implementing order book optimizations (e.g., tree-based price levels)")
    
    if not recommendations:
        recommendations.append("🎉 All performance targets achieved! System is well-optimized.")
    
    for rec in recommendations:
        print(f"  {rec}")


def main():
    """Main benchmark execution"""
    print_header("HFT SIMULATOR PERFORMANCE OPTIMIZATION BENCHMARKS")
    print("\n🚀 Running comprehensive performance tests...")
    print("⏱️  This may take a few minutes to complete.\n")
    
    # Run all benchmarks
    try:
        analytics_results = run_analytics_benchmarks()
        cache_results = run_caching_benchmarks()
        orderbook_results = run_order_book_benchmarks()
        memory_results = run_memory_benchmarks()
        
        # Generate comprehensive report
        generate_performance_report(analytics_results, cache_results, orderbook_results, memory_results)
        
        print_header("BENCHMARK COMPLETED")
        print("\n✅ All performance benchmarks completed successfully!")
        print("📊 Check the results above to see optimization improvements.")
        print("🔧 Review recommendations for potential further optimizations.")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
