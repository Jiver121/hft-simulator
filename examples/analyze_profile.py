#!/usr/bin/env python3
"""
Profile Analysis Script
Analyzes the cProfile output and generates performance bottleneck analysis
"""

import pstats
import io
from pathlib import Path

def analyze_profile_stats(profile_file: str = "profile.stats"):
    """Analyze profile statistics and extract key performance metrics"""
    
    # Create a Stats object
    stats = pstats.Stats(profile_file)
    
    # Redirect stdout to capture the output
    output = io.StringIO()
    
    # Get total time and call count
    print("="*80)
    print("PERFORMANCE PROFILING ANALYSIS")
    print("="*80)
    
    # Get basic stats
    stats.print_stats(0)  # Just header info
    total_calls = stats.total_calls
    total_time = stats.total_tt
    
    print(f"Total function calls: {total_calls:,}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per call: {(total_time/total_calls)*1000000:.2f} microseconds")
    print()
    
    # Sort by cumulative time and get top functions
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 80)
    stats.sort_stats('cumtime')
    stats.print_stats(20)
    print()
    
    # Sort by total time (internal time) and get top functions
    print("TOP 20 FUNCTIONS BY INTERNAL TIME:")
    print("-" * 80)
    stats.sort_stats('tottime')
    stats.print_stats(20)
    print()
    
    # Sort by number of calls
    print("TOP 20 MOST CALLED FUNCTIONS:")
    print("-" * 80)
    stats.sort_stats('ncalls')
    stats.print_stats(20)
    print()
    
    # Get detailed breakdown of key bottlenecks
    print("DETAILED BOTTLENECK ANALYSIS:")
    print("-" * 80)
    
    # Analyze specific modules/functions that might be bottlenecks
    bottleneck_patterns = [
        'matching_engine',
        'order_book',
        'execution',
        'strategy',
        'data_ingestion',
        'preprocessor',
        'pandas',
        'numpy'
    ]
    
    for pattern in bottleneck_patterns:
        print(f"\nFunctions containing '{pattern}':")
        try:
            stats.print_stats(f'.*{pattern}.*')
        except:
            print(f"No functions found matching pattern '{pattern}'")
    
    return stats

def generate_performance_report(stats):
    """Generate a structured performance report"""
    
    # Get the raw stats data
    raw_stats = stats.get_stats_profile().stats
    
    # Sort by cumulative time
    sorted_stats = sorted(raw_stats.items(), 
                         key=lambda x: x[1].cumulative, 
                         reverse=True)
    
    print("\n" + "="*80)
    print("TOP 10 PERFORMANCE BOTTLENECKS")
    print("="*80)
    
    bottlenecks = []
    
    for i, (func, stat) in enumerate(sorted_stats[:10], 1):
        filename = func[0]
        line_number = func[1] 
        function_name = func[2]
        
        cumtime = stat.cumulative
        tottime = stat.totaltime
        ncalls = stat.callcount
        
        # Calculate percentages
        total_cumtime = sorted_stats[0][1].cumulative
        cumtime_percent = (cumtime / total_cumtime) * 100
        
        bottleneck = {
            'rank': i,
            'function': f"{function_name} ({Path(filename).name}:{line_number})",
            'cumulative_time': cumtime,
            'internal_time': tottime,
            'calls': ncalls,
            'cumtime_percent': cumtime_percent,
            'avg_time_per_call': cumtime / ncalls if ncalls > 0 else 0
        }
        
        bottlenecks.append(bottleneck)
        
        print(f"{i:2d}. {bottleneck['function']}")
        print(f"    Cumulative Time: {cumtime:.4f}s ({cumtime_percent:.2f}%)")
        print(f"    Internal Time:   {tottime:.4f}s")
        print(f"    Calls:           {ncalls:,}")
        print(f"    Avg per call:    {bottleneck['avg_time_per_call']*1000:.2f}ms")
        print()
    
    return bottlenecks

if __name__ == "__main__":
    try:
        print("Starting profile analysis...")
        stats = analyze_profile_stats()
        bottlenecks = generate_performance_report(stats)
        
        # Save bottlenecks data for the documentation
        print(f"Analysis complete. Found {len(bottlenecks)} major bottlenecks.")
        
    except Exception as e:
        print(f"Error analyzing profile: {e}")
        import traceback
        traceback.print_exc()
