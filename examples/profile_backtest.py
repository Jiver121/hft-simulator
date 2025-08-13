#!/usr/bin/env python3
"""
Quick Backtesting Profiler
Runs cProfile on the backtesting engine and generates analysis reports
"""

import subprocess
import sys
import os
from pathlib import Path

def run_profile_analysis():
    """Run complete profiling analysis"""
    
    print("🔄 Starting HFT Simulator Performance Profiling...")
    print("=" * 60)
    
    # Step 1: Run cProfile on backtesting
    print("📊 Step 1: Running cProfile on backtesting engine...")
    cmd = [
        sys.executable, "-m", "cProfile", "-o", "profile.stats",
        "main.py", 
        "--mode", "backtest", 
        "--data", "data/aapl_data.csv", 
        "--strategy", "market_making",
        "--output", "profile_backtest_results.json"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Profiling completed successfully")
        else:
            print(f"❌ Profiling failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Profiling timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running profiler: {e}")
        return False
    
    # Step 2: Verify profile file exists
    if not os.path.exists("profile.stats"):
        print("❌ Profile stats file not generated")
        return False
    
    print(f"📁 Profile file size: {os.path.getsize('profile.stats')} bytes")
    
    # Step 3: Run analysis script
    print("\n📈 Step 2: Analyzing profile statistics...")
    try:
        result = subprocess.run([sys.executable, "analyze_profile.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Profile analysis completed")
            # Print key results
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Total function calls:' in line or 'Total time:' in line:
                    print(f"  {line}")
        else:
            print(f"❌ Analysis failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Analysis script error: {e}")
    
    # Step 4: Generate visual report info
    print("\n🌐 Step 3: Visual profiling report available...")
    print("  To view visual analysis:")
    print(f"    snakeviz profile.stats")
    print(f"  Or install and run:")
    print(f"    pip install snakeviz")
    print(f"    snakeviz profile.stats")
    
    # Step 5: Check documentation
    print("\n📝 Step 4: Performance analysis documentation...")
    doc_path = Path("docs/performance_analysis.md")
    if doc_path.exists():
        print(f"✅ Documentation available: {doc_path}")
        print(f"  File size: {doc_path.stat().st_size} bytes")
    else:
        print("❌ Documentation not found")
    
    print("\n" + "=" * 60)
    print("🎯 Profiling Summary:")
    print("  • cProfile data: profile.stats")
    print("  • Analysis output: console + analyze_profile.py")
    print("  • Visual report: snakeviz profile.stats")
    print("  • Documentation: docs/performance_analysis.md")
    print("  • Backtest results: profile_backtest_results.json")
    print("🚀 Performance profiling complete!")
    
    return True

def quick_profile():
    """Quick profiling without full analysis"""
    print("⚡ Quick Profile Mode")
    cmd = [
        sys.executable, "-m", "cProfile", "-o", "quick_profile.stats",
        "main.py", 
        "--mode", "backtest", 
        "--data", "data/aapl_data.csv", 
        "--strategy", "market_making",
        "--output", "quick_results.json"
    ]
    
    subprocess.run(cmd)
    print("✅ Quick profile saved to: quick_profile.stats")
    print("   View with: python -m pstats quick_profile.stats")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile HFT Simulator Backtesting")
    parser.add_argument("--quick", action="store_true", help="Run quick profile only")
    parser.add_argument("--data", default="data/aapl_data.csv", help="Data file to use")
    parser.add_argument("--strategy", default="market_making", help="Strategy to profile")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_profile()
    else:
        success = run_profile_analysis()
        sys.exit(0 if success else 1)
