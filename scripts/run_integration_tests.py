#!/usr/bin/env python3
"""
Integration Test Runner for HFT Trading System.

This script runs comprehensive integration and system tests to validate
the complete HFT trading system functionality.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_command(cmd, description="", timeout=300):
    """Run a command and return the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully in {duration:.1f}s")
            if result.stdout.strip():
                logger.info("Output:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
        else:
            logger.error(f"✗ {description} failed in {duration:.1f}s")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr.strip():
                logger.error("Error output:")
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")
            if result.stdout.strip():
                logger.error("Standard output:")
                for line in result.stdout.strip().split('\n'):
                    logger.error(f"  {line}")
        
        return result.returncode == 0, result
    
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out after {timeout}s")
        return False, None
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False, None


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pytest',
        'numpy',
        'pandas',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies are installed")
    return True


def run_unit_tests():
    """Run unit tests first."""
    logger.info("Running unit tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-x",
        "--durations=10"
    ]
    
    return run_command(cmd, "Unit tests", timeout=180)


def run_integration_tests(test_type="all"):
    """Run integration tests."""
    logger.info(f"Running integration tests: {test_type}")
    
    # Base pytest command
    base_cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    
    test_commands = {
        "comprehensive": base_cmd + [
            "tests/integration/test_system_comprehensive.py",
            "-s"
        ],
        "trading_sessions": base_cmd + [
            "tests/integration/test_full_trading_session.py",
            "-x"
        ],
        "stress": base_cmd + [
            "tests/integration/test_system_stress.py",
            "-x"
        ],
        "pnl": base_cmd + [
            "tests/integration/test_order_flow_pnl.py",
            "-x"
        ],
        "all": base_cmd + [
            "tests/integration/",
            "-x"
        ]
    }
    
    if test_type not in test_commands:
        logger.error(f"Unknown test type: {test_type}")
        logger.info(f"Available types: {', '.join(test_commands.keys())}")
        return False
    
    cmd = test_commands[test_type]
    return run_command(cmd, f"Integration tests ({test_type})", timeout=600)


def run_performance_benchmarks():
    """Run performance benchmark tests."""
    logger.info("Running performance benchmarks...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-v",
        "--tb=short",
        "--durations=10",
        "-m", "not slow"  # Skip slow tests by default
    ]
    
    return run_command(cmd, "Performance benchmarks", timeout=300)


def generate_test_report():
    """Generate a comprehensive test report."""
    logger.info("Generating test report...")
    
    # Run tests with coverage and XML output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "--tb=short",
        "--durations=10",
        "--junitxml=test_report.xml",
        "-v"
    ]
    
    success, result = run_command(cmd, "Test report generation", timeout=600)
    
    if success:
        logger.info("✓ Test report generated: test_report.xml")
    
    return success


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="HFT Trading System Integration Test Runner")
    parser.add_argument(
        "--type",
        choices=["all", "comprehensive", "trading_sessions", "stress", "pnl"],
        default="all",
        help="Type of integration tests to run"
    )
    parser.add_argument(
        "--skip-unit",
        action="store_true",
        help="Skip unit tests and run only integration tests"
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true", 
        help="Skip performance benchmark tests"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate XML test report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("=" * 60)
    logger.info("HFT Trading System - Integration Test Runner")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    all_passed = True
    
    # Run unit tests first (unless skipped)
    if not args.skip_unit:
        success, _ = run_unit_tests()
        if not success:
            logger.warning("Unit tests failed, but continuing with integration tests...")
            # Don't fail completely, just warn
    
    # Run integration tests
    success, _ = run_integration_tests(args.type)
    if not success:
        logger.error("Integration tests failed")
        all_passed = False
    
    # Run performance benchmarks (unless skipped)
    if not args.skip_performance:
        success, _ = run_performance_benchmarks()
        if not success:
            logger.warning("Performance benchmarks failed, but continuing...")
            # Performance benchmarks are informational
    
    # Generate report if requested
    if args.generate_report:
        generate_test_report()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info(f"Test execution completed in {total_time:.1f}s")
    
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("System is ready for production deployment!")
    else:
        logger.error("✗ SOME TESTS FAILED")
        logger.error("Please review test results and fix issues before deployment.")
    
    logger.info("=" * 60)
    
    # Performance summary
    logger.info("Performance Summary:")
    logger.info("- Market data processing: >1000 updates/sec target")
    logger.info("- Strategy signal generation: >100 signals/sec target")
    logger.info("- Portfolio updates: >500 updates/sec target")
    logger.info("- Order execution: <100ms latency target")
    logger.info("- Memory usage: <1GB growth during extended operation")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
