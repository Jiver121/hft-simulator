"""
Test Runner for HFT Simulator

This script provides a convenient way to run different types of tests
with appropriate configurations and reporting.

Educational Notes:
- Automated test running ensures consistent test execution
- Different test suites can be run independently
- Test results are properly formatted and reported
- Performance benchmarks can be run separately from unit tests
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed after {duration:.2f}s")
        print(f"Exit code: {e.returncode}")
        return False


def run_unit_tests():
    """Run unit tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-m", "not slow"
    ]
    return run_command(cmd, "Running Unit Tests")


def run_integration_tests():
    """Run integration tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    return run_command(cmd, "Running Integration Tests")


def run_performance_tests():
    """Run performance benchmarks"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-v",
        "--tb=short",
        "-m", "performance",
        "-s"  # Don't capture output for benchmarks
    ]
    return run_command(cmd, "Running Performance Benchmarks")


def run_all_tests():
    """Run all tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running All Tests")


def run_quick_tests():
    """Run quick tests (unit tests only, no slow tests)"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=line",
        "-m", "not slow",
        "-x"  # Stop on first failure
    ]
    return run_command(cmd, "Running Quick Tests")


def run_coverage_tests():
    """Run tests with coverage reporting"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-v"
    ]
    return run_command(cmd, "Running Tests with Coverage")


def run_specific_test(test_path: str):
    """Run a specific test file or test function"""
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-s"
    ]
    return run_command(cmd, f"Running Specific Test: {test_path}")


def check_test_environment():
    """Check if test environment is properly set up"""
    print("🔍 Checking Test Environment")
    print("-" * 60)
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not installed. Run: pip install pytest")
        return False
    
    # Check if source code is importable
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        import src.engine.order_book
        print("✅ Source code is importable")
    except ImportError as e:
        print(f"❌ Cannot import source code: {e}")
        return False
    
    # Check test directory structure
    test_dirs = ["tests/unit", "tests/integration", "tests/performance"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"✅ {test_dir} directory exists")
        else:
            print(f"❌ {test_dir} directory missing")
            return False
    
    print("\n✅ Test environment is properly configured")
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="HFT Simulator Test Runner")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "all", "quick", "coverage", "check"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--specific",
        help="Run a specific test file or function (e.g., tests/unit/test_order_book.py::TestOrderBook::test_initialization)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Don't capture output (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    print("🚀 HFT Simulator Test Runner")
    print("=" * 60)
    
    # Check environment first
    if not check_test_environment():
        print("\n❌ Test environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    success = True
    
    if args.specific:
        success = run_specific_test(args.specific)
    elif args.test_type == "unit":
        success = run_unit_tests()
    elif args.test_type == "integration":
        success = run_integration_tests()
    elif args.test_type == "performance":
        success = run_performance_tests()
    elif args.test_type == "all":
        success = run_all_tests()
    elif args.test_type == "quick":
        success = run_quick_tests()
    elif args.test_type == "coverage":
        success = run_coverage_tests()
    elif args.test_type == "check":
        # Environment check already done above
        pass
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()