import sys
import subprocess


def main():
    print("Running automated performance regression tests...")
    try:
        # Run pytest on the performance directory
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/performance/test_regression.py', '--tb=short', '-q', '--disable-warnings'
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("[Benchmark Validation PASSED] All performance requirements have been met.")
        else:
            print("[Benchmark Validation FAILED] Some performance requirements were not met.")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Exception running benchmarks: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()

