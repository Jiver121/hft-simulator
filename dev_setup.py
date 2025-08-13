"""
Development Setup Script for HFT Simulator

This script provides utilities for setting up the development environment,
running tests, and maintaining code quality.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

class DevSetup:
    """Development setup and maintenance utilities."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        
    def install_dependencies(self, extras=None):
        """Install project dependencies."""
        print("🔧 Installing dependencies...")
        
        # Install main dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        
        # Install extras if specified
        if extras:
            for extra in extras:
                print(f"Installing {extra} extras...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-e", f".[{extra}]"
                ], check=True)
        
        print("✅ Dependencies installed successfully!")
    
    def setup_pre_commit(self):
        """Setup pre-commit hooks."""
        print("🔧 Setting up pre-commit hooks...")
        
        try:
            subprocess.run(["pre-commit", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Installing pre-commit...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        
        subprocess.run(["pre-commit", "install"], check=True)
        print("✅ Pre-commit hooks installed!")
    
    def run_tests(self, coverage=True, verbose=False):
        """Run test suite."""
        print("🧪 Running tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.append("tests/")
        
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return result.returncode == 0
    
    def format_code(self):
        """Format code using black and isort."""
        print("🎨 Formatting code...")
        
        # Black formatting
        subprocess.run(["black", str(self.src_path), "tests/"], check=True)
        
        # Import sorting
        subprocess.run(["isort", str(self.src_path), "tests/"], check=True)
        
        print("✅ Code formatted!")
    
    def lint_code(self):
        """Run code linting."""
        print("🔍 Linting code...")
        
        # Flake8
        result = subprocess.run(["flake8", str(self.src_path)], capture_output=True)
        if result.returncode != 0:
            print("❌ Linting issues found:")
            print(result.stdout.decode())
        else:
            print("✅ No linting issues found!")
        
        return result.returncode == 0
    
    def type_check(self):
        """Run type checking with mypy."""
        print("🔍 Type checking...")
        
        result = subprocess.run(["mypy", str(self.src_path)], capture_output=True)
        if result.returncode != 0:
            print("❌ Type checking issues found:")
            print(result.stdout.decode())
        else:
            print("✅ No type issues found!")
        
        return result.returncode == 0
    
    def clean_cache(self):
        """Clean Python cache files."""
        print("🧹 Cleaning cache files...")
        
        # Remove __pycache__ directories
        for cache_dir in self.project_root.rglob("__pycache__"):
            if cache_dir.is_dir():
                subprocess.run(["rm", "-rf", str(cache_dir)])
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            pyc_file.unlink()
        
        # Remove pytest cache
        pytest_cache = self.project_root / ".pytest_cache"
        if pytest_cache.exists():
            subprocess.run(["rm", "-rf", str(pytest_cache)])
        
        print("✅ Cache cleaned!")
    
    def build_docs(self):
        """Build documentation."""
        print("📚 Building documentation...")
        
        docs_dir = self.project_root / "docs"
        if not docs_dir.exists():
            print("❌ Docs directory not found!")
            return False
        
        # Build with Sphinx or MkDocs based on config
        if (docs_dir / "conf.py").exists():
            # Sphinx
            subprocess.run(["sphinx-build", "-b", "html", str(docs_dir), str(docs_dir / "_build")])
        elif (docs_dir / "mkdocs.yml").exists():
            # MkDocs
            subprocess.run(["mkdocs", "build"])
        
        print("✅ Documentation built!")
        return True
    
    def check_security(self):
        """Run security checks."""
        print("🔒 Running security checks...")
        
        try:
            # Safety check
            result = subprocess.run(["safety", "check"], capture_output=True)
            if result.returncode != 0:
                print("❌ Security vulnerabilities found:")
                print(result.stdout.decode())
            else:
                print("✅ No security issues found!")
            
            return result.returncode == 0
        except FileNotFoundError:
            print("⚠️ Safety not installed, skipping security check")
            return True
    
    def full_check(self):
        """Run full code quality check."""
        print("🚀 Running full quality check...")
        
        checks = [
            ("Format", self.format_code),
            ("Lint", self.lint_code),
            ("Type Check", self.type_check),
            ("Tests", self.run_tests),
            ("Security", self.check_security),
        ]
        
        results = {}
        for name, check_func in checks:
            print(f"\n--- {name} ---")
            try:
                results[name] = check_func()
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = False
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY:")
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{name}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All checks passed! Ready for commit.")
        else:
            print("\n💥 Some checks failed. Please fix issues before committing.")
        
        return all_passed


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="HFT Simulator Development Tools")
    parser.add_argument("command", nargs="?", default="help", 
                       choices=["install", "test", "format", "lint", "type-check", 
                               "clean", "docs", "security", "check-all", "help"])
    parser.add_argument("--extras", nargs="*", help="Extra dependencies to install")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage in tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    dev = DevSetup()
    
    if args.command == "help":
        print("""
HFT Simulator Development Tools

Available commands:
  install      Install dependencies
  test         Run test suite
  format       Format code with black and isort
  lint         Run flake8 linting
  type-check   Run mypy type checking
  clean        Clean cache files
  docs         Build documentation
  security     Run security checks
  check-all    Run all quality checks
  help         Show this help message

Examples:
  python dev_setup.py install --extras dev ml
  python dev_setup.py test --verbose
  python dev_setup.py check-all
        """)
    
    elif args.command == "install":
        dev.install_dependencies(args.extras)
        dev.setup_pre_commit()
    
    elif args.command == "test":
        dev.run_tests(coverage=not args.no_coverage, verbose=args.verbose)
    
    elif args.command == "format":
        dev.format_code()
    
    elif args.command == "lint":
        dev.lint_code()
    
    elif args.command == "type-check":
        dev.type_check()
    
    elif args.command == "clean":
        dev.clean_cache()
    
    elif args.command == "docs":
        dev.build_docs()
    
    elif args.command == "security":
        dev.check_security()
    
    elif args.command == "check-all":
        success = dev.full_check()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
