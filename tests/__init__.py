"""
Test Suite for HFT Simulator

This package contains comprehensive tests for all components of the HFT simulator,
including unit tests, integration tests, and performance benchmarks.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- performance/: Performance and benchmark tests
- fixtures/: Test data and fixtures
- utils/: Testing utilities and helpers

Educational Notes:
- Testing is crucial for ensuring system reliability and correctness
- Unit tests verify individual component behavior
- Integration tests ensure components work together properly
- Performance tests validate system efficiency and scalability
- Comprehensive testing builds confidence in the system
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures" / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

__version__ = "1.0.0"