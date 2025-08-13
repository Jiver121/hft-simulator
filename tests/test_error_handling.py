"""
Test suite for error handling and recovery mechanisms.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.validation import validate_price, validate_quantity, validate_symbol, OrderValidationError
from src.utils.error_handler import ErrorHandler
from src.performance.circuit_breaker import CircuitBreaker
from src.engine.recovery import OrderBookRecovery


class TestValidation(unittest.TestCase):
    def test_validate_price(self):
        # Valid prices
        validate_price(100.50)
        validate_price(1)
        
        # Invalid prices
        with self.assertRaises(OrderValidationError):
            validate_price(-10)
        with self.assertRaises(OrderValidationError):
            validate_price(0)
        with self.assertRaises(OrderValidationError):
            validate_price("invalid")

    def test_validate_quantity(self):
        # Valid quantities
        validate_quantity(100)
        validate_quantity(1)
        
        # Invalid quantities
        with self.assertRaises(OrderValidationError):
            validate_quantity(-10)
        with self.assertRaises(OrderValidationError):
            validate_quantity(0)
        with self.assertRaises(OrderValidationError):
            validate_quantity(10.5)

    def test_validate_symbol(self):
        # Valid symbols
        validate_symbol("AAPL")
        validate_symbol("MSFT")
        validate_symbol("GOOGL")
        
        # Invalid symbols
        with self.assertRaises(OrderValidationError):
            validate_symbol("invalid_symbol")
        with self.assertRaises(OrderValidationError):
            validate_symbol("aapl")  # lowercase
        with self.assertRaises(OrderValidationError):
            validate_symbol("")


class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.error_handler = ErrorHandler()

    @patch('src.utils.error_handler.logger')
    def test_log_error(self, mock_logger):
        correlation_id = self.error_handler.log_error(
            ValueError("Test error"), 
            extra_info={"test": "data"}
        )
        
        self.assertIsInstance(correlation_id, str)
        mock_logger.error.assert_called_once()

    @patch('src.utils.error_handler.logger')
    def test_log_warning(self, mock_logger):
        correlation_id = self.error_handler.log_warning("Test warning")
        
        self.assertIsInstance(correlation_id, str)
        mock_logger.warning.assert_called_once()


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.circuit_breaker = CircuitBreaker(threshold_pct=10)

    def test_price_move_threshold(self):
        # First check - no previous price, should pass
        result = self.circuit_breaker.check(100.0, 100.0, 1000)
        self.assertFalse(result)
        
        # Check with large price move - should trip
        result = self.circuit_breaker.check(120.0, 100.0, 1000)
        self.assertTrue(result)
        self.assertTrue(self.circuit_breaker.tripped)

    def test_zero_liquidity(self):
        # Check with zero liquidity - should trip
        result = self.circuit_breaker.check(100.0, 100.0, 0)
        self.assertTrue(result)
        self.assertTrue(self.circuit_breaker.tripped)

    def test_reset(self):
        # Trip the breaker
        self.circuit_breaker.check(120.0, 100.0, 1000)
        self.assertTrue(self.circuit_breaker.tripped)
        
        # Reset
        self.circuit_breaker.reset()
        self.assertFalse(self.circuit_breaker.tripped)


class TestOrderBookRecovery(unittest.TestCase):
    def setUp(self):
        self.mock_order_book = MagicMock()
        self.mock_order_book.symbol = "TEST"

    @patch('builtins.open', create=True)
    @patch('src.engine.recovery.pickle')
    def test_snapshot_success(self, mock_pickle, mock_open):
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        result = OrderBookRecovery.snapshot(self.mock_order_book)
        
        self.assertTrue(result)
        mock_pickle.dump.assert_called_once()

    @patch('src.engine.recovery.os.path.exists')
    @patch('builtins.open', create=True)
    @patch('src.engine.recovery.pickle')
    def test_restore_success(self, mock_pickle, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_pickle.load.return_value = self.mock_order_book
        
        result = OrderBookRecovery.restore()
        
        self.assertEqual(result, self.mock_order_book)
        mock_pickle.load.assert_called_once()

    @patch('src.engine.recovery.os.path.exists')
    def test_restore_no_snapshot(self, mock_exists):
        mock_exists.return_value = False
        
        result = OrderBookRecovery.restore()
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
