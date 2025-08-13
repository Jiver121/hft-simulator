"""
Order book state recovery: snapshotting/restoring mechanism, for corruption/failure scenarios.
"""
import copy
import os
import pickle
from src.utils.error_handler import ErrorHandler

SNAPSHOT_PATH = 'order_book_snapshot.pkl'

class OrderBookRecovery:
    @staticmethod
    def snapshot(order_book, correlation_id=None):
        try:
            with open(SNAPSHOT_PATH, 'wb') as f:
                pickle.dump(copy.deepcopy(order_book), f)
            ErrorHandler.log_info('Order book snapshot saved.', correlation_id)
            return True
        except Exception as e:
            ErrorHandler.log_error(e, correlation_id)
            return False

    @staticmethod
    def restore(correlation_id=None):
        try:
            if not os.path.exists(SNAPSHOT_PATH):
                ErrorHandler.log_warning('No snapshot found.', correlation_id)
                return None
            with open(SNAPSHOT_PATH, 'rb') as f:
                order_book = pickle.load(f)
            ErrorHandler.log_info('Order book snapshot restored.', correlation_id)
            return order_book
        except Exception as e:
            ErrorHandler.log_error(e, correlation_id)
            return None

