"""
Centralized error handling and structured logging with correlation IDs.
"""
import uuid
import logging

logger = logging.getLogger("hft_simulator")
logger.setLevel(logging.INFO)

class ErrorHandler:
    @staticmethod
    def log_error(error, correlation_id=None, extra_info=None):
        corr_id = correlation_id or str(uuid.uuid4())
        logger.error(f"[CorrID: {corr_id}] {repr(error)} | Extra: {extra_info}")
        return corr_id

    @staticmethod
    def log_warning(message, correlation_id=None):
        corr_id = correlation_id or str(uuid.uuid4())
        logger.warning(f"[CorrID: {corr_id}] {message}")
        return corr_id

    @staticmethod
    def log_info(message, correlation_id=None):
        corr_id = correlation_id or str(uuid.uuid4())
        logger.info(f"[CorrID: {corr_id}] {message}")
        return corr_id

