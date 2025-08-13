"""
WebSocket reconnection utility with exponential backoff and error logging.
"""
import asyncio
import time
import random
from typing import Callable, Optional, Dict, Any
from src.utils.logger import get_logger


class WebSocketReconnectHandler:
    def __init__(self, 
                 max_attempts: int = 10,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.attempt_count = 0
        self.logger = get_logger("WebSocketReconnectHandler")
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        # Add jitter (±20%)
        jitter = delay * 0.2 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
    
    def should_attempt_reconnect(self) -> bool:
        """Check if reconnection should be attempted"""
        return self.attempt_count < self.max_attempts
    
    def record_attempt(self):
        """Record a reconnection attempt"""
        self.attempt_count += 1
        self.logger.info(f"Reconnection attempt {self.attempt_count}/{self.max_attempts}")
    
    def reset(self):
        """Reset reconnection state after successful connection"""
        self.attempt_count = 0
        self.logger.info("Reconnection state reset - connection stable")
