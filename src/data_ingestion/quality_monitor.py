"""
Data Quality Monitor

Monitors and tracks data quality metrics including latency, throughput,
data integrity, and connection stability for real-time market data feeds.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from .websocket_client import MarketDataPoint


@dataclass
class QualityMetrics:
    """Data quality metrics snapshot"""
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    # Latency metrics (milliseconds)
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_max: float = 0.0
    
    # Throughput metrics
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    # Data quality metrics
    duplicate_rate: float = 0.0
    error_rate: float = 0.0
    validation_pass_rate: float = 0.0
    
    # Connection stability
    uptime_percentage: float = 0.0
    reconnection_count: int = 0
    connection_drops: int = 0
    
    # Market data specific
    symbols_active: int = 0
    price_staleness_seconds: float = 0.0
    spread_consistency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'latency': {
                'avg_ms': self.latency_avg,
                'p50_ms': self.latency_p50,
                'p95_ms': self.latency_p95,
                'p99_ms': self.latency_p99,
                'max_ms': self.latency_max
            },
            'throughput': {
                'messages_per_second': self.messages_per_second,
                'bytes_per_second': self.bytes_per_second
            },
            'quality': {
                'duplicate_rate': self.duplicate_rate,
                'error_rate': self.error_rate,
                'validation_pass_rate': self.validation_pass_rate
            },
            'connection': {
                'uptime_percentage': self.uptime_percentage,
                'reconnection_count': self.reconnection_count,
                'connection_drops': self.connection_drops
            },
            'market_data': {
                'symbols_active': self.symbols_active,
                'price_staleness_seconds': self.price_staleness_seconds,
                'spread_consistency': self.spread_consistency
            }
        }


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring for real-time market data feeds
    """
    
    def __init__(self, monitoring_interval: float = 10.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.logger = get_logger("DataQualityMonitor")
        
        # Metrics tracking
        self.metrics_history: Deque[QualityMetrics] = deque(maxlen=history_size)
        self.current_metrics = QualityMetrics()
        
        # Data collection
        self.latency_samples: Deque[float] = deque(maxlen=10000)
        self.message_timestamps: Deque[float] = deque(maxlen=1000)
        self.error_count = 0
        self.duplicate_count = 0
        self.validation_failures = 0
        self.total_messages = 0
        
        # Connection tracking
        self.connection_start_time = time.time()
        self.total_downtime = 0.0
        self.last_connection_drop = None
        self.reconnection_count = 0
        self.connection_drops = 0
        
        # Market data tracking
        self.symbol_last_update = {}  # symbol -> timestamp
        self.symbol_spreads = {}  # symbol -> [spread_values]
        
        # Monitoring state
        self.running = False
        self.monitor_task = None
    
    async def start_monitoring(self):
        """Start quality monitoring"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting data quality monitoring")
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop quality monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
        
        self.logger.info("Stopped data quality monitoring")
    
    def record_message(self, data_point: MarketDataPoint):
        """Record a received message for quality analysis"""
        current_time = time.time()
        self.message_timestamps.append(current_time)
        self.total_messages += 1
        
        # Record latency if available
        if data_point.latency_ms is not None:
            self.latency_samples.append(data_point.latency_ms)
        
        # Track symbol updates
        if data_point.symbol:
            self.symbol_last_update[data_point.symbol] = current_time
            
            # Track spread consistency for quotes
            if (data_point.event_type in ['ticker', 'book_ticker'] and 
                data_point.best_bid and data_point.best_ask):
                spread = data_point.best_ask - data_point.best_bid
                if data_point.symbol not in self.symbol_spreads:
                    self.symbol_spreads[data_point.symbol] = deque(maxlen=100)
                self.symbol_spreads[data_point.symbol].append(spread)
    
    def record_error(self):
        """Record a data processing error"""
        self.error_count += 1
    
    def record_duplicate(self):
        """Record a duplicate message"""
        self.duplicate_count += 1
    
    def record_validation_failure(self):
        """Record a data validation failure"""
        self.validation_failures += 1
    
    def record_connection_drop(self):
        """Record a connection drop event"""
        self.connection_drops += 1
        self.last_connection_drop = time.time()
    
    def record_reconnection(self):
        """Record a reconnection attempt"""
        self.reconnection_count += 1
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._calculate_metrics()
                self._log_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _calculate_metrics(self):
        """Calculate current quality metrics"""
        current_time = time.time()
        
        # Calculate latency metrics
        if self.latency_samples:
            latency_array = np.array(list(self.latency_samples))
            self.current_metrics.latency_avg = np.mean(latency_array)
            self.current_metrics.latency_p50 = np.percentile(latency_array, 50)
            self.current_metrics.latency_p95 = np.percentile(latency_array, 95)
            self.current_metrics.latency_p99 = np.percentile(latency_array, 99)
            self.current_metrics.latency_max = np.max(latency_array)
        
        # Calculate throughput metrics
        if self.message_timestamps:
            # Messages per second over last minute
            recent_messages = [ts for ts in self.message_timestamps if current_time - ts <= 60]
            self.current_metrics.messages_per_second = len(recent_messages) / 60.0
            
            # Estimate bytes per second (rough calculation)
            avg_message_size = 500  # bytes (estimate for market data)
            self.current_metrics.bytes_per_second = self.current_metrics.messages_per_second * avg_message_size
        
        # Calculate quality metrics
        if self.total_messages > 0:
            self.current_metrics.duplicate_rate = self.duplicate_count / self.total_messages
            self.current_metrics.error_rate = self.error_count / self.total_messages
            self.current_metrics.validation_pass_rate = 1.0 - (self.validation_failures / self.total_messages)
        
        # Calculate connection stability
        total_runtime = current_time - self.connection_start_time
        if total_runtime > 0:
            self.current_metrics.uptime_percentage = (1.0 - self.total_downtime / total_runtime) * 100
        
        self.current_metrics.reconnection_count = self.reconnection_count
        self.current_metrics.connection_drops = self.connection_drops
        
        # Calculate market data metrics
        self.current_metrics.symbols_active = len(self.symbol_last_update)
        
        # Price staleness (max time since last update for any symbol)
        if self.symbol_last_update:
            staleness_times = [current_time - ts for ts in self.symbol_last_update.values()]
            self.current_metrics.price_staleness_seconds = max(staleness_times)
        
        # Spread consistency (coefficient of variation for spreads)
        spread_consistency_scores = []
        for symbol, spreads in self.symbol_spreads.items():
            if len(spreads) >= 10:  # Need enough samples
                spreads_array = np.array(list(spreads))
                if np.mean(spreads_array) > 0:
                    cv = np.std(spreads_array) / np.mean(spreads_array)
                    spread_consistency_scores.append(1.0 / (1.0 + cv))  # Inverse CV for consistency score
        
        if spread_consistency_scores:
            self.current_metrics.spread_consistency = np.mean(spread_consistency_scores)
        
        # Update timestamp
        self.current_metrics.timestamp = pd.Timestamp.now()
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
    
    def _log_metrics(self):
        """Log current metrics"""
        m = self.current_metrics
        
        # Log summary every monitoring interval
        self.logger.info(
            f"Quality metrics: "
            f"{m.messages_per_second:.1f} msg/s, "
            f"{m.latency_p95:.1f}ms P95 latency, "
            f"{m.validation_pass_rate:.2%} validation pass rate, "
            f"{m.symbols_active} symbols active"
        )
        
        # Log warnings for poor quality
        if m.latency_p95 > 1000:  # High latency
            self.logger.warning(f"High latency detected: P95 = {m.latency_p95:.1f}ms")
        
        if m.error_rate > 0.01:  # Error rate > 1%
            self.logger.warning(f"High error rate: {m.error_rate:.2%}")
        
        if m.validation_pass_rate < 0.95:  # Validation pass rate < 95%
            self.logger.warning(f"Low validation pass rate: {m.validation_pass_rate:.2%}")
        
        if m.price_staleness_seconds > 30:  # Price data older than 30 seconds
            self.logger.warning(f"Stale price data: {m.price_staleness_seconds:.1f}s since last update")
        
        if m.uptime_percentage < 99.0:  # Uptime < 99%
            self.logger.warning(f"Low uptime: {m.uptime_percentage:.1f}%")
    
    def get_current_metrics(self) -> QualityMetrics:
        """Get current quality metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[QualityMetrics]:
        """Get historical metrics"""
        history = list(self.metrics_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score (0-100)"""
        m = self.current_metrics
        
        # Weight different factors
        latency_score = max(0, 100 - m.latency_p95 / 10)  # Penalty for high latency
        error_score = max(0, 100 - m.error_rate * 1000)  # Penalty for errors
        validation_score = m.validation_pass_rate * 100
        uptime_score = m.uptime_percentage
        staleness_score = max(0, 100 - m.price_staleness_seconds / 5)  # Penalty for stale data
        
        # Weighted average
        weights = {
            'latency': 0.25,
            'error': 0.20,
            'validation': 0.20,
            'uptime': 0.20,
            'staleness': 0.15
        }
        
        quality_score = (
            latency_score * weights['latency'] +
            error_score * weights['error'] +
            validation_score * weights['validation'] +
            uptime_score * weights['uptime'] +
            staleness_score * weights['staleness']
        )
        
        return min(100, max(0, quality_score))
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        m = self.current_metrics
        quality_score = self.get_quality_score()
        
        # Determine overall status
        if quality_score >= 90:
            status = "EXCELLENT"
        elif quality_score >= 80:
            status = "GOOD"
        elif quality_score >= 70:
            status = "FAIR"
        else:
            status = "POOR"
        
        return {
            'overall': {
                'quality_score': quality_score,
                'status': status,
                'timestamp': m.timestamp.isoformat()
            },
            'performance': {
                'throughput_msg_per_sec': m.messages_per_second,
                'latency_p95_ms': m.latency_p95,
                'latency_avg_ms': m.latency_avg
            },
            'reliability': {
                'uptime_percentage': m.uptime_percentage,
                'error_rate': m.error_rate,
                'validation_pass_rate': m.validation_pass_rate,
                'connection_drops': m.connection_drops,
                'reconnections': m.reconnection_count
            },
            'data_quality': {
                'symbols_active': m.symbols_active,
                'price_staleness_seconds': m.price_staleness_seconds,
                'duplicate_rate': m.duplicate_rate,
                'spread_consistency': m.spread_consistency
            },
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving data quality"""
        recommendations = []
        m = self.current_metrics
        
        if m.latency_p95 > 500:
            recommendations.append("Consider optimizing network connection or switching to closer data center")
        
        if m.error_rate > 0.05:
            recommendations.append("Review error handling and data validation logic")
        
        if m.validation_pass_rate < 0.95:
            recommendations.append("Check data source quality and validation rules")
        
        if m.connection_drops > 5:
            recommendations.append("Investigate connection stability and implement better reconnection logic")
        
        if m.price_staleness_seconds > 60:
            recommendations.append("Monitor data source health and subscription status")
        
        if m.messages_per_second < 1.0:
            recommendations.append("Check if data subscriptions are active and functioning")
        
        if not recommendations:
            recommendations.append("Data quality is good - maintain current configuration")
        
        return recommendations
