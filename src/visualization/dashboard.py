"""
Real-time Dashboard for HFT Simulator

This module provides a comprehensive real-time dashboard for monitoring
HFT trading activities, performance metrics, risk indicators, and market data.

Educational Notes:
- Real-time monitoring is essential for HFT operations
- Dashboards provide at-a-glance views of critical metrics
- Interactive dashboards enable quick decision making
- Multiple views serve different stakeholder needs
- Alert systems help identify issues quickly
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import json
from enum import Enum

from src.utils.logger import get_logger
from src.visualization.charts import ChartGenerator, ChartType, ChartTheme
from src.performance.metrics import PerformanceAnalyzer, PerformanceMetrics
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.engine.order_book import OrderBook
from src.engine.market_data import BookSnapshot


class DashboardMode(Enum):
    """Dashboard display modes"""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    RISK = "risk"
    TRADING = "trading"
    MARKET = "market"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DashboardConfig:
    """Configuration for dashboard display and behavior"""
    
    # Display settings
    theme: ChartTheme = ChartTheme.PROFESSIONAL
    refresh_interval: int = 1  # seconds
    max_data_points: int = 1000
    
    # Layout settings
    show_sidebar: bool = True
    show_alerts: bool = True
    show_performance: bool = True
    show_risk: bool = True
    show_trading: bool = True
    show_market: bool = True
    
    # Alert settings
    enable_alerts: bool = True
    alert_sound: bool = False
    email_alerts: bool = False
    
    # Data retention
    history_days: int = 30
    snapshot_interval: int = 60  # seconds


@dataclass
class Alert:
    """Dashboard alert"""
    
    timestamp: pd.Timestamp
    level: AlertLevel
    title: str
    message: str
    source: str
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'acknowledged': self.acknowledged
        }


class Dashboard:
    """
    Comprehensive real-time dashboard for HFT simulator
    
    This class provides a complete dashboard interface for monitoring
    all aspects of HFT trading operations including performance,
    risk, market data, and trading activity.
    
    Key Features:
    - Real-time data updates
    - Multiple dashboard views
    - Interactive charts and metrics
    - Alert system with notifications
    - Historical data tracking
    - Export and reporting capabilities
    - Customizable layouts
    
    Educational Notes:
    - Dashboards are the control center for trading operations
    - Real-time monitoring prevents issues from escalating
    - Different views serve different purposes and users
    - Alert systems ensure critical issues are noticed immediately
    - Historical tracking enables performance analysis and improvement
    """
    
    def __init__(self, 
                 config: DashboardConfig = None,
                 portfolio: Portfolio = None,
                 risk_manager: RiskManager = None):
        """
        Initialize dashboard
        
        Args:
            config: Dashboard configuration
            portfolio: Portfolio to monitor
            risk_manager: Risk manager to monitor
        """
        self.config = config or DashboardConfig()
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        
        self.logger = get_logger(__name__)
        
        # Chart generator
        self.chart_generator = ChartGenerator(theme=self.config.theme)
        
        # Data storage
        self.market_data: Dict[str, List[BookSnapshot]] = {}
        self.performance_history: List[Tuple[pd.Timestamp, Dict[str, Any]]] = []
        self.risk_history: List[Tuple[pd.Timestamp, Dict[str, Any]]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Dashboard state
        self.current_mode = DashboardMode.OVERVIEW
        self.is_running = False
        self.last_update = pd.Timestamp.now()
        
        # Threading for real-time updates
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Metrics cache
        self._cached_metrics: Dict[str, Any] = {}
        self._cache_timestamp: Optional[pd.Timestamp] = None
        self._cache_ttl = timedelta(seconds=5)  # Cache for 5 seconds
        
        self.logger.info("Dashboard initialized")
    
    def start(self) -> None:
        """Start real-time dashboard updates"""
        if self.is_running:
            self.logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Dashboard started")
    
    def stop(self) -> None:
        """Stop dashboard updates"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        self.logger.info("Dashboard stopped")
    
    def _update_loop(self) -> None:
        """Main update loop for real-time data"""
        while not self.stop_event.is_set():
            try:
                self._update_data()
                self._check_alerts()
                self.last_update = pd.Timestamp.now()
                
                # Wait for next update
                self.stop_event.wait(self.config.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def _update_data(self) -> None:
        """Update all dashboard data"""
        current_time = pd.Timestamp.now()
        
        # Update performance data
        if self.portfolio:
            performance_data = {
                'total_value': self.portfolio.total_value,
                'total_pnl': self.portfolio.total_pnl,
                'cash': self.portfolio.current_cash,
                'active_positions': len(self.portfolio.get_active_positions()),
                'trade_count': self.portfolio.trade_count
            }
            
            self.performance_history.append((current_time, performance_data))
            
            # Limit history size
            if len(self.performance_history) > self.config.max_data_points:
                self.performance_history = self.performance_history[-self.config.max_data_points//2:]
        
        # Update risk data
        if self.risk_manager:
            risk_data = self.risk_manager.get_risk_summary()
            self.risk_history.append((current_time, risk_data))
            
            # Limit history size
            if len(self.risk_history) > self.config.max_data_points:
                self.risk_history = self.risk_history[-self.config.max_data_points//2:]
        
        # Clear cache
        self._cached_metrics.clear()
        self._cache_timestamp = None
    
    def _check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self.config.enable_alerts:
            return
        
        current_time = pd.Timestamp.now()
        
        # Check portfolio alerts
        if self.portfolio:
            # Large drawdown alert
            if hasattr(self.portfolio, 'performance_analyzer'):
                metrics = self.portfolio.performance_analyzer.calculate_metrics()
                if metrics.max_drawdown > 0.1:  # 10% drawdown
                    self._create_alert(
                        AlertLevel.WARNING,
                        "Large Drawdown",
                        f"Portfolio drawdown: {metrics.max_drawdown:.2%}",
                        "portfolio"
                    )
        
        # Check risk alerts
        if self.risk_manager:
            risk_summary = self.risk_manager.get_risk_summary()
            
            # High volatility alert
            if risk_summary.get('portfolio_volatility', 0) > 0.3:  # 30% volatility
                self._create_alert(
                    AlertLevel.WARNING,
                    "High Volatility",
                    f"Portfolio volatility: {risk_summary['portfolio_volatility']:.2%}",
                    "risk"
                )
            
            # Recent risk events
            recent_events = self.risk_manager.get_risk_events(hours=1)
            if len(recent_events) > 5:  # More than 5 events in last hour
                self._create_alert(
                    AlertLevel.ERROR,
                    "Multiple Risk Events",
                    f"{len(recent_events)} risk events in the last hour",
                    "risk"
                )
    
    def _create_alert(self, 
                     level: AlertLevel, 
                     title: str, 
                     message: str, 
                     source: str) -> None:
        """Create a new alert"""
        # Check if similar alert already exists (avoid spam)
        recent_alerts = [a for a in self.alerts 
                        if a.timestamp > pd.Timestamp.now() - timedelta(minutes=5)]
        
        for alert in recent_alerts:
            if alert.title == title and alert.source == source:
                return  # Don't create duplicate alert
        
        alert = Alert(
            timestamp=pd.Timestamp.now(),
            level=level,
            title=title,
            message=message,
            source=source
        )
        
        self.alerts.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.info(f"Alert created: {title} - {message}")
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_overview_data(self) -> Dict[str, Any]:
        """Get overview dashboard data"""
        data = {
            'timestamp': self.last_update.isoformat(),
            'status': 'running' if self.is_running else 'stopped',
            'alerts': {
                'total': len(self.alerts),
                'unacknowledged': len([a for a in self.alerts if not a.acknowledged]),
                'critical': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
                'recent': [a.to_dict() for a in self.alerts[-5:]]
            }
        }
        
        # Add portfolio data
        if self.portfolio:
            portfolio_summary = self.portfolio.get_portfolio_summary()
            data['portfolio'] = portfolio_summary
        
        # Add risk data
        if self.risk_manager:
            risk_summary = self.risk_manager.get_risk_summary()
            data['risk'] = risk_summary
        
        return data
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        if not self.portfolio:
            return {'error': 'No portfolio data available'}
        
        # Get cached metrics if available
        if (self._cache_timestamp and 
            pd.Timestamp.now() - self._cache_timestamp < self._cache_ttl and
            'performance' in self._cached_metrics):
            return self._cached_metrics['performance']
        
        # Calculate fresh metrics
        metrics = self.portfolio.calculate_performance_metrics()
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Get historical performance data
        if self.performance_history:
            df = pd.DataFrame([data for _, data in self.performance_history])
            timestamps = [ts for ts, _ in self.performance_history]
            df.index = timestamps
            
            performance_data = {
                'summary': portfolio_summary,
                'metrics': metrics.to_dict(),
                'history': {
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'values': df['total_value'].tolist(),
                    'pnl': df['total_pnl'].tolist(),
                    'positions': df['active_positions'].tolist(),
                    'trades': df['trade_count'].tolist()
                },
                'positions': self.portfolio.get_position_breakdown().to_dict('records')
            }
        else:
            performance_data = {
                'summary': portfolio_summary,
                'metrics': metrics.to_dict(),
                'history': {'timestamps': [], 'values': [], 'pnl': [], 'positions': [], 'trades': []},
                'positions': []
            }
        
        # Cache the result
        self._cached_metrics['performance'] = performance_data
        self._cache_timestamp = pd.Timestamp.now()
        
        return performance_data
    
    def get_risk_data(self) -> Dict[str, Any]:
        """Get risk dashboard data"""
        if not self.risk_manager:
            return {'error': 'No risk manager available'}
        
        # Get cached metrics if available
        if (self._cache_timestamp and 
            pd.Timestamp.now() - self._cache_timestamp < self._cache_ttl and
            'risk' in self._cached_metrics):
            return self._cached_metrics['risk']
        
        # Calculate fresh metrics
        risk_summary = self.risk_manager.get_risk_summary()
        position_breakdown = self.risk_manager.get_position_risk_breakdown()
        recent_events = self.risk_manager.get_risk_events(hours=24)
        
        # Get historical risk data
        if self.risk_history:
            timestamps = [ts for ts, _ in self.risk_history]
            risk_values = [data for _, data in self.risk_history]
            
            risk_data = {
                'summary': risk_summary,
                'position_breakdown': position_breakdown.to_dict('records'),
                'recent_events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'risk_type': event.risk_type.value,
                        'level': event.risk_level.value,
                        'description': event.description,
                        'current_value': event.current_value,
                        'threshold': event.threshold
                    }
                    for event in recent_events
                ],
                'history': {
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'drawdown': [data.get('current_drawdown', 0) for data in risk_values],
                    'volatility': [data.get('portfolio_volatility', 0) for data in risk_values],
                    'var': [data.get('portfolio_var', 0) for data in risk_values]
                }
            }
        else:
            risk_data = {
                'summary': risk_summary,
                'position_breakdown': [],
                'recent_events': [],
                'history': {'timestamps': [], 'drawdown': [], 'volatility': [], 'var': []}
            }
        
        # Cache the result
        self._cached_metrics['risk'] = risk_data
        self._cache_timestamp = pd.Timestamp.now()
        
        return risk_data
    
    def get_trading_data(self) -> Dict[str, Any]:
        """Get trading dashboard data"""
        if not self.portfolio:
            return {'error': 'No portfolio data available'}
        
        # Get recent trades
        recent_trades = self.trade_history[-100:] if self.trade_history else []
        
        # Calculate trading statistics
        if recent_trades:
            df = pd.DataFrame(recent_trades)
            
            trading_stats = {
                'total_trades': len(recent_trades),
                'avg_trade_size': df['volume'].mean() if 'volume' in df.columns else 0,
                'total_volume': df['volume'].sum() if 'volume' in df.columns else 0,
                'win_rate': len(df[df.get('pnl', 0) > 0]) / len(df) * 100 if 'pnl' in df.columns else 0,
                'avg_pnl': df['pnl'].mean() if 'pnl' in df.columns else 0
            }
            
            # Trading activity by hour
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                hourly_activity = df.groupby('hour').size().to_dict()
            else:
                hourly_activity = {}
        else:
            trading_stats = {
                'total_trades': 0,
                'avg_trade_size': 0,
                'total_volume': 0,
                'win_rate': 0,
                'avg_pnl': 0
            }
            hourly_activity = {}
        
        trading_data = {
            'statistics': trading_stats,
            'recent_trades': recent_trades,
            'hourly_activity': hourly_activity,
            'active_orders': []  # Would need order manager integration
        }
        
        return trading_data
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get market data dashboard"""
        market_data = {
            'symbols': list(self.market_data.keys()),
            'snapshots': {}
        }
        
        for symbol, snapshots in self.market_data.items():
            if snapshots:
                latest = snapshots[-1]
                market_data['snapshots'][symbol] = {
                    'timestamp': latest.timestamp.isoformat(),
                    'best_bid': latest.best_bid,
                    'best_ask': latest.best_ask,
                    'spread': (latest.best_ask[0] - latest.best_bid[0]) if latest.best_bid and latest.best_ask else None,
                    'mid_price': (latest.best_ask[0] + latest.best_bid[0]) / 2 if latest.best_bid and latest.best_ask else None
                }
        
        return market_data
    
    def add_market_snapshot(self, symbol: str, snapshot: BookSnapshot) -> None:
        """Add market data snapshot"""
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        
        self.market_data[symbol].append(snapshot)
        
        # Limit history size
        if len(self.market_data[symbol]) > self.config.max_data_points:
            self.market_data[symbol] = self.market_data[symbol][-self.config.max_data_points//2:]
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add trade to history"""
        self.trade_history.append(trade_data)
        
        # Limit history size
        if len(self.trade_history) > self.config.max_data_points:
            self.trade_history = self.trade_history[-self.config.max_data_points//2:]
    
    def set_mode(self, mode: DashboardMode) -> None:
        """Set dashboard display mode"""
        self.current_mode = mode
        self.logger.info(f"Dashboard mode changed to {mode.value}")
    
    def acknowledge_alert(self, alert_index: int) -> None:
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
            self.logger.info(f"Alert acknowledged: {self.alerts[alert_index].title}")
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self.alerts.clear()
        self.logger.info("All alerts cleared")
    
    def export_data(self, 
                   data_type: str = 'all',
                   format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export dashboard data
        
        Args:
            data_type: Type of data to export ('all', 'performance', 'risk', 'trading')
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data
        """
        
        if data_type == 'all':
            export_data = {
                'overview': self.get_overview_data(),
                'performance': self.get_performance_data(),
                'risk': self.get_risk_data(),
                'trading': self.get_trading_data(),
                'market': self.get_market_data()
            }
        elif data_type == 'performance':
            export_data = self.get_performance_data()
        elif data_type == 'risk':
            export_data = self.get_risk_data()
        elif data_type == 'trading':
            export_data = self.get_trading_data()
        elif data_type == 'market':
            export_data = self.get_market_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        elif format == 'dict':
            return export_data
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status information"""
        return {
            'is_running': self.is_running,
            'current_mode': self.current_mode.value,
            'last_update': self.last_update.isoformat(),
            'data_points': {
                'performance': len(self.performance_history),
                'risk': len(self.risk_history),
                'trades': len(self.trade_history),
                'market_symbols': len(self.market_data)
            },
            'alerts': {
                'total': len(self.alerts),
                'unacknowledged': len([a for a in self.alerts if not a.acknowledged])
            },
            'config': {
                'refresh_interval': self.config.refresh_interval,
                'max_data_points': self.config.max_data_points,
                'theme': self.config.theme.value
            }
        }


# Utility functions
def create_dashboard(portfolio: Portfolio = None,
                    risk_manager: RiskManager = None,
                    config: DashboardConfig = None) -> Dashboard:
    """Create a dashboard with default settings"""
    return Dashboard(config=config, portfolio=portfolio, risk_manager=risk_manager)


def create_alert_handler(dashboard: Dashboard) -> Callable[[Alert], None]:
    """Create a simple alert handler that logs alerts"""
    def handle_alert(alert: Alert):
        logger = get_logger("alert_handler")
        logger.info(f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.message}")
    
    return handle_alert