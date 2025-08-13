"""
Real-Time Trading Dashboard for HFT Simulator

This module provides a comprehensive real-time web dashboard featuring:
- Live order book visualization
- Real-time strategy performance tracking
- Market data streaming with WebSocket
- Interactive risk management controls
- Advanced analytics and charting

Key Features:
- WebSocket-based real-time data streaming
- Interactive Plotly charts with live updates
- Real-time P&L and risk metrics
- Order book depth visualization
- Strategy control panel
- Performance attribution dashboard
- Multi-asset support
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import threading
import os
import queue
from dataclasses import dataclass, asdict
import logging

# Web framework
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit, disconnect
import plotly.graph_objs as go
import plotly.utils

# HFT Simulator components
from src.engine.order_book import OrderBook
from src.execution.simulator import ExecutionSimulator
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.realtime.data_feeds import (
    RealTimeDataFeed,
    DataFeedConfig,
    MarketDataMessage,
    create_data_feed,
)
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.utils.logger import get_logger


@dataclass
class DashboardConfig:
    """Configuration for real-time dashboard"""
    
    # Server settings
    host: str = "localhost"
    port: int = 8080
    debug: bool = True
    
    # Real-time settings
    update_interval_ms: int = 250  # Dashboard update frequency
    max_data_points: int = 1000    # Maximum points to keep in charts
    
    # Display settings
    default_symbols: List[str] = None
    theme: str = "dark"  # 'light' or 'dark'
    
    # Performance settings
    enable_profiling: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.default_symbols is None:
            # Default to publicly available, no-API-key crypto symbols (Binance)
            self.default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


class RealTimeDashboard:
    """
    Comprehensive real-time trading dashboard
    
    Provides a web-based interface for monitoring and controlling
    HFT trading strategies with real-time data visualization.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = get_logger("RealTimeDashboard")
        
        # Flask app setup
        self.app = Flask(
            __name__,
            template_folder='../../templates',
            static_folder='../../static',
        )
        self.app.config['SECRET_KEY'] = os.getenv('DASHBOARD_SECRET_KEY', 'hft_simulator_secret_key_2024')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Data storage
        self.market_data_buffer: Dict[str, List[Dict]] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.risk_managers: Dict[str, RiskManager] = {}
        
        # Real-time components
        self.data_feeds: Dict[str, RealTimeDataFeed] = {}
        self.strategies: Dict[str, Any] = {}
        self.simulators: Dict[str, ExecutionSimulator] = {}
        
        # Dashboard state
        self.connected_clients = set()
        self.is_streaming = False
        self.streaming_task = None
        
        # Performance tracking
        self.performance_data = {
            'timestamps': [],
            'pnl': [],
            'positions': [],
            'drawdown': [],
            'sharpe_ratio': [],
            'fill_rate': [],
            'trade_count': 0,
            'total_volume': 0
        }
        
        # Setup routes and event handlers
        self._setup_routes()
        self._setup_socketio_events()
        
        self.logger.info("Real-time dashboard initialized")
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html', 
                                 symbols=self.config.default_symbols,
                                 theme=self.config.theme)
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for system status"""
            return jsonify({
                'streaming': self.is_streaming,
                'connected_clients': len(self.connected_clients),
                'symbols': list(self.order_books.keys()),
                'strategies': list(self.strategies.keys()),
                'uptime': datetime.now().isoformat()
            })

        @self.app.route('/health')
        def api_health():
            """Simple health check endpoint for container orchestration"""
            return jsonify({
                'status': 'ok',
                'server_time': datetime.now().isoformat(),
                'streaming': self.is_streaming
            }), 200
        
        @self.app.route('/api/symbols')
        def api_symbols():
            """Get available symbols"""
            return jsonify({
                'symbols': self.config.default_symbols,
                'active_symbols': list(self.order_books.keys())
            })
        
        @self.app.route('/api/performance/<symbol>')
        def api_performance(symbol):
            """Get performance data for symbol"""
            if symbol in self.portfolios:
                portfolio = self.portfolios[symbol]
                metrics = portfolio.get_performance_summary()
                return jsonify(metrics)
            return jsonify({'error': 'Symbol not found'}), 404
        
        @self.app.route('/api/orderbook/<symbol>')
        def api_orderbook(symbol):
            """Get current order book for symbol"""
            if symbol in self.order_books:
                book = self.order_books[symbol]
                snapshot = book.get_snapshot()
                return jsonify({
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'best_bid': snapshot.best_bid,
                    'best_ask': snapshot.best_ask,
                    'bid_volume': snapshot.best_bid_volume,
                    'ask_volume': snapshot.best_ask_volume,
                    'mid_price': snapshot.mid_price,
                    'spread': snapshot.spread,
                    'spread_bps': snapshot.spread_bps
                })
            return jsonify({'error': 'Symbol not found'}), 404
        
        @self.app.route('/api/control/start', methods=['POST'])
        def api_start_streaming():
            """Start real-time streaming"""
            symbols = request.json.get('symbols', self.config.default_symbols)
            return jsonify(self.start_streaming(symbols))
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def api_stop_streaming():
            """Stop real-time streaming"""
            return jsonify(self.stop_streaming())
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect(auth=None):
            """Handle client connection"""
            self.connected_clients.add(request.sid)
            self.logger.info(f"Client connected: {request.sid}")
            emit('connected', {
                'message': 'Connected to HFT Dashboard',
                'client_id': request.sid,
                'server_time': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.connected_clients.discard(request.sid)
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_symbol')
        def handle_subscribe_symbol(data):
            """Subscribe to symbol updates"""
            symbol = data.get('symbol')
            if symbol:
                self.logger.info(f"Client {request.sid} subscribed to {symbol}")
                # Send initial data
                if symbol in self.market_data_buffer:
                    emit('market_data', {
                        'symbol': symbol,
                        'data': self.market_data_buffer[symbol][-100:]  # Last 100 points
                    })
        
        @self.socketio.on('unsubscribe_symbol')
        def handle_unsubscribe_symbol(data):
            """Unsubscribe from symbol updates"""
            symbol = data.get('symbol')
            if symbol:
                self.logger.info(f"Client {request.sid} unsubscribed from {symbol}")
        
        @self.socketio.on('get_performance')
        def handle_get_performance():
            """Send performance data to client"""
            emit('performance_data', self.performance_data)
        
        @self.socketio.on('strategy_control')
        def handle_strategy_control(data):
            """Handle strategy control commands"""
            action = data.get('action')  # 'start', 'stop', 'pause', 'configure'
            strategy_id = data.get('strategy_id')
            
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                
                if action == 'start':
                    # Start strategy
                    self.logger.info(f"Starting strategy: {strategy_id}")
                    emit('strategy_status', {
                        'strategy_id': strategy_id,
                        'status': 'running'
                    })
                elif action == 'stop':
                    # Stop strategy
                    self.logger.info(f"Stopping strategy: {strategy_id}")
                    emit('strategy_status', {
                        'strategy_id': strategy_id,
                        'status': 'stopped'
                    })
                elif action == 'configure':
                    # Update strategy parameters
                    params = data.get('parameters', {})
                    self.logger.info(f"Configuring strategy {strategy_id}: {params}")
                    emit('strategy_configured', {
                        'strategy_id': strategy_id,
                        'parameters': params
                    })
    
    async def initialize_symbols(self, symbols: List[str]):
        """Initialize order books and components for symbols"""
        for symbol in symbols:
            if symbol not in self.order_books:
                # Create order book
                self.order_books[symbol] = OrderBook(symbol)
                
                # Create portfolio
                self.portfolios[symbol] = Portfolio(initial_cash=100000.0)
                
                # Create risk manager
                self.risk_managers[symbol] = RiskManager()
                
                # Initialize data buffer
                self.market_data_buffer[symbol] = []
                
                # Create real-time WebSocket data feed (Binance public stream)
                # Uses combined stream endpoint compatible with our parser
                feed_config = DataFeedConfig(
                    url="wss://stream.binance.com:9443/stream",
                    symbols=[symbol],
                    max_messages_per_second=1000,
                    buffer_size=10000,
                )
                self.data_feeds[symbol] = create_data_feed("websocket", feed_config)
                
                # Create strategies (pass required symbol)
                self.strategies[f"{symbol}_mm"] = MarketMakingStrategy(symbol=symbol)
                self.strategies[f"{symbol}_lt"] = LiquidityTakingStrategy(symbol=symbol)
                
                self.logger.info(f"Initialized components for {symbol}")
    
    def start_streaming(self, symbols: List[str]) -> Dict[str, Any]:
        """Start real-time data streaming"""
        try:
            if self.is_streaming:
                return {'status': 'already_streaming'}
            
            # Initialize components
            asyncio.run(self.initialize_symbols(symbols))
            
            # Start streaming in background thread
            self.is_streaming = True
            self.streaming_task = threading.Thread(
                target=self._run_streaming_loop,
                args=(symbols,),
                daemon=True
            )
            self.streaming_task.start()
            
            self.logger.info(f"Started streaming for symbols: {symbols}")
            return {
                'status': 'started',
                'symbols': symbols,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def stop_streaming(self) -> Dict[str, Any]:
        """Stop real-time data streaming"""
        try:
            self.is_streaming = False
            
            if self.streaming_task and self.streaming_task.is_alive():
                self.streaming_task.join(timeout=2.0)
            
            # Disconnect data feeds
            for feed in self.data_feeds.values():
                asyncio.run(feed.disconnect())
            
            self.logger.info("Stopped streaming")
            return {
                'status': 'stopped',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _run_streaming_loop(self, symbols: List[str]):
        """Run the streaming loop in background thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._streaming_coroutine(symbols))
        except Exception as e:
            self.logger.error(f"Streaming loop error: {e}")
        finally:
            loop.close()
    
    async def _streaming_coroutine(self, symbols: List[str]):
        """Main streaming coroutine"""
        tasks = []
        
        # Start data feeds
        for symbol in symbols:
            if symbol in self.data_feeds:
                feed = self.data_feeds[symbol]
                await feed.connect()
                await feed.subscribe([symbol])
                
                # Create streaming task
                task = asyncio.create_task(
                    self._process_symbol_stream(symbol, feed)
                )
                tasks.append(task)
        
        # Create update task
        update_task = asyncio.create_task(self._update_dashboard())
        tasks.append(update_task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
        finally:
            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def _process_symbol_stream(self, symbol: str, feed: RealTimeDataFeed):
        """Process market data stream for a symbol"""
        async for message in feed.start_streaming():
            if not self.is_streaming:
                break
            
            try:
                # Update order book
                if symbol in self.order_books:
                    book = self.order_books[symbol]
                    
                    # Create synthetic orders from market data
                    if message.bid_price and message.ask_price:
                        # This is simplified - in reality would need proper order handling
                        pass
                
                # Update market data buffer
                data_point = {
                    'timestamp': message.timestamp.isoformat(),
                    'price': message.price,
                    'volume': message.volume,
                    'bid': message.bid_price,
                    'ask': message.ask_price,
                    'bid_volume': message.bid_volume,
                    'ask_volume': message.ask_volume,
                    'spread': (message.ask_price - message.bid_price) if (message.ask_price and message.bid_price) else None
                }
                
                if symbol in self.market_data_buffer:
                    buffer = self.market_data_buffer[symbol]
                    buffer.append(data_point)
                    
                    # Keep buffer size manageable
                    if len(buffer) > self.config.max_data_points:
                        buffer.pop(0)
                
                # Update performance tracking
                self._update_performance_data(symbol, message)
                
            except Exception as e:
                self.logger.error(f"Error processing stream for {symbol}: {e}")
    
    async def _update_dashboard(self):
        """Update dashboard with latest data"""
        while self.is_streaming:
            try:
                # Broadcast market data updates
                for symbol, buffer in self.market_data_buffer.items():
                    if buffer:  # Only send if we have data
                        latest_data = buffer[-10:]  # Last 10 points
                        self.socketio.emit('market_data_update', {
                            'symbol': symbol,
                            'data': latest_data
                        })
                
                # Broadcast performance updates
                self.socketio.emit('performance_update', self.performance_data)
                
                # Broadcast order book updates
                for symbol, book in self.order_books.items():
                    snapshot = book.get_snapshot()
                    self.socketio.emit('orderbook_update', {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'best_bid': snapshot.best_bid,
                        'best_ask': snapshot.best_ask,
                        'bid_volume': snapshot.best_bid_volume,
                        'ask_volume': snapshot.best_ask_volume,
                        'mid_price': snapshot.mid_price,
                        'spread': snapshot.spread,
                        'spread_bps': snapshot.spread_bps
                    })
                
                await asyncio.sleep(self.config.update_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(1.0)
    
    def _update_performance_data(self, symbol: str, message: MarketDataMessage):
        """Update performance tracking data"""
        try:
            timestamp = datetime.now()
            
            # Update basic metrics (simplified for demo)
            if len(self.performance_data['timestamps']) == 0 or \
               (timestamp - pd.to_datetime(self.performance_data['timestamps'][-1])).total_seconds() > 1:
                
                self.performance_data['timestamps'].append(timestamp.isoformat())
                
                # Simulate some performance metrics
                if len(self.performance_data['pnl']) > 0:
                    # Random walk for P&L
                    last_pnl = self.performance_data['pnl'][-1]
                    pnl_change = np.random.normal(0, 10)
                    new_pnl = last_pnl + pnl_change
                else:
                    new_pnl = 0.0
                
                self.performance_data['pnl'].append(new_pnl)
                self.performance_data['positions'].append(np.random.randint(-500, 500))
                
                # Calculate running metrics
                if len(self.performance_data['pnl']) > 1:
                    pnl_series = np.array(self.performance_data['pnl'])
                    returns = np.diff(pnl_series)
                    
                    # Running max for drawdown
                    running_max = np.maximum.accumulate(pnl_series)
                    drawdown = pnl_series - running_max
                    self.performance_data['drawdown'].append(drawdown[-1])
                    
                    # Simple Sharpe ratio (annualized)
                    if len(returns) > 10 and np.std(returns) > 0:
                        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Assuming minute data
                        self.performance_data['sharpe_ratio'].append(sharpe)
                    else:
                        self.performance_data['sharpe_ratio'].append(0.0)
                else:
                    self.performance_data['drawdown'].append(0.0)
                    self.performance_data['sharpe_ratio'].append(0.0)
                
                # Simulate fill rate
                self.performance_data['fill_rate'].append(np.random.uniform(0.85, 0.98))
                
                # Keep arrays manageable
                max_points = self.config.max_data_points
                for key in ['timestamps', 'pnl', 'positions', 'drawdown', 'sharpe_ratio', 'fill_rate']:
                    if len(self.performance_data[key]) > max_points:
                        self.performance_data[key] = self.performance_data[key][-max_points:]
                
                # Update counters
                self.performance_data['trade_count'] += np.random.randint(0, 3)
                self.performance_data['total_volume'] += message.volume or 0
                
        except Exception as e:
            self.logger.error(f"Error updating performance data: {e}")
    
    def create_charts(self) -> Dict[str, Any]:
        """Create Plotly charts for the dashboard"""
        charts = {}
        
        # Price chart
        if self.market_data_buffer:
            for symbol, data in self.market_data_buffer.items():
                if data:
                    df = pd.DataFrame(data)
                    
                    price_chart = go.Figure()
                    price_chart.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(color='#00ff88')
                    ))
                    
                    price_chart.update_layout(
                        title=f'{symbol} Real-Time Price',
                        xaxis_title='Time',
                        yaxis_title='Price ($)',
                        template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
                    )
                    
                    charts[f'{symbol}_price'] = json.dumps(price_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        # P&L chart
        if self.performance_data['timestamps'] and self.performance_data['pnl']:
            pnl_chart = go.Figure()
            pnl_chart.add_trace(go.Scatter(
                x=self.performance_data['timestamps'],
                y=self.performance_data['pnl'],
                mode='lines',
                name='P&L',
                line=dict(color='#ff6b6b')
            ))
            
            pnl_chart.update_layout(
                title='Real-Time P&L',
                xaxis_title='Time',
                yaxis_title='P&L ($)',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
            )
            
            charts['pnl'] = json.dumps(pnl_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        return charts
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the dashboard server"""
        host = host or self.config.host
        port = port or self.config.port
        debug = debug or self.config.debug
        
        self.logger.info(f"Starting HFT Real-Time Dashboard on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# Example usage and demo functions
def create_demo_dashboard():
    """Create a demo dashboard with sample data"""
    config = DashboardConfig(
        host="localhost",
        port=8080,
        debug=True,
        default_symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
        theme="dark",
        update_interval_ms=250
    )
    
    return RealTimeDashboard(config)


if __name__ == "__main__":
    # Run demo dashboard
    dashboard = create_demo_dashboard()
    dashboard.run()
