"""
Web-based Dashboard Interface for HFT Simulator

This module provides a web-based interface for the HFT simulator dashboard,
enabling remote monitoring and control through a browser interface.

Educational Notes:
- Web interfaces enable remote monitoring and collaboration
- Real-time updates through WebSocket connections
- RESTful APIs provide programmatic access to data
- Interactive web dashboards are user-friendly and accessible
- Modern web technologies enable rich data visualization
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

from src.utils.logger import get_logger
from src.visualization.dashboard import Dashboard, DashboardConfig, DashboardMode
from src.visualization.charts import ChartGenerator, ChartTheme
from src.visualization.reports import ReportGenerator, ReportType, ReportFormat
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager


class WebDashboard:
    """
    Web-based dashboard interface for HFT simulator
    
    This class provides a complete web interface for monitoring and controlling
    the HFT simulator, including real-time charts, performance metrics,
    risk monitoring, and report generation.
    
    Key Features:
    - Real-time web dashboard
    - WebSocket-based live updates
    - RESTful API endpoints
    - Interactive charts and visualizations
    - Report generation and download
    - Mobile-responsive design
    - Multi-user support
    
    Educational Notes:
    - Web dashboards provide accessible monitoring interfaces
    - Real-time updates keep users informed of current status
    - APIs enable integration with external systems
    - Modern web technologies enhance user experience
    - Responsive design works across different devices
    """
    
    def __init__(self, 
                 dashboard: Dashboard,
                 host: str = "localhost",
                 port: int = 5000,
                 debug: bool = False):
        """
        Initialize web dashboard
        
        Args:
            dashboard: Dashboard instance to serve
            host: Host address to bind to
            port: Port number to listen on
            debug: Enable debug mode
        """
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO are required for web interface. "
                            "Install with: pip install flask flask-socketio")
        
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self.debug = debug
        
        self.logger = get_logger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder=self._get_template_dir(),
                        static_folder=self._get_static_dir())
        self.app.config['SECRET_KEY'] = 'hft-simulator-secret-key'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Chart generator for web charts
        self.chart_generator = ChartGenerator(theme=ChartTheme.PROFESSIONAL)
        
        # Report generator
        self.report_generator = ReportGenerator()
        
        # Connected clients
        self.connected_clients = set()
        
        # Update thread
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        self.logger.info(f"Web dashboard initialized on {host}:{port}")
    
    def _get_template_dir(self) -> str:
        """Get templates directory path"""
        return str(Path(__file__).parent / "templates")
    
    def _get_static_dir(self) -> str:
        """Get static files directory path"""
        return str(Path(__file__).parent / "static")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get dashboard status"""
            try:
                status = self.dashboard.get_dashboard_status()
                return jsonify({'success': True, 'data': status})
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/overview')
        def api_overview():
            """Get overview data"""
            try:
                data = self.dashboard.get_overview_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                self.logger.error(f"Error getting overview: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        def api_performance():
            """Get performance data"""
            try:
                data = self.dashboard.get_performance_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                self.logger.error(f"Error getting performance data: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/risk')
        def api_risk():
            """Get risk data"""
            try:
                data = self.dashboard.get_risk_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                self.logger.error(f"Error getting risk data: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading')
        def api_trading():
            """Get trading data"""
            try:
                data = self.dashboard.get_trading_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                self.logger.error(f"Error getting trading data: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/market')
        def api_market():
            """Get market data"""
            try:
                data = self.dashboard.get_market_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                self.logger.error(f"Error getting market data: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """Get alerts"""
            try:
                alerts = [alert.to_dict() for alert in self.dashboard.alerts[-50:]]  # Last 50 alerts
                return jsonify({'success': True, 'data': alerts})
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
        def api_acknowledge_alert(alert_id):
            """Acknowledge an alert"""
            try:
                self.dashboard.acknowledge_alert(alert_id)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Error acknowledging alert: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/reports/generate', methods=['POST'])
        def api_generate_report():
            """Generate a report"""
            try:
                data = request.get_json()
                report_type = ReportType(data.get('type', 'performance_summary'))
                format_type = ReportFormat(data.get('format', 'html'))
                
                # Generate report based on type
                if report_type == ReportType.PERFORMANCE_SUMMARY and self.dashboard.portfolio:
                    report_data = self.report_generator.generate_performance_report(self.dashboard.portfolio)
                elif report_type == ReportType.RISK_ANALYSIS and self.dashboard.risk_manager:
                    report_data = self.report_generator.generate_risk_report(self.dashboard.risk_manager)
                elif report_type == ReportType.COMPREHENSIVE and self.dashboard.portfolio:
                    report_data = self.report_generator.generate_comprehensive_report(
                        self.dashboard.portfolio, self.dashboard.risk_manager
                    )
                else:
                    return jsonify({'success': False, 'error': 'Invalid report type or missing data'}), 400
                
                # Export report
                filepath = self.report_generator.export_report(report_data, format_type)
                
                return jsonify({
                    'success': True, 
                    'data': {
                        'filepath': filepath,
                        'download_url': f'/api/reports/download/{Path(filepath).name}'
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error generating report: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/reports/download/<filename>')
        def api_download_report(filename):
            """Download a generated report"""
            try:
                return send_from_directory(
                    self.report_generator.config.output_directory,
                    filename,
                    as_attachment=True
                )
            except Exception as e:
                self.logger.error(f"Error downloading report: {e}")
                return jsonify({'success': False, 'error': str(e)}), 404
        
        @self.app.route('/api/dashboard/start', methods=['POST'])
        def api_start_dashboard():
            """Start dashboard updates"""
            try:
                self.dashboard.start()
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Error starting dashboard: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/stop', methods=['POST'])
        def api_stop_dashboard():
            """Stop dashboard updates"""
            try:
                self.dashboard.stop()
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Error stopping dashboard: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/mode', methods=['POST'])
        def api_set_mode():
            """Set dashboard mode"""
            try:
                data = request.get_json()
                mode = DashboardMode(data.get('mode', 'overview'))
                self.dashboard.set_mode(mode)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Error setting mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _setup_socketio_handlers(self) -> None:
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.connected_clients.add(request.sid)
            self.logger.info(f"Client connected: {request.sid}")
            
            # Send initial data
            emit('dashboard_status', self.dashboard.get_dashboard_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.connected_clients.discard(request.sid)
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to data updates"""
            data_type = data.get('type', 'overview')
            self.logger.info(f"Client {request.sid} subscribed to {data_type}")
            
            # Send initial data for subscription
            if data_type == 'overview':
                emit('overview_update', self.dashboard.get_overview_data())
            elif data_type == 'performance':
                emit('performance_update', self.dashboard.get_performance_data())
            elif data_type == 'risk':
                emit('risk_update', self.dashboard.get_risk_data())
            elif data_type == 'trading':
                emit('trading_update', self.dashboard.get_trading_data())
            elif data_type == 'market':
                emit('market_update', self.dashboard.get_market_data())
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscription from data updates"""
            data_type = data.get('type', 'overview')
            self.logger.info(f"Client {request.sid} unsubscribed from {data_type}")
    
    def start(self) -> None:
        """Start the web dashboard"""
        
        # Start dashboard if not already running
        if not self.dashboard.is_running:
            self.dashboard.start()
        
        # Start update broadcast thread
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._broadcast_updates, daemon=True)
        self.update_thread.start()
        
        # Create template and static directories if they don't exist
        self._create_web_assets()
        
        self.logger.info(f"Starting web dashboard on http://{self.host}:{self.port}")
        
        # Start Flask-SocketIO server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            allow_unsafe_werkzeug=True
        )
    
    def stop(self) -> None:
        """Stop the web dashboard"""
        self.stop_event.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        self.dashboard.stop()
        self.logger.info("Web dashboard stopped")
    
    def _broadcast_updates(self) -> None:
        """Broadcast updates to connected clients"""
        
        while not self.stop_event.is_set():
            try:
                if self.connected_clients:
                    # Broadcast overview data
                    overview_data = self.dashboard.get_overview_data()
                    self.socketio.emit('overview_update', overview_data)
                    
                    # Broadcast performance data
                    if self.dashboard.portfolio:
                        performance_data = self.dashboard.get_performance_data()
                        self.socketio.emit('performance_update', performance_data)
                    
                    # Broadcast risk data
                    if self.dashboard.risk_manager:
                        risk_data = self.dashboard.get_risk_data()
                        self.socketio.emit('risk_update', risk_data)
                    
                    # Broadcast trading data
                    trading_data = self.dashboard.get_trading_data()
                    self.socketio.emit('trading_update', trading_data)
                    
                    # Broadcast market data
                    market_data = self.dashboard.get_market_data()
                    self.socketio.emit('market_update', market_data)
                
                # Wait before next update
                self.stop_event.wait(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error broadcasting updates: {e}")
                time.sleep(1)
    
    def _create_web_assets(self) -> None:
        """Create web templates and static files"""
        
        # Create directories
        template_dir = Path(self._get_template_dir())
        static_dir = Path(self._get_static_dir())
        
        template_dir.mkdir(parents=True, exist_ok=True)
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main dashboard template
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFT Simulator Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .warning { color: #ffc107; }
        .chart-container {
            height: 400px;
            margin: 20px 0;
        }
        .alert-item {
            border-left: 4px solid #dc3545;
            margin: 5px 0;
            padding: 10px;
            background: #f8f9fa;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running { background-color: #28a745; }
        .status-stopped { background-color: #dc3545; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">HFT Simulator Dashboard</span>
            <div class="d-flex">
                <span id="status-indicator" class="status-indicator status-stopped"></span>
                <span id="status-text">Disconnected</span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- Navigation Tabs -->
        <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button">Overview</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="performance-tab" data-bs-toggle="tab" data-bs-target="#performance" type="button">Performance</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="risk-tab" data-bs-toggle="tab" data-bs-target="#risk" type="button">Risk</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="trading-tab" data-bs-toggle="tab" data-bs-target="#trading" type="button">Trading</button>
            </li>
        </ul>

        <!-- Tab Content -->
        <div class="tab-content" id="dashboardTabContent">
            <!-- Overview Tab -->
            <div class="tab-pane fade show active" id="overview" role="tabpanel">
                <div class="row mt-3">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value" id="portfolio-value">$0</div>
                            <div class="metric-label">Portfolio Value</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value" id="total-pnl">$0</div>
                            <div class="metric-label">Total P&L</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value" id="active-positions">0</div>
                            <div class="metric-label">Active Positions</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value" id="total-trades">0</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">Portfolio Performance</div>
                            <div class="card-body">
                                <div id="performance-chart" class="chart-container"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">Recent Alerts</div>
                            <div class="card-body">
                                <div id="alerts-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Tab -->
            <div class="tab-pane fade" id="performance" role="tabpanel">
                <div class="row mt-3">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Performance Metrics</div>
                            <div class="card-body">
                                <div id="performance-metrics"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Risk Tab -->
            <div class="tab-pane fade" id="risk" role="tabpanel">
                <div class="row mt-3">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Risk Analysis</div>
                            <div class="card-body">
                                <div id="risk-metrics"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Trading Tab -->
            <div class="tab-pane fade" id="trading" role="tabpanel">
                <div class="row mt-3">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Trading Activity</div>
                            <div class="card-body">
                                <div id="trading-stats"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Connection status
        socket.on('connect', function() {
            document.getElementById('status-indicator').className = 'status-indicator status-running';
            document.getElementById('status-text').textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('status-indicator').className = 'status-indicator status-stopped';
            document.getElementById('status-text').textContent = 'Disconnected';
        });
        
        // Data updates
        socket.on('overview_update', function(data) {
            updateOverview(data);
        });
        
        socket.on('performance_update', function(data) {
            updatePerformance(data);
        });
        
        socket.on('risk_update', function(data) {
            updateRisk(data);
        });
        
        socket.on('trading_update', function(data) {
            updateTrading(data);
        });
        
        // Update functions
        function updateOverview(data) {
            if (data.portfolio) {
                document.getElementById('portfolio-value').textContent = '$' + (data.portfolio.total_value || 0).toLocaleString();
                document.getElementById('total-pnl').textContent = '$' + (data.portfolio.total_pnl || 0).toLocaleString();
                document.getElementById('active-positions').textContent = data.portfolio.active_positions || 0;
                document.getElementById('total-trades').textContent = data.portfolio.total_trades || 0;
            }
            
            if (data.alerts && data.alerts.recent) {
                updateAlerts(data.alerts.recent);
            }
        }
        
        function updatePerformance(data) {
            // Update performance metrics display
            const metricsDiv = document.getElementById('performance-metrics');
            if (data.metrics) {
                let html = '<div class="row">';
                Object.entries(data.metrics).forEach(([key, value]) => {
                    html += `<div class="col-md-4"><div class="metric-card">
                        <div class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</div>
                        <div class="metric-label">${key.replace('_', ' ').toUpperCase()}</div>
                    </div></div>`;
                });
                html += '</div>';
                metricsDiv.innerHTML = html;
            }
        }
        
        function updateRisk(data) {
            // Update risk metrics display
            const metricsDiv = document.getElementById('risk-metrics');
            if (data.summary) {
                let html = '<div class="row">';
                Object.entries(data.summary).forEach(([key, value]) => {
                    if (typeof value === 'number') {
                        html += `<div class="col-md-4"><div class="metric-card">
                            <div class="metric-value">${value.toFixed(4)}</div>
                            <div class="metric-label">${key.replace('_', ' ').toUpperCase()}</div>
                        </div></div>`;
                    }
                });
                html += '</div>';
                metricsDiv.innerHTML = html;
            }
        }
        
        function updateTrading(data) {
            // Update trading statistics display
            const statsDiv = document.getElementById('trading-stats');
            if (data.statistics) {
                let html = '<div class="row">';
                Object.entries(data.statistics).forEach(([key, value]) => {
                    html += `<div class="col-md-4"><div class="metric-card">
                        <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                        <div class="metric-label">${key.replace('_', ' ').toUpperCase()}</div>
                    </div></div>`;
                });
                html += '</div>';
                statsDiv.innerHTML = html;
            }
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alerts-list');
            let html = '';
            alerts.forEach(alert => {
                html += `<div class="alert-item">
                    <strong>${alert.title}</strong><br>
                    <small>${alert.message}</small><br>
                    <small class="text-muted">${new Date(alert.timestamp).toLocaleString()}</small>
                </div>`;
            });
            alertsList.innerHTML = html || '<p class="text-muted">No recent alerts</p>';
        }
        
        // Subscribe to data updates
        socket.emit('subscribe', {type: 'overview'});
        socket.emit('subscribe', {type: 'performance'});
        socket.emit('subscribe', {type: 'risk'});
        socket.emit('subscribe', {type: 'trading'});
    </script>
</body>
</html>
        """
        
        # Write dashboard template
        with open(template_dir / "dashboard.html", "w") as f:
            f.write(dashboard_html)
        
        self.logger.info("Web assets created successfully")


# Utility functions
def create_web_dashboard(dashboard: Dashboard,
                        host: str = "localhost",
                        port: int = 5000,
                        debug: bool = False) -> WebDashboard:
    """Create a web dashboard with specified configuration"""
    return WebDashboard(dashboard=dashboard, host=host, port=port, debug=debug)


def run_web_dashboard(portfolio: Portfolio = None,
                     risk_manager: RiskManager = None,
                     host: str = "localhost",
                     port: int = 5000) -> None:
    """
    Quick function to run a web dashboard
    
    Args:
        portfolio: Portfolio to monitor
        risk_manager: Risk manager to monitor
        host: Host address
        port: Port number
    """
    
    # Create dashboard
    dashboard_config = DashboardConfig()
    dashboard = Dashboard(config=dashboard_config, 
                         portfolio=portfolio, 
                         risk_manager=risk_manager)
    
    # Create web interface
    web_dashboard = WebDashboard(dashboard=dashboard, host=host, port=port)
    
    try:
        web_dashboard.start()
    except KeyboardInterrupt:
        print("\nShutting down web dashboard...")
        web_dashboard.stop()