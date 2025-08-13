"""
Enhanced HFT Real-Time Dashboard Launcher

This script starts the enhanced real-time HFT dashboard with production-ready features:
- Multi-source data feeds with automatic failover
- Real-time data quality monitoring
- Advanced performance analytics
- System health monitoring
- Enhanced error handling
"""

import asyncio
from src.visualization.realtime_dashboard import RealTimeDashboard, DashboardConfig
from src.realtime.enhanced_data_feeds import EnhancedDataFeedConfig, create_enhanced_data_feed
from src.utils.logger import setup_main_logger

def create_enhanced_demo_dashboard():
    """Create enhanced demo dashboard with real-time features"""
    
    # Enhanced dashboard configuration
    config = DashboardConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
        update_interval_ms=200,  # Higher frequency updates
        max_data_points=2000,    # More data points
        default_symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],  # Multi-asset
        theme="dark",
        enable_profiling=False,
        log_level="INFO"
    )
    
    dashboard = RealTimeDashboard(config)
    
    # Setup enhanced data feeds for each symbol
    for symbol in config.default_symbols:
        enhanced_config = EnhancedDataFeedConfig(
            url="wss://stream.binance.com:9443/stream",
            symbols=[symbol],
            buffer_size=15000,
            max_messages_per_second=1500,
            # Enhanced features
            primary_source="binance",
            backup_sources=["mock"],  # Use mock as backup
            enable_redundancy=True,
            enable_data_validation=True,
            enable_outlier_detection=True,
            max_price_deviation=0.05,
            error_threshold=8,
        )
        
        try:
            feed = create_enhanced_data_feed("enhanced_websocket", enhanced_config)
            dashboard.data_feeds[symbol] = feed
            print(f"[OK] Enhanced data feed configured for {symbol}")
        except Exception as e:
            print(f"[WARNING] Failed to setup enhanced feed for {symbol}: {e}")
    
    return dashboard


if __name__ == "__main__":
    print("Starting Enhanced HFT Real-Time Dashboard...")
    print("=" * 50)
    print("Enhanced Features:")
    print("   * Multi-asset real-time streaming")
    print("   * Advanced data quality monitoring")
    print("   * Production-grade error handling")
    print("   * System health dashboard")
    print("   * Enhanced performance analytics")
    print("=" * 50)
    
    logger = setup_main_logger()
    
    try:
        dashboard = create_enhanced_demo_dashboard()
        print(f"\nDashboard starting at: http://127.0.0.1:8080")
        print(f"Health check endpoint: http://127.0.0.1:8080/health")
        print(f"Ready for live multi-asset trading data!")
        print(f"\nTip: Open multiple browser tabs to compare assets")
        
        # Run the enhanced dashboard
        dashboard.socketio.run(
            dashboard.app, 
            host="127.0.0.1", 
            port=8080, 
            debug=True,
            use_reloader=False,  # Prevent issues with async components
            allow_unsafe_werkzeug=True
        )
    
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nDashboard error: {e}")
        logger.error(f"Dashboard startup error: {e}")
    
    print("\nEnhanced HFT Dashboard shutdown complete!")
