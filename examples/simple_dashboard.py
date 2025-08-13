#!/usr/bin/env python3
"""
Simple HFT Dashboard Launcher
Minimal dashboard to test real-time components without complex configurations
"""

import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.realtime_dashboard import RealTimeDashboard, DashboardConfig

def create_simple_dashboard():
    """Create a simple dashboard with basic configuration"""
    print("Creating simple dashboard...")
    
    config = DashboardConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
        update_interval_ms=1000,  # 1 second updates
        max_data_points=500,
        default_symbols=["BTCUSDT", "ETHUSDT"],
        theme="dark",
        enable_profiling=False,
        log_level="ERROR"  # Minimal logging
    )
    
    dashboard = RealTimeDashboard(config)
    return dashboard

def main():
    print("=" * 50)
    print("HFT Simple Dashboard Test")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        dashboard = create_simple_dashboard()
        
        print("Starting dashboard server...")
        print("URL: http://127.0.0.1:8080")
        print("Health: http://127.0.0.1:8080/health")
        print()
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        # Run the dashboard
        dashboard.socketio.run(
            dashboard.app,
            host="127.0.0.1",
            port=8080,
            debug=False,  # Disable debug to reduce output
            use_reloader=False,
            allow_unsafe_werkzeug=True,
            log_output=False  # Reduce log output
        )
        
    except KeyboardInterrupt:
        print("\nShutdown by user")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("Dashboard stopped")

if __name__ == "__main__":
    main()
