#!/usr/bin/env python3
"""
HFT Simulator - Unified Main Interface
=====================================

This is the main entry point that connects ALL implemented features:
- Advanced ML-based trading strategies with 500+ features
- Real-time data processing and enhanced feeds
- Multi-asset trading capabilities
- Professional risk management
- Advanced performance analytics
- Real-time dashboard and visualizations

Usage Examples:
    # Advanced ML Strategy Backtesting
    python hft_main.py --mode backtest --strategy ml --data ./data/BTCUSDT_sample.csv
    
    # Real-time Multi-Asset Trading
    python hft_main.py --mode realtime --symbols BTCUSDT,ETHUSDT --duration 60
    
    # Complete System Demo
    python hft_main.py --mode demo --advanced
    
    # Launch Professional Dashboard
    python hft_main.py --mode dashboard --enhanced
    
    # Performance Analysis
    python hft_main.py --mode analysis --input ./results/backtest_results.json
"""

import argparse
import asyncio
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Advanced Strategy Imports
from src.strategies.ml_strategy import MLTradingStrategy, MLFeatureEngineer
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.strategies.arbitrage_strategy import ArbitrageStrategy

# Advanced Engine and Execution
from src.engine.order_book import OrderBook
from src.execution.simulator import ExecutionSimulator
from src.execution.fill_models import RealisticFillModel, AdvancedFillModel

# Real-time and Data Processing
from src.realtime.enhanced_data_feeds import EnhancedDataFeedConfig, create_enhanced_data_feed
from src.realtime.trading_system import RealTimeTradingSystem
from src.data_ingestion.data_pipeline import DataPipeline
from src.data_ingestion.quality_monitor import DataQualityMonitor

# Advanced Analytics and Performance
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager, RiskConfig
from src.performance.metrics import PerformanceAnalyzer, calculate_all_metrics

# Visualization and Reporting
from src.visualization.realtime_dashboard import RealTimeDashboard, DashboardConfig
from src.visualization.performance_dashboard import PerformanceDashboard
from src.visualization.reports import ReportGenerator

# ML and Advanced Features
from src.ml.feature_store.feature_store import FeatureStore
from src.ml.ensemble_models import EnsembleModelManager
from src.ml.anomaly_detection import AnomalyDetector

# Multi-asset Support
from src.assets.core.base_asset import AssetManager
from src.assets.crypto.crypto_asset import CryptoAsset
from src.assets.options.options_asset import OptionsAsset

# Utilities
from src.utils.logger import setup_main_logger, get_logger
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.constants import OrderSide, OrderType


class UnifiedHFTSystem:
    """Unified HFT System connecting all advanced features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_main_logger()
        self.performance_monitor = PerformanceMonitor()
        
        # System components
        self.asset_manager = AssetManager()
        self.feature_store = FeatureStore()
        self.ensemble_manager = EnsembleModelManager()
        self.anomaly_detector = AnomalyDetector()
        self.data_quality_monitor = DataQualityMonitor()
        self.report_generator = ReportGenerator()
        
        # Trading components
        self.portfolios = {}
        self.risk_managers = {}
        self.strategies = {}
        self.data_feeds = {}
        self.order_books = {}
        
        self.logger.info("🚀 Unified HFT System initialized with ALL advanced features")
    
    async def run_advanced_backtest(self, strategy_type: str, data_path: str, 
                                  symbols: List[str] = None) -> Dict[str, Any]:
        """Run advanced backtesting with ML strategies and comprehensive analytics"""
        
        self.logger.info(f"🧠 Starting Advanced {strategy_type.upper()} Strategy Backtest")
        
        if not symbols:
            symbols = ["BTCUSDT"]  # Default symbol
        
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"📊 Processing {symbol}...")
            
            # Load and prepare data with advanced preprocessing
            data_pipeline = DataPipeline()
            raw_data = await data_pipeline.load_data(data_path)
            
            # Advanced feature engineering (500+ features)
            feature_engineer = MLFeatureEngineer()
            feature_data = feature_engineer.create_features(raw_data)
            self.logger.info(f"✨ Generated {len(feature_engineer.feature_names)} features")
            
            # Store features in feature store
            await self.feature_store.store_features(symbol, feature_data)
            
            # Initialize advanced strategy
            strategy = self._create_advanced_strategy(strategy_type, symbol)
            
            # Initialize advanced portfolio and risk management
            portfolio = Portfolio(initial_cash=1000000)
            risk_config = RiskConfig(
                max_portfolio_risk=0.02,
                max_position_size=50000,
                max_drawdown=0.15,
                var_confidence=0.99
            )
            risk_manager = RiskManager(risk_config)
            
            # Run advanced execution simulation
            order_book = OrderBook(symbol, tick_size=0.01)
            fill_model = AdvancedFillModel(
                fill_probability=0.95,
                slippage_model="realistic",
                market_impact_factor=0.001
            )
            
            simulator = ExecutionSimulator(
                order_book=order_book,
                strategy=strategy,
                portfolio=portfolio,
                risk_manager=risk_manager,
                fill_model=fill_model
            )
            
            # Execute backtest with performance monitoring
            with self.performance_monitor.measure("backtest_execution"):
                backtest_result = await simulator.run_advanced_backtest(
                    feature_data, 
                    enable_analytics=True,
                    enable_ml_insights=True
                )
            
            # Advanced performance analysis
            analyzer = PerformanceAnalyzer()
            performance_metrics = analyzer.calculate_all_metrics(
                backtest_result.trades,
                backtest_result.portfolio_values,
                benchmark_returns=None  # Could add benchmark
            )
            
            # Anomaly detection on results
            anomalies = self.anomaly_detector.detect_anomalies(
                backtest_result.portfolio_values
            )
            
            results[symbol] = {
                'backtest_result': backtest_result,
                'performance_metrics': performance_metrics,
                'feature_count': len(feature_engineer.feature_names),
                'anomalies_detected': len(anomalies),
                'execution_stats': self.performance_monitor.get_stats("backtest_execution")
            }
            
            self.logger.info(f"✅ {symbol} backtest complete: "
                           f"PnL=${performance_metrics['total_return']:.2f}, "
                           f"Sharpe={performance_metrics['sharpe_ratio']:.2f}")
        
        return results
    
    async def run_realtime_trading(self, symbols: List[str], duration: int = 60):
        """Run real-time multi-asset trading with advanced features"""
        
        self.logger.info(f"⚡ Starting Real-time Multi-Asset Trading")
        self.logger.info(f"📈 Symbols: {', '.join(symbols)}")
        self.logger.info(f"⏱️  Duration: {duration}s")
        
        # Initialize real-time trading system
        trading_system = RealTimeTradingSystem()
        
        # Setup enhanced data feeds for each symbol
        for symbol in symbols:
            enhanced_config = EnhancedDataFeedConfig(
                url="wss://stream.binance.com:9443/stream",
                symbols=[symbol],
                buffer_size=20000,
                max_messages_per_second=2000,
                primary_source="binance",
                backup_sources=["mock"],
                enable_redundancy=True,
                enable_data_validation=True,
                enable_outlier_detection=True
            )
            
            feed = create_enhanced_data_feed("enhanced_websocket", enhanced_config)
            await feed.connect()
            await feed.subscribe([symbol])
            self.data_feeds[symbol] = feed
            
            # Initialize components for each symbol
            self.order_books[symbol] = OrderBook(symbol, tick_size=0.01)
            self.portfolios[symbol] = Portfolio(initial_cash=500000)
            
            # Advanced ML strategy for real-time trading
            self.strategies[symbol] = self._create_advanced_strategy("ml", symbol)
            
            # Real-time risk management
            risk_config = RiskConfig(max_position_size=10000, max_portfolio_risk=0.015)
            self.risk_managers[symbol] = RiskManager(risk_config)
            
            self.logger.info(f"🔗 {symbol} real-time components initialized")
        
        # Run real-time trading simulation
        await trading_system.run_multi_asset_trading(
            symbols=symbols,
            data_feeds=self.data_feeds,
            strategies=self.strategies,
            portfolios=self.portfolios,
            risk_managers=self.risk_managers,
            duration=duration
        )
        
        return {"status": "completed", "symbols": symbols, "duration": duration}
    
    async def run_enhanced_dashboard(self):
        """Launch enhanced real-time dashboard with all features"""
        
        self.logger.info("🖥️ Launching Enhanced Real-time Dashboard")
        
        config = DashboardConfig(
            host="127.0.0.1",
            port=8080,
            debug=True,
            update_interval_ms=100,
            max_data_points=5000,
            default_symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
            theme="professional_dark",
            enable_advanced_charts=True,
            enable_ml_insights=True,
            enable_risk_analytics=True
        )
        
        dashboard = RealTimeDashboard(config)
        
        # Add advanced features to dashboard
        dashboard.enable_feature_monitoring(self.feature_store)
        dashboard.enable_anomaly_detection(self.anomaly_detector)
        dashboard.enable_performance_analytics()
        
        self.logger.info("🚀 Enhanced Dashboard starting at: http://127.0.0.1:8080")
        self.logger.info("📊 Features: ML Insights, Risk Analytics, Multi-Asset")
        
        # Run enhanced dashboard
        await dashboard.run_enhanced()
    
    async def run_complete_demo(self):
        """Run complete system demonstration showcasing all features"""
        
        self.logger.info("🎬 Running Complete System Demonstration")
        print("=" * 70)
        print("🚀 HFT SIMULATOR - COMPLETE FEATURE DEMONSTRATION")
        print("=" * 70)
        print("📋 Features being demonstrated:")
        print("   ✨ Advanced ML strategies with 500+ features")
        print("   ✨ Real-time multi-asset processing")
        print("   ✨ Professional risk management")
        print("   ✨ Advanced performance analytics")
        print("   ✨ Anomaly detection and monitoring")
        print("   ✨ Enhanced data quality control")
        print("=" * 70)
        
        # 1. Advanced Backtesting Demo
        print("\n🧠 PHASE 1: Advanced ML Strategy Backtesting")
        print("-" * 50)
        
        backtest_results = await self.run_advanced_backtest(
            strategy_type="ml",
            data_path="./data/BTCUSDT_sample.csv",
            symbols=["BTCUSDT"]
        )
        
        for symbol, results in backtest_results.items():
            metrics = results['performance_metrics']
            print(f"📊 {symbol} Results:")
            print(f"   💰 Total Return: ${metrics['total_return']:,.2f}")
            print(f"   📈 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   🎯 Features Used: {results['feature_count']}")
            print(f"   ⚠️  Anomalies: {results['anomalies_detected']}")
        
        # 2. Real-time Trading Demo (short duration)
        print("\n⚡ PHASE 2: Real-time Multi-Asset Demo (30s)")
        print("-" * 50)
        
        realtime_results = await self.run_realtime_trading(
            symbols=["BTCUSDT", "ETHUSDT"],
            duration=30
        )
        print(f"✅ Real-time demo completed: {realtime_results['status']}")
        
        # 3. Generate Comprehensive Report
        print("\n📄 PHASE 3: Generating Comprehensive Report")
        print("-" * 50)
        
        report = await self.report_generator.generate_complete_report({
            'backtest_results': backtest_results,
            'realtime_results': realtime_results,
            'system_performance': self.performance_monitor.get_all_stats(),
            'feature_analysis': await self.feature_store.get_feature_summary()
        })
        
        print(f"📋 Complete report generated: {report['report_path']}")
        print(f"📊 Total features utilized: {report['total_features']}")
        print(f"🔍 System components tested: {report['components_tested']}")
        
        print("\n🎉 COMPLETE SYSTEM DEMONSTRATION FINISHED!")
        print("=" * 70)
        
        return {
            'demo_completed': True,
            'backtest_results': backtest_results,
            'realtime_results': realtime_results,
            'report': report
        }
    
    def _create_advanced_strategy(self, strategy_type: str, symbol: str):
        """Create advanced strategy instance"""
        
        if strategy_type == "ml":
            # Advanced ML strategy with all features
            return MLTradingStrategy(
                symbol=symbol,
                feature_engineer=MLFeatureEngineer(),
                ensemble_manager=self.ensemble_manager,
                lookback_periods=[5, 10, 20, 50, 100],
                prediction_horizon=5,
                min_confidence=0.65,
                enable_online_learning=True
            )
        
        elif strategy_type == "market_making":
            config = MarketMakingConfig(
                target_spread=0.02,
                max_inventory=5000,
                base_quote_size=200,
                enable_ml_signals=True,
                volatility_adjustment=True
            )
            return MarketMakingStrategy(symbol, config)
        
        elif strategy_type == "arbitrage":
            return ArbitrageStrategy(
                primary_symbol=symbol,
                secondary_symbols=[f"{symbol}"],  # Could add related symbols
                min_profit_bps=5.0,
                max_position=2000
            )
        
        else:
            # Default to liquidity taking
            return LiquidityTakingStrategy(
                symbol=symbol,
                signal_threshold=0.01,
                max_position=3000,
                order_size=100
            )


async def main():
    """Main entry point for unified HFT system"""
    
    parser = argparse.ArgumentParser(
        description="HFT Simulator - Unified Interface for ALL Advanced Features"
    )
    
    parser.add_argument("--mode", required=True, 
                       choices=["backtest", "realtime", "dashboard", "demo", "analysis"],
                       help="Operation mode")
    
    parser.add_argument("--strategy", default="ml",
                       choices=["ml", "market_making", "liquidity_taking", "arbitrage"],
                       help="Trading strategy type")
    
    parser.add_argument("--data", help="Data file path for backtesting")
    parser.add_argument("--symbols", default="BTCUSDT", 
                       help="Comma-separated list of symbols")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration in seconds for real-time mode")
    parser.add_argument("--advanced", action="store_true",
                       help="Enable all advanced features")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    # Initialize unified system
    system = UnifiedHFTSystem(config)
    
    try:
        if args.mode == "backtest":
            if not args.data:
                print("Error: --data required for backtest mode")
                return 1
            
            results = await system.run_advanced_backtest(
                strategy_type=args.strategy,
                data_path=args.data,
                symbols=symbols
            )
            
            # Save results
            output_file = f"results/advanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path("results").mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Advanced backtest completed. Results saved to: {output_file}")
        
        elif args.mode == "realtime":
            results = await system.run_realtime_trading(symbols, args.duration)
            print(f"✅ Real-time trading completed: {results}")
        
        elif args.mode == "dashboard":
            await system.run_enhanced_dashboard()
        
        elif args.mode == "demo":
            results = await system.run_complete_demo()
            print(f"✅ Complete demo finished: {results['demo_completed']}")
        
        elif args.mode == "analysis":
            # Performance analysis mode
            if args.data:
                analyzer = PerformanceAnalyzer()
                results = analyzer.analyze_results_file(args.data)
                print(f"✅ Analysis complete: {results}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
