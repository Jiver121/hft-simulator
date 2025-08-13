"""
Real-Time Trading System Demo

This example demonstrates the complete real-time HFT trading system in action,
showing how all components work together to process live market data,
execute trades, and manage risk in real-time.

Features Demonstrated:
- Real-time market data ingestion
- Live order execution through brokers
- Risk management and controls
- Stream processing pipeline
- Performance monitoring
- Configuration management

Usage:
    python examples/realtime_trading_demo.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path so we can import the package under `src`
sys.path.append(str(Path(__file__).parent.parent))

from src.realtime.trading_system import RealTimeTradingSystem
from src.realtime.config import RealTimeConfig, Environment, DataFeedConfig, BrokerConfig, BrokerType
from src.realtime.types import OrderRequest, ExecutionAlgorithm, OrderPriority, RiskViolation
from src.realtime.data_feeds import MarketDataMessage
from src.utils.constants import OrderSide, OrderType


class SimpleMarketMakingStrategy:
    """
    Simple market making strategy for demonstration
    
    This strategy places buy and sell orders around the current market price
    to capture the bid-ask spread while managing inventory risk.
    """
    
    def __init__(self, symbol: str, spread_bps: float = 10.0, order_size: int = 100):
        self.symbol = symbol
        self.spread_bps = spread_bps  # Spread in basis points
        self.order_size = order_size
        
        # Strategy state
        self.current_mid_price = None
        self.inventory = 0
        self.max_inventory = 500
        
        # Performance tracking
        self.orders_placed = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        
        print(f"[STRATEGY] Initialized market making for {symbol}")
    
    async def on_market_data(self, market_data: MarketDataMessage, trading_system: RealTimeTradingSystem):
        """Handle market data update"""
        if market_data.symbol != self.symbol:
            return
        
        # Update current market price
        if market_data.bid_price and market_data.ask_price:
            self.current_mid_price = (market_data.bid_price + market_data.ask_price) / 2
            
            # Generate trading signals
            await self._generate_quotes(trading_system)
    
    async def _generate_quotes(self, trading_system: RealTimeTradingSystem):
        """Generate buy and sell quotes"""
        if not self.current_mid_price:
            return
        
        # Calculate spread
        spread = self.current_mid_price * (self.spread_bps / 10000)
        
        # Adjust for inventory (inventory skew)
        inventory_skew = (self.inventory / self.max_inventory) * (spread / 2)
        
        # Calculate quote prices
        bid_price = self.current_mid_price - (spread / 2) - inventory_skew
        ask_price = self.current_mid_price + (spread / 2) - inventory_skew
        
        # Only quote if inventory is within limits
        if abs(self.inventory) < self.max_inventory:
            # Place buy order (if not too long)
            if self.inventory < self.max_inventory:
                buy_order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=self.order_size,
                    order_type=OrderType.LIMIT,
                    price=bid_price,
                    execution_algorithm=ExecutionAlgorithm.LIMIT,
                    priority=OrderPriority.NORMAL,
                    strategy_id="market_making_demo"
                )
                
                try:
                    order_id = await trading_system.submit_order(buy_order)
                    self.orders_placed += 1
                    print(f"[STRATEGY] Placed buy order: {order_id} @ {bid_price:.4f}")
                except Exception as e:
                    print(f"[STRATEGY] Failed to place buy order: {e}")
            
            # Place sell order (if not too short)
            if self.inventory > -self.max_inventory:
                sell_order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=self.order_size,
                    order_type=OrderType.LIMIT,
                    price=ask_price,
                    execution_algorithm=ExecutionAlgorithm.LIMIT,
                    priority=OrderPriority.NORMAL,
                    strategy_id="market_making_demo"
                )
                
                try:
                    order_id = await trading_system.submit_order(sell_order)
                    self.orders_placed += 1
                    print(f"[STRATEGY] Placed sell order: {order_id} @ {ask_price:.4f}")
                except Exception as e:
                    print(f"[STRATEGY] Failed to place sell order: {e}")
    
    def on_order_filled(self, order_state):
        """Handle order fill"""
        if order_state.request.strategy_id != "market_making_demo":
            return
        
        # Update inventory
        if order_state.request.side == OrderSide.BUY:
            self.inventory += order_state.filled_quantity
        else:
            self.inventory -= order_state.filled_quantity
        
        self.trades_executed += 1
        
        print(f"[STRATEGY] Order filled: {order_state.order.order_id}, "
              f"Inventory: {self.inventory}, Trades: {self.trades_executed}")
    
    def get_performance_summary(self):
        """Get strategy performance summary"""
        return {
            'symbol': self.symbol,
            'orders_placed': self.orders_placed,
            'trades_executed': self.trades_executed,
            'current_inventory': self.inventory,
            'total_pnl': self.total_pnl
        }


class TradingSystemDemo:
    """
    Main demo class that orchestrates the real-time trading system demonstration
    """
    
    def __init__(self):
        self.trading_system = None
        self.strategy = None
        self.demo_duration = 60  # Run demo for 60 seconds
        self.start_time = None
        
        # Demo statistics
        self.market_data_received = 0
        self.orders_submitted = 0
        self.risk_violations = 0
        
    async def setup_demo_configuration(self) -> RealTimeConfig:
        """Setup configuration for the demo"""
        print("[DEMO] Setting up configuration...")
        
        # Create demo configuration
        config = RealTimeConfig(
            environment=Environment.DEVELOPMENT,
            debug_mode=True,
            system_id="hft-demo-001"
        )
        
        # Configure real-time WebSocket data feed (Binance; public no key)
        config.data_feeds["demo_feed"] = DataFeedConfig(
            url="wss://stream.binance.com:9443/stream",
            symbols=["BTCUSDT"],
            max_messages_per_second=1000,
            buffer_size=10000
        )
        
        # Configure mock broker
        config.brokers["demo_broker"] = BrokerConfig(
            broker_type=BrokerType.MOCK,
            api_key="demo_key",
            sandbox_mode=True,
            enable_paper_trading=True
        )
        
        # Configure stream processing
        config.stream_processing.update({
            'num_workers': 2,
            'queue_size': 10000,
            'enable_monitoring': True
        })
        
        # Configure trading parameters
        config.trading.update({
            'enable_live_trading': True,
            'paper_trading_mode': True,
            'max_orders_per_second': 10
        })
        
        print("[DEMO] Configuration setup complete")
        return config
    
    async def setup_event_handlers(self):
        """Setup event handlers for monitoring"""
        
        def on_market_data_received(message: MarketDataMessage):
            self.market_data_received += 1
            if self.market_data_received % 50 == 0:  # Log every 50 messages
                print(f"[DEMO] Market data received: {self.market_data_received} messages")
        
        def on_order_filled(order_state):
            self.orders_submitted += 1
            print(f"[DEMO] Order filled: {order_state.order.symbol} "
                  f"{order_state.request.side.value} {order_state.filled_quantity}")
            
            # Notify strategy
            if self.strategy:
                self.strategy.on_order_filled(order_state)
        
        def on_risk_violation(violation: RiskViolation):
            self.risk_violations += 1
            print(f"[DEMO] Risk violation: {violation.violation_type.value} - {violation.message}")
        
        def on_system_started(system):
            print("[DEMO] ✅ Trading system started successfully!")
        
        def on_system_stopped(system):
            print("[DEMO] 🛑 Trading system stopped")
        
        # Register event handlers
        self.trading_system.add_event_callback('market_data_received', on_market_data_received)
        self.trading_system.add_event_callback('order_filled', on_order_filled)
        self.trading_system.add_event_callback('risk_violation', on_risk_violation)
        self.trading_system.add_event_callback('system_started', on_system_started)
        self.trading_system.add_event_callback('system_stopped', on_system_stopped)
        
        print("[DEMO] Event handlers configured")
    
    async def setup_strategy(self):
        """Setup trading strategy"""
        print("[DEMO] Setting up market making strategy...")
        
        # Create simple market making strategy (match feed symbol)
        self.strategy = SimpleMarketMakingStrategy(
            symbol="BTCUSDT",
            spread_bps=20.0,  # 20 basis points spread
            order_size=100
        )
        
        # Add market data handler to strategy
        async def market_data_handler(message: MarketDataMessage):
            await self.strategy.on_market_data(message, self.trading_system)
        
        self.trading_system.add_event_callback('market_data_received', market_data_handler)
        
        print("[DEMO] Strategy setup complete")
    
    async def run_demo(self):
        """Run the complete trading system demo"""
        print("=" * 60)
        print("🚀 HFT REAL-TIME TRADING SYSTEM DEMO")
        print("=" * 60)
        
        try:
            # Setup configuration
            config = await self.setup_demo_configuration()
            
            # Initialize trading system
            print("[DEMO] Initializing trading system...")
            self.trading_system = RealTimeTradingSystem(config)
            await self.trading_system.initialize()
            
            # Setup event handlers and strategy
            await self.setup_event_handlers()
            await self.setup_strategy()
            
            # Start the trading system
            print("[DEMO] Starting trading system...")
            self.start_time = datetime.now()
            await self.trading_system.start()
            
            # Run demo for specified duration
            print(f"[DEMO] Running demo for {self.demo_duration} seconds...")
            print("[DEMO] Watch for:")
            print("  📊 Market data processing")
            print("  📈 Strategy order generation")
            print("  ⚡ Real-time execution")
            print("  🛡️  Risk management")
            print("  📋 Performance monitoring")
            print()
            
            # Monitor system during demo
            await self._monitor_demo()
            
        except KeyboardInterrupt:
            print("\n[DEMO] Demo interrupted by user")
        except Exception as e:
            print(f"[DEMO] Demo error: {e}")
        finally:
            # Stop the system
            if self.trading_system:
                print("[DEMO] Stopping trading system...")
                await self.trading_system.stop()
            
            # Print final summary
            await self._print_demo_summary()
    
    async def _monitor_demo(self):
        """Monitor the demo and print periodic updates"""
        demo_start = datetime.now()
        last_status_time = demo_start
        
        while (datetime.now() - demo_start).total_seconds() < self.demo_duration:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Print status update
            elapsed = (datetime.now() - demo_start).total_seconds()
            remaining = self.demo_duration - elapsed
            
            if (datetime.now() - last_status_time).total_seconds() >= 10:  # Every 10 seconds
                system_status = self.trading_system.get_system_status()
                
                print(f"\n[DEMO STATUS] Time remaining: {remaining:.0f}s")
                print(f"  📊 Market data: {self.market_data_received} messages")
                print(f"  📋 Orders: {self.orders_submitted} submitted")
                print(f"  🛡️  Risk violations: {self.risk_violations}")
                print(f"  ⚡ System state: {system_status['system_state']}")
                print(f"  💾 Memory usage: {system_status['metrics']['memory_usage_mb']:.1f} MB")
                
                if self.strategy:
                    perf = self.strategy.get_performance_summary()
                    print(f"  📈 Strategy: {perf['orders_placed']} orders, "
                          f"{perf['trades_executed']} trades, "
                          f"inventory: {perf['current_inventory']}")
                
                last_status_time = datetime.now()
    
    async def _print_demo_summary(self):
        """Print final demo summary"""
        print("\n" + "=" * 60)
        print("📊 DEMO SUMMARY")
        print("=" * 60)
        
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
            print(f"Runtime: {runtime:.1f} seconds")
        
        print(f"Market Data Messages: {self.market_data_received}")
        print(f"Orders Submitted: {self.orders_submitted}")
        print(f"Risk Violations: {self.risk_violations}")
        
        if self.strategy:
            perf = self.strategy.get_performance_summary()
            print(f"\nStrategy Performance:")
            print(f"  Symbol: {perf['symbol']}")
            print(f"  Orders Placed: {perf['orders_placed']}")
            print(f"  Trades Executed: {perf['trades_executed']}")
            print(f"  Final Inventory: {perf['current_inventory']}")
            print(f"  Total P&L: ${perf['total_pnl']:.2f}")
        
        if self.trading_system:
            system_status = self.trading_system.get_system_status()
            print(f"\nSystem Metrics:")
            print(f"  Messages/sec: {system_status['metrics']['messages_per_second']:.1f}")
            print(f"  Avg Latency: {system_status['metrics']['avg_processing_latency_us']:.1f}μs")
            print(f"  Active Components: {len([c for c in system_status['component_status'].values() if c['is_healthy']])}")
        
        print("\n✅ Demo completed successfully!")
        print("=" * 60)


async def main():
    """Main demo entry point"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some verbose logging for demo
    logging.getLogger('realtime.stream_processing').setLevel(logging.WARNING)
    logging.getLogger('realtime.data_feeds').setLevel(logging.WARNING)
    
    # Run the demo
    demo = TradingSystemDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("Starting HFT Real-Time Trading System Demo...")
    print("Press Ctrl+C to stop the demo early\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)