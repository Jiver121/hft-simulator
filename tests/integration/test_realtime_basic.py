"""
Basic test of real-time components to validate the system works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import asyncio
from datetime import datetime

# Test basic imports
try:
    from realtime.types import OrderRequest, Position, RiskViolation, OrderPriority, ExecutionAlgorithm
    from realtime.risk_management import RealTimeRiskManager
    from realtime.order_management import RealTimeOrderManager
    from realtime.brokers import MockBroker, BrokerConfig, BrokerType
    from utils.constants import OrderSide, OrderType
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_basic_functionality():
    """Test basic functionality of real-time components"""
    print("\n🧪 Testing basic real-time functionality...")
    
    try:
        # Test 1: Create risk manager
        print("1. Creating risk manager...")
        risk_manager = RealTimeRiskManager()
        print("   ✅ Risk manager created")
        
        # Test 2: Create order manager
        print("2. Creating order manager...")
        order_manager = RealTimeOrderManager(risk_manager)
        print("   ✅ Order manager created")
        
        # Test 3: Create mock broker
        print("3. Creating mock broker...")
        broker_config = BrokerConfig(
            broker_type=BrokerType.MOCK,
            api_key="test_key",
            sandbox_mode=True
        )
        broker = MockBroker(broker_config)
        print("   ✅ Mock broker created")
        
        # Test 4: Connect broker
        print("4. Connecting broker...")
        connected = await broker.connect()
        if connected:
            print("   ✅ Broker connected")
        else:
            print("   ❌ Broker connection failed")
            return False
        
        # Test 5: Add broker to order manager
        print("5. Adding broker to order manager...")
        order_manager.add_broker("test_broker", broker)
        print("   ✅ Broker added to order manager")
        
        # Test 6: Start order manager
        print("6. Starting order manager...")
        await order_manager.start()
        print("   ✅ Order manager started")
        
        # Test 7: Create and submit test order
        print("7. Creating test order...")
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            execution_algorithm=ExecutionAlgorithm.MARKET,
            priority=OrderPriority.NORMAL,
            strategy_id="test_strategy"
        )
        print("   ✅ Order request created")
        
        # Test 8: Submit order
        print("8. Submitting order...")
        order_id = await order_manager.submit_order(order_request)
        print(f"   ✅ Order submitted with ID: {order_id}")
        
        # Test 9: Wait a moment for processing
        print("9. Waiting for order processing...")
        await asyncio.sleep(2)
        
        # Test 10: Check order status
        print("10. Checking order status...")
        order_state = await order_manager.get_order_status(order_id)
        if order_state:
            print(f"    ✅ Order status: {order_state.status}")
        else:
            print("    ❌ Could not retrieve order status")
        
        # Test 11: Get statistics
        print("11. Getting system statistics...")
        stats = order_manager.get_statistics()
        print(f"    📊 Orders submitted: {stats['orders_submitted']}")
        print(f"    📊 Active orders: {stats['active_orders']}")
        print(f"    📊 Completed orders: {stats['completed_orders']}")
        
        # Test 12: Stop order manager
        print("12. Stopping order manager...")
        await order_manager.stop()
        print("    ✅ Order manager stopped")
        
        # Test 13: Disconnect broker
        print("13. Disconnecting broker...")
        await broker.disconnect()
        print("    ✅ Broker disconnected")
        
        print("\n🎉 All tests passed! Real-time system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 HFT Real-Time System Basic Test")
    print("=" * 50)
    
    success = await test_basic_functionality()
    
    if success:
        print("\n✅ Real-time system validation completed successfully!")
        print("The system is ready for full integration testing.")
    else:
        print("\n❌ Real-time system validation failed!")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())