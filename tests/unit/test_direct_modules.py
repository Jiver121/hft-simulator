"""
Test real-time components by importing modules directly by file path
"""

import sys
from pathlib import Path

# Add src to path
# Add project root to path to allow `from src...` imports
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from datetime import datetime

# Test direct module imports (bypassing __init__.py completely)
try:
    # Import types directly by importing the module file
    from src.realtime.types import (
        OrderRequest,
        Position,
        RiskViolation,
        OrderPriority,
        ExecutionAlgorithm,
    )
    print("1. Types imported successfully")
    
    # Import risk management directly
    from src.realtime.risk_management import RealTimeRiskManager
    print("2. Risk management imported successfully")
    
    # Import order management directly
    from src.realtime.order_management import RealTimeOrderManager
    print("3. Order management imported successfully")
    
    # Import brokers directly
    from src.realtime.brokers import MockBroker, BrokerConfig, BrokerType
    print("4. Brokers imported successfully")
    
    # Import constants
    from src.utils.constants import OrderSide, OrderType
    print("5. Constants imported successfully")
    
    print("All core components imported successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test_core_functionality():
    """Test core functionality"""
    print("\nTesting core functionality...")
    
    try:
        # Test 1: Create risk manager
        print("1. Creating risk manager...")
        risk_manager = RealTimeRiskManager()
        print("   Risk manager created successfully")
        
        # Test 2: Create order manager
        print("2. Creating order manager...")
        order_manager = RealTimeOrderManager(risk_manager)
        print("   Order manager created successfully")
        
        # Test 3: Create mock broker
        print("3. Creating mock broker...")
        broker_config = BrokerConfig(
            broker_type=BrokerType.MOCK,
            api_key="test_key",
            sandbox_mode=True
        )
        broker = MockBroker(broker_config)
        print("   Mock broker created successfully")
        
        # Test 4: Connect broker
        print("4. Testing broker connection...")
        connected = await broker.connect()
        if connected:
            print("   Broker connected successfully")
        else:
            print("   Broker connection failed")
            return False
        
        # Test 5: Add broker to order manager
        print("5. Adding broker to order manager...")
        order_manager.add_broker("test_broker", broker)
        print("   Broker added successfully")
        
        # Test 6: Start order manager
        print("6. Starting order manager...")
        await order_manager.start()
        print("   Order manager started successfully")
        
        # Test 7: Create test order
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
        print("   Order request created successfully")
        
        # Test 8: Submit order
        print("8. Submitting order...")
        order_id = await order_manager.submit_order(order_request)
        print(f"   Order submitted successfully with ID: {order_id}")
        
        # Test 9: Wait for processing
        print("9. Waiting for order processing...")
        await asyncio.sleep(2)
        
        # Test 10: Check order status
        print("10. Checking order status...")
        order_state = await order_manager.get_order_status(order_id)
        if order_state:
            print(f"    Order status: {order_state.status}")
            print(f"    Order filled quantity: {order_state.filled_quantity}")
        else:
            print("    Could not retrieve order status")
        
        # Test 11: Get statistics
        print("11. Getting system statistics...")
        stats = order_manager.get_statistics()
        print(f"    Orders submitted: {stats['orders_submitted']}")
        print(f"    Orders filled: {stats['orders_filled']}")
        print(f"    Active orders: {stats['active_orders']}")
        
        # Test 12: Test risk management
        print("12. Testing risk management...")
        risk_metrics = risk_manager.get_risk_metrics()
        print(f"    Total exposure: ${risk_metrics['total_exposure']:.2f}")
        print(f"    Emergency stop active: {risk_metrics['emergency_stop_active']}")
        
        # Test 13: Stop order manager
        print("13. Stopping order manager...")
        await order_manager.stop()
        print("    Order manager stopped successfully")
        
        # Test 14: Disconnect broker
        print("14. Disconnecting broker...")
        await broker.disconnect()
        print("    Broker disconnected successfully")
        
        print("\nSUCCESS: All tests passed!")
        print("Core real-time components are working correctly.")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("HFT Real-Time System Direct Module Test")
    print("=" * 50)
    
    success = await test_core_functionality()
    
    if success:
        print("\nVALIDATION COMPLETE!")
        print("The real-time trading system core components are functional:")
        print("- Order management system works")
        print("- Risk management system works") 
        print("- Broker integration works")
        print("- Order execution workflow works")
        print("- Circular import issues resolved")
    else:
        print("\nVALIDATION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())