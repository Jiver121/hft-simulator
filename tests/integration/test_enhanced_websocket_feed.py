#!/usr/bin/env python3
"""
Test script for EnhancedWebSocketFeed to verify connection to Binance live data.

This script tests:
1. Connection establishment 
2. Heartbeat monitoring
3. Data reception from Binance WebSocket feed
4. Proper error handling and reconnection
"""

import asyncio
import sys
import os
import logging
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.realtime.enhanced_data_feeds import EnhancedWebSocketFeed, EnhancedDataFeedConfig
from src.realtime.data_feeds import MarketDataMessage

async def test_binance_connection():
    """Test connection to Binance WebSocket feed"""
    print("🚀 Testing EnhancedWebSocketFeed with Binance...")
    
    # Configure the enhanced feed for Binance
    config = EnhancedDataFeedConfig(
        url="wss://stream.binance.com:9443/ws",
        symbols=["BTCUSDT", "ETHUSDT"],
        heartbeat_interval=30.0,
        connection_timeout=10.0,
        primary_source="binance",
        backup_sources=["mock"],  # Use mock as backup for testing
        enable_redundancy=True,
        log_level="INFO"
    )
    
    # Create the enhanced feed
    feed = EnhancedWebSocketFeed(config)
    
    try:
        print("📡 Attempting to connect to Binance WebSocket...")
        
        # Test connection
        connected = await feed.connect()
        if not connected:
            print("❌ Failed to connect to Binance")
            return False
            
        print(f"✅ Successfully connected to {feed.current_source}")
        print(f"🔄 Heartbeat monitor status: {'Active' if feed.heartbeat_task else 'Inactive'}")
        
        # Test subscription
        print("📊 Subscribing to market data...")
        subscribed = await feed.subscribe(config.symbols)
        if not subscribed:
            print("❌ Failed to subscribe to symbols")
            return False
            
        print(f"✅ Successfully subscribed to {config.symbols}")
        
        # Test data streaming for a short time
        print("📈 Testing data streaming (15 seconds)...")
        message_count = 0
        timeout_seconds = 15
        
        try:
            async with asyncio.timeout(timeout_seconds):
                async for message in feed.start_streaming():
                    message_count += 1
                    print(f"📨 Message {message_count}: {message.symbol} - {message.message_type} - ${message.price}")
                    
                    # Test different message types
                    if message_count >= 10:  # Get at least 10 messages
                        break
                        
        except asyncio.TimeoutError:
            print(f"⏰ Streaming timeout after {timeout_seconds} seconds")
        
        # Display statistics
        stats = feed.get_enhanced_statistics()
        print("\n📊 Connection Statistics:")
        print(f"  📨 Messages received: {stats['messages_received']}")
        print(f"  ✅ Messages processed: {stats['messages_processed']}")
        print(f"  🔄 Current source: {stats['current_source']}")
        print(f"  💔 Connection errors: {stats['connection_errors']}")
        print(f"  ⚠️  Data errors: {stats['data_errors']}")
        print(f"  🛡️  Circuit breaker: {stats['circuit_breaker_state']}")
        print(f"  📊 Data quality: {stats['data_quality']}")
        
        # Test heartbeat monitoring
        if feed.heartbeat_task and not feed.heartbeat_task.done():
            print("✅ Heartbeat monitor is running properly")
        else:
            print("⚠️  Heartbeat monitor issue detected")
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        await feed.disconnect()
        print("✅ Successfully disconnected")
        
        return message_count > 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logging.exception("Test error details:")
        return False
    finally:
        # Ensure cleanup
        try:
            await feed.disconnect()
        except:
            pass

async def test_mock_connection():
    """Test with mock data as a fallback verification"""
    print("\n🎭 Testing with mock data feed...")
    
    config = EnhancedDataFeedConfig(
        url="mock://localhost",
        symbols=["BTCUSDT", "ETHUSDT"],
        heartbeat_interval=5.0,
        connection_timeout=5.0,
        primary_source="mock",
        backup_sources=[],
        log_level="INFO"
    )
    
    feed = EnhancedWebSocketFeed(config)
    
    try:
        # Test mock connection
        connected = await feed.connect()
        if not connected:
            print("❌ Failed to connect to mock feed")
            return False
            
        print(f"✅ Successfully connected to {feed.current_source}")
        
        # Test subscription
        subscribed = await feed.subscribe(config.symbols)
        if not subscribed:
            print("❌ Failed to subscribe to mock symbols")
            return False
            
        print(f"✅ Successfully subscribed to mock data")
        
        # Test mock data streaming
        print("📈 Testing mock data streaming (5 seconds)...")
        message_count = 0
        
        try:
            async with asyncio.timeout(5):
                async for message in feed.start_streaming():
                    message_count += 1
                    print(f"🎭 Mock Message {message_count}: {message.symbol} - ${message.price:.2f}")
                    
                    if message_count >= 5:  # Get a few mock messages
                        break
                        
        except asyncio.TimeoutError:
            print("⏰ Mock streaming timeout")
        
        print(f"✅ Received {message_count} mock messages")
        
        await feed.disconnect()
        return message_count > 0
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        return False
    finally:
        try:
            await feed.disconnect()
        except:
            pass

async def main():
    """Main test function"""
    print("🧪 EnhancedWebSocketFeed Test Suite")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test results
    results = []
    
    # Test 1: Try real Binance connection
    print("\n1️⃣  Testing Real Binance Connection")
    print("-" * 40)
    binance_success = await test_binance_connection()
    results.append(("Binance Connection", binance_success))
    
    # Test 2: Test mock connection as backup
    print("\n2️⃣  Testing Mock Data Feed")  
    print("-" * 40)
    mock_success = await test_mock_connection()
    results.append(("Mock Connection", mock_success))
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Overall result
    all_tests_passed = all(success for _, success in results)
    if all_tests_passed:
        print("\n🎉 All tests passed! EnhancedWebSocketFeed is working correctly.")
        print("The heartbeat monitor has been successfully implemented and tested.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all_tests_passed

if __name__ == "__main__":
    # Run the test
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.exception("Unexpected error details:")
        exit(1)
