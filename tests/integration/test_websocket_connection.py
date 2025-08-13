#!/usr/bin/env python3
"""
Quick test script to verify Binance WebSocket connections and symbol formatting
"""

import asyncio
import json
import websockets
import time

async def test_binance_websocket():
    """Test Binance WebSocket connection with proper symbol formatting"""
    
    # Test symbols - these should work with Binance
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    
    print("🔗 Testing Binance WebSocket Connection...")
    print(f"📡 Testing symbols: {test_symbols}")
    print("=" * 50)
    
    try:
        # Connect to Binance WebSocket API
        uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
        print(f"🔌 Connecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established!")
            
            # Listen for a few messages
            message_count = 0
            start_time = time.time()
            
            print("\n📊 Receiving real-time data...")
            print("-" * 50)
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    
                    # Extract key information
                    symbol = data.get('s', 'N/A')
                    price = float(data.get('c', 0))
                    volume = float(data.get('v', 0))
                    change = float(data.get('P', 0))
                    
                    print(f"#{message_count:2d} {symbol}: ${price:,.2f} | Vol: {volume:,.0f} | Change: {change:+.2f}%")
                    
                    # Test for 10 messages or 30 seconds
                    if message_count >= 10 or (time.time() - start_time) > 30:
                        break
                        
                except json.JSONDecodeError:
                    print("⚠️  Failed to parse JSON message")
                except Exception as e:
                    print(f"⚠️  Error processing message: {e}")
            
            print(f"\n✅ Successfully received {message_count} real-time messages!")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
    
    return True

async def test_multiple_symbols():
    """Test multiple symbol subscriptions using combined stream"""
    
    symbols = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"]
    streams = [f"{symbol}@ticker" for symbol in symbols]
    
    print("\n🚀 Testing Multi-Symbol WebSocket...")
    print(f"📡 Testing symbols: {[s.upper() for s in symbols]}")
    print("=" * 50)
    
    try:
        # Use combined stream endpoint
        uri = "wss://stream.binance.com:9443/stream"
        
        async with websockets.connect(uri) as websocket:
            print("✅ Combined stream connection established!")
            
            # Subscribe to multiple streams
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time())
            }
            
            await websocket.send(json.dumps(subscribe_message))
            print(f"📤 Subscribed to {len(streams)} symbol streams")
            
            # Listen for messages from different symbols
            symbol_data = {}
            message_count = 0
            start_time = time.time()
            
            print("\n📊 Receiving multi-asset data...")
            print("-" * 60)
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Skip subscription confirmation messages
                    if 'result' in data or 'id' in data:
                        continue
                    
                    if 'stream' in data and 'data' in data:
                        message_count += 1
                        stream_name = data['stream']
                        stream_data = data['data']
                        
                        symbol = stream_data.get('s', 'N/A')
                        price = float(stream_data.get('c', 0))
                        change = float(stream_data.get('P', 0))
                        
                        # Track unique symbols
                        symbol_data[symbol] = {
                            'price': price,
                            'change': change,
                            'messages': symbol_data.get(symbol, {}).get('messages', 0) + 1
                        }
                        
                        print(f"#{message_count:2d} {symbol}: ${price:,.2f} ({change:+.2f}%) | Stream: {stream_name}")
                        
                        # Test until we get data from all symbols or timeout
                        if len(symbol_data) >= len(symbols) and message_count >= 20:
                            break
                        if (time.time() - start_time) > 45:
                            break
                            
                except json.JSONDecodeError:
                    print("⚠️  Failed to parse JSON message")
                except Exception as e:
                    print(f"⚠️  Error processing message: {e}")
            
            print(f"\n✅ Multi-asset test complete!")
            print(f"📊 Received {message_count} total messages from {len(symbol_data)} symbols:")
            for symbol, data in symbol_data.items():
                print(f"   • {symbol}: {data['messages']} messages, latest price: ${data['price']:,.2f}")
            
            return len(symbol_data) >= len(symbols)
            
    except Exception as e:
        print(f"❌ Multi-symbol WebSocket test failed: {e}")
        return False

async def main():
    """Run all WebSocket tests"""
    print("🧪 Binance WebSocket API Testing Suite")
    print("=" * 60)
    
    # Test 1: Single symbol connection
    success1 = await test_binance_websocket()
    
    # Test 2: Multiple symbol subscriptions
    success2 = await test_multiple_symbols()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"   • Single Symbol Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   • Multi-Symbol Test:  {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All WebSocket tests passed! Dashboard should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check network connectivity and Binance API status.")

if __name__ == "__main__":
    asyncio.run(main())
