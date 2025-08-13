"""
Message serialization utilities for distributed HFT system.
Handles efficient serialization/deserialization of trading messages.
"""

import json
import pickle
import gzip
import logging
from typing import Any, Dict, Union, Optional
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass


class MessageSerializer:
    """High-performance message serializer for HFT systems"""
    
    def __init__(self, 
                 format: str = "json",
                 compression: bool = False,
                 compression_threshold: int = 1024):
        """
        Initialize serializer
        
        Args:
            format: Serialization format ('json', 'msgpack', 'pickle')
            compression: Whether to compress large messages
            compression_threshold: Minimum size in bytes to trigger compression
        """
        self.format = format.lower()
        self.compression = compression
        self.compression_threshold = compression_threshold
        
        # Validate format availability
        if self.format == "msgpack" and not MSGPACK_AVAILABLE:
            logger.warning("msgpack not available, falling back to json")
            self.format = "json"
        
        logger.info(f"MessageSerializer initialized: format={self.format}, compression={compression}")
    
    def serialize(self, obj: Any) -> bytes:
        """
        Serialize object to bytes
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized bytes
        """
        try:
            # Convert to serializable format
            serializable_obj = self._make_serializable(obj)
            
            # Serialize based on format
            if self.format == "json":
                data = self._serialize_json(serializable_obj)
            elif self.format == "msgpack":
                data = self._serialize_msgpack(serializable_obj)
            elif self.format == "pickle":
                data = self._serialize_pickle(serializable_obj)
            else:
                raise SerializationError(f"Unsupported format: {self.format}")
            
            # Apply compression if needed
            if self.compression and len(data) > self.compression_threshold:
                data = self._compress(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise SerializationError(f"Failed to serialize object: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to object
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized object
        """
        try:
            # Check if data is compressed
            if self.compression and self._is_compressed(data):
                data = self._decompress(data)
            
            # Deserialize based on format
            if self.format == "json":
                obj = self._deserialize_json(data)
            elif self.format == "msgpack":
                obj = self._deserialize_msgpack(data)
            elif self.format == "pickle":
                obj = self._deserialize_pickle(data)
            else:
                raise SerializationError(f"Unsupported format: {self.format}")
            
            return obj
            
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise SerializationError(f"Failed to deserialize data: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if obj is None:
            return None
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return {
                "_type": "numpy_array",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                "_type": "pandas_dataframe",
                "data": obj.to_dict(orient="records"),
                "columns": list(obj.columns),
                "index": list(obj.index)
            }
        elif isinstance(obj, pd.Series):
            return {
                "_type": "pandas_series",
                "data": obj.tolist(),
                "index": list(obj.index),
                "name": obj.name
            }
        elif is_dataclass(obj):
            return {
                "_type": "dataclass",
                "class_name": obj.__class__.__name__,
                "data": asdict(obj)
            }
        else:
            # Fallback: convert to string representation
            logger.warning(f"Converting non-serializable object to string: {type(obj)}")
            return str(obj)
    
    def _serialize_json(self, obj: Any) -> bytes:
        """Serialize using JSON"""
        if ORJSON_AVAILABLE:
            return orjson.dumps(obj)
        else:
            return json.dumps(obj, separators=(',', ':')).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize from JSON"""
        if ORJSON_AVAILABLE:
            return orjson.loads(data)
        else:
            return json.loads(data.decode('utf-8'))
    
    def _serialize_msgpack(self, obj: Any) -> bytes:
        """Serialize using MessagePack"""
        return msgpack.packb(obj, use_bin_type=True)
    
    def _deserialize_msgpack(self, data: bytes) -> Any:
        """Deserialize from MessagePack"""
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    
    def _serialize_pickle(self, obj: Any) -> bytes:
        """Serialize using Pickle"""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize from Pickle"""
        return pickle.loads(data)
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        compressed = gzip.compress(data)
        # Add compression marker
        return b"GZIP" + compressed
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        if data.startswith(b"GZIP"):
            return gzip.decompress(data[4:])
        return data
    
    def _is_compressed(self, data: bytes) -> bool:
        """Check if data is compressed"""
        return data.startswith(b"GZIP")


class MessageEncoder:
    """Specialized encoder for trading messages"""
    
    @staticmethod
    def encode_market_data(symbol: str, 
                          bid: float, 
                          ask: float,
                          bid_size: Optional[float] = None,
                          ask_size: Optional[float] = None,
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Encode market data message"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return {
            "type": "market_data",
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread": ask - bid,
            "mid_price": (bid + ask) / 2,
            "timestamp": timestamp.isoformat()
        }
    
    @staticmethod
    def encode_order_event(order_id: str,
                          symbol: str,
                          side: str,
                          quantity: float,
                          price: Optional[float] = None,
                          order_type: str = "LIMIT",
                          status: str = "NEW",
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Encode order event message"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return {
            "type": "order_event",
            "order_id": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price,
            "order_type": order_type.upper(),
            "status": status.upper(),
            "timestamp": timestamp.isoformat()
        }
    
    @staticmethod
    def encode_execution(execution_id: str,
                        order_id: str,
                        symbol: str,
                        side: str,
                        quantity: float,
                        price: float,
                        commission: float = 0.0,
                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Encode execution message"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return {
            "type": "execution",
            "execution_id": execution_id,
            "order_id": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price,
            "notional": quantity * price,
            "commission": commission,
            "timestamp": timestamp.isoformat()
        }
    
    @staticmethod
    def encode_position_update(symbol: str,
                              position: float,
                              average_price: float,
                              unrealized_pnl: float,
                              realized_pnl: float = 0.0,
                              timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Encode position update message"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return {
            "type": "position_update",
            "symbol": symbol,
            "position": position,
            "average_price": average_price,
            "market_value": position * average_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_pnl": unrealized_pnl + realized_pnl,
            "timestamp": timestamp.isoformat()
        }


class MessageValidator:
    """Validates message structure and content"""
    
    REQUIRED_FIELDS = {
        "market_data": ["symbol", "bid", "ask", "timestamp"],
        "order_event": ["order_id", "symbol", "side", "quantity", "timestamp"],
        "execution": ["execution_id", "order_id", "symbol", "side", "quantity", "price", "timestamp"],
        "position_update": ["symbol", "position", "average_price", "timestamp"]
    }
    
    @classmethod
    def validate_message(cls, message: Dict[str, Any]) -> bool:
        """
        Validate message structure
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            message_type = message.get("type")
            if not message_type:
                logger.error("Message missing 'type' field")
                return False
            
            required_fields = cls.REQUIRED_FIELDS.get(message_type)
            if not required_fields:
                logger.warning(f"Unknown message type: {message_type}")
                return True  # Allow unknown types
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in message]
            if missing_fields:
                logger.error(f"Message missing required fields: {missing_fields}")
                return False
            
            # Validate timestamp format
            if "timestamp" in message:
                try:
                    datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
                except ValueError:
                    logger.error("Invalid timestamp format")
                    return False
            
            # Type-specific validation
            if message_type == "market_data":
                return cls._validate_market_data(message)
            elif message_type == "order_event":
                return cls._validate_order_event(message)
            elif message_type == "execution":
                return cls._validate_execution(message)
            elif message_type == "position_update":
                return cls._validate_position_update(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    @staticmethod
    def _validate_market_data(message: Dict[str, Any]) -> bool:
        """Validate market data message"""
        try:
            bid = message["bid"]
            ask = message["ask"]
            
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                logger.error("Bid and ask must be numeric")
                return False
            
            if bid <= 0 or ask <= 0:
                logger.error("Bid and ask must be positive")
                return False
            
            if bid >= ask:
                logger.error("Bid must be less than ask")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Market data validation error: {e}")
            return False
    
    @staticmethod
    def _validate_order_event(message: Dict[str, Any]) -> bool:
        """Validate order event message"""
        try:
            side = message["side"].upper()
            if side not in ["BUY", "SELL"]:
                logger.error(f"Invalid side: {side}")
                return False
            
            quantity = message["quantity"]
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                logger.error("Quantity must be positive numeric")
                return False
            
            if "price" in message and message["price"] is not None:
                price = message["price"]
                if not isinstance(price, (int, float)) or price <= 0:
                    logger.error("Price must be positive numeric")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order event validation error: {e}")
            return False
    
    @staticmethod
    def _validate_execution(message: Dict[str, Any]) -> bool:
        """Validate execution message"""
        try:
            quantity = message["quantity"]
            price = message["price"]
            
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                logger.error("Execution quantity must be positive numeric")
                return False
            
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error("Execution price must be positive numeric")
                return False
            
            side = message["side"].upper()
            if side not in ["BUY", "SELL"]:
                logger.error(f"Invalid execution side: {side}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Execution validation error: {e}")
            return False
    
    @staticmethod
    def _validate_position_update(message: Dict[str, Any]) -> bool:
        """Validate position update message"""
        try:
            position = message["position"]
            average_price = message["average_price"]
            
            if not isinstance(position, (int, float)):
                logger.error("Position must be numeric")
                return False
            
            if not isinstance(average_price, (int, float)):
                logger.error("Average price must be numeric")
                return False
            
            if position != 0 and average_price <= 0:
                logger.error("Average price must be positive for non-zero position")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Position update validation error: {e}")
            return False


# Example usage and testing
def test_serialization():
    """Test serialization functionality"""
    # Test different serializers
    serializers = [
        MessageSerializer("json"),
        MessageSerializer("json", compression=True),
    ]
    
    if MSGPACK_AVAILABLE:
        serializers.append(MessageSerializer("msgpack"))
    
    # Test data
    test_data = {
        "simple": {"a": 1, "b": 2.5, "c": "hello"},
        "market_data": MessageEncoder.encode_market_data("BTCUSDT", 45000.0, 45001.0),
        "order": MessageEncoder.encode_order_event("order123", "BTCUSDT", "BUY", 0.1, 45000.0),
        "execution": MessageEncoder.encode_execution("exec123", "order123", "BTCUSDT", "BUY", 0.1, 45000.0)
    }
    
    for serializer in serializers:
        print(f"\nTesting {serializer.format} serializer (compression={serializer.compression}):")
        
        for name, data in test_data.items():
            try:
                # Serialize
                serialized = serializer.serialize(data)
                print(f"  {name}: {len(serialized)} bytes")
                
                # Deserialize
                deserialized = serializer.deserialize(serialized)
                
                # Validate
                if name in ["market_data", "order", "execution"]:
                    is_valid = MessageValidator.validate_message(deserialized)
                    print(f"    Valid: {is_valid}")
                
                # Check roundtrip
                if data == deserialized:
                    print(f"    Roundtrip: SUCCESS")
                else:
                    print(f"    Roundtrip: FAILED")
                    
            except Exception as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_serialization()
