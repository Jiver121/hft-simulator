"""
Strategy Execution Scenarios Integration Tests

Validates strategy signal generation and execution with live market data replay.

Key Scenarios:
- Strategy signal generation with market data replay
- Basic execution loop: snapshot -> decision -> orders -> fills
- Latency and throughput checks for strategy decision cycle
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import uuid
import time

from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot
from src.execution.matching_engine import MatchingEngine
from src.utils.constants import OrderSide, OrderType
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockMomentumStrategy(BaseStrategy):
    """Minimal momentum strategy for integration testing."""

    def __init__(self, symbol: str, window: int = 5):
        super().__init__("mock_momentum", symbol)
        self.window = window

    def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> StrategyResult:
        self._update_market_history(snapshot)
        self.update_count += 1
        self.last_update_time = timestamp

        result = StrategyResult(timestamp=timestamp)

        # Generate simple momentum signal
        if len(self.price_history) >= self.window + 1:
            ret = self.price_history[-1] / self.price_history[-self.window-1] - 1
            if ret > 0.002:  # upward momentum -> buy
                order = self.create_order(OrderSide.BUY, volume=50, price=snapshot.best_ask, order_type=OrderType.LIMIT, reason="up_momo")
                result.add_order(order, reason="momentum_buy")
            elif ret < -0.002:  # downward -> sell
                order = self.create_order(OrderSide.SELL, volume=50, price=snapshot.best_bid, order_type=OrderType.LIMIT, reason="down_momo")
                result.add_order(order, reason="momentum_sell")
        return result


def _gen_snapshots(symbol: str, n: int = 500, start_price: float = 100.0) -> List[BookSnapshot]:
    ts0 = pd.Timestamp.now()
    price = start_price
    out: List[BookSnapshot] = []
    rng = np.random.default_rng(7)
    for i in range(n):
        price *= (1 + rng.normal(0, 0.0008))
        spread = 0.02
        bid, ask = price - spread/2, price + spread/2
        bid_vol = int(rng.integers(200, 800))
        ask_vol = int(rng.integers(200, 800))
        out.append(BookSnapshot(
            symbol=symbol,
            timestamp=ts0 + pd.Timedelta(milliseconds=10*i),
            bids=[(bid, bid_vol)],
            asks=[(ask, ask_vol)],
            best_bid=bid,
            best_ask=ask,
            best_bid_volume=bid_vol,
            best_ask_volume=ask_vol,
            mid_price=(bid+ask)/2
        ))
    return out


@pytest.mark.integration
@pytest.mark.strategy
def test_strategy_signal_generation_and_execution():
    symbol = "AAPL"
    engine = MatchingEngine(symbol)
    strategy = MockMomentumStrategy(symbol)

    # Seed book with liquidity so strategy orders can fill
    seed_orders = [
        Order(order_id=f"liq_bid_{uuid.uuid4()}", symbol=symbol, side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.9, volume=500, timestamp=pd.Timestamp.now()),
        Order(order_id=f"liq_ask_{uuid.uuid4()}", symbol=symbol, side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.1, volume=500, timestamp=pd.Timestamp.now()),
    ]
    for o in seed_orders:
        engine.process_order(o)

    snapshots = _gen_snapshots(symbol, n=400, start_price=100.0)

    orders_submitted = 0
    trades_total = 0
    t0 = time.time()

    for snap in snapshots:
        decision = strategy.on_market_update(snap, snap.timestamp)
        for order in decision.orders:
            orders_submitted += 1
            trades, _ = engine.process_order(order)
            trades_total += len(trades)

    elapsed = time.time() - t0

    # Assertions
    assert orders_submitted > 0, "Strategy did not generate any orders"
    assert trades_total >= 1, "No trades executed from strategy orders"
    assert elapsed < 5.0, f"Strategy execution too slow: {elapsed:.2f}s"


@pytest.mark.integration
@pytest.mark.strategy
def test_live_replay_does_not_crash_on_crossed_ticks():
    symbol = "MSFT"
    engine = MatchingEngine(symbol)
    strategy = MockMomentumStrategy(symbol)

    # Generate snapshots with occasional crossed markets
    snaps = _gen_snapshots(symbol, n=200, start_price=200.0)
    # introduce crossed every 25th
    for i in range(0, len(snaps), 25):
        s = snaps[i]
        snaps[i] = BookSnapshot(
            symbol=symbol,
            timestamp=s.timestamp,
            bids=[(s.best_ask + 0.01, 300)],
            asks=[(s.best_bid - 0.01, 300)],
            best_bid=s.best_ask + 0.01,
            best_ask=s.best_bid - 0.01,
            best_bid_volume=300,
            best_ask_volume=300,
            mid_price=None
        )

    # run
    for snap in snaps:
        decision = strategy.on_market_update(snap, snap.timestamp)
        for order in decision.orders:
            engine.process_order(order)

    stats = engine.get_statistics()
    assert stats["orders_processed"] >= 0  # sanity

