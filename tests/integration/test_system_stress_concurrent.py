"""
System Stress Tests - Concurrent Orders

Validates high-load behavior with 10,000+ concurrent orders and measures throughput and rejection rates.
"""

import pytest
import time
import uuid
import concurrent.futures as cf
import pandas as pd
from typing import List, Tuple

from src.execution.matching_engine import MatchingEngine
from src.engine.order_types import Order
from src.utils.constants import OrderSide, OrderType


def _make_order(symbol: str, i: int) -> Order:
    side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
    price = 100.0 + ((i % 100) * 0.01)
    return Order(
        order_id=f"stress_{i}_{uuid.uuid4()}",
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        volume=10,
        timestamp=pd.Timestamp.now(),
    )


@pytest.mark.integration
@pytest.mark.stress
def test_high_load_concurrent_order_processing():
    symbol = "AAPL"
    engine = MatchingEngine(symbol)

    # Pre-seed book with baseline liquidity
    for i in range(1000):
        o = _make_order(symbol, i)
        engine.process_order(o)

    N = 12000  # 10k+ orders
    t0 = time.time()

    def submit(i: int) -> Tuple[int, int]:
        o = _make_order(symbol, i + 1000)
        trades, _ = engine.process_order(o)
        return (1, len(trades))

    # Simulate concurrency using thread pool (engine is in-process)
    orders_count = 0
    trades_count = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for submitted, trades in ex.map(submit, range(N)):
            orders_count += submitted
            trades_count += trades

    elapsed = time.time() - t0
    throughput = orders_count / max(elapsed, 1e-6)

    stats = engine.get_statistics()

    # Assertions: ensure it finished quickly and processed all orders
    assert orders_count == N, "Not all orders were processed"
    assert throughput > 5000, f"Throughput too low: {throughput:.0f} orders/sec"
    # Ensure engine tracked processing
    assert stats["orders_processed"] >= orders_count

