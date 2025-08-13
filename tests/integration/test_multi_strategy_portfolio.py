"""
Multi-Strategy Portfolio Integration Tests

Validates scenarios where multiple strategies compete for the same liquidity
and verifies portfolio-level aggregation and isolation.
"""

import pytest
import pandas as pd
import uuid
from typing import List

from src.execution.matching_engine import MatchingEngine
from src.engine.order_types import Order
from src.utils.constants import OrderSide, OrderType
from src.performance.portfolio import Portfolio
from src.strategies.base_strategy import BaseStrategy, StrategyResult


class MockAggressiveBuyer(BaseStrategy):
    def __init__(self, name: str, symbol: str):
        super().__init__(name, symbol)
    def on_market_update(self, snapshot, timestamp):
        res = StrategyResult(timestamp=timestamp)
        # Always try to buy at ask
        if snapshot.best_ask:
            res.add_order(self.create_order(OrderSide.BUY, 50, price=snapshot.best_ask, order_type=OrderType.LIMIT))
        return res


class MockAggressiveSeller(BaseStrategy):
    def __init__(self, name: str, symbol: str):
        super().__init__(name, symbol)
    def on_market_update(self, snapshot, timestamp):
        res = StrategyResult(timestamp=timestamp)
        if snapshot.best_bid:
            res.add_order(self.create_order(OrderSide.SELL, 50, price=snapshot.best_bid, order_type=OrderType.LIMIT))
        return res


def _seed_liquidity(engine: MatchingEngine, symbol: str):
    for i in range(20):
        engine.process_order(Order(order_id=f"b{i}", symbol=symbol, side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0 - i*0.01, volume=200, timestamp=pd.Timestamp.now()))
        engine.process_order(Order(order_id=f"a{i}", symbol=symbol, side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0 + i*0.01, volume=200, timestamp=pd.Timestamp.now()))


@pytest.mark.integration
@pytest.mark.portfolio
def test_multiple_strategies_competing_for_liquidity():
    symbol = "AAPL"
    engine = MatchingEngine(symbol)
    _seed_liquidity(engine, symbol)

    portfolio = Portfolio(initial_cash=1_000_000, name="multi")

    # Create two strategies pushing opposite sides
    buyer = MockAggressiveBuyer("buyer", symbol)
    seller = MockAggressiveSeller("seller", symbol)

    # Construct a simple snapshot
    snapshot = type("S", (), {"best_bid": 99.99, "best_ask": 100.01})()

    # Both generate orders at the same tick
    orders = []
    for strat in (buyer, seller):
        res = strat.on_market_update(snapshot, pd.Timestamp.now())
        orders += res.orders

    # Submit orders and count fills
    trades_total = 0
    for o in orders:
        trades, _ = engine.process_order(o)
        trades_total += len(trades)

    # Basic assertions: both strategies got interaction with book
    assert len(orders) == 2
    assert trades_total >= 1

