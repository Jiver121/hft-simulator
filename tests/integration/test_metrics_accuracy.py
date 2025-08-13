"""
Metrics Accuracy Integration Tests

Validates performance metric calculations: Sharpe ratio, PnL, drawdown.
"""

import pytest
import numpy as np
import pandas as pd

from src.performance.metrics import (
    PerformanceAnalyzer,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
)


@pytest.mark.integration
@pytest.mark.metrics
def test_sharpe_ratio_accuracy_on_synthetic_returns():
    # Create synthetic returns with known properties
    rng = np.random.default_rng(123)
    daily_ret = rng.normal(0.001, 0.01, 252)  # ~25.2% annualized mean approx

    sharpe = calculate_sharpe_ratio(daily_ret)

    # Expected ballpark: mean/std * sqrt(252)
    expected = np.mean(daily_ret - 0.0) / np.std(daily_ret) * np.sqrt(252)
    assert np.isfinite(sharpe)
    assert abs(sharpe - expected) < 0.2  # within tolerance


@pytest.mark.integration
@pytest.mark.metrics
def test_drawdown_calculation_from_monotonic_series():
    # Portfolio rises then falls
    values = [100, 110, 120, 115, 112, 130, 90, 95, 140]
    dd = calculate_max_drawdown(values)
    # Max drawdown occurs from 130 -> 90 = 40/130 ~ 30.77%
    assert 0.30 <= dd <= 0.32


@pytest.mark.integration
@pytest.mark.metrics
def test_performance_analyzer_pnl_and_metrics():
    analyzer = PerformanceAnalyzer(initial_capital=100000)

    # Simulate portfolio value path
    ts0 = pd.Timestamp("2024-01-01")
    vals = [100000, 100500, 100250, 101000, 100800, 102000]
    for i, v in enumerate(vals):
        analyzer.update_portfolio_value(v, ts0 + pd.Timedelta(days=i))

    metrics = analyzer.calculate_metrics()
    assert metrics.total_return == (vals[-1] - vals[0]) / vals[0]
    assert metrics.max_drawdown >= 0
    assert metrics.sharpe_ratio is not None
