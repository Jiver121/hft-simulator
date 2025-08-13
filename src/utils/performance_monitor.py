import time
import functools
import tracemalloc
from memory_profiler import memory_usage
from threading import Lock
from collections import defaultdict

# Thread-safe storage for metrics
_PERF_METRICS = defaultdict(list)
_METRICS_LOCK = Lock()

def performance_monitor(section=None):
    """
    Decorator for measuring execution time and memory of functions.
    Args:
        section (str): Optional section name for grouping metrics.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label = section or func.__name__
            tracemalloc.start()
            t0 = time.perf_counter()
            mem_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
            tracemalloc.stop()

            with _METRICS_LOCK:
                _PERF_METRICS[label].append({
                    'time': elapsed,
                    'memory_profiled': float(mem_after) - float(mem_before),
                    'mem_peak_kb': peak / 1024,
                })
            return result
        return wrapper
    return decorator


def get_performance_metrics():
    """Return a copy of all collected metrics (thread-safe)."""
    with _METRICS_LOCK:
        import copy
        return copy.deepcopy(_PERF_METRICS)


def clear_performance_metrics():
    """Clear all collected metrics."""
    with _METRICS_LOCK:
        _PERF_METRICS.clear()

