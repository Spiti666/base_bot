"""High-performance compiled backtest engine helpers."""

from .compiled_core import run_fast_backtest_loop, run_fast_backtest_loop_detailed

__all__ = ("run_fast_backtest_loop", "run_fast_backtest_loop_detailed")

