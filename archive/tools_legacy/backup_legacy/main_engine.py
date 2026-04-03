from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import json
import os
from contextlib import contextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from itertools import product
from multiprocessing import Pool, TimeoutError as PoolTimeoutError, get_context
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from threading import RLock
from time import sleep
import time
from typing import Callable, Sequence

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
import sys

from config import (
    COIN_STRATEGIES,
    MAX_BACKTEST_CANDLES,
    PRODUCTION_PROFILE_REGISTRY,
    settings,
)
from core.api.bitunix import BitunixClient
from core.api.websocket import MultiTimeframeWebSocketManager
from core.data.db import CandleRecord, Database, PaperTrade
from core.data.history import HistoryManager
from core.patterns.setup_gate import SmartSetupGate
from core.regime.hmm_regime_detector import HMMRegimeDetector
from core.paper_trading.engine import PaperTradingEngine
from strategies.python.dual_thrust_breakout import (
    build_dual_thrust_signal_frame,
    run_python_dual_thrust,
    run_python_dual_thrust_breakout,
    should_exit_python_dual_thrust_breakout,
)
from strategies.python.ema_cross_volume import (
    build_ema_cross_volume_signal_frame,
    run_python_ema_cross_volume,
)
from strategies.python.frama_cross import (
    build_frama_cross_signal_frame,
    run_python_frama_cross,
)


StrategyRunner = Callable[[pd.DataFrame], int]
OptimizationProfile = dict[str, float]
STRATEGY_NAME_ALIASES = {
    "dual_thrust_breakout": "dual_thrust",
}
STRATEGY_RUNNERS: dict[str, StrategyRunner] = {
    "ema_cross_volume": run_python_ema_cross_volume,
    "frama_cross": run_python_frama_cross,
    "dual_thrust": run_python_dual_thrust,
}
SETUP_GATE_SETTLED_WARMUP_CANDLES = 500
LIVE_SETTLED_WARMUP_CANDLES = SETUP_GATE_SETTLED_WARMUP_CANDLES
MAX_OPTIMIZATION_WORKERS = 30
OPTIMIZER_WORKER_MAX_TASKS_PER_CHILD = 100
OPTIMIZER_MEMORY_RESERVE_GB = 10.0
OPTIMIZER_MEMORY_PER_WORKER_GB = 2.45
FORCE_MAX_OPTIMIZATION_WORKERS = True
OPTIMIZER_SAMPLE_RANDOM_SEED = 20260321
OPTIMIZER_INCREMENTAL_SYNC_MIN_CANDLES = 10_000
MIN_SAMPLE_SCAN_PROFILES_PER_STRATEGY = 10_000
BACKTEST_FEE_STRESS_MULTIPLIER = 1.25
MAX_OPTIMIZATION_GRID_PROFILES = 2_000_000
MIN_TRADES_15M = 50
MIN_TRADES_1H = 20
RANKING_MIN_PROFIT_FACTOR = 1.8
RANKING_MAX_DRAWDOWN_PCT = 12.0
EMA_VOLUME_MULTIPLIER_OPTIONS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0)
FRAMA_VOLUME_MULTIPLIER_OPTIONS = (1.5, 2.5, 4.0)
TRACE_VALUE_UNAVAILABLE = "unavailable"
_OPTIMIZER_SETTINGS_LOCK = RLock()
_WORKER_CANDLES_DF: pd.DataFrame | None = None
_WORKER_CANDLE_ROWS: list[dict[str, object]] | None = None
_WORKER_REGIME_MASK: list[int] | None = None


def _resolve_backtest_history_start_utc() -> datetime:
    parsed = datetime.fromisoformat(
        settings.trading.backtest_history_start_utc.replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(tzinfo=None)


BACKTEST_HISTORY_START_UTC = _resolve_backtest_history_start_utc()


def _resolve_optimizer_history_start_utc() -> datetime:
    parsed = datetime.fromisoformat(
        settings.trading.optimizer_history_start_utc.replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(tzinfo=None)


OPTIMIZER_HISTORY_START_UTC = _resolve_optimizer_history_start_utc()


def _history_start_for_mode(*, optimize_profile: bool) -> datetime:
    return OPTIMIZER_HISTORY_START_UTC if optimize_profile else BACKTEST_HISTORY_START_UTC


def _resolve_backtest_fee_pct() -> float:
    base_fee_pct = float(settings.trading.taker_fee_pct)
    return max(0.0, base_fee_pct * BACKTEST_FEE_STRESS_MULTIPLIER)


def _resolve_optimization_history_sync_start(
    *,
    db: Database,
    symbol: str,
    interval: str,
    strategy_name: str,
    base_start_time: datetime,
    use_setup_gate: bool,
) -> datetime:
    """Return a smaller sync start for optimizer runs when recent cache already exists."""
    requested_start = base_start_time
    _existing_count, _existing_oldest, existing_newest = db.get_candle_stats(symbol, interval)
    if existing_newest is None:
        return requested_start

    search_window_candles = max(1, int(settings.trading.optimization_search_window_candles))
    required_candles = _required_candle_count_for_strategy(
        strategy_name,
        use_setup_gate=use_setup_gate,
    )
    hmm_warmup_candles = (
        2500
        if strategy_name == "frama_cross"
        and bool(getattr(settings.trading, "use_hmm_regime_filter", False))
        else 0
    )
    safety_buffer_candles = max(required_candles, 100)
    needed_candles = (
        search_window_candles
        + required_candles
        + hmm_warmup_candles
        + safety_buffer_candles
    )
    needed_candles = max(needed_candles, OPTIMIZER_INCREMENTAL_SYNC_MIN_CANDLES)
    interval_seconds = max(_interval_total_seconds(interval), 1)
    optimized_start = existing_newest - timedelta(seconds=needed_candles * interval_seconds)
    if optimized_start > requested_start:
        return optimized_start
    return requested_start


def _optimization_risk_profiles() -> list[tuple[float, float, float, float]]:
    # Use universal risk grid when present
    tp_opts = getattr(settings.trading, "universal_take_profit_pct_options", settings.trading.optimization_take_profit_pct_options)
    sl_opts = getattr(settings.trading, "universal_stop_loss_pct_options", settings.trading.optimization_stop_loss_pct_options)
    ta_opts = getattr(settings.trading, "universal_trailing_activation_pct_options", settings.trading.optimization_trailing_activation_pct_options)
    td_opts = getattr(settings.trading, "universal_trailing_distance_pct_options", settings.trading.optimization_trailing_distance_pct_options)
    return [
        (
            float(take_profit_pct),
            float(stop_loss_pct),
            float(trailing_activation_pct),
            float(trailing_distance_pct),
        )
        for take_profit_pct, stop_loss_pct, trailing_activation_pct, trailing_distance_pct in product(
            tp_opts,
            sl_opts,
            ta_opts,
            td_opts,
        )
        if float(trailing_distance_pct) < float(trailing_activation_pct)
    ]


def generate_trade_management_optimization_grid() -> list[OptimizationProfile]:
    return [
        {
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "trailing_activation_pct": trailing_activation_pct,
            "trailing_distance_pct": trailing_distance_pct,
        }
        for (
            take_profit_pct,
            stop_loss_pct,
            trailing_activation_pct,
            trailing_distance_pct,
        ) in _optimization_risk_profiles()
    ]


def generate_ema_optimization_grid() -> list[OptimizationProfile]:
    ema_fast_periods = settings.strategy.ema_fast_options
    ema_slow_periods = settings.strategy.ema_slow_options
    volume_multiplier_options = EMA_VOLUME_MULTIPLIER_OPTIONS
    risk_profiles = _optimization_risk_profiles()
    profiles: list[OptimizationProfile] = []
    for ema_fast_period, ema_slow_period, volume_multiplier in product(
        ema_fast_periods,
        ema_slow_periods,
        volume_multiplier_options,
    ):
        if int(ema_fast_period) >= int(ema_slow_period):
            continue
        for take_profit_pct, stop_loss_pct, trailing_activation_pct, trailing_distance_pct in risk_profiles:
            profiles.append(
                {
                    "take_profit_pct": take_profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                    "trailing_activation_pct": trailing_activation_pct,
                    "trailing_distance_pct": trailing_distance_pct,
                    "ema_fast_period": float(ema_fast_period),
                    "ema_slow_period": float(ema_slow_period),
                    "volume_multiplier": float(volume_multiplier),
                }
            )
    return _cap_profiles(profiles)


def generate_frama_optimization_grid() -> list[OptimizationProfile]:
    frama_fast_periods = settings.strategy.frama_fast_options
    frama_slow_periods = settings.strategy.frama_slow_options
    volume_multiplier_options = FRAMA_VOLUME_MULTIPLIER_OPTIONS
    risk_profiles = _optimization_risk_profiles()
    profiles: list[OptimizationProfile] = []
    for frama_fast_period, frama_slow_period, volume_multiplier in product(
        frama_fast_periods,
        frama_slow_periods,
        volume_multiplier_options,
    ):
        if int(frama_fast_period) >= int(frama_slow_period):
            continue
        for take_profit_pct, stop_loss_pct, trailing_activation_pct, trailing_distance_pct in risk_profiles:
            profiles.append(
                {
                    "take_profit_pct": take_profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                    "trailing_activation_pct": trailing_activation_pct,
                    "trailing_distance_pct": trailing_distance_pct,
                    "frama_fast_period": float(frama_fast_period),
                    "frama_slow_period": float(frama_slow_period),
                    "volume_multiplier": float(volume_multiplier),
                }
            )
    return _cap_profiles(profiles)


def generate_dual_thrust_optimization_grid(symbol: str | None = None) -> list[OptimizationProfile]:
    normalized_symbol = symbol.strip().upper() if symbol else None
    symbol_specific_grid = (
        {}
        if normalized_symbol is None
        else settings.strategy.dual_thrust_symbol_optimization_grids.get(normalized_symbol, {})
    )
    take_profit_options = tuple(
        symbol_specific_grid.get(
            "take_profit_pct_options",
            settings.strategy.dual_thrust_take_profit_pct_options,
        )
    )
    stop_loss_options = tuple(
        symbol_specific_grid.get(
            "stop_loss_pct_options",
            settings.strategy.dual_thrust_stop_loss_pct_options,
        )
    )
    trailing_distance_options = settings.strategy.dual_thrust_trailing_distance_pct_options
    trailing_activation_options = settings.strategy.dual_thrust_trailing_activation_pct_options
    period_options = settings.strategy.dual_thrust_opt_periods
    k_options = settings.strategy.dual_thrust_opt_k_values
    risk_profiles: list[tuple[float, float, float, float]] = []
    for take_profit_pct, stop_loss_pct, trailing_activation_pct, trailing_distance_pct in product(
        take_profit_options,
        stop_loss_options,
        trailing_activation_options,
        trailing_distance_options,
    ):
        take_profit_value = float(take_profit_pct)
        stop_loss_value = float(stop_loss_pct)
        trailing_activation_value = float(trailing_activation_pct)
        trailing_distance_value = float(trailing_distance_pct)

        if trailing_distance_value >= trailing_activation_value:
            continue

        risk_profiles.append(
            (
                take_profit_value,
                stop_loss_value,
                trailing_activation_value,
                trailing_distance_value,
            )
        )

    if not risk_profiles:
        return []

    period_count = len(period_options)
    k_count = len(k_options)
    risk_count = len(risk_profiles)
    variant_count = period_count * k_count * k_count
    total_profiles = variant_count * risk_count
    if total_profiles <= 0:
        return []

    capped_target = min(total_profiles, MAX_OPTIMIZATION_GRID_PROFILES)

    if total_profiles <= MAX_OPTIMIZATION_GRID_PROFILES:
        profiles: list[OptimizationProfile] = []
        for dual_thrust_period, dual_thrust_k1, dual_thrust_k2 in product(
            period_options,
            k_options,
            k_options,
        ):
            for (
                take_profit_pct,
                stop_loss_pct,
                trailing_activation_pct,
                trailing_distance_pct,
            ) in risk_profiles:
                profiles.append(
                    {
                        "take_profit_pct": take_profit_pct,
                        "stop_loss_pct": stop_loss_pct,
                        "trailing_activation_pct": trailing_activation_pct,
                        "trailing_distance_pct": trailing_distance_pct,
                        "dual_thrust_period": float(dual_thrust_period),
                        "dual_thrust_k1": float(dual_thrust_k1),
                        "dual_thrust_k2": float(dual_thrust_k2),
                    }
                )
        return profiles

    profiles: list[OptimizationProfile] = []
    step = total_profiles / float(capped_target)
    last_flat_index = -1
    variant_block = k_count * k_count

    for sample_index in range(capped_target):
        flat_index = int(sample_index * step)
        if flat_index <= last_flat_index:
            flat_index = last_flat_index + 1
        if flat_index >= total_profiles:
            flat_index = total_profiles - 1
        last_flat_index = flat_index

        variant_index, risk_index = divmod(flat_index, risk_count)
        period_index, k_variant_index = divmod(variant_index, variant_block)
        k1_index, k2_index = divmod(k_variant_index, k_count)

        take_profit_pct, stop_loss_pct, trailing_activation_pct, trailing_distance_pct = risk_profiles[
            risk_index
        ]
        profiles.append(
            {
                "take_profit_pct": take_profit_pct,
                "stop_loss_pct": stop_loss_pct,
                "trailing_activation_pct": trailing_activation_pct,
                "trailing_distance_pct": trailing_distance_pct,
                "dual_thrust_period": float(period_options[period_index]),
                "dual_thrust_k1": float(k_options[k1_index]),
                "dual_thrust_k2": float(k_options[k2_index]),
            }
        )
    return profiles


def _cap_profiles(
    profiles: list[OptimizationProfile],
    max_total: int = MAX_OPTIMIZATION_GRID_PROFILES,
) -> list[OptimizationProfile]:
    """Cap the returned profile list so total profiles does not exceed max_total.
    If over the limit, sample evenly across the existing list to reach the cap.
    """
    if not profiles:
        return profiles
    if len(profiles) <= max_total:
        return profiles
    # Sample evenly
    step = len(profiles) / float(max_total)
    selected: list[OptimizationProfile] = []
    i = 0.0
    for _ in range(max_total):
        idx = int(i)
        if idx >= len(profiles):
            idx = len(profiles) - 1
        selected.append(profiles[idx])
        i += step
    return selected


def generate_optimization_grid(
    strategy_name: str | None = None,
    *,
    symbol: str | None = None,
) -> list[OptimizationProfile]:
    resolved_strategy = (
        settings.strategy.default_strategy_name
        if strategy_name is None
        else _validate_strategy_name(strategy_name)
    )
    if resolved_strategy == "ema_cross_volume":
        return generate_ema_optimization_grid()
    if resolved_strategy == "frama_cross":
        return generate_frama_optimization_grid()
    if resolved_strategy == "dual_thrust":
        return generate_dual_thrust_optimization_grid(symbol=symbol)
    return generate_trade_management_optimization_grid()


def _validate_strategy_name(strategy_name: str) -> str:
    strategy_name = STRATEGY_NAME_ALIASES.get(strategy_name, strategy_name)
    if strategy_name not in settings.strategy.available_strategies:
        raise ValueError(f"Unsupported strategy_name: {strategy_name}")
    if strategy_name not in STRATEGY_RUNNERS:
        raise ValueError(f"Strategy runner is not implemented: {strategy_name}")
    return strategy_name


def _validate_interval_name(interval: str) -> str:
    if interval not in settings.api.timeframes:
        raise ValueError(f"Unsupported interval: {interval}")
    return interval


def resolve_optimizer_scan_interval_for_symbol(
    symbol: str,
    default_interval: str | None = None,
) -> str:
    return resolve_interval_for_symbol(
        str(symbol).strip().upper(),
        default_interval,
        use_coin_override=False,
    )


def resolve_optimizer_strategy_for_symbol(
    symbol: str,
    default_strategy_name: str | None = None,
    *,
    apply_backtest_optimizer_overrides: bool = False,
) -> str:
    normalized_symbol = str(symbol).strip().upper()
    _ = apply_backtest_optimizer_overrides
    fallback_strategy = (
        settings.strategy.default_strategy_name
        if default_strategy_name is None
        else default_strategy_name
    )
    return resolve_strategy_for_symbol(
        normalized_symbol,
        fallback_strategy,
        use_coin_override=True,
    )


def resolve_interval_for_symbol(
    symbol: str,
    default_interval: str | None = None,
    *,
    use_coin_override: bool = True,
) -> str:
    fallback_interval = settings.trading.interval if default_interval is None else default_interval
    if use_coin_override:
        profile = settings.trading.coin_profiles.get(symbol)
        if profile is not None and profile.interval is not None:
            return _validate_interval_name(profile.interval)
    return _validate_interval_name(fallback_interval)


def _load_latest_backtest_candles(
    db: Database,
    *,
    symbol: str,
    interval: str,
    limit: int = MAX_BACKTEST_CANDLES,
) -> list[CandleRecord]:
    return db.fetch_recent_candles(symbol, interval, limit=limit)


def _required_candle_count_for_strategy(strategy_name: str, *, use_setup_gate: bool = False) -> int:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "ema_cross_volume":
        required_count = max(
            settings.strategy.ema_slow_period + 1,
            settings.strategy.volume_sma_period,
        )
    elif strategy_name == "frama_cross":
        required_count = max(settings.strategy.frama_slow_period + 1, 6)
    elif strategy_name == "dual_thrust":
        required_count = max(settings.strategy.dual_thrust_period + 2, 6)
    else:
        raise ValueError(f"Unsupported strategy_name for candle count: {strategy_name}")
    if use_setup_gate:
        required_count = max(
            required_count,
            SmartSetupGate.required_candle_count(),
            SETUP_GATE_SETTLED_WARMUP_CANDLES,
        )
    return required_count

def _live_analysis_window_for_strategy(strategy_name: str, *, use_setup_gate: bool = False) -> int:
    required_count = _required_candle_count_for_strategy(
        strategy_name,
        use_setup_gate=use_setup_gate,
    )
    if use_setup_gate:
        return max(
            required_count + 3,
            SmartSetupGate.required_candle_count() * 2,
            LIVE_SETTLED_WARMUP_CANDLES,
        )
    return max(required_count + 3, 50)


def _interval_total_seconds(interval: str) -> int:
    normalized = interval.strip()
    if normalized.endswith("M"):
        return int(normalized[:-1]) * 2592000
    normalized = normalized.lower()
    if normalized.endswith("m"):
        return int(normalized[:-1]) * 60
    if normalized.endswith("h"):
        return int(normalized[:-1]) * 3600
    if normalized.endswith("d"):
        return int(normalized[:-1]) * 86400
    if normalized.endswith("w"):
        return int(normalized[:-1]) * 604800
    if normalized == "1m":
        return 60
    return 900


def _clear_strategy_indicator_cache(candles_dataframe: pd.DataFrame | None) -> None:
    if candles_dataframe is None:
        return
    indicator_columns = (
        "ema_fast",
        "ema_slow",
        "frama_fast",
        "frama_slow",
        "frama_volume_sma_20",
        "volume_sma",
        "volume_threshold",
        "recent_volume_spike",
        "ema_long_entry",
        "ema_short_entry",
        "frama_long_entry",
        "frama_short_entry",
        "dual_range",
        "dual_buy_line",
        "dual_sell_line",
        "dual_long_entry",
        "dual_short_entry",
        "dual_long_exit",
        "dual_short_exit",
    )
    for column_name in indicator_columns:
        if column_name in candles_dataframe.columns:
            # Use `del` to release indicator arrays immediately without creating drop-copies.
            del candles_dataframe[column_name]
    with suppress(Exception):
        candles_dataframe._item_cache.clear()  # pandas cache, best effort.


def _initialize_optimizer_worker(
    candles_records: list[dict[str, object]],
    regime_mask: list[int] | None = None,
) -> None:
    global _WORKER_CANDLES_DF, _WORKER_CANDLE_ROWS, _WORKER_REGIME_MASK
    _WORKER_CANDLES_DF = pd.DataFrame(candles_records)
    _WORKER_CANDLE_ROWS = PaperTradingEngine._extract_backtest_rows(_WORKER_CANDLES_DF)
    _WORKER_REGIME_MASK = None if regime_mask is None else list(regime_mask)


def _worker_candles_dataframe(*, copy_deep: bool = True) -> pd.DataFrame:
    if _WORKER_CANDLES_DF is None:
        raise RuntimeError("Optimizer worker candles were not initialized.")
    return _WORKER_CANDLES_DF.copy(deep=copy_deep)


def _worker_candle_rows() -> list[dict[str, object]]:
    if _WORKER_CANDLE_ROWS is None:
        raise RuntimeError("Optimizer worker candle rows were not initialized.")
    return list(_WORKER_CANDLE_ROWS)


def _worker_regime_mask() -> list[int] | None:
    if _WORKER_REGIME_MASK is None:
        return None
    return list(_WORKER_REGIME_MASK)


def _available_memory_bytes() -> int | None:
    # Windows path (preferred for this environment).
    with suppress(Exception):
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = _MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):  # type: ignore[attr-defined]
            return int(memory_status.ullAvailPhys)

    # POSIX fallback.
    with suppress(Exception):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and avail_pages > 0:
            return page_size * avail_pages
    return None


def _memory_safe_worker_count(total_profiles: int) -> int:
    if total_profiles <= 0:
        return 1
    requested = min(MAX_OPTIMIZATION_WORKERS, total_profiles)
    if FORCE_MAX_OPTIMIZATION_WORKERS:
        return max(1, requested)
    available_memory = _available_memory_bytes()
    if available_memory is None:
        return max(1, requested)

    reserve_bytes = int(OPTIMIZER_MEMORY_RESERVE_GB * (1024**3))
    per_worker_bytes = max(1, int(OPTIMIZER_MEMORY_PER_WORKER_GB * (1024**3)))
    usable_bytes = max(0, available_memory - reserve_bytes)
    by_memory = max(1, usable_bytes // per_worker_bytes)
    return max(1, min(requested, int(by_memory)))


def _create_optimizer_pool(
    worker_count: int,
    candles_records: list[dict[str, object]],
    *,
    regime_mask: Sequence[int] | None = None,
) -> Pool:
    return get_context("spawn").Pool(
        processes=worker_count,
        initializer=_initialize_optimizer_worker,
        initargs=(candles_records, None if regime_mask is None else list(regime_mask)),
        maxtasksperchild=OPTIMIZER_WORKER_MAX_TASKS_PER_CHILD,
    )


def _finalize_signal_series(
    candles_df: pd.DataFrame,
    raw_signals: Sequence[int],
    *,
    required_candles: int,
    setup_gate: SmartSetupGate | None,
    strategy_name: str,
    regime_mask: Sequence[int] | None = None,
) -> tuple[list[int], int, int, int]:
    signals: list[int] = []
    total_signals = 0
    approved_signals = 0
    blocked_signals = 0

    for index, raw_signal in enumerate(raw_signals):
        signal_direction = int(raw_signal)
        if (
            signal_direction != 0
            and regime_mask is not None
            and index < len(regime_mask)
            and int(regime_mask[index]) <= 0
        ):
            signal_direction = 0
        if index + 1 < required_candles or signal_direction == 0:
            signals.append(0)
            continue
        total_signals += 1
        if setup_gate is not None:
            candles_slice = candles_df.iloc[: index + 1]
            is_approved, _score, _reason = setup_gate.evaluate_signal(
                candles_slice,
                signal_direction,
                strategy_name,
            )
            if not is_approved:
                blocked_signals += 1
                signals.append(0)
                continue
        approved_signals += 1
        signals.append(signal_direction)
    return signals, total_signals, approved_signals, blocked_signals


def _build_vectorized_strategy_cache_payload(
    candles_df: pd.DataFrame,
    *,
    strategy_name: str,
    required_candles: int,
    setup_gate: SmartSetupGate | None,
    strategy_profile: OptimizationProfile | None = None,
    regime_mask: Sequence[int] | None = None,
) -> dict[str, object] | None:
    if strategy_name == "ema_cross_volume":
        with _temporary_strategy_profile(strategy_profile, candles_df):
            working_df = build_ema_cross_volume_signal_frame(candles_df)
            raw_signals = (
                working_df["ema_long_entry"].astype("int8")
                - working_df["ema_short_entry"].astype("int8")
            ).tolist()
            payload: dict[str, object] = {}
        signals, total_signals, approved_signals, blocked_signals = _finalize_signal_series(
            working_df,
            raw_signals,
            required_candles=required_candles,
            setup_gate=setup_gate,
            strategy_name=strategy_name,
            regime_mask=regime_mask,
        )
        payload.update({
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
        })
        return payload

    if strategy_name == "frama_cross":
        with _temporary_strategy_profile(strategy_profile, candles_df):
            working_df = build_frama_cross_signal_frame(candles_df)
            raw_signals = (
                working_df["frama_long_entry"].astype("int8")
                - working_df["frama_short_entry"].astype("int8")
            ).tolist()
        signals, total_signals, approved_signals, blocked_signals = _finalize_signal_series(
            working_df,
            raw_signals,
            required_candles=required_candles,
            setup_gate=setup_gate,
            strategy_name=strategy_name,
            regime_mask=regime_mask,
        )
        return {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
        }

    if strategy_name == "dual_thrust":
        with _temporary_strategy_profile(strategy_profile, candles_df):
            working_df = build_dual_thrust_signal_frame(candles_df)
            raw_signals = (
                working_df["dual_long_entry"].astype("int8")
                - working_df["dual_short_entry"].astype("int8")
            ).tolist()
            long_exit_flags = working_df["dual_long_exit"].fillna(False).astype(bool).tolist()
            short_exit_flags = working_df["dual_short_exit"].fillna(False).astype(bool).tolist()
        signals, total_signals, approved_signals, blocked_signals = _finalize_signal_series(
            working_df,
            raw_signals,
            required_candles=required_candles,
            setup_gate=setup_gate,
            strategy_name=strategy_name,
            regime_mask=regime_mask,
        )
        return {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
            "precomputed_long_exit_flags": long_exit_flags,
            "precomputed_short_exit_flags": short_exit_flags,
        }

    return None


@contextmanager
def _temporary_strategy_profile(
    strategy_profile: OptimizationProfile | None,
    candles_dataframe: pd.DataFrame | None = None,
):
    if strategy_profile is None and candles_dataframe is None:
        yield
        return

    effective_strategy_profile: OptimizationProfile = (
        {}
        if strategy_profile is None
        else dict(strategy_profile)
    )
    original_values = {
        "dual_thrust_period": settings.strategy.dual_thrust_period,
        "dual_thrust_k1": settings.strategy.dual_thrust_k1,
        "dual_thrust_k2": settings.strategy.dual_thrust_k2,
        "ema_fast_period": settings.strategy.ema_fast_period,
        "ema_slow_period": settings.strategy.ema_slow_period,
        "frama_fast_period": settings.strategy.frama_fast_period,
        "frama_slow_period": settings.strategy.frama_slow_period,
        "volume_multiplier": settings.strategy.volume_multiplier,
    }
    with _OPTIMIZER_SETTINGS_LOCK:
        if candles_dataframe is not None:
            _clear_strategy_indicator_cache(candles_dataframe)
        for field_name, original_value in original_values.items():
            if field_name not in effective_strategy_profile:
                continue
            profile_value = effective_strategy_profile[field_name]
            if isinstance(original_value, bool):
                converted_value = bool(profile_value)
            elif isinstance(original_value, int):
                converted_value = int(profile_value)
            else:
                converted_value = float(profile_value)
            object.__setattr__(settings.strategy, field_name, converted_value)
        # Apply per-coin strategy parameter overrides when available (live overrides)
        if candles_dataframe is not None:
            try:
                symbol = candles_dataframe["symbol"].iloc[0]
            except Exception:
                symbol = None
            coin_params = getattr(settings.strategy, "coin_strategy_params", {}) or {}
            overrides = coin_params.get(symbol)
            if overrides:
                for field_name, param_value in overrides.items():
                    # Do not override values explicitly provided by strategy_profile
                    if field_name in effective_strategy_profile:
                        continue
                    if field_name not in original_values:
                        # only support known original fields
                        continue
                    original_value = original_values[field_name]
                    if isinstance(original_value, bool):
                        converted_value = bool(param_value)
                    elif isinstance(original_value, int):
                        converted_value = int(param_value)
                    else:
                        converted_value = float(param_value)
                    object.__setattr__(settings.strategy, field_name, converted_value)
        try:
            yield
        finally:
            for field_name, original_value in original_values.items():
                object.__setattr__(settings.strategy, field_name, original_value)
            if candles_dataframe is not None:
                _clear_strategy_indicator_cache(candles_dataframe)


def _evaluate_strategy_signal(
    strategy_name: str,
    candles_dataframe: pd.DataFrame,
    strategy_profile: OptimizationProfile | None = None,
) -> int:
    validated_strategy_name = _validate_strategy_name(strategy_name)
    with _temporary_strategy_profile(strategy_profile, candles_dataframe):
        return STRATEGY_RUNNERS[validated_strategy_name](candles_dataframe)


def resolve_strategy_for_symbol(
    symbol: str,
    default_strategy_name: str | None = None,
    *,
    use_coin_override: bool = True,
) -> str:
    fallback_strategy = (
        settings.strategy.default_strategy_name if default_strategy_name is None else default_strategy_name
    )
    if use_coin_override and symbol in COIN_STRATEGIES:
        return _validate_strategy_name(COIN_STRATEGIES[symbol])
    return _validate_strategy_name(fallback_strategy)


def get_strategy_badge(
    symbol: str,
    default_strategy_name: str | None = None,
    *,
    use_coin_override: bool = True,
) -> str:
    strategy_name = resolve_strategy_for_symbol(
        symbol,
        default_strategy_name,
        use_coin_override=use_coin_override,
    )
    if strategy_name in {"ema_cross_volume", "frama_cross", "dual_thrust"}:
        return "TREND"
    return strategy_name.upper()


def _should_exit_strategy_position(
    strategy_name: str,
    candles_dataframe: pd.DataFrame,
    *,
    side: str,
) -> bool:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "dual_thrust":
        return should_exit_python_dual_thrust_breakout(candles_dataframe, side)
    return False


def _build_strategy_exit_rule(
    strategy_name: str,
    candles_dataframe: pd.DataFrame,
    *,
    strategy_profile: OptimizationProfile | None = None,
) -> Callable[[PaperTrade, dict[str, object]], str | None] | None:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "dual_thrust":
        open_time_index = {
            candle_row["open_time"]: index
            for index, candle_row in enumerate(candles_dataframe.to_dict("records"))
        }

        def strategy_exit_rule(trade: PaperTrade, candle_row: dict[str, object]) -> str | None:
            candle_index = open_time_index.get(candle_row["open_time"])
            if candle_index is None or candle_index < 0:
                return None
            candles_slice = candles_dataframe.iloc[: candle_index + 1]
            with _temporary_strategy_profile(strategy_profile):
                should_exit = _should_exit_strategy_position(strategy_name, candles_slice, side=trade.side)
            if should_exit:
                return "STRATEGY_EXIT"
            return None

        return strategy_exit_rule

    return None


def _resolve_leverage_override(leverage: int | None) -> int | None:
    if leverage is None or leverage == settings.trading.default_leverage:
        return None
    return leverage


def _format_leverage_log(
    *,
    prefix: str,
    symbol: str,
    effective_leverage: int,
    configured_leverage: int,
    leverage_override: int | None,
    manual_override_forced: bool = False,
) -> str:
    if manual_override_forced:
        return (
            f"{prefix} {symbol}: "
            f"effective_leverage={effective_leverage}x (forced by GUI manual override)"
        )
    details = [f"gui={configured_leverage}x"]
    if leverage_override is None:
        details.append("override=profile/default")
    else:
        details.append(f"override=manual {leverage_override}x")
    profile = settings.trading.coin_profiles.get(symbol)
    if profile is not None and profile.default_leverage is not None:
        details.append(f"coin_profile={profile.default_leverage}x")
    return f"{prefix} {symbol}: effective_leverage={effective_leverage}x ({', '.join(details)})"


def _format_runtime_settings_log(
    *,
    symbol: str,
    interval: str,
    runtime_leverage: int,
    configured_leverage: int,
    leverage_override: int | None,
    manual_override_forced: bool = False,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_activation_pct: float,
    trailing_distance_pct: float,
) -> str:
    base = _format_leverage_log(
        prefix="Runtime profile",
        symbol=symbol,
        effective_leverage=runtime_leverage,
        configured_leverage=configured_leverage,
        leverage_override=leverage_override,
        manual_override_forced=manual_override_forced,
    )
    return (
        f"{base} "
        f"interval={interval} "
        f"sl={stop_loss_pct:.1f}% tp={take_profit_pct:.1f}% "
        f"trail_on={trailing_activation_pct:.1f}% trail_gap={trailing_distance_pct:.1f}%"
    )


def _trace_datetime_text(value: object) -> str:
    if value is None:
        return TRACE_VALUE_UNAVAILABLE
    if isinstance(value, datetime):
        resolved = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return resolved.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return TRACE_VALUE_UNAVAILABLE
    return text.replace("T", " ").replace("Z", "")


def _trace_normalize_value(value: object) -> object:
    if value is None:
        return TRACE_VALUE_UNAVAILABLE
    if isinstance(value, datetime):
        return _trace_datetime_text(value)
    if isinstance(value, float):
        if pd.isna(value):
            return TRACE_VALUE_UNAVAILABLE
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return value


def _trace_json_payload(payload: dict[str, object]) -> str:
    normalized_payload = {
        key: _trace_normalize_value(value)
        for key, value in payload.items()
    }
    return json.dumps(
        normalized_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _optimization_sort_key(
    summary: dict[str, float],
    *,
    strategy_name: str | None = None,
) -> tuple[float, ...]:
    profit_factor = float(summary.get("profit_factor", 0.0))
    max_dd = float(summary.get("max_drawdown_pct", 0.0))
    win_rate = float(summary.get("win_rate_pct", 0.0))
    pf_pass = 1.0 if profit_factor > RANKING_MIN_PROFIT_FACTOR else 0.0
    dd_pass = 1.0 if max_dd < RANKING_MAX_DRAWDOWN_PCT else 0.0
    return (
        pf_pass,
        dd_pass,
        profit_factor,
        -max_dd,
        win_rate,
        float(summary.get("real_rrr", 0.0)),
        float(summary.get("total_pnl_usd", 0.0)),
    )


OPTIMIZATION_METRIC_FIELDS = {
    "total_pnl_usd",
    "win_rate_pct",
    "profit_factor",
    "total_trades",
    "real_rrr",
    "average_win_usd",
    "average_loss_usd",
}


def _extract_profile_from_summary(summary: dict[str, float]) -> OptimizationProfile:
    return {
        key: float(value)
        for key, value in summary.items()
        if key not in OPTIMIZATION_METRIC_FIELDS
    }


def _strategy_variant_field_names(strategy_name: str) -> tuple[str, ...]:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "ema_cross_volume":
        return ("ema_fast_period", "ema_slow_period", "volume_multiplier")
    if strategy_name == "frama_cross":
        return ("frama_fast_period", "frama_slow_period", "volume_multiplier")
    if strategy_name == "dual_thrust":
        return ("dual_thrust_period", "dual_thrust_k1", "dual_thrust_k2")
    return ()


def _collect_strategy_variant_profiles(
    strategy_name: str,
    profiles: Sequence[OptimizationProfile],
) -> list[OptimizationProfile]:
    variant_fields = _strategy_variant_field_names(strategy_name)
    if not variant_fields:
        return []

    unique_profiles: list[OptimizationProfile] = []
    seen_keys: set[object] = set()
    for profile in profiles:
        variant_profile = {
            field_name: float(profile[field_name])
            for field_name in variant_fields
            if field_name in profile
        }
        if not variant_profile:
            continue
        cache_key = _strategy_cache_key(strategy_name, variant_profile)
        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)
        unique_profiles.append(variant_profile)
    if not unique_profiles:
        return _strategy_cache_profiles(strategy_name)
    return unique_profiles


def _sample_optimization_profiles(
    profiles: Sequence[OptimizationProfile],
    *,
    max_profiles: int,
    symbol: str,
    strategy_name: str,
    interval: str,
    random_seed: int,
) -> list[OptimizationProfile]:
    if len(profiles) <= max_profiles:
        return [dict(profile) for profile in profiles]

    seed_material = (
        f"{int(random_seed)}|{symbol}|{strategy_name}|{interval}|{len(profiles)}|{max_profiles}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False)
    randomizer = Random(seed)
    sampled_indices = sorted(randomizer.sample(range(len(profiles)), max_profiles))
    return [dict(profiles[index]) for index in sampled_indices]


def _validation_sort_key(
    validation_summary: dict[str, float],
    search_summary: dict[str, float] | None,
) -> tuple[float, float, float, float, float]:
    search_summary = {} if search_summary is None else search_summary
    return (
        float(validation_summary.get("profit_factor", 0.0)),
        float(validation_summary.get("real_rrr", 0.0)),
        float(validation_summary.get("total_pnl_usd", 0.0)),
        float(search_summary.get("profit_factor", 0.0)),
        float(search_summary.get("real_rrr", 0.0)),
    )


def _format_profile_dict(profile: OptimizationProfile) -> str:
    formatted_parts: list[str] = []
    integer_fields = {
        "dual_thrust_period",
        "ema_fast_period",
        "ema_slow_period",
        "frama_fast_period",
        "frama_slow_period",
    }
    for key, value in profile.items():
        numeric_value = float(value)
        if key in integer_fields:
            formatted_parts.append(f"'{key}': {int(round(numeric_value))}")
        else:
            formatted_parts.append(f"'{key}': {numeric_value:g}")
    formatted_items = ", ".join(formatted_parts)
    return "{" + formatted_items + "}"


def _strategy_cache_profiles(strategy_name: str) -> list[OptimizationProfile]:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "ema_cross_volume":
        return [
            {
                "ema_fast_period": float(ema_fast_period),
                "ema_slow_period": float(ema_slow_period),
                "volume_multiplier": float(volume_multiplier),
            }
            for ema_fast_period, ema_slow_period, volume_multiplier in product(
                settings.strategy.ema_fast_options,
                settings.strategy.ema_slow_options,
                EMA_VOLUME_MULTIPLIER_OPTIONS,
            )
            if int(ema_fast_period) < int(ema_slow_period)
        ]
    if strategy_name == "frama_cross":
        return [
            {
                "frama_fast_period": float(frama_fast_period),
                "frama_slow_period": float(frama_slow_period),
                "volume_multiplier": float(volume_multiplier),
            }
            for frama_fast_period, frama_slow_period, volume_multiplier in product(
                settings.strategy.frama_fast_options,
                settings.strategy.frama_slow_options,
                FRAMA_VOLUME_MULTIPLIER_OPTIONS,
            )
            if int(frama_fast_period) < int(frama_slow_period)
        ]
    if strategy_name == "dual_thrust":
        return [
            {
                "dual_thrust_period": float(dual_thrust_period),
                "dual_thrust_k1": float(dual_thrust_k1),
                "dual_thrust_k2": float(dual_thrust_k2),
            }
            for dual_thrust_period, dual_thrust_k1, dual_thrust_k2 in product(
                settings.strategy.dual_thrust_opt_periods,
                settings.strategy.dual_thrust_opt_k_values,
                settings.strategy.dual_thrust_opt_k_values,
            )
        ]
    return []


def _strategy_cache_key(strategy_name: str, strategy_profile: OptimizationProfile) -> object:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "ema_cross_volume":
        return (
            int(strategy_profile.get("ema_fast_period", 0.0)),
            int(strategy_profile.get("ema_slow_period", 0.0)),
            float(strategy_profile.get("volume_multiplier", 0.0)),
        )
    if strategy_name == "frama_cross":
        return (
            int(strategy_profile.get("frama_fast_period", 0.0)),
            int(strategy_profile.get("frama_slow_period", 0.0)),
            float(strategy_profile.get("volume_multiplier", 0.0)),
        )
    if strategy_name == "dual_thrust":
        return (
            int(strategy_profile.get("dual_thrust_period", 0.0)),
            float(strategy_profile.get("dual_thrust_k1", 0.0)),
            float(strategy_profile.get("dual_thrust_k2", 0.0)),
        )
    return strategy_name


def _candles_to_dataframe(candles: Sequence[CandleRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": candle.symbol,
                "interval": candle.interval,
                "open_time": candle.open_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )


class _NoopDatabase:
    pass


def _calculate_trade_metrics_for_closed_trades(closed_trades: Sequence[dict]) -> dict[str, float]:
    wins = [float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) > 0.0]
    losses = [-float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) < 0.0]

    average_win_usd = sum(wins) / len(wins) if wins else 0.0
    average_loss_usd = sum(losses) / len(losses) if losses else 0.0
    if average_loss_usd == 0.0:
        real_rrr = float("inf") if average_win_usd > 0.0 else 0.0
    else:
        real_rrr = average_win_usd / average_loss_usd

    return {
        "average_win_usd": average_win_usd,
        "average_loss_usd": average_loss_usd,
        "real_rrr": real_rrr,
    }


def _generate_signals_for_worker(
    candles_df: pd.DataFrame,
    *,
    strategy_name: str,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    strategy_profile: OptimizationProfile | None = None,
    regime_mask: Sequence[int] | None = None,
) -> tuple[list[int], int, int, int]:
    with _temporary_strategy_profile(strategy_profile):
        required_candles = _required_candle_count_for_strategy(
            strategy_name,
            use_setup_gate=use_setup_gate,
        )
    setup_gate = SmartSetupGate(min_confidence_pct=min_confidence_pct) if use_setup_gate else None

    vectorized_payload = _build_vectorized_strategy_cache_payload(
        candles_df,
        strategy_name=strategy_name,
        required_candles=required_candles,
        setup_gate=setup_gate,
        strategy_profile=strategy_profile,
        regime_mask=regime_mask,
    )
    if vectorized_payload is not None:
        return (
            list(vectorized_payload["signals"]),
            int(vectorized_payload["total_signals"]),
            int(vectorized_payload["approved_signals"]),
            int(vectorized_payload["blocked_signals"]),
        )

    signals: list[int] = []
    total_signals = 0
    approved_signals = 0
    blocked_signals = 0
    for index in range(len(candles_df)):
        if index + 1 < required_candles:
            signals.append(0)
            continue
        candles_slice = candles_df.iloc[: index + 1]
        signal_direction = _evaluate_strategy_signal(
            strategy_name,
            candles_slice,
            strategy_profile,
        )
        if (
            signal_direction != 0
            and regime_mask is not None
            and index < len(regime_mask)
            and int(regime_mask[index]) <= 0
        ):
            signal_direction = 0
        if signal_direction != 0 and setup_gate is not None:
            total_signals += 1
            is_approved, _score, _reason = setup_gate.evaluate_signal(
                candles_slice,
                signal_direction,
                strategy_name,
            )
            if not is_approved:
                blocked_signals += 1
                signal_direction = 0
            else:
                approved_signals += 1
        elif signal_direction != 0:
            total_signals += 1
            approved_signals += 1
        signals.append(signal_direction)
    return signals, total_signals, approved_signals, blocked_signals


def _task_candles_dataframe(task: dict[str, object], *, copy_deep: bool = False) -> pd.DataFrame:
    candles_records = task.get("candles_records")
    if candles_records is not None:
        # avoid unnecessary deep copies inside worker processes
        df = pd.DataFrame(candles_records)
        return df.copy(deep=copy_deep) if copy_deep else df
    return _worker_candles_dataframe(copy_deep=copy_deep)


def _task_candle_rows(task: dict[str, object]) -> list[dict[str, object]]:
    candles_records = task.get("candles_records")
    if candles_records is not None:
        candles_df = pd.DataFrame(candles_records)
        return PaperTradingEngine._extract_backtest_rows(candles_df)
    return _worker_candle_rows()


def _task_regime_mask(task: dict[str, object]) -> list[int] | None:
    task_regime_mask = task.get("regime_mask")
    if task_regime_mask is not None:
        return list(task_regime_mask)
    return _worker_regime_mask()


def _build_precomputed_exit_rule(
    candle_rows: Sequence[dict[str, object]],
    *,
    long_exit_flags: Sequence[bool] | None = None,
    short_exit_flags: Sequence[bool] | None = None,
) -> Callable[[PaperTrade, dict[str, object]], str | None] | None:
    if (
        long_exit_flags is None
        and short_exit_flags is None
    ):
        return None

    open_time_index = {
        candle_row["open_time"]: index
        for index, candle_row in enumerate(candle_rows)
    }

    def strategy_exit_rule(trade: PaperTrade, candle_row: dict[str, object]) -> str | None:
        candle_index = open_time_index.get(candle_row["open_time"])
        if candle_index is None or candle_index < 0:
            return None
        if (
            trade.side.upper() == "LONG"
            and long_exit_flags is not None
            and candle_index < len(long_exit_flags)
            and bool(long_exit_flags[candle_index])
        ):
            return "STRATEGY_EXIT"
        if (
            trade.side.upper() == "SHORT"
            and short_exit_flags is not None
            and candle_index < len(short_exit_flags)
            and bool(short_exit_flags[candle_index])
        ):
            return "STRATEGY_EXIT"
        return None

    return strategy_exit_rule


def _run_optimization_profile_worker(task: dict[str, object]) -> dict[str, object]:
    candle_rows = _task_candle_rows(task)
    strategy_name = str(task["strategy_name"])
    strategy_profile = dict(task["strategy_profile"])
    use_setup_gate = bool(task["use_setup_gate"])
    min_confidence_pct = (
        None if task["min_confidence_pct"] is None else float(task["min_confidence_pct"])
    )
    regime_mask = _task_regime_mask(task)
    leverage_override = (
        None if task["leverage_override"] is None else int(task["leverage_override"])
    )
    symbol = str(task["symbol"])
    interval = str(task.get("interval") or settings.trading.interval)
    precomputed_signals = task.get("precomputed_signals")
    precomputed_exit_rule = _build_precomputed_exit_rule(
        candle_rows,
        long_exit_flags=task.get("precomputed_long_exit_flags"),
        short_exit_flags=task.get("precomputed_short_exit_flags"),
    )
    candles_df: pd.DataFrame | None = None
    if precomputed_signals is not None:
        signals = list(precomputed_signals)
        total_signals = int(task.get("total_signals", 0))
        approved_signals = int(task.get("approved_signals", 0))
        blocked_signals = int(task.get("blocked_signals", 0))
    else:
        candles_df = _task_candles_dataframe(task)
        signals = []
        total_signals = int(task.get("total_signals", 0))
        approved_signals = int(task.get("approved_signals", 0))
        blocked_signals = int(task.get("blocked_signals", 0))

    engine = PaperTradingEngine(
        _NoopDatabase(),
        symbol=symbol,
        interval=interval,
        leverage=leverage_override,
        take_profit_pct=float(strategy_profile["take_profit_pct"]),
        stop_loss_pct=float(strategy_profile["stop_loss_pct"]),
        trailing_activation_pct=float(strategy_profile["trailing_activation_pct"]),
        trailing_distance_pct=float(strategy_profile["trailing_distance_pct"]),
        enable_persistence=False,
    )
    if candles_df is None and precomputed_exit_rule is None:
        candles_df = _task_candles_dataframe(task)
    result = engine.run_historical_backtest(
        candles_df if precomputed_exit_rule is None else None,
        signals,
        strategy_exit_rule=(
            precomputed_exit_rule
            if precomputed_exit_rule is not None
            else _build_strategy_exit_rule(
                strategy_name,
                candles_df,
                strategy_profile=strategy_profile,
            )
        ),
        candle_rows=candle_rows,
    )
    result.update(_calculate_trade_metrics_for_closed_trades(result["closed_trades"]))

    summary = {
        **{key: float(value) for key, value in strategy_profile.items()},
        "total_pnl_usd": float(result["total_pnl_usd"]),
        "win_rate_pct": float(result["win_rate_pct"]),
        "profit_factor": float(result["profit_factor"]),
        "total_trades": float(result["total_trades"]),
        "average_win_usd": float(result["average_win_usd"]),
        "average_loss_usd": float(result["average_loss_usd"]),
        "real_rrr": float(result["real_rrr"]),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
        "longest_consecutive_losses": float(result.get("longest_consecutive_losses", 0.0)),
    }
    with suppress(Exception):
        result.pop("closed_trades", None)
    with suppress(Exception):
        result.clear()
    return {
        "summary": summary,
        "total_signals": total_signals,
        "approved_signals": approved_signals,
        "blocked_signals": blocked_signals,
    }


def _precompute_signal_cache_worker(task: dict[str, object]) -> dict[str, object]:
    candles_df = _task_candles_dataframe(task)
    strategy_name = str(task["strategy_name"])
    strategy_profile = dict(task["strategy_profile"])
    use_setup_gate = bool(task["use_setup_gate"])
    min_confidence_pct = (
        None if task["min_confidence_pct"] is None else float(task["min_confidence_pct"])
    )
    regime_mask = _task_regime_mask(task)
    with _temporary_strategy_profile(strategy_profile):
        required_candles = _required_candle_count_for_strategy(
            strategy_name,
            use_setup_gate=use_setup_gate,
        )
    setup_gate = SmartSetupGate(min_confidence_pct=min_confidence_pct) if use_setup_gate else None
    vectorized_payload = _build_vectorized_strategy_cache_payload(
        candles_df,
        strategy_name=strategy_name,
        required_candles=required_candles,
        setup_gate=setup_gate,
        strategy_profile=strategy_profile,
        regime_mask=regime_mask,
    )
    if vectorized_payload is None:
        (
            signals,
            total_signals,
            approved_signals,
            blocked_signals,
        ) = _generate_signals_for_worker(
            candles_df,
            strategy_name=strategy_name,
            use_setup_gate=use_setup_gate,
            min_confidence_pct=min_confidence_pct,
            strategy_profile=strategy_profile,
            regime_mask=regime_mask,
        )
        payload: dict[str, object] = {
            "cache_key": _strategy_cache_key(strategy_name, strategy_profile),
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
        }
    else:
        payload = {
            "cache_key": _strategy_cache_key(strategy_name, strategy_profile),
            **vectorized_payload,
        }
    return payload


def _precompute_dual_thrust_signal_cache(
    candles_df: pd.DataFrame,
    *,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
) -> dict[tuple[int, float, float], dict[str, object]]:
    signal_cache: dict[tuple[int, float, float], dict[str, object]] = {}
    for dual_thrust_period, dual_thrust_k1, dual_thrust_k2 in product(
        settings.strategy.dual_thrust_opt_periods,
        settings.strategy.dual_thrust_opt_k_values,
        settings.strategy.dual_thrust_opt_k_values,
    ):
        strategy_profile = {
            "dual_thrust_period": float(dual_thrust_period),
            "dual_thrust_k1": float(dual_thrust_k1),
            "dual_thrust_k2": float(dual_thrust_k2),
        }
        (
            signals,
            total_signals,
            approved_signals,
            blocked_signals,
        ) = _generate_signals_for_worker(
            candles_df.copy(deep=True),
            strategy_name="dual_thrust",
            use_setup_gate=use_setup_gate,
            min_confidence_pct=min_confidence_pct,
            strategy_profile=strategy_profile,
        )
        signal_cache[(int(dual_thrust_period), float(dual_thrust_k1), float(dual_thrust_k2))] = {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
        }
    return signal_cache


class BotEngineThread(QThread):
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)
    trade_opened = pyqtSignal(dict)
    positions_updated = pyqtSignal(list)
    price_update = pyqtSignal(str, float)
    heartbeat_status = pyqtSignal(dict)

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        strategy_name: str | None = None,
        intervals: Sequence[str] | None = None,
        symbol_intervals: dict[str, str] | None = None,
        leverage: int | None = None,
        min_confidence_pct: float | None = None,
        db_path: str | Path = "data/paper_trading.duckdb",
    ) -> None:
        super().__init__()
        if not symbols:
            raise ValueError("At least one symbol must be configured.")
        normalized_symbols = [
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        ]
        if not normalized_symbols:
            raise ValueError("At least one non-empty symbol must be configured.")
        self._symbols = list(dict.fromkeys(normalized_symbols))
        self._strategy_name = _validate_strategy_name(
            settings.strategy.default_strategy_name if strategy_name is None else strategy_name
        )
        fallback_interval = (
            settings.live.default_interval
            if not intervals
            else _validate_interval_name(next(iter(intervals)))
        )
        normalized_symbol_intervals = {
            str(symbol).strip().upper(): str(interval)
            for symbol, interval in (symbol_intervals or {}).items()
            if str(symbol).strip()
        }
        if symbol_intervals:
            self._symbol_intervals = {
                symbol: _validate_interval_name(normalized_symbol_intervals.get(symbol, fallback_interval))
                for symbol in self._symbols
            }
        else:
            self._symbol_intervals = {
                symbol: resolve_interval_for_symbol(symbol, fallback_interval)
                for symbol in self._symbols
            }
        self._intervals = list(dict.fromkeys(self._symbol_intervals.values()))
        self._configured_leverage = settings.trading.default_leverage if leverage is None else leverage
        self._leverage_override = _resolve_leverage_override(leverage)
        self._use_setup_gate = settings.trading.use_setup_gate
        self._min_confidence_pct = (
            None
            if not self._use_setup_gate
            else (
                settings.trading.min_confidence_pct
                if min_confidence_pct is None
                else float(min_confidence_pct)
            )
        )
        self._setup_gate = (
            SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
            if self._use_setup_gate
            else None
        )
        if any(symbol in PRODUCTION_PROFILE_REGISTRY for symbol in self._symbols):
            # Production registry enforces Phase 19 gate path for these symbols.
            self._use_setup_gate = True
            if self._min_confidence_pct is None:
                self._min_confidence_pct = settings.trading.min_confidence_pct
            self._setup_gate = SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
        self._db_path = Path(db_path)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._db: Database | None = None
        self._paper_engine: PaperTradingEngine | None = None
        self._ws_manager: MultiTimeframeWebSocketManager | None = None
        self._last_closed_candle_times: dict[tuple[str, str], datetime] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._latest_market_prices: dict[str, float] = {}
        # Throttling state for in-place terminal progress printing
        self._last_signal_cache_print_time = 0.0
        self._last_optimization_print_time = 0.0
        self._last_signal_cache_progress_ratio = 0.0
        self._last_optimization_progress_ratio = 0.0

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as exc:
            self.log_message.emit(f"BotEngineThread stopped with error: {exc}")
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            self._loop = None

    def stop(self) -> None:
        self.requestInterruption()
        if self._loop is not None and self._ws_manager is not None:
            future = asyncio.run_coroutine_threadsafe(self._ws_manager.stop(), self._loop)
            future.add_done_callback(self._handle_stop_future)

    def request_manual_close(self, symbol: str, exit_price: float) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._close_position_manually, symbol, float(exit_price))

    async def _run_async(self) -> None:
        self._db = Database(self._db_path)
        self._paper_engine = PaperTradingEngine(
            self._db,
            symbol=self._symbols[0] if len(self._symbols) == 1 else None,
            interval=self._symbol_intervals.get(self._symbols[0]) if len(self._symbols) == 1 else None,
            leverage=self._leverage_override,
        )
        self._include_recovered_position_symbols()
        self.log_message.emit(
            f"[RECOVERY] Restored {len(self._paper_engine.active_trades)} active trades from storage."
        )
        self._seed_recent_candles()
        self._emit_latest_prices_from_db()
        self._initialize_reconnect_baseline()
        self._ws_manager = MultiTimeframeWebSocketManager(
            self._symbols,
            self._intervals,
            symbol_intervals=self._symbol_intervals,
            on_candle_closed=self._on_candle_closed,
            on_candle_update=self._on_candle_update,
            on_log=self._on_ws_log,
            on_reconnect=self._on_ws_reconnect,
        )

        self.log_message.emit(
            "BotEngineThread started for "
            f"symbols={','.join(self._symbols)} intervals={','.join(self._intervals)} "
            f"default_strategy={self._strategy_name} leverage={self._configured_leverage}x"
        )
        for symbol in self._symbols:
            resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            runtime_settings = self._paper_engine.get_runtime_settings(symbol)
            if symbol in PRODUCTION_PROFILE_REGISTRY:
                profile_strategy = str(PRODUCTION_PROFILE_REGISTRY[symbol].get("strategy_name", ""))
                profile_strategy = STRATEGY_NAME_ALIASES.get(profile_strategy, profile_strategy)
                self.log_message.emit(
                    f"[PRODUCTION] Loaded optimized profile for {symbol} "
                    f"({profile_strategy} + HMM enabled)."
                )
            self.log_message.emit(f"Strategy switcher: {symbol} -> {resolved_strategy_name}")
            self.log_message.emit(
                _format_leverage_log(
                    prefix="Runtime leverage",
                    symbol=symbol,
                    effective_leverage=runtime_settings.leverage,
                    configured_leverage=self._configured_leverage,
                    leverage_override=self._leverage_override,
                )
            )
            self.log_message.emit(
                _format_runtime_settings_log(
                    symbol=symbol,
                    interval=runtime_settings.interval,
                    runtime_leverage=runtime_settings.leverage,
                    configured_leverage=self._configured_leverage,
                    leverage_override=self._leverage_override,
                    stop_loss_pct=runtime_settings.stop_loss_pct,
                    take_profit_pct=runtime_settings.take_profit_pct,
                    trailing_activation_pct=runtime_settings.trailing_activation_pct,
                    trailing_distance_pct=runtime_settings.trailing_distance_pct,
                )
            )
            self.log_message.emit(
                f"Live interval routing: {symbol} -> {self._target_interval_for_symbol(symbol)}"
            )
        self._emit_positions_snapshot()
        await self._perform_heartbeat_check()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.progress_update.emit(100, "Live Stream active")
        try:
            await self._ws_manager.start()
        finally:
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._heartbeat_task
                self._heartbeat_task = None
            if self._db is not None:
                self._db.close()
                self._db = None
            self._paper_engine = None
            self._ws_manager = None
            self.log_message.emit("BotEngineThread stopped.")

    def _include_recovered_position_symbols(self) -> None:
        if self._paper_engine is None:
            return
        recovered_symbols = [
            str(trade.symbol).strip().upper()
            for trade in self._paper_engine.active_trades
            if str(trade.symbol).strip()
        ]
        missing_symbols = [
            symbol
            for symbol in dict.fromkeys(recovered_symbols)
            if symbol not in self._symbols
        ]
        if not missing_symbols:
            return

        self._symbols.extend(missing_symbols)
        for symbol in missing_symbols:
            self._symbol_intervals[symbol] = resolve_interval_for_symbol(
                symbol,
                settings.live.default_interval,
            )
        self._intervals = list(dict.fromkeys(self._symbol_intervals[symbol] for symbol in self._symbols))
        self.log_message.emit(
            "[RECOVERY] Added symbols from restored open positions to live stream: "
            + ", ".join(missing_symbols)
        )

    def _close_position_manually(self, symbol: str, exit_price: float) -> None:
        if self._db is None or self._paper_engine is None:
            return

        closed_trade_id = self._paper_engine.close_position_at_price(
            symbol,
            exit_price,
            status="MANUAL_CLOSE",
        )
        if closed_trade_id is None:
            self.log_message.emit(f"Manual close skipped: no open position for {symbol}.")
            return

        closed_trade = self._db.fetch_trade_by_id(closed_trade_id)
        if closed_trade is None:
            self.log_message.emit(
                f"Trade manually closed: symbol={symbol} price={exit_price:.4f}"
            )
        else:
            self.log_message.emit(
                "Trade closed: "
                f"id={closed_trade.id} price={exit_price:.4f} "
                f"reason={closed_trade.status} net_pnl={closed_trade.pnl:.4f} fees={closed_trade.total_fees:.4f}"
            )
        self._emit_positions_snapshot()

    async def _on_candle_closed(
        self,
        symbol: str,
        candle: CandleRecord,
        *,
        recovered: bool = False,
    ) -> None:
        if self._db is None or self._paper_engine is None:
            return

        target_interval = self._target_interval_for_symbol(symbol)
        if candle.interval != target_interval:
            return

        try:
            resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            self._db.upsert_candles([candle])
            # Keep a fallback market price for startup/recovery paths when no live tick update is present yet.
            self._latest_market_prices.setdefault(symbol, float(candle.close))
            if not recovered:
                self.log_message.emit(
                    f"Candle closed: {symbol} {candle.interval} @ {candle.open_time.isoformat()} close={candle.close:.4f}"
                )
            else:
                self.log_message.emit(
                    f"Recovered candle: {symbol} {candle.interval} @ {candle.open_time.isoformat()} close={candle.close:.4f}"
                )

            closed_trade_ids = self._paper_engine.update_positions(candle.close, symbol=symbol)
            for trade_id in closed_trade_ids:
                closed_trade = self._db.fetch_trade_by_id(trade_id)
                if closed_trade is None:
                    self.log_message.emit(f"Trade closed: id={trade_id} price={candle.close:.4f}")
                    continue
                self.log_message.emit(
                    "Trade closed: "
                    f"id={closed_trade.id} price={candle.close:.4f} "
                    f"reason={closed_trade.status} net_pnl={closed_trade.pnl:.4f} fees={closed_trade.total_fees:.4f}"
                )
            if closed_trade_ids:
                self._emit_positions_snapshot()

            required_candle_count = _required_candle_count_for_strategy(
                resolved_strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
            analysis_window = _live_analysis_window_for_strategy(
                resolved_strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
            recent_candles = self._db.fetch_recent_candles(
                symbol,
                candle.interval,
                limit=analysis_window,
            )
            if len(recent_candles) < required_candle_count:
                return

            candles_dataframe = _candles_to_dataframe(recent_candles)
            open_trades = self._db.fetch_open_trades(symbol=symbol)
            if open_trades:
                open_trade = open_trades[0]
                if _should_exit_strategy_position(
                    resolved_strategy_name,
                    candles_dataframe,
                    side=open_trade.side,
                ):
                    closed_trade_id = self._paper_engine.close_position_at_price(
                        symbol,
                        candle.close,
                        status="STRATEGY_EXIT",
                    )
                    if closed_trade_id is not None:
                        closed_trade = self._db.fetch_trade_by_id(closed_trade_id)
                        if closed_trade is not None:
                            self.log_message.emit(
                                "Trade closed: "
                                f"id={closed_trade.id} price={candle.close:.4f} "
                                f"reason={closed_trade.status} net_pnl={closed_trade.pnl:.4f} fees={closed_trade.total_fees:.4f}"
                            )
                        self._emit_positions_snapshot()
                    return

            if recovered:
                # Recovered candles are historical gap fills; avoid opening fresh entries on stale prices.
                return

            execution_price = float(self._latest_market_prices.get(symbol, candle.close))
            signal_direction = _evaluate_strategy_signal(
                resolved_strategy_name,
                candles_dataframe,
            )
            signal_name = "LONG" if signal_direction > 0 else "SHORT" if signal_direction < 0 else "NEUTRAL"
            self.log_message.emit(
                f"Signal evaluation: {signal_name} on {symbol} {candle.interval} via {resolved_strategy_name}"
            )
            if signal_direction != 0 and self._setup_gate is not None:
                is_approved, score, reason = self._setup_gate.evaluate_signal(
                    candles_dataframe,
                    signal_direction,
                    resolved_strategy_name,
                )
                if not is_approved:
                    self.log_message.emit(
                        f"Signal filtered by Setup Gate (Score: {score:.1f}%): {reason}"
                    )
                    return
            trade_id = self._paper_engine.process_signal(
                symbol=symbol,
                current_price=execution_price,
                signal_direction=signal_direction,
            )
            if trade_id is None:
                return

            trade = self._db.fetch_trade_by_id(trade_id)
            if trade is None:
                self.log_message.emit(f"Trade opened but could not be reloaded: id={trade_id}")
                return

            self.trade_opened.emit(self._trade_to_payload(trade))
            self._emit_positions_snapshot()
            self.log_message.emit(
                "Trade opened: "
                f"id={trade.id} symbol={trade.symbol} side={trade.side} leverage={trade.leverage}x "
                f"entry_price={trade.entry_price:.4f} signal_close={candle.close:.4f} "
                f"qty={trade.qty:.6f} entry_fee={trade.total_fees:.4f} high_water_mark={trade.high_water_mark:.4f}"
            )
        finally:
            self._last_closed_candle_times[(symbol, candle.interval)] = candle.open_time

    async def _on_candle_update(self, symbol: str, candle: CandleRecord) -> None:
        target_interval = self._target_interval_for_symbol(symbol)
        if candle.interval != target_interval:
            return
        self._latest_market_prices[symbol] = float(candle.close)
        self.price_update.emit(symbol, candle.close)

    def _on_ws_log(self, message: str) -> None:
        self.log_message.emit(message)

    async def _heartbeat_loop(self) -> None:
        while not self.isInterruptionRequested():
            await asyncio.sleep(300)
            if self.isInterruptionRequested():
                return
            await self._perform_heartbeat_check()

    async def _perform_heartbeat_check(self) -> None:
        if self._db is None:
            return

        now = datetime.now(tz=UTC).replace(tzinfo=None)
        status_payload: dict[str, dict[str, object]] = {}
        lagging_symbols: list[str] = []
        latest_summaries: list[str] = []

        for symbol in self._symbols:
            interval = self._target_interval_for_symbol(symbol)
            symbol_status = "OK"
            latest_time: datetime | None = None
            latest_interval: str | None = None
            worst_lag_seconds = 0
            interval_statuses: dict[str, dict[str, object]] = {}

            _count, _oldest_time, newest_time = self._db.get_candle_stats(symbol, interval)
            interval_seconds = _interval_total_seconds(interval)
            threshold_seconds = interval_seconds * 2
            lag_seconds = None if newest_time is None else max(
                0,
                int((now - newest_time).total_seconds()),
            )
            interval_state = (
                "OK"
                if lag_seconds is not None and lag_seconds < threshold_seconds
                else "LAGGING"
            )
            if interval_state == "LAGGING":
                symbol_status = "LAGGING"
            if lag_seconds is not None:
                worst_lag_seconds = max(worst_lag_seconds, lag_seconds)
            if newest_time is not None and (latest_time is None or newest_time > latest_time):
                latest_time = newest_time
                latest_interval = interval
            interval_statuses[interval] = {
                "status": interval_state,
                "latest_open_time": (
                    None if newest_time is None else newest_time.isoformat()
                ),
                "lag_seconds": lag_seconds,
                "threshold_seconds": threshold_seconds,
            }

            if symbol_status == "LAGGING":
                lagging_symbols.append(symbol)

            status_payload[symbol] = {
                "status": symbol_status,
                "latest_open_time": None if latest_time is None else latest_time.isoformat(),
                "latest_interval": latest_interval,
                "lag_seconds": worst_lag_seconds,
                "interval_statuses": interval_statuses,
            }

            latest_label = "--:--" if latest_time is None else latest_time.strftime("%H:%M")
            latest_summaries.append(f"{symbol} {interval}: {latest_label}")

        self.heartbeat_status.emit(status_payload)

        if lagging_symbols:
            self.log_message.emit(
                "[HEARTBEAT] "
                f"LAGGING {len(lagging_symbols)}/{len(self._symbols)} symbols. "
                f"Affected: {', '.join(lagging_symbols)}. "
                f"Latest {' | '.join(latest_summaries)}."
            )
        else:
            self.log_message.emit(
                "[HEARTBEAT] "
                f"All {len(self._symbols)} symbols synced. "
                f"Latest {' | '.join(latest_summaries)}."
            )

    async def _on_ws_reconnect(self) -> None:
        if self._db is None or self._paper_engine is None or self.isInterruptionRequested():
            return

        self._sync_recent_candles(progress_label="Re-syncing gap history...")
        recovered_total = 0
        current_timestamp_ms = int(datetime.now(tz=UTC).timestamp() * 1000)

        for interval in self._intervals:
            interval_symbols = self._symbols_for_interval(interval)
            if not interval_symbols:
                continue
            current_bucket_open_time = MultiTimeframeWebSocketManager._bucket_open_time(
                current_timestamp_ms,
                interval,
            )
            for symbol in interval_symbols:
                analysis_window = _live_analysis_window_for_strategy(
                    resolve_strategy_for_symbol(symbol, self._strategy_name),
                    use_setup_gate=self._use_setup_gate,
                )
                recent_candles = self._db.fetch_recent_candles(
                    symbol,
                    interval,
                    limit=analysis_window,
                )
                last_processed_time = self._last_closed_candle_times.get((symbol, interval))
                recovered_candles = [
                    candle
                    for candle in recent_candles
                    if candle.open_time < current_bucket_open_time
                    and (last_processed_time is None or candle.open_time > last_processed_time)
                ]
                for candle in recovered_candles:
                    await self._on_candle_closed(symbol, candle, recovered=True)
                    recovered_total += 1

        if recovered_total == 0:
            self.log_message.emit("Reconnect gap recovery finished: no missing closed candles found.")
        else:
            self.log_message.emit(
                f"Reconnect gap recovery finished: processed {recovered_total} missing closed candle(s)."
            )

    def set_leverage(self, leverage: int) -> None:
        self._configured_leverage = leverage
        self._leverage_override = _resolve_leverage_override(leverage)
        if self._paper_engine is not None:
            self._paper_engine.set_leverage(self._leverage_override)
            if self._leverage_override is None:
                self.log_message.emit(
                    f"Configured leverage reset to default {self._configured_leverage}x; coin profiles are active."
                )
            else:
                self.log_message.emit(
                    f"Configured leverage override updated to {self._configured_leverage}x"
                )
            for symbol in self._symbols:
                runtime_settings = self._paper_engine.get_runtime_settings(symbol)
                self.log_message.emit(
                    _format_leverage_log(
                        prefix="Runtime leverage",
                        symbol=symbol,
                        effective_leverage=runtime_settings.leverage,
                        configured_leverage=self._configured_leverage,
                        leverage_override=self._leverage_override,
                    )
                )
                self.log_message.emit(
                    _format_runtime_settings_log(
                        symbol=symbol,
                        interval=runtime_settings.interval,
                        runtime_leverage=runtime_settings.leverage,
                        configured_leverage=self._configured_leverage,
                        leverage_override=self._leverage_override,
                        stop_loss_pct=runtime_settings.stop_loss_pct,
                        take_profit_pct=runtime_settings.take_profit_pct,
                        trailing_activation_pct=runtime_settings.trailing_activation_pct,
                        trailing_distance_pct=runtime_settings.trailing_distance_pct,
                    )
                )

    @staticmethod
    def _trade_to_payload(trade: PaperTrade) -> dict:
        payload = asdict(trade)
        for key in ("entry_time", "exit_time"):
            if payload[key] is not None:
                payload[key] = payload[key].isoformat()
        return payload

    def _emit_positions_snapshot(self) -> None:
        if self._db is None:
            return
        payload = [self._trade_to_payload(trade) for trade in self._db.fetch_open_trades()]
        self.positions_updated.emit(payload)

    def _handle_stop_future(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            self.log_message.emit(f"BotEngineThread stop warning: {exc}")

    def _seed_recent_candles(self) -> None:
        if self._db is None:
            return
        self.log_message.emit(
            f"Live warmup target per symbol/interval: {self._live_history_target_count()} candles for settled indicators."
        )
        self._sync_recent_candles(progress_label="Preloading candles...", emit_progress=True)

    def _emit_latest_prices_from_db(self) -> None:
        if self._db is None:
            return
        for symbol in self._symbols:
            interval = self._target_interval_for_symbol(symbol)
            recent_candles = self._db.fetch_recent_candles(symbol, interval, limit=1)
            if not recent_candles:
                continue
            last_price = float(recent_candles[-1].close)
            self._latest_market_prices[symbol] = last_price
            self.price_update.emit(symbol, last_price)

    def _live_history_target_count(self) -> int:
        preload_counts = [
            _live_analysis_window_for_strategy(
                resolve_strategy_for_symbol(symbol, self._strategy_name),
                use_setup_gate=self._use_setup_gate,
            )
            for symbol in self._symbols
        ]
        minimum_target = 50
        if any(symbol in PRODUCTION_PROFILE_REGISTRY for symbol in self._symbols):
            minimum_target = LIVE_SETTLED_WARMUP_CANDLES
        return max(max(preload_counts, default=minimum_target), minimum_target)

    def _sync_recent_candles(
        self,
        *,
        progress_label: str,
        emit_progress: bool = False,
    ) -> None:
        if self._db is None:
            return

        preload_count = self._live_history_target_count()
        if emit_progress:
            self.progress_update.emit(10, progress_label)
        with BitunixClient(max_retries=6, backoff_factor=1.0) as client:
            with HistoryManager(self._db, client) as history_manager:
                total_intervals = max(len(self._intervals), 1)
                for interval_index, interval in enumerate(self._intervals):
                    interval_symbols = self._symbols_for_interval(interval)
                    if not interval_symbols:
                        continue
                    try:
                        saved_counts = history_manager.sync_recent_candles(
                            symbols=interval_symbols,
                            interval=interval,
                            candles_per_symbol=preload_count,
                            on_progress=(
                                None
                                if not emit_progress
                                else lambda progress_value, text, current_interval_index=interval_index: self._emit_seed_history_progress(
                                    interval_index=current_interval_index,
                                    total_intervals=total_intervals,
                                    progress_value=progress_value,
                                    text=text,
                                )
                            ),
                        )
                    except Exception as exc:
                        self.log_message.emit(
                            f"Incremental history sync failed for interval {interval}: {exc}"
                        )
                        continue

                    for symbol in interval_symbols:
                        saved_count = saved_counts.get(symbol, 0)
                        self.log_message.emit(
                            f"Incremental history sync ready for {symbol} {interval}. "
                            f"New candles added: {saved_count}."
                        )

    def _initialize_reconnect_baseline(self) -> None:
        if self._db is None:
            return

        current_timestamp_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        for interval in self._intervals:
            interval_symbols = self._symbols_for_interval(interval)
            if not interval_symbols:
                continue
            current_bucket_open_time = MultiTimeframeWebSocketManager._bucket_open_time(
                current_timestamp_ms,
                interval,
            )
            for symbol in interval_symbols:
                analysis_window = _live_analysis_window_for_strategy(
                    resolve_strategy_for_symbol(symbol, self._strategy_name),
                    use_setup_gate=self._use_setup_gate,
                )
                recent_candles = self._db.fetch_recent_candles(
                    symbol,
                    interval,
                    limit=analysis_window,
                )
                closed_candles = [
                    candle for candle in recent_candles if candle.open_time < current_bucket_open_time
                ]
                if closed_candles:
                    self._last_closed_candle_times[(symbol, interval)] = closed_candles[-1].open_time

    def _target_interval_for_symbol(self, symbol: str) -> str:
        return self._symbol_intervals.get(
            symbol,
            resolve_interval_for_symbol(symbol, settings.live.default_interval),
        )

    def _symbols_for_interval(self, interval: str) -> list[str]:
        return [
            symbol
            for symbol in self._symbols
            if self._target_interval_for_symbol(symbol) == interval
        ]

    def _emit_seed_history_progress(
        self,
        *,
        interval_index: int,
        total_intervals: int,
        progress_value: int,
        text: str,
    ) -> None:
        normalized_progress = max(0, min(100, progress_value))
        interval_ratio = interval_index / max(total_intervals, 1)
        overall_ratio = (interval_ratio + (normalized_progress / 100.0) / max(total_intervals, 1))
        scaled_value = 10 + int(overall_ratio * 80)
        self.progress_update.emit(min(scaled_value, 90), text)


class BacktestThread(QThread):
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)
    backtest_finished = pyqtSignal(dict)
    backtest_error = pyqtSignal(str)

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        strategy_name: str,
        leverage: int | None = None,
        min_confidence_pct: float | None = None,
        optimize_profile: bool = False,
        isolated_db: bool = False,
        db_path: str | Path = "data/paper_trading.duckdb",
    ) -> None:
        super().__init__()
        self._symbol = symbol
        resolved_interval = _validate_interval_name(interval)
        self._optimize_profile = optimize_profile
        self._interval = resolved_interval
        self._strategy_name = _validate_strategy_name(strategy_name)
        self._configured_leverage = (
            settings.trading.default_leverage
            if leverage is None
            else int(leverage)
        )
        self._manual_leverage_override = leverage is not None
        self._leverage_override = (
            None
            if not self._manual_leverage_override
            else int(leverage)
        )
        self._use_setup_gate = settings.trading.use_setup_gate
        self._min_confidence_pct = (
            None
            if not self._use_setup_gate
            else (
                settings.trading.min_confidence_pct
                if min_confidence_pct is None
                else float(min_confidence_pct)
            )
        )
        self._setup_gate = (
            SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
            if self._use_setup_gate
            else None
        )
        self._optimization_grid: list[OptimizationProfile] = []
        self._db_path = Path(db_path)
        self._isolated_db = isolated_db
        self._active_pool: object | None = None
        self._stop_requested = False
        self._last_precomputed_exit_cache: dict[str, object] | None = None
        self._hmm_regime_mask: list[int] | None = None
        self._last_runtime_trace: dict[str, object] | None = None
        self._last_strategy_diagnostics: dict[str, object] | None = None

    def _should_stop(self) -> bool:
        return self._stop_requested or self.isInterruptionRequested()

    def stop(self) -> None:
        self._stop_requested = True
        self.requestInterruption()
        active_pool = self._active_pool
        if active_pool is None:
            return
        with suppress(Exception):
            active_pool.terminate()

    def _resolve_effective_backtest_leverage(self) -> int:
        if self._leverage_override is not None:
            return int(self._leverage_override)
        profile = settings.trading.coin_profiles.get(self._symbol)
        if profile is not None and profile.default_leverage is not None:
            return int(profile.default_leverage)
        return int(settings.trading.default_leverage)

    def _emit_trace_event(self, event_name: str, payload: dict[str, object]) -> None:
        self.log_message.emit(
            f"BACKTEST_TRACE|{event_name}|{_trace_json_payload(payload)}"
        )

    def _resolve_required_runtime_candles(
        self,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None = None,
    ) -> tuple[int, int]:
        with _temporary_strategy_profile(strategy_profile):
            required_candles = _required_candle_count_for_strategy(
                strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
        required_warmup = (
            int(SETUP_GATE_SETTLED_WARMUP_CANDLES)
            if self._use_setup_gate
            else 0
        )
        return required_warmup, int(required_candles)

    def _resolve_strategy_diagnostics_payload(
        self,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None = None,
        diagnostics_source: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        _ = strategy_name, strategy_profile, diagnostics_source
        return None

    def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):
        """Run a robust multi-coin multi-strategy sweep according to project policy.

        This method enforces Phase 19 warmup, EMA-200 macro context via SmartSetupGate,
        per-(coin,strategy) cache clearing and immediate persistence of winners.
        """
        base_strategies = [
            "ema_cross_volume",
            "frama_cross",
            "dual_thrust",
        ]
        for symbol in coins:
            symbol_strategies = list(base_strategies)
            if not symbol_strategies:
                self.log_message.emit(
                    f"Sweep skip: no eligible strategies remain for {symbol}."
                )
                continue
            try:
                for strategy_name in symbol_strategies:
                    if self._should_stop():
                        return
                    self.log_message.emit(f"Sweep start: {symbol} -> {strategy_name}")
                    candles_df: pd.DataFrame | None = None
                    candles: list[CandleRecord] | None = None
                    result: dict[str, object] | None = None
                    try:
                        interval = resolve_optimizer_scan_interval_for_symbol(
                            symbol,
                            settings.trading.interval,
                        )
                        with Database(self._db_path) as db:
                            # Ensure history is present
                            try:
                                with BitunixClient(max_retries=6, backoff_factor=1.0) as client:
                                    with HistoryManager(db, client) as history_manager:
                                        self.log_message.emit(
                                            f"Incremental history sync active for {symbol} {interval}: "
                                            f"refreshing latest tail window ({MAX_BACKTEST_CANDLES} candles)."
                                        )
                                        history_manager.sync_recent_candles(
                                            symbols=[symbol],
                                            interval=interval,
                                            candles_per_symbol=MAX_BACKTEST_CANDLES,
                                            on_progress=self._emit_history_download_progress,
                                        )
                            except Exception:
                                # Continue even if history sync fails; assume DB may already have candles
                                pass

                            candles = _load_latest_backtest_candles(
                                db,
                                symbol=symbol,
                                interval=interval,
                                limit=MAX_BACKTEST_CANDLES,
                            )
                            if not candles:
                                self.log_message.emit(f"No candles for {symbol} {interval}; skipping.")
                                continue
                            candles_df = _candles_to_dataframe(candles).copy(deep=True)
                            required_candles = _required_candle_count_for_strategy(strategy_name, use_setup_gate=True)
                            if len(candles_df) < required_candles:
                                self.log_message.emit(
                                    f"Not enough candles for {symbol} {strategy_name}: have {len(candles_df)}, need {required_candles}."
                                )
                                continue

                            # Configure runtime state for this run
                            self._symbol = symbol
                            self._interval = interval
                            self._strategy_name = _validate_strategy_name(strategy_name)
                            self._use_setup_gate = True
                            self._min_confidence_pct = settings.trading.min_confidence_pct
                            self._setup_gate = SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
                            self._hmm_regime_mask = self._prepare_hmm_regime_mask(
                                candles_df,
                                strategy_name=self._strategy_name,
                            )

                            # Build and possibly cap optimization grid
                            self._optimization_grid = generate_optimization_grid(
                                self._strategy_name,
                                symbol=symbol,
                            )
                            # _optimization_grid entries are capped by generator, but ensure global cap
                            if len(self._optimization_grid) > MAX_OPTIMIZATION_GRID_PROFILES:
                                self._optimization_grid = _cap_profiles(
                                    self._optimization_grid,
                                    max_total=MAX_OPTIMIZATION_GRID_PROFILES,
                                )

                            self.log_message.emit(
                                f"Prepared grid for {symbol} {strategy_name}: {len(self._optimization_grid)} profiles"
                            )

                            # Run optimization with error isolation
                            try:
                                result = self._run_profile_optimization(
                                    db,
                                    candles_df,
                                    strategy_name=self._strategy_name,
                                    regime_mask=self._hmm_regime_mask,
                                )
                                if result:
                                    self.log_message.emit(f"Sweep finished: {symbol} {strategy_name}")
                            except Exception as exc:
                                self.log_message.emit(f"Error during optimization {symbol} {strategy_name}: {exc}")
                    except Exception as exc_outer:
                        self.log_message.emit(f"Sweep outer error for {symbol} {strategy_name}: {exc_outer}")
                        # continue to next strategy
                        continue
                    finally:
                        # Enforce aggressive cleanup after each (coin, strategy)
                        try:
                            _clear_strategy_indicator_cache(candles_df)
                        except Exception:
                            pass
                        self._clear_signal_caches_and_gc()
                        with suppress(Exception):
                            if candles_df is not None:
                                del candles_df
                        with suppress(Exception):
                            if candles is not None:
                                del candles
                        with suppress(Exception):
                            if result is not None:
                                del result
                        gc.collect()
            finally:
                # Aggressive cleanup after each completed coin run.
                self._hmm_regime_mask = None
                self._optimization_grid = []
                self._clear_signal_caches_and_gc()
                gc.collect()
                self.log_message.emit(f"Sweep memory cleanup complete for coin {symbol}.")

    def run(self) -> None:
        temp_workspace: TemporaryDirectory[str] | None = None
        try:
            working_db_path = self._db_path
            if self._isolated_db:
                temp_workspace = TemporaryDirectory(prefix="backtest_")
                working_db_path = Path(temp_workspace.name) / "isolated_backtest.duckdb"
                self.log_message.emit(
                    "Live bot active: running backtest on isolated temp DB to avoid lock contention."
                )

            with Database(working_db_path) as db:
                resolved_strategy_name = resolve_strategy_for_symbol(
                    self._symbol,
                    self._strategy_name,
                    use_coin_override=False,
                )
                self._last_runtime_trace = None
                self._last_strategy_diagnostics = None
                shared_history_db: Database | None = None
                history_sources: list[tuple[Database, str]] = []
                if self._isolated_db:
                    try:
                        shared_history_db = Database(self._db_path)
                        history_sources.append((shared_history_db, "shared incremental cache DB"))
                    except Exception as exc:
                        self.log_message.emit(
                            "Warning: could not open shared history cache DB. "
                            f"Falling back to isolated backtest DB. ({exc})"
                        )
                history_sources.append(
                    (
                        db,
                        "isolated backtest DB" if self._isolated_db else "backtest DB",
                    )
                )
                backtest_start_time = _history_start_for_mode(
                    optimize_profile=self._optimize_profile
                )
                required_warmup_candles, required_candles = self._resolve_required_runtime_candles(
                    strategy_name=resolved_strategy_name,
                    strategy_profile=None,
                )
                effective_history_start_time = backtest_start_time
                self._optimization_grid = generate_optimization_grid(
                    resolved_strategy_name,
                    symbol=self._symbol,
                )
                self.log_message.emit(
                    f"Backtest strategy selected: {self._symbol} -> {resolved_strategy_name}"
                )
                if self._optimize_profile:
                    self.log_message.emit(
                        "Backtest history window base: "
                        f"{backtest_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC -> latest available "
                        "(optimizer uses incremental recent-window sync when cache exists)."
                    )
                else:
                    self.log_message.emit(
                        "Backtest history window fixed: "
                        f"{backtest_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC -> latest available."
                    )
                if self._use_setup_gate:
                    if self._optimize_profile:
                        self.log_message.emit(
                            "Phase 19 active: enforcing a 500-candle settled warmup before Setup Gate approval "
                            f"with optimizer history base start at {backtest_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
                        )
                    else:
                        self.log_message.emit(
                            "Phase 19 active: enforcing a 500-candle settled warmup before Setup Gate approval "
                            f"with fixed backtest history start at {backtest_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
                        )
                self._emit_trace_event(
                    "run_start",
                    {
                        "symbol": self._symbol,
                        "resolved_strategy_name": resolved_strategy_name,
                        "resolved_interval": self._interval,
                        "configured_leverage": int(self._configured_leverage),
                        "required_warmup_candles": int(required_warmup_candles),
                        "required_candles": int(required_candles),
                        "history_requested_start_utc": _trace_datetime_text(backtest_start_time),
                        "optimization_mode": bool(self._optimize_profile),
                    },
                )
                self.progress_update.emit(10, "Downloading history...")
                candles: list[CandleRecord] = []
                total_candles = 0
                oldest_time: datetime | None = None
                newest_time: datetime | None = None
                last_history_error: Exception | None = None
                try:
                    for source_index, (history_db, source_label) in enumerate(history_sources):
                        try:
                            self.log_message.emit(
                                f"Downloading historical data for backtest via {source_label}..."
                            )
                            with BitunixClient(max_retries=6, backoff_factor=1.0) as client:
                                with HistoryManager(history_db, client) as history_manager:
                                    self.log_message.emit(
                                        "Incremental history sync active: "
                                        f"refreshing latest tail window ({MAX_BACKTEST_CANDLES} candles)."
                                    )
                                    history_manager.sync_recent_candles(
                                        symbols=[self._symbol],
                                        interval=self._interval,
                                        candles_per_symbol=MAX_BACKTEST_CANDLES,
                                        on_progress=self._emit_history_download_progress,
                                    )
                            candles = _load_latest_backtest_candles(
                                history_db,
                                symbol=self._symbol,
                                interval=self._interval,
                                limit=MAX_BACKTEST_CANDLES,
                            )
                            effective_history_start_time = backtest_start_time
                            total_candles = len(candles)
                            if candles:
                                oldest_time = candles[0].open_time
                                newest_time = candles[-1].open_time
                            break
                        except Exception as exc:
                            last_history_error = exc
                            if source_index < len(history_sources) - 1:
                                next_label = history_sources[source_index + 1][1]
                                self.log_message.emit(
                                    f"History sync via {source_label} failed: {exc}. "
                                    f"Retrying via {next_label}."
                                )
                                continue
                            raise
                finally:
                    if shared_history_db is not None:
                        with suppress(Exception):
                            shared_history_db.close()
                        shared_history_db = None
                if not candles and last_history_error is not None:
                    raise last_history_error

                if total_candles > 0 and oldest_time is not None and newest_time is not None:
                    self.log_message.emit(
                        "Historical data range loaded: "
                        f"{oldest_time.strftime('%Y-%m-%d %H:%M:%S')} -> "
                        f"{newest_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({total_candles} candles, interval={self._interval})"
                    )
                    self.log_message.emit(
                        "Backtest loader mode: "
                        f"using latest tail window ({min(total_candles, MAX_BACKTEST_CANDLES)}/{MAX_BACKTEST_CANDLES} candles)."
                    )
                if candles:
                    self.log_message.emit(
                        "Backtest window in use: "
                        f"{self._format_candle_range(candles)} "
                        f"({len(candles)} candles)"
                    )
                self._emit_trace_event(
                    "history_loaded",
                    {
                        "symbol": self._symbol,
                        "resolved_strategy_name": resolved_strategy_name,
                        "resolved_interval": self._interval,
                        "history_requested_start_utc": _trace_datetime_text(
                            effective_history_start_time
                        ),
                        "history_start_utc": _trace_datetime_text(oldest_time),
                        "history_end_utc": _trace_datetime_text(newest_time),
                        "history_candles": int(total_candles),
                        "required_warmup_candles": int(required_warmup_candles),
                        "required_candles": int(required_candles),
                    },
                )
                if len(candles) < required_candles:
                    raise RuntimeError(
                        f"Not enough candles for backtest: have {len(candles)}, need {required_candles}."
                    )

                candles_df = _candles_to_dataframe(candles).copy(deep=True)
                self._hmm_regime_mask = self._prepare_hmm_regime_mask(
                    candles_df,
                    strategy_name=resolved_strategy_name,
                )
                if self._optimize_profile:
                    result = self._run_profile_optimization(
                        db,
                        candles_df,
                        strategy_name=resolved_strategy_name,
                        regime_mask=self._hmm_regime_mask,
                    )
                else:
                    (
                        signals,
                        total_signals,
                        approved_signals,
                        blocked_signals,
                    ) = self._generate_signals(
                        candles_df,
                        strategy_name=resolved_strategy_name,
                        regime_mask=self._hmm_regime_mask,
                    )
                    if self._should_stop():
                        return
                    self.log_message.emit(
                        "Backtest summary: "
                        f"Total Signals: {total_signals} | Approved by Gate: {approved_signals} | Blocked: {blocked_signals}"
                    )
                    self._emit_trace_event(
                        "signal_summary",
                        {
                            "symbol": self._symbol,
                            "resolved_strategy_name": resolved_strategy_name,
                            "resolved_interval": self._interval,
                            "total_signals": int(total_signals),
                            "approved_signals": int(approved_signals),
                            "blocked_signals": int(blocked_signals),
                        },
                    )
                    self.progress_update.emit(50, "Running simulation...")
                    result = self._run_single_backtest(
                        db,
                        candles_df,
                        signals,
                        strategy_name=resolved_strategy_name,
                        strategy_profile=None,
                        total_signals=total_signals,
                        approved_signals=approved_signals,
                        blocked_signals=blocked_signals,
                    )
                result = dict(result)
                strategy_diagnostics = result.get("strategy_diagnostics")
                result.update(
                    {
                        "symbol": self._symbol,
                        "interval": self._interval,
                        "strategy_name": resolved_strategy_name,
                        "optimization_mode": bool(result.get("optimization_mode", False)),
                        "configured_leverage": int(self._configured_leverage),
                        "effective_leverage": int(
                            result.get(
                                "effective_leverage",
                                self._resolve_effective_backtest_leverage(),
                            )
                        ),
                        "history_requested_start_utc": effective_history_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "history_start_utc": (
                            oldest_time.strftime("%Y-%m-%d %H:%M:%S")
                            if oldest_time is not None
                            else None
                        ),
                        "history_end_utc": (
                            newest_time.strftime("%Y-%m-%d %H:%M:%S")
                            if newest_time is not None
                            else None
                        ),
                        "history_candles": int(total_candles),
                        "history_loader_mode": "latest_tail_window",
                        "tail_window_limit_candles": int(MAX_BACKTEST_CANDLES),
                        "required_warmup_candles": int(
                            result.get("required_warmup_candles", required_warmup_candles)
                        ),
                        "required_candles": int(result.get("required_candles", required_candles)),
                    }
                )
                self._emit_trace_event(
                    "result_summary",
                    {
                        "symbol": self._symbol,
                        "resolved_strategy_name": resolved_strategy_name,
                        "resolved_interval": self._interval,
                        "configured_leverage": result.get("configured_leverage"),
                        "effective_leverage": result.get("effective_leverage"),
                        "stop_loss_pct": result.get("stop_loss_pct"),
                        "take_profit_pct": result.get("take_profit_pct"),
                        "trailing_activation_pct": result.get("trailing_activation_pct"),
                        "trailing_distance_pct": result.get("trailing_distance_pct"),
                        "history_requested_start_utc": result.get("history_requested_start_utc"),
                        "history_start_utc": result.get("history_start_utc"),
                        "history_end_utc": result.get("history_end_utc"),
                        "history_candles": result.get("history_candles"),
                        "required_warmup_candles": result.get("required_warmup_candles"),
                        "required_candles": result.get("required_candles"),
                        "total_signals": result.get("total_signals"),
                        "approved_signals": result.get("approved_signals"),
                        "blocked_signals": result.get("blocked_signals"),
                        "final_pnl": result.get("total_pnl_usd"),
                        "win_rate": result.get("win_rate_pct"),
                        "profit_factor": result.get("profit_factor"),
                        "total_trades": result.get("total_trades"),
                        "average_win": result.get("average_win_usd"),
                        "average_loss": result.get("average_loss_usd"),
                        "real_rrr": result.get("real_rrr"),
                        "max_drawdown": result.get("max_drawdown_pct"),
                    },
                )
                if self._isolated_db:
                    persist_error: Exception | None = None
                    for attempt in range(5):
                        try:
                            with Database(self._db_path) as result_db:
                                self._persist_backtest_result(
                                    result_db,
                                    result,
                                    strategy_name=resolved_strategy_name,
                                )
                            persist_error = None
                            break
                        except Exception as exc:
                            persist_error = exc
                            if attempt < 4:
                                sleep(0.25)
                    if persist_error is not None:
                        raise persist_error
                else:
                    self._persist_backtest_result(
                        db,
                        result,
                        strategy_name=resolved_strategy_name,
                    )
                self.log_message.emit(
                    f"Backtest result saved to DB for {self._symbol} {self._interval} ({resolved_strategy_name})."
                )
        except Exception as exc:
            if not self._should_stop():
                self.backtest_error.emit(str(exc))
            return
        finally:
            self._hmm_regime_mask = None
            self._clear_signal_caches_and_gc()
            gc.collect()
            if temp_workspace is not None:
                temp_workspace.cleanup()

        if not self._should_stop():
            self.progress_update.emit(100, "Done")
            self.backtest_finished.emit(result)

    def _run_single_backtest(
        self,
        db: Database,
        candles_df: pd.DataFrame,
        signals: list[int],
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None = None,
        total_signals: int | None = None,
        approved_signals: int | None = None,
        blocked_signals: int | None = None,
    ) -> dict:
        profile_tp = (
            float(strategy_profile["take_profit_pct"])
            if isinstance(strategy_profile, dict) and "take_profit_pct" in strategy_profile
            else None
        )
        profile_sl = (
            float(strategy_profile["stop_loss_pct"])
            if isinstance(strategy_profile, dict) and "stop_loss_pct" in strategy_profile
            else None
        )
        profile_trailing_activation = (
            float(strategy_profile["trailing_activation_pct"])
            if isinstance(strategy_profile, dict) and "trailing_activation_pct" in strategy_profile
            else None
        )
        profile_trailing_distance = (
            float(strategy_profile["trailing_distance_pct"])
            if isinstance(strategy_profile, dict) and "trailing_distance_pct" in strategy_profile
            else None
        )
        engine = self._build_backtest_engine(
            db,
            take_profit_pct=profile_tp,
            stop_loss_pct=profile_sl,
            trailing_activation_pct=profile_trailing_activation,
            trailing_distance_pct=profile_trailing_distance,
        )
        runtime_trace = self._log_backtest_runtime_settings(
            engine,
            prefix="Backtest runtime leverage",
        )
        self._last_runtime_trace = dict(runtime_trace)
        precomputed_exit_rule = _build_precomputed_exit_rule(
            PaperTradingEngine._extract_backtest_rows(candles_df),
            long_exit_flags=(
                None
                if self._last_precomputed_exit_cache is None
                else self._last_precomputed_exit_cache.get("precomputed_long_exit_flags")
            ),
            short_exit_flags=(
                None
                if self._last_precomputed_exit_cache is None
                else self._last_precomputed_exit_cache.get("precomputed_short_exit_flags")
            ),
        )
        result = engine.run_historical_backtest(
            candles_df,
            signals,
            strategy_exit_rule=(
                precomputed_exit_rule
                if precomputed_exit_rule is not None
                else _build_strategy_exit_rule(
                    strategy_name,
                    candles_df,
                    strategy_profile=strategy_profile,
                )
            ),
        )
        result.update(self._calculate_trade_metrics(result["closed_trades"]))
        required_warmup_candles, required_candles = self._resolve_required_runtime_candles(
            strategy_name=strategy_name,
            strategy_profile=strategy_profile,
        )
        strategy_diagnostics_payload = self._resolve_strategy_diagnostics_payload(
            strategy_name=strategy_name,
            strategy_profile=strategy_profile,
            diagnostics_source=self._last_strategy_diagnostics,
        )
        result.update(
            {
                "configured_leverage": int(self._configured_leverage),
                "effective_leverage": int(
                    runtime_trace.get(
                        "effective_leverage",
                        self._resolve_effective_backtest_leverage(),
                    )
                ),
                "stop_loss_pct": runtime_trace.get("stop_loss_pct"),
                "take_profit_pct": runtime_trace.get("take_profit_pct"),
                "trailing_activation_pct": runtime_trace.get("trailing_activation_pct"),
                "trailing_distance_pct": runtime_trace.get("trailing_distance_pct"),
                "effective_taker_fee_pct": runtime_trace.get("effective_taker_fee_pct"),
                "min_confidence_pct": (
                    float(self._min_confidence_pct)
                    if self._min_confidence_pct is not None
                    else None
                ),
                "required_warmup_candles": int(required_warmup_candles),
                "required_candles": int(required_candles),
            }
        )
        if total_signals is not None:
            result["total_signals"] = int(total_signals)
        if approved_signals is not None:
            result["approved_signals"] = int(approved_signals)
        if blocked_signals is not None:
            result["blocked_signals"] = int(blocked_signals)
        if strategy_diagnostics_payload is not None:
            result["strategy_diagnostics"] = strategy_diagnostics_payload
        self._log_backtest_metrics(result)
        return result

    def _run_profile_optimization(
        self,
        db: Database,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        regime_mask: Sequence[int] | None = None,
    ) -> dict:
        if not self._use_setup_gate:
            self.log_message.emit(
                "Phase 19 override active: forcing SmartSetupGate for every optimizer worker."
            )
            self._use_setup_gate = True
            if self._min_confidence_pct is None:
                self._min_confidence_pct = settings.trading.min_confidence_pct
            self._setup_gate = SmartSetupGate(min_confidence_pct=self._min_confidence_pct)

        theoretical_profiles = len(self._optimization_grid)
        if theoretical_profiles == 0:
            raise RuntimeError("Optimization grid is empty.")

        configured_search_window = int(settings.trading.optimization_search_window_candles)
        if 0 < configured_search_window < len(candles_df):
            optimization_candles_df = candles_df.iloc[-configured_search_window:].copy(deep=True)
            optimization_uses_full_history = False
        else:
            optimization_candles_df = candles_df
            optimization_uses_full_history = True

        optimization_regime_mask: list[int] | None = None
        if regime_mask is not None:
            if len(regime_mask) == len(candles_df):
                optimization_regime_mask = list(regime_mask)[-len(optimization_candles_df):]
            else:
                self.log_message.emit(
                    "Warning: regime mask length mismatch; disabling regime filter for optimizer window."
                )

        strategy_label = strategy_name.replace("_", " ").title()
        full_grid_profiles = [dict(profile) for profile in self._optimization_grid]
        configured_max_sample_profiles = max(
            1,
            int(settings.trading.optimization_max_sample_profiles),
        )
        max_sample_profiles = max(
            configured_max_sample_profiles,
            MIN_SAMPLE_SCAN_PROFILES_PER_STRATEGY,
        )
        if max_sample_profiles > configured_max_sample_profiles:
            self.log_message.emit(
                "Sample scan coverage floor active: "
                f"raised per-strategy sample size from {configured_max_sample_profiles} "
                f"to {max_sample_profiles} profiles."
            )
        sampled_grid_profiles = _sample_optimization_profiles(
            full_grid_profiles,
            max_profiles=max_sample_profiles,
            symbol=self._symbol,
            strategy_name=strategy_name,
            interval=self._interval,
            random_seed=OPTIMIZER_SAMPLE_RANDOM_SEED,
        )
        scan_profile_count = len(sampled_grid_profiles)
        is_sample_scan = scan_profile_count < theoretical_profiles
        coverage_ratio = scan_profile_count / float(theoretical_profiles)
        initial_worker_count = _memory_safe_worker_count(scan_profile_count)

        self.log_message.emit(
            f"Optimizer worker lock active: reserving up to {MAX_OPTIMIZATION_WORKERS} isolated worker processes."
        )
        if FORCE_MAX_OPTIMIZATION_WORKERS:
            self.log_message.emit(
                "Optimizer worker policy active: "
                f"forced up to {MAX_OPTIMIZATION_WORKERS} worker process(es); "
                f"{initial_worker_count} selected for this run."
            )
        else:
            self.log_message.emit(
                "Optimizer RAM guard active: "
                f"{initial_worker_count} worker process(es) selected "
                f"(reserve={OPTIMIZER_MEMORY_RESERVE_GB:.1f}GB, est/worker={OPTIMIZER_MEMORY_PER_WORKER_GB:.1f}GB)."
            )
        if is_sample_scan:
            self.log_message.emit(
                f"Sample Scan active for {self._symbol} {strategy_label}: "
                f"{scan_profile_count}/{theoretical_profiles} profiles "
                f"({coverage_ratio * 100.0:.2f}% coverage), random_seed={OPTIMIZER_SAMPLE_RANDOM_SEED}."
            )
        else:
            self.log_message.emit(
                f"Full Scan active for {self._symbol} {strategy_label}: "
                f"{theoretical_profiles}/{theoretical_profiles} profiles "
                f"(100.00% coverage), random_seed={OPTIMIZER_SAMPLE_RANDOM_SEED}."
            )
        self.log_message.emit(
            f"Optimization mode active for {self._symbol}: evaluating {scan_profile_count} profiles "
            f"(grid_total={theoretical_profiles}) on {initial_worker_count} worker processes."
        )
        if self._use_setup_gate:
            self.log_message.emit(
                "Phase 19 optimizer guarantee active: every worker instantiates its own SmartSetupGate "
                f"and validates signals only after {SETUP_GATE_SETTLED_WARMUP_CANDLES} settled candles."
            )
        if optimization_uses_full_history:
            self.log_message.emit(
                f"Full-history optimizer window: evaluating all {len(optimization_candles_df)} candles per profile."
            )
        else:
            self.log_message.emit(
                "Search-window optimizer active: "
                f"evaluating last {len(optimization_candles_df)}/{len(candles_df)} candles per profile."
            )

        try:
            optimization_phase_summaries = self._evaluate_optimization_phase(
                optimization_candles_df,
                strategy_name=strategy_name,
                profiles=sampled_grid_profiles,
                phase_label=f"Optimizing {strategy_label} profiles",
                cache_progress_start=20,
                cache_progress_span=25,
                optimization_progress_start=46,
                optimization_progress_end=99,
                regime_mask=optimization_regime_mask,
            )
        except Exception as exc:
            # Do not allow one strategy optimization error to abort the entire run
            self.log_message.emit(f"Optimization for {strategy_label} failed: {exc}")
            # Clear caches and return empty result so higher-level orchestrator can continue
            self._clear_signal_caches_and_gc()
            return {}
        if self._should_stop():
            return {}
        if not optimization_phase_summaries:
            raise RuntimeError("Optimization did not produce any result.")

        min_required_trades = (
            MIN_TRADES_15M if str(self._interval).strip().lower() == "15m" else MIN_TRADES_1H
        )
        for profile_result in optimization_phase_summaries:
            if float(profile_result.get("total_trades", 0.0)) < float(min_required_trades):
                profile_result["profit_factor"] = -999999.0
                profile_result["real_rrr"] = -999999.0
                profile_result["total_pnl_usd"] = -999999.0

        eligible_optimization_summaries = [
            summary
            for summary in optimization_phase_summaries
            if float(summary.get("total_trades", 0.0)) >= float(min_required_trades)
        ]
        if not eligible_optimization_summaries:
            self._clear_signal_caches_and_gc()
            raise RuntimeError(
                f"No optimization profile passed minimum sample filter: "
                f"need >= {min_required_trades} trades for interval {self._interval}."
            )
        ranked_optimization_summaries = sorted(
            eligible_optimization_summaries,
            key=lambda summary: _optimization_sort_key(summary, strategy_name=strategy_name),
            reverse=True,
        )
        best_summary = dict(ranked_optimization_summaries[0])
        best_profile = _extract_profile_from_summary(best_summary)
        (
            best_signals,
            best_total_signals,
            best_approved_signals,
            best_blocked_signals,
        ) = self._generate_signals(
            optimization_candles_df,
            strategy_name=strategy_name,
            strategy_profile=best_profile,
            regime_mask=optimization_regime_mask,
        )
        best_result = self._run_single_backtest(
            db,
            optimization_candles_df,
            best_signals,
            strategy_name=strategy_name,
            strategy_profile=best_profile,
            total_signals=best_total_signals,
            approved_signals=best_approved_signals,
            blocked_signals=best_blocked_signals,
        )
        self._log_backtest_metrics(best_result, prefix="Optimization metrics")
        self.log_message.emit(
            f"Best Profile for {self._symbol}: "
            f"{_format_profile_dict(best_profile)} "
            f"| profit_factor={self._format_profit_factor(float(best_summary['profit_factor']))} "
            f"| real_rrr={self._format_profit_factor(float(best_summary['real_rrr']))} "
            f"| pnl={float(best_summary['total_pnl_usd']):.2f} "
            f"| win_rate={float(best_summary['win_rate_pct']):.2f}% "
            f"| max_dd={float(best_summary.get('max_drawdown_pct', 0.0)):.2f}%"
        )

        keep_top_count = min(
            len(ranked_optimization_summaries),
            max(1, int(settings.trading.optimization_validation_top_n)),
        )
        optimization_summaries_compact = [
            dict(summary)
            for summary in ranked_optimization_summaries[:keep_top_count]
        ]
        best_result.update(
            {
                "optimization_mode": True,
                "best_profile": best_profile,
                "optimization_results": optimization_summaries_compact,
                "evaluated_profiles": scan_profile_count,
                "sampled_profiles": scan_profile_count,
                "theoretical_profiles": theoretical_profiles,
                "search_window_candles": len(optimization_candles_df),
                "validated_profiles": keep_top_count,
                "full_history_optimization": optimization_uses_full_history,
                "sampling_mode": "sample" if is_sample_scan else "full",
                "sampling_coverage_pct": coverage_ratio * 100.0,
                "sampling_random_seed": int(OPTIMIZER_SAMPLE_RANDOM_SEED),
            }
        )
        # Persist winner immediately (batch-saving) so each strategy/coin result is durable
        try:
            if db is not None:
                self._persist_backtest_result(db, best_result, strategy_name=strategy_name)
        except Exception:
            # Don't fail if persistence fails; continue to clear caches.
            self.log_message.emit("Warning: failed to persist optimization winner to DB.")

        # Free large worker-side caches to avoid memory growth between coin runs
        self._clear_signal_caches_and_gc()
        return best_result

    def _evaluate_optimization_phase(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        profiles: Sequence[OptimizationProfile],
        phase_label: str,
        cache_progress_start: int,
        cache_progress_span: int,
        optimization_progress_start: int,
        optimization_progress_end: int,
        regime_mask: Sequence[int] | None = None,
    ) -> list[dict[str, float]]:
        total_profiles = len(profiles)
        if total_profiles == 0:
            return []

        worker_count = _memory_safe_worker_count(total_profiles)
        strategy_signal_cache: dict[object, dict[str, object]] | None = None
        precomputed_signals: list[int] | None = None
        total_signals = 0
        approved_signals = 0
        blocked_signals = 0

        if strategy_name in {
            "ema_cross_volume",
            "frama_cross",
            "dual_thrust",
        }:
            strategy_signal_cache = self._build_parallel_strategy_signal_cache(
                candles_df,
                strategy_name=strategy_name,
                worker_count=worker_count,
                variant_profiles=_collect_strategy_variant_profiles(strategy_name, profiles),
                progress_start=cache_progress_start,
                progress_span=cache_progress_span,
                regime_mask=regime_mask,
            )
            if self._should_stop():
                return []
            self.progress_update.emit(
                optimization_progress_start,
                f"{phase_label} 0/{total_profiles}",
            )
        else:
            self.progress_update.emit(
                optimization_progress_start,
                f"{phase_label} 0/{total_profiles}",
            )
            (
                precomputed_signals,
                total_signals,
                approved_signals,
                blocked_signals,
            ) = self._generate_signals(
                candles_df,
                strategy_name=strategy_name,
                regime_mask=regime_mask,
            )
            self.log_message.emit(
                "Backtest summary: "
                f"Total Signals: {total_signals} | Approved by Gate: {approved_signals} | Blocked: {blocked_signals}"
            )

        candles_records = candles_df.to_dict("records")
        first_summary_logged = strategy_signal_cache is None
        completed_profiles = 0
        batch_size = worker_count
        pool = _create_optimizer_pool(
            worker_count,
            candles_records,
            regime_mask=regime_mask,
        )
        self._active_pool = pool
        interrupted = False
        phase_results: list[dict[str, float]] = []
        best_summary: dict[str, float] | None = None
        best_profile: OptimizationProfile | None = None
        try:
            for batch_start in range(0, total_profiles, batch_size):
                if self._should_stop():
                    interrupted = True
                    return []

                batch_profiles = list(profiles[batch_start : batch_start + batch_size])
                tasks: list[dict[str, object]] = []
                for profile in batch_profiles:
                    cache_key = (
                        _strategy_cache_key(strategy_name, profile)
                        if strategy_signal_cache is not None
                        else None
                    )
                    cached_group = (
                        strategy_signal_cache.get(cache_key)
                        if strategy_signal_cache is not None and cache_key is not None
                        else None
                    )
                    tasks.append(
                        {
                            "strategy_name": strategy_name,
                            "strategy_profile": profile,
                            "symbol": self._symbol,
                            "interval": self._interval,
                            "leverage_override": self._leverage_override,
                            "use_setup_gate": self._use_setup_gate,
                            "min_confidence_pct": self._min_confidence_pct,
                            "precomputed_signals": (
                                cached_group["signals"]
                                if cached_group is not None
                                else precomputed_signals
                            ),
                            "precomputed_long_exit_flags": (
                                cached_group.get("precomputed_long_exit_flags")
                                if cached_group is not None
                                else None
                            ),
                            "precomputed_short_exit_flags": (
                                cached_group.get("precomputed_short_exit_flags")
                                if cached_group is not None
                                else None
                            ),
                            "total_signals": (
                                int(cached_group["total_signals"])
                                if cached_group is not None
                                else total_signals
                            ),
                            "approved_signals": (
                                int(cached_group["approved_signals"])
                                if cached_group is not None
                                else approved_signals
                            ),
                            "blocked_signals": (
                                int(cached_group["blocked_signals"])
                                if cached_group is not None
                                else blocked_signals
                            ),
                        }
                    )

                async_result = pool.map_async(_run_optimization_profile_worker, tasks)
                while True:
                    if self._should_stop():
                        interrupted = True
                        return []
                    try:
                        batch_results = async_result.get(timeout=0.25)
                        break
                    except PoolTimeoutError:
                        continue

                for batch_result in batch_results:
                    summary = dict(batch_result["summary"])
                    phase_results.append(summary)
                    if not first_summary_logged and strategy_signal_cache is not None:
                        self.log_message.emit(
                            "Backtest summary: "
                            f"Total Signals: {int(batch_result['total_signals'])} | "
                            f"Approved by Gate: {int(batch_result['approved_signals'])} | "
                            f"Blocked: {int(batch_result['blocked_signals'])}"
                        )
                        first_summary_logged = True
                    if best_summary is None or _optimization_sort_key(
                        summary,
                        strategy_name=strategy_name,
                    ) > _optimization_sort_key(
                        best_summary,
                        strategy_name=strategy_name,
                    ):
                        best_summary = summary
                        best_profile = _extract_profile_from_summary(summary)

                completed_profiles += len(batch_results)
                best_profile_text = _format_profile_dict(best_profile) if best_profile is not None else "{}"
                # Throttle progress logging to configured modulo (or on completion)
                try:
                    modulo = settings.trading.optimizer_progress_print_modulo
                    should_log = completed_profiles >= total_profiles or (
                        modulo > 0 and (completed_profiles % modulo == 0)
                    )
                except Exception:
                    should_log = True
                if should_log:
                    self.log_message.emit(
                        f"{phase_label} {completed_profiles}/{total_profiles}: "
                        f"current_best={best_profile_text} "
                        f"pf={self._format_profit_factor(float(best_summary['profit_factor'])) if best_summary is not None else '0.00'}"
                    )
                    self._emit_optimization_progress(
                        completed_profiles,
                        total_profiles,
                        start_value=optimization_progress_start,
                        end_value=optimization_progress_end,
                        prefix=phase_label,
                    )
            # Clear large strategy signal cache after finishing this phase to free memory
            with suppress(Exception):
                if strategy_signal_cache is not None:
                    try:
                        strategy_signal_cache.clear()
                    finally:
                        strategy_signal_cache = None
            gc.collect()
        except Exception:
            if self._should_stop():
                interrupted = True
                return []
            raise
        finally:
            self._active_pool = None
            with suppress(Exception):
                if interrupted or self._should_stop():
                    pool.terminate()
                else:
                    pool.close()
            with suppress(Exception):
                pool.join()
            gc.collect()

        return phase_results

    def _build_parallel_strategy_signal_cache(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        worker_count: int,
        variant_profiles: Sequence[OptimizationProfile] | None = None,
        progress_start: int = 50,
        progress_span: int = 25,
        regime_mask: Sequence[int] | None = None,
    ) -> dict[object, dict[str, object]]:
        variant_profiles = (
            list(variant_profiles)
            if variant_profiles is not None
            else _strategy_cache_profiles(strategy_name)
        )
        if not variant_profiles:
            return {}

        cache_worker_count = min(
            worker_count,
            _memory_safe_worker_count(len(variant_profiles)),
            len(variant_profiles),
        )
        strategy_label = strategy_name.replace("_", " ").title()
        self.log_message.emit(
            f"Precomputing {strategy_label} signal cache for {len(variant_profiles)} strategy variants "
            f"on {cache_worker_count} worker processes."
        )
        self.progress_update.emit(
            progress_start,
            f"Precomputing {strategy_label} signals... 0/{len(variant_profiles)}",
        )

        tasks = [
            {
                "strategy_name": strategy_name,
                "strategy_profile": profile,
                "use_setup_gate": self._use_setup_gate,
                "min_confidence_pct": self._min_confidence_pct,
            }
            for profile in variant_profiles
        ]

        pool = _create_optimizer_pool(
            cache_worker_count,
            candles_df.to_dict("records"),
            regime_mask=regime_mask,
        )
        previous_active_pool = self._active_pool
        self._active_pool = pool
        interrupted = False
        cache_results: list[dict[str, object]] = []
        completed_variants = 0
        next_log_percent = 5
        try:
            async_result = pool.imap_unordered(
                _precompute_signal_cache_worker,
                tasks,
                chunksize=1,
            )
            while True:
                if self._should_stop():
                    interrupted = True
                    return {}
                try:
                    cache_result = async_result.next(timeout=0.25)
                except PoolTimeoutError:
                    continue
                except StopIteration:
                    break
                cache_results.append(cache_result)
                completed_variants += 1
                self._emit_signal_cache_progress(
                    completed_variants,
                    len(variant_profiles),
                    strategy_label=strategy_label,
                    start_value=progress_start,
                    progress_span=progress_span,
                )
                progress_percent = int((completed_variants / len(variant_profiles)) * 100)
                if completed_variants == len(variant_profiles) or progress_percent >= next_log_percent:
                    self.log_message.emit(
                        f"{strategy_label} signal cache progress: "
                        f"{completed_variants}/{len(variant_profiles)} ({progress_percent}%)"
                    )
                    while next_log_percent <= progress_percent:
                        next_log_percent += 5
        finally:
            self._active_pool = previous_active_pool
            with suppress(Exception):
                if interrupted or self._should_stop():
                    pool.terminate()
                else:
                    pool.close()
            with suppress(Exception):
                pool.join()
            gc.collect()

        strategy_signal_cache: dict[object, dict[str, object]] = {}
        for cache_result in cache_results:
            cache_key = cache_result.pop("cache_key")
            strategy_signal_cache[cache_key] = cache_result

        self.log_message.emit(
            f"{strategy_label} signal cache prepared: {len(strategy_signal_cache)} strategy variants in RAM "
            f"using {cache_worker_count} worker processes."
        )
        return strategy_signal_cache

    def _clear_signal_caches_and_gc(self) -> None:
        self._last_precomputed_exit_cache = None
        self._hmm_regime_mask = None
        gc.collect()

    def _build_backtest_engine(
        self,
        db: Database,
        *,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_activation_pct: float | None = None,
        trailing_distance_pct: float | None = None,
    ) -> PaperTradingEngine:
        return PaperTradingEngine(
            db,
            symbol=self._symbol,
            interval=self._interval,
            leverage=self._leverage_override,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            trailing_activation_pct=trailing_activation_pct,
            trailing_distance_pct=trailing_distance_pct,
            fee_pct_override=_resolve_backtest_fee_pct(),
            enable_persistence=False,
        )

    def _log_backtest_runtime_settings(
        self,
        engine: PaperTradingEngine,
        *,
        prefix: str,
    ) -> dict[str, object]:
        runtime_settings = engine.get_runtime_settings(self._symbol)
        self.log_message.emit(
            _format_leverage_log(
                prefix=prefix,
                symbol=self._symbol,
                effective_leverage=runtime_settings.leverage,
                configured_leverage=self._configured_leverage,
                leverage_override=self._leverage_override,
                manual_override_forced=self._manual_leverage_override,
            )
        )
        self.log_message.emit(
            _format_runtime_settings_log(
                symbol=self._symbol,
                interval=runtime_settings.interval,
                runtime_leverage=runtime_settings.leverage,
                configured_leverage=self._configured_leverage,
                leverage_override=self._leverage_override,
                manual_override_forced=self._manual_leverage_override,
                stop_loss_pct=runtime_settings.stop_loss_pct,
                take_profit_pct=runtime_settings.take_profit_pct,
                trailing_activation_pct=runtime_settings.trailing_activation_pct,
                trailing_distance_pct=runtime_settings.trailing_distance_pct,
            )
        )
        effective_fee_pct = engine.get_effective_taker_fee_pct()
        self.log_message.emit(
            "Backtest fee stress active: "
            f"taker_fee={effective_fee_pct:.4f}% per side "
            f"(base={float(settings.trading.taker_fee_pct):.4f}%, "
            f"multiplier={BACKTEST_FEE_STRESS_MULTIPLIER:.2f}x)."
        )
        runtime_trace = {
            "symbol": self._symbol,
            "resolved_interval": runtime_settings.interval,
            "configured_leverage": int(self._configured_leverage),
            "effective_leverage": int(runtime_settings.leverage),
            "stop_loss_pct": float(runtime_settings.stop_loss_pct),
            "take_profit_pct": float(runtime_settings.take_profit_pct),
            "trailing_activation_pct": float(runtime_settings.trailing_activation_pct),
            "trailing_distance_pct": float(runtime_settings.trailing_distance_pct),
            "effective_taker_fee_pct": float(effective_fee_pct),
        }
        self._emit_trace_event(
            "runtime_settings",
            runtime_trace,
        )
        return runtime_trace

    def _log_backtest_metrics(self, result: dict, *, prefix: str = "Backtest metrics") -> None:
        self.log_message.emit(
            f"{prefix}: "
            f"Profit Factor = {self._format_profit_factor(float(result['profit_factor']))} | "
            f"Average Win = {float(result['average_win_usd']):.2f} USDT | "
            f"Average Loss = {float(result['average_loss_usd']):.2f} USDT | "
            f"Real RRR = {self._format_profit_factor(float(result['real_rrr']))}"
        )

    def _emit_signal_cache_progress(
        self,
        completed_variants: int,
        total_variants: int,
        *,
        strategy_label: str,
        start_value: int = 50,
        progress_span: int = 25,
    ) -> None:
        if total_variants <= 0:
            return
        progress_ratio = completed_variants / total_variants
        scaled_value = start_value + min(max(progress_span, 0), int(progress_ratio * progress_span))
        message = f"Precomputing {strategy_label} signals... {completed_variants}/{total_variants}"
        self.progress_update.emit(scaled_value, message)
        # Throttle terminal printing to avoid I/O blocking worker processes.
        try:
            if sys.stdout is None or not sys.stdout.isatty():
                return
            now = time.time()
            progress_ratio = completed_variants / total_variants
            # Print at most every `optimizer_progress_print_modulo` variants, or on >=1% progress increments, or every 2 seconds.
            should_print = False
            try:
                modulo = settings.trading.optimizer_progress_print_modulo
            except Exception:
                modulo = 500
            if completed_variants == 0 or completed_variants >= total_variants:
                should_print = True
            elif modulo > 0 and total_variants >= modulo and completed_variants % modulo == 0:
                should_print = True
            elif (progress_ratio - self._last_signal_cache_progress_ratio) >= 0.01:
                should_print = True
            elif (now - self._last_signal_cache_print_time) >= 2.0:
                should_print = True
            if should_print:
                end_char = "\n" if completed_variants >= total_variants else ""
                print(f"\r{message}", end=end_char, flush=True)
                self._last_signal_cache_print_time = now
                self._last_signal_cache_progress_ratio = progress_ratio
        except Exception:
            pass

    def _emit_optimization_progress(
        self,
        completed_profiles: int,
        total_profiles: int,
        *,
        start_value: int = 50,
        end_value: int = 99,
        prefix: str = "Optimizing profiles...",
    ) -> None:
        if total_profiles <= 0:
            return
        progress_ratio = completed_profiles / total_profiles
        max_span = max(0, end_value - start_value)
        scaled_value = start_value + min(max_span, int(progress_ratio * max_span))
        message = f"{prefix} {completed_profiles}/{total_profiles}"
        self.progress_update.emit(scaled_value, message)
        # Throttle terminal printing to avoid I/O blocking worker processes.
        try:
            if sys.stdout is None or not sys.stdout.isatty():
                return
            now = time.time()
            progress_ratio = completed_profiles / total_profiles
            # Print at most every `optimizer_progress_print_modulo` profiles, or on >=1% progress increments, or every 2 seconds.
            should_print = False
            try:
                modulo = settings.trading.optimizer_progress_print_modulo
            except Exception:
                modulo = 500
            if completed_profiles == 0 or completed_profiles >= total_profiles:
                should_print = True
            elif modulo > 0 and total_profiles >= modulo and completed_profiles % modulo == 0:
                should_print = True
            elif (progress_ratio - self._last_optimization_progress_ratio) >= 0.01:
                should_print = True
            elif (now - self._last_optimization_print_time) >= 2.0:
                should_print = True
            if should_print:
                end_char = "\n" if completed_profiles >= total_profiles else ""
                print(f"\r{message}", end=end_char, flush=True)
                self._last_optimization_print_time = now
                self._last_optimization_progress_ratio = progress_ratio
        except Exception:
            pass

    def _generate_signals(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None = None,
        regime_mask: Sequence[int] | None = None,
    ) -> tuple[list[int], int, int, int]:
        self._last_precomputed_exit_cache = None
        self._last_strategy_diagnostics = None
        with _temporary_strategy_profile(strategy_profile):
            required_candles = _required_candle_count_for_strategy(
                strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
        setup_gate = (
            None
            if self._setup_gate is None
            else SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
        )
        vectorized_payload = _build_vectorized_strategy_cache_payload(
            candles_df,
            strategy_name=strategy_name,
            required_candles=required_candles,
            setup_gate=setup_gate,
            strategy_profile=strategy_profile,
            regime_mask=regime_mask,
        )
        if vectorized_payload is not None:
            self._last_precomputed_exit_cache = {
                key: value
                for key, value in vectorized_payload.items()
                if key in {"precomputed_long_exit_flags", "precomputed_short_exit_flags"}
            }
            diagnostics_payload = vectorized_payload.get("strategy_diagnostics")
            if isinstance(diagnostics_payload, dict):
                self._last_strategy_diagnostics = dict(diagnostics_payload)
            return (
                list(vectorized_payload["signals"]),
                int(vectorized_payload["total_signals"]),
                int(vectorized_payload["approved_signals"]),
                int(vectorized_payload["blocked_signals"]),
            )
        signals: list[int] = []
        total_signals = 0
        approved_signals = 0
        blocked_signals = 0
        for index in range(len(candles_df)):
            if self._should_stop():
                return signals, total_signals, approved_signals, blocked_signals
            if index + 1 < required_candles:
                signals.append(0)
                continue
            candles_slice = candles_df.iloc[: index + 1]
            signal_direction = _evaluate_strategy_signal(
                strategy_name,
                candles_slice,
                strategy_profile,
            )
            if (
                signal_direction != 0
                and regime_mask is not None
                and index < len(regime_mask)
                and int(regime_mask[index]) <= 0
            ):
                signal_direction = 0
            if signal_direction != 0 and self._setup_gate is not None:
                total_signals += 1
                is_approved, _score, _reason = self._setup_gate.evaluate_signal(
                    candles_slice,
                    signal_direction,
                    strategy_name,
                )
                if not is_approved:
                    blocked_signals += 1
                    signal_direction = 0
                else:
                    approved_signals += 1
            elif signal_direction != 0:
                total_signals += 1
                approved_signals += 1
            signals.append(signal_direction)
        return signals, total_signals, approved_signals, blocked_signals

    def _persist_backtest_result(
        self,
        db: Database,
        result: dict[str, object],
        *,
        strategy_name: str,
    ) -> None:
        summary_payload = {
            key: value
            for key, value in result.items()
            if key not in {"closed_trades", "optimization_results"}
        }
        db.insert_backtest_run(
            symbol=self._symbol,
            strategy_name=strategy_name,
            interval=self._interval,
            optimization_mode=bool(result.get("optimization_mode", False)),
            leverage=self._configured_leverage,
            min_confidence_pct=self._min_confidence_pct,
            total_pnl_usd=float(result.get("total_pnl_usd", 0.0) or 0.0),
            win_rate_pct=float(result.get("win_rate_pct", 0.0) or 0.0),
            profit_factor=float(result.get("profit_factor", 0.0) or 0.0),
            total_trades=int(result.get("total_trades", 0) or 0),
            evaluated_profiles=(
                int(result["evaluated_profiles"])
                if result.get("evaluated_profiles") is not None
                else None
            ),
            best_profile=(
                dict(result.get("best_profile", {}))
                if isinstance(result.get("best_profile"), dict)
                else None
            ),
            summary=summary_payload,
        )

    def _prepare_hmm_regime_mask(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
    ) -> list[int] | None:
        if strategy_name != "frama_cross":
            return None

        use_hmm_filter = bool(getattr(settings.trading, "use_hmm_regime_filter", False))
        if not use_hmm_filter:
            return None

        allowed_regimes = tuple(getattr(settings.trading, "hmm_allowed_regimes", ()))
        if not allowed_regimes:
            self.log_message.emit(
                "HMM regime filter is enabled but hmm_allowed_regimes is empty. "
                "Skipping HMM filter for this run."
            )
            return None

        try:
            detector = HMMRegimeDetector(
                n_components=4,
                train_window_candles=8000,
                apply_window_candles=2000,
                warmup_candles=2000,
                stability_candles=5,
                allowed_regimes=allowed_regimes,
            )
            detection = detector.detect(candles_df)
            regime_mask = detection.regime_mask.astype("int8").tolist()
            allowed_candles = int(sum(regime_mask))
            total_candles = max(len(regime_mask), 1)
            self.log_message.emit(
                "HMM regime filter active: "
                f"{allowed_candles}/{len(regime_mask)} candles allowed "
                f"({(allowed_candles / total_candles) * 100:.1f}%), "
                f"walk-forward windows={detection.window_count}, "
                f"allowed_regimes={', '.join(allowed_regimes)}."
            )
            return regime_mask
        except Exception as exc:
            self.log_message.emit(
                f"HMM regime detector failed ({exc}). Falling back to unfiltered FRAMA signals."
            )
            return None

    def _emit_history_download_progress(self, progress_value: int, text: str) -> None:
        normalized_progress = max(0, min(100, progress_value))
        scaled_value = min(50, 10 + int(normalized_progress * 0.4))
        self.progress_update.emit(scaled_value, text)

    @staticmethod
    def _format_candle_range(candles: Sequence[CandleRecord]) -> str:
        if not candles:
            return "no candles"
        start = candles[0].open_time.strftime("%Y-%m-%d %H:%M:%S")
        end = candles[-1].open_time.strftime("%Y-%m-%d %H:%M:%S")
        return f"{start} -> {end}"

    @staticmethod
    def _format_profit_factor(value: float) -> str:
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"

    @staticmethod
    def _calculate_trade_metrics(closed_trades: Sequence[dict]) -> dict[str, float]:
        return _calculate_trade_metrics_for_closed_trades(closed_trades)
