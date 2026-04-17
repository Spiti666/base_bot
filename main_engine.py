from __future__ import annotations

import asyncio
from collections import Counter, deque
import ctypes
import gc
import hashlib
import json
import math
import os
import traceback
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
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
import sys

from config import (
    BACKTEST_ONLY_AVAILABLE_STRATEGIES,
    BACKTEST_OPTIMIZER_COIN_STRATEGIES,
    COIN_STRATEGIES,
    EMA_BAND_REJECTION_1H_EXCLUDED_COINS,
    EMA_BAND_REJECTION_1H_WINNERS,
    MAX_BACKTEST_CANDLES,
    PRODUCTION_PROFILE_REGISTRY,
    settings,
)
from core.api.bitunix import BitunixClient
from core.api.websocket import MultiTimeframeWebSocketManager
from core.data.db import (
    AdaptationLogCreate,
    CandleRecord,
    Database,
    PaperTrade,
    PaperTradeUpdate,
    RegimeObservationCreate,
    StrategyHealthSnapshot,
    TradeReviewCreate,
)
from core.data_pipeline import PolarsDataLoader
from core.data.history import HistoryManager
from core.patterns.setup_gate import SmartSetupGate
from core.regime.hmm_regime_detector import HMMRegimeDetector
from core.paper_trading.engine import PaperTradingEngine
from engine.compiled_core import run_fast_backtest_loop, run_fast_backtest_loop_detailed
from strategies.python.dual_thrust_breakout import (
    build_dual_thrust_signal_frame,
    run_python_dual_thrust,
    should_exit_python_dual_thrust_breakout,
)
from strategies.python.ema_cross_volume import (
    build_ema_cross_volume_signal_frame,
    run_python_ema_cross_volume,
)
from strategies.python.ema_band_rejection import (
    build_ema_band_rejection_signal_frame,
    run_python_ema_band_rejection,
)
from strategies.python.frama_cross import (
    build_frama_cross_signal_frame,
    run_python_frama_cross,
)
from strategies.jit_indicators import (
    compute_ema_band_rejection_signals,
    generate_strategy_signals,
)


StrategyRunner = Callable[[pd.DataFrame], int]
OptimizationProfile = dict[str, float]
STRATEGY_NAME_ALIASES = {
    "dual_thrust_breakout": "dual_thrust",
}
STRATEGY_RUNNERS: dict[str, StrategyRunner] = {
    "ema_cross_volume": run_python_ema_cross_volume,
    "ema_band_rejection": run_python_ema_band_rejection,
    "frama_cross": run_python_frama_cross,
    "dual_thrust": run_python_dual_thrust,
}
BACKTEST_EXECUTION_STRATEGIES: frozenset[str] = frozenset(
    {
        "ema_cross_volume",
        "ema_band_rejection",
        "frama_cross",
        "dual_thrust",
    }
)
BACKTEST_STRATEGY_REGISTRY: dict[str, dict[str, object]] = {}
SETUP_GATE_SETTLED_WARMUP_CANDLES = 500
LIVE_SETTLED_WARMUP_CANDLES = SETUP_GATE_SETTLED_WARMUP_CANDLES
MAX_OPTIMIZATION_WORKERS = 28
OPTIMIZER_WORKER_MAX_TASKS_PER_CHILD = 100
OPTIMIZER_MEMORY_RESERVE_GB = 2.0
OPTIMIZER_MEMORY_PER_WORKER_GB = 0.75
FORCE_MAX_OPTIMIZATION_WORKERS = False
OPTIMIZER_GUI_DEFAULT_MAX_WORKERS = 12
OPTIMIZER_SIGNAL_CACHE_MAX_PARENT_BYTES = 512 * 1024 * 1024
OPTIMIZER_SIGNAL_CACHE_MAX_REPLICATED_BYTES = 2 * 1024 * 1024 * 1024
OPTIMIZER_SAMPLE_RANDOM_SEED = 20260321
OPTIMIZER_INCREMENTAL_SYNC_MIN_CANDLES = 10_000
OPTIMIZER_INTERMEDIATE_CACHE_MIN_PROFILES = 20_000
BACKTEST_FEE_STRESS_MULTIPLIER = 1.25
LIVE_TAKER_FEE_PCT_STRESS = 0.0625
LIVE_BREAKEVEN_ACTIVATION_PCT = 3.5
LIVE_BREAKEVEN_BUFFER_PCT = 0.2
LIVE_EMA_BAND_REJECTION_CANDIDATE_SYMBOL = "FILUSDT"
LIVE_EMA_BAND_REJECTION_CANDIDATE_POLICY: dict[str, float | int | str] = {
    "strategy_name": "ema_band_rejection",
    "interval": "1h",
    "target_leverage": 5,
    "risk_multiplier": 0.35,
    "max_loss_streak": 4,
    "max_drawdown_pct": 6.0,
    "pf_window_size": 20,
    "pf_minimum": 1.0,
    "fill_slippage_window_size": 20,
    "max_avg_fill_slippage_pct": 0.15,
}
MAX_OPTIMIZATION_GRID_PROFILES = 600_000
OPTIMIZER_MIN_TOTAL_TRADES = 20
OPTIMIZER_MIN_BREAKEVEN_ACTIVATION_PCT = 3.5
OPTIMIZER_MIN_AVERAGE_WIN_TO_COST_RATIO = 3.0
OPTIMIZER_MIN_AVG_PROFIT_PER_TRADE_NET_PCT = 0.05
OPTIMIZER_SORT_PROFIT_FACTOR_CAP = 100.0
RANKING_MIN_PROFIT_FACTOR = 1.25
OPTIMIZER_STRONG_SAMPLE_MIN_TRADES = 100
RANKING_MAX_DRAWDOWN_PCT = 12.0
OPTIMIZER_PROFILE_MODE_ALL_COINS_PASS1 = "all_coins_pass1"
OPTIMIZER_PROFILE_MODE_STRICT_VERIFICATION = "strict_verification"
OPTIMIZER_PROFILE_MODE_DEFAULT = OPTIMIZER_PROFILE_MODE_ALL_COINS_PASS1
OPTIMIZER_ALL_COINS_PASS1_MIN_BREAKEVEN_ACTIVATION_PCT = 2.0
OPTIMIZER_ALL_COINS_PASS1_MIN_AVERAGE_WIN_TO_COST_RATIO = 2.25
OPTIMIZER_ALL_COINS_PASS1_MIN_AVG_PROFIT_PER_TRADE_NET_PCT = 0.05
OPTIMIZER_ALL_COINS_PASS1_RANKING_MIN_PROFIT_FACTOR = 1.15
OPTIMIZER_ALL_COINS_PASS1_RANKING_MAX_DRAWDOWN_PCT = 15.0
OPTIMIZER_ALL_COINS_PASS1_STRONG_SAMPLE_MIN_TRADES = 80
OPTIMIZER_ALL_COINS_PASS1_SEARCH_WINDOW_CANDLES = 35_000
OPTIMIZER_ALL_COINS_PASS1_VALIDATION_TOP_N = 5
OPTIMIZER_ALL_COINS_PASS1_MAX_SAMPLE_PROFILES = 5_000
OPTIMIZER_ALL_COINS_PASS1_RANDOM_SEARCH_SAMPLES = 5_000
OPTIMIZER_ALL_COINS_PASS1_TWO_STAGE_ENABLED = True
OPTIMIZER_ALL_COINS_PASS1_TWO_STAGE_STRATEGIES: frozenset[str] = frozenset(
    {"ema_cross_volume", "ema_band_rejection", "frama_cross", "dual_thrust"}
)
OPTIMIZER_ALL_COINS_PASS1_STAGE1_TARGET_PROFILES = 1_500
OPTIMIZER_ALL_COINS_PASS1_STAGE2_TOP_N = 50
OPTIMIZER_ALL_COINS_PASS1_STAGE1_SEARCH_WINDOW_CANDLES = 25_000
OPTIMIZER_ALL_COINS_PASS1_STAGE2_SEARCH_WINDOW_CANDLES = 60_000
OPTIMIZER_ALL_COINS_PASS1_FORCE_FULL_VERIFICATION_FOR_WINNER = True
BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE = float(
    PaperTradingEngine.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE
)
BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE = float(
    PaperTradingEngine.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE
)
OPTIMIZER_SHORT_INTERVAL_WINDOW_OVERRIDES: dict[str, int] = {
    "1m": 60_000,
    "5m": 60_000,
}
TRACE_VALUE_UNAVAILABLE = "unavailable"
MAX_STRATEGY_FAMILY_GRID_PROFILES = 64_000
OPTIMIZER_BREAKEVEN_MIN_BY_STRATEGY: dict[str, float | None] = {
    "ema_band_rejection": 1.5,
}
OPTIMIZER_TWO_STAGE_DEFAULT_ENABLED = True
OPTIMIZER_TWO_STAGE_DEFAULT_STRATEGIES: frozenset[str] = frozenset(
    {"ema_band_rejection"}
)
OPTIMIZER_TWO_STAGE_STAGE1_TARGET_PROFILES_DEFAULT = 2_500
OPTIMIZER_TWO_STAGE_STAGE2_TOP_N_DEFAULT = 100
OPTIMIZER_TWO_STAGE_STAGE1_SEARCH_WINDOW_CANDLES_DEFAULT = 25_000
OPTIMIZER_TWO_STAGE_STAGE2_SEARCH_WINDOW_CANDLES_DEFAULT = 70_000
OPTIMIZER_WORKER_CAP_BY_STRATEGY_DEFAULT: dict[str, int] = {
    "ema_band_rejection": 16,
}
OPTIMIZER_FORCE_FULL_VERIFICATION_FOR_WINNER_DEFAULT = True
OPTIMIZER_JIT_STRATEGY_CONSISTENT_BACKTEST: frozenset[str] = frozenset(
    {
        "ema_band_rejection",
        "frama_cross",
    }
)
ENTRY_SNAPSHOT_SCHEMA_VERSION = "entry_snapshot_v1"
REGIME_ENGINE_VERSION = "regime_engine_v1"
TRADE_REVIEW_ENGINE_VERSION = "trade_review_v1"
STRATEGY_HEALTH_ENGINE_VERSION = "strategy_health_v1"
REGIME_LABEL_UNIVERSE: frozenset[str] = frozenset(
    {
        "trend_clean_up",
        "trend_clean_down",
        "trend_exhausted",
        "range_balanced",
        "range_volatile",
        "breakout_transition",
        "compression_pre_breakout",
        "panic_expansion",
        "illiquid_noise",
    }
)
REVIEW_ERROR_CATALOG: tuple[str, ...] = (
    "late_entry_after_expansion",
    "entry_into_exhaustion",
    "entry_against_regime",
    "entry_without_sufficient_confirmation",
    "stop_too_tight",
    "stop_too_wide",
    "tp_too_conservative",
    "tp_too_ambitious",
    "strategy_regime_mismatch",
    "strategy_coin_mismatch",
    "should_not_have_traded",
)
META_REPORTS_DIRECTORY = Path("data/meta_reports")
META_MAX_DAILY_LOSS_USD = float(settings.trading.start_capital) * 0.05
META_MAX_SYMBOL_LOSS_STREAK = 5
META_MAX_STRATEGY_DRAWDOWN_PCT = 12.0
META_MAX_OPEN_POSITIONS_GUARD = int(max(1, settings.trading.max_open_positions))
META_MAX_CORRELATED_RISK = 4
META_MAX_PAUSE_EVENTS_PER_DAY = 12
META_COOLDOWN_AFTER_PAUSE_MINUTES = 90
META_MIN_SAMPLE_SIZE_FOR_DEMOTION = 12
META_MIN_SAMPLE_SIZE_FOR_PROMOTION = 25
META_OPERATIONAL_ERROR_FAILSAFE_THRESHOLD = 5
META_OPERATIONAL_ERROR_WINDOW_MINUTES = 30
META_OPERATIONAL_ERROR_DEDUP_SECONDS = 120
META_OPERATIONAL_ERROR_MIN_DISTINCT_SYMBOLS = 2
META_OPERATIONAL_ERROR_MIN_DISTINCT_SIGNATURES = 2
META_FAILSAFE_AUTO_RECOVER_MINUTES = 60
REGIME_HMM_NON_CONVERGENCE_WARN_INTERVAL_MINUTES = 30
META_WARN_HOOK_CHANNELS: tuple[str, ...] = ("log", "email_stub", "telegram_stub")
META_STATE_ORDER: tuple[str, ...] = (
    "healthy",
    "degraded",
    "watchlist",
    "paused",
)

FRAMA_FIXED_GRID_OPTIONS: dict[str, tuple[float | int, ...]] = {
    "frama_fast_period": (8, 13, 21),
    "frama_slow_period": (60, 100, 150, 200),
    "volume_multiplier": (1.0, 1.25, 1.5, 2.0),
    "chandelier_period": (14, 30),
    "chandelier_multiplier": (2.5, 3.5),
    "stop_loss_pct": (0.5, 1.5, 3.0),
    "take_profit_pct": (3.5, 5.0, 7.5),
    "trailing_activation_pct": (3.0, 5.0, 7.5),
    "trailing_distance_pct": (0.1, 0.2),
    "breakeven_activation_pct": (2.0, 3.0, 4.0),
    "breakeven_buffer_pct": (0.2,),
}
FRAMA_GRID_TRIM_ORDER: tuple[str, ...] = (
    "chandelier_multiplier",
    "chandelier_period",
    "trailing_distance_pct",
    "trailing_activation_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "volume_multiplier",
)

EMA_CROSS_VOLUME_FIXED_GRID_OPTIONS: dict[str, tuple[float | int, ...]] = {
    "ema_fast_period": (9, 13, 21),
    "ema_slow_period": (100, 200, 300),
    "volume_multiplier": (1.25, 1.5, 2.0, 2.5),
    "stop_loss_pct": (0.5, 1.0, 1.5, 3.0),
    "take_profit_pct": (3.0, 5.0, 7.5, 10.0),
    "trailing_activation_pct": (3.0, 5.0, 8.0),
    "trailing_distance_pct": (0.1, 0.3, 0.5),
    "breakeven_activation_pct": (2.0, 3.5, 5.0),
    "breakeven_buffer_pct": (0.2,),
    "tight_trailing_activation_pct": (8.0,),
    "tight_trailing_distance_pct": (0.3,),
}
EMA_CROSS_VOLUME_GRID_TRIM_ORDER: tuple[str, ...] = (
    "trailing_distance_pct",
    "trailing_activation_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "volume_multiplier",
    "ema_slow_period",
)

EMA_BAND_REJECTION_FIXED_GRID_OPTIONS: dict[str, tuple[float | int, ...]] = {
    "ema_fast": (5,),
    "ema_mid": (10,),
    "ema_slow": (20,),
    "slope_lookback": (3, 5),
    "min_ema_spread_pct": (0.03, 0.05, 0.08),
    "min_slow_slope_pct": (0.0, 0.01, 0.02),
    "pullback_requires_outer_band_touch": (0, 1),
    "use_rejection_quality_filter": (0, 1),
    "rejection_wick_min_ratio": (0.35,),
    "rejection_body_min_ratio": (0.20,),
    "use_rsi_filter": (0, 1),
    "rsi_length": (14,),
    "rsi_midline": (50.0,),
    "use_rsi_cross_filter": (0, 1),
    "rsi_midline_margin": (0.0, 1.0),
    "use_volume_filter": (0, 1),
    "volume_ma_length": (20,),
    "volume_multiplier": (1.0, 1.25),
    "use_atr_stop_buffer": (0, 1),
    "atr_length": (14,),
    "atr_stop_buffer_mult": (0.5,),
    "signal_cooldown_bars": (0, 1, 2),
    "stop_loss_pct": (1.0, 1.5, 2.0),
    "take_profit_pct": (3.0, 5.0),
    "trailing_activation_pct": (2.0, 3.0),
    "trailing_distance_pct": (0.1, 0.3),
    "breakeven_activation_pct": (2.0, 3.5),
    "breakeven_buffer_pct": (0.2,),
}
EMA_BAND_REJECTION_GRID_TRIM_ORDER: tuple[str, ...] = (
    "signal_cooldown_bars",
    "pullback_requires_outer_band_touch",
    "use_rejection_quality_filter",
    "use_atr_stop_buffer",
    "use_rsi_filter",
    "use_volume_filter",
    "trailing_distance_pct",
    "trailing_activation_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "min_slow_slope_pct",
    "min_ema_spread_pct",
    "slope_lookback",
)

DUAL_THRUST_FIXED_GRID_OPTIONS: dict[str, tuple[float | int, ...]] = {
    "dual_thrust_k1": (0.2, 0.35, 0.5, 0.8),
    "dual_thrust_k2": (0.2, 0.3, 0.5),
    "dual_thrust_period": (34, 55, 89),
    "chandelier_period": (14, 30),
    "chandelier_multiplier": (1.5, 2.5),
    "stop_loss_pct": (0.5, 1.5, 3.0),
    "take_profit_pct": (3.0, 5.0, 10.0, 15.0),
    "trailing_activation_pct": (3.0, 6.0, 8.0),
    "trailing_distance_pct": (0.15, 0.3, 0.5),
    "breakeven_activation_pct": (2.0, 3.5, 5.0),
    "breakeven_buffer_pct": (0.2,),
    "tight_trailing_activation_pct": (8.0,),
    "tight_trailing_distance_pct": (0.3,),
}
DUAL_THRUST_GRID_TRIM_ORDER: tuple[str, ...] = (
    "chandelier_multiplier",
    "chandelier_period",
    "trailing_distance_pct",
    "trailing_activation_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "dual_thrust_period",
)


def _summarize_profile_grid_bounds(
    profiles: Sequence[Mapping[str, object]] | None,
) -> dict[str, dict[str, float]]:
    if not profiles:
        return {}
    bounds: dict[str, dict[str, float]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            continue
        for field_name, raw_value in profile.items():
            with suppress(Exception):
                numeric_value = float(raw_value)
                if not math.isfinite(numeric_value):
                    continue
                current = bounds.get(str(field_name))
                if current is None:
                    bounds[str(field_name)] = {
                        "min": float(numeric_value),
                        "max": float(numeric_value),
                    }
                else:
                    current["min"] = float(min(float(current["min"]), numeric_value))
                    current["max"] = float(max(float(current["max"]), numeric_value))
    return bounds
_OPTIMIZER_SETTINGS_LOCK = RLock()
_WORKER_CANDLES_DF: pd.DataFrame | None = None
_WORKER_CANDLE_ROWS: list[dict[str, object]] | None = None
_WORKER_REGIME_MASK: list[int] | None = None
_WORKER_STRATEGY_SIGNAL_CACHE: dict[object, dict[str, object]] | None = None
_WORKER_OHLCV_ARRAYS: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
_NUMBA_WARMUP_DONE = False


def _resolve_backtest_history_start_utc() -> datetime:
    parsed = datetime.fromisoformat(
        settings.trading.backtest_history_start_utc.replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(tzinfo=None)


BACKTEST_HISTORY_START_UTC = _resolve_backtest_history_start_utc()
BACKTEST_HISTORY_SEED_STATE_PATH = Path("data/backtest_history_seed_state.json")
_BACKTEST_HISTORY_SEED_STATE_LOCK = RLock()
_BACKTEST_HISTORY_SEED_STATE_VERSION = 1


def _resolve_optimizer_history_start_utc() -> datetime:
    parsed = datetime.fromisoformat(
        settings.trading.optimizer_history_start_utc.replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(tzinfo=None)


OPTIMIZER_HISTORY_START_UTC = _resolve_optimizer_history_start_utc()


def _history_seed_key(symbol: str, interval: str) -> str:
    normalized_symbol = str(symbol).strip().upper()
    normalized_interval = _validate_interval_name(str(interval))
    return f"{normalized_symbol}|{normalized_interval}"


def _history_seed_parse_utc_naive(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    with suppress(Exception):
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(UTC).replace(tzinfo=None)
    return None


def _history_seed_load_state() -> dict[str, object]:
    path = BACKTEST_HISTORY_SEED_STATE_PATH
    if not path.exists():
        return {"version": _BACKTEST_HISTORY_SEED_STATE_VERSION, "entries": {}}
    with suppress(Exception):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            entries = payload.get("entries")
            if not isinstance(entries, dict):
                entries = {}
            return {
                "version": int(payload.get("version", _BACKTEST_HISTORY_SEED_STATE_VERSION) or _BACKTEST_HISTORY_SEED_STATE_VERSION),
                "entries": dict(entries),
            }
    return {"version": _BACKTEST_HISTORY_SEED_STATE_VERSION, "entries": {}}


def _history_seed_save_state(state: Mapping[str, object]) -> None:
    path = BACKTEST_HISTORY_SEED_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        "version": int(state.get("version", _BACKTEST_HISTORY_SEED_STATE_VERSION) or _BACKTEST_HISTORY_SEED_STATE_VERSION),
        "entries": dict(state.get("entries", {})) if isinstance(state.get("entries"), Mapping) else {},
    }
    path.write_text(
        json.dumps(serialized, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _is_full_history_seed_required(
    *,
    symbol: str,
    interval: str,
    requested_start_utc: datetime,
    requested_end_utc: datetime,
) -> tuple[bool, str]:
    key = _history_seed_key(symbol, interval)
    with _BACKTEST_HISTORY_SEED_STATE_LOCK:
        state = _history_seed_load_state()
        entries = state.get("entries", {})
        if not isinstance(entries, Mapping):
            return True, "seed state invalid"
        entry = entries.get(key)
        if not isinstance(entry, Mapping):
            return True, "no seed state"
        seeded_start_utc = _history_seed_parse_utc_naive(entry.get("seeded_start_utc"))
        if seeded_start_utc is None:
            return True, "seed start missing"
        interval_seconds = max(_interval_total_seconds(interval), 60)
        tolerance = timedelta(seconds=interval_seconds * 2)
        if seeded_start_utc > (requested_start_utc + tolerance):
            return True, "requested start extends further back than seeded baseline"
        return False, "seed baseline already available"


def _mark_history_seed_completed(
    *,
    symbol: str,
    interval: str,
    seeded_start_utc: datetime,
    seeded_end_utc: datetime,
    mode: str,
) -> None:
    key = _history_seed_key(symbol, interval)
    now_text = datetime.now(tz=UTC).replace(tzinfo=None).isoformat(sep=" ")
    with _BACKTEST_HISTORY_SEED_STATE_LOCK:
        state = _history_seed_load_state()
        entries_raw = state.get("entries", {})
        entries: dict[str, object] = (
            dict(entries_raw)
            if isinstance(entries_raw, Mapping)
            else {}
        )
        existing_entry = entries.get(key)
        if isinstance(existing_entry, Mapping):
            existing_seeded_start_utc = _history_seed_parse_utc_naive(
                existing_entry.get("seeded_start_utc")
            )
            existing_seeded_end_utc = _history_seed_parse_utc_naive(
                existing_entry.get("seeded_end_utc")
            )
            if existing_seeded_start_utc is not None:
                seeded_start_utc = min(seeded_start_utc, existing_seeded_start_utc)
            if existing_seeded_end_utc is not None:
                seeded_end_utc = max(seeded_end_utc, existing_seeded_end_utc)
        entries[key] = {
            "symbol": str(symbol).strip().upper(),
            "interval": _validate_interval_name(interval),
            "seeded_start_utc": seeded_start_utc.isoformat(sep=" "),
            "seeded_end_utc": seeded_end_utc.isoformat(sep=" "),
            "seed_mode": str(mode).strip().lower() or "date_range",
            "updated_at_utc": now_text,
        }
        state["entries"] = entries
        _history_seed_save_state(state)


def _history_start_for_mode(*, optimize_profile: bool) -> datetime:
    return OPTIMIZER_HISTORY_START_UTC if optimize_profile else BACKTEST_HISTORY_START_UTC


def resolve_optimizer_profile_mode_name() -> str:
    raw_mode = str(
        getattr(settings.trading, "optimizer_profile_mode", OPTIMIZER_PROFILE_MODE_DEFAULT)
        or OPTIMIZER_PROFILE_MODE_DEFAULT
    ).strip().lower()
    if raw_mode in {
        OPTIMIZER_PROFILE_MODE_ALL_COINS_PASS1,
        OPTIMIZER_PROFILE_MODE_STRICT_VERIFICATION,
    }:
        return raw_mode
    return OPTIMIZER_PROFILE_MODE_DEFAULT


def _is_optimizer_all_coins_pass1_mode() -> bool:
    return resolve_optimizer_profile_mode_name() == OPTIMIZER_PROFILE_MODE_ALL_COINS_PASS1


def _resolve_optimizer_min_average_win_to_cost_ratio() -> float:
    if _is_optimizer_all_coins_pass1_mode():
        return float(OPTIMIZER_ALL_COINS_PASS1_MIN_AVERAGE_WIN_TO_COST_RATIO)
    return float(OPTIMIZER_MIN_AVERAGE_WIN_TO_COST_RATIO)


def _resolve_optimizer_min_avg_profit_per_trade_net_pct() -> float:
    if _is_optimizer_all_coins_pass1_mode():
        return float(OPTIMIZER_ALL_COINS_PASS1_MIN_AVG_PROFIT_PER_TRADE_NET_PCT)
    return float(OPTIMIZER_MIN_AVG_PROFIT_PER_TRADE_NET_PCT)


def _resolve_optimizer_ranking_min_profit_factor() -> float:
    if _is_optimizer_all_coins_pass1_mode():
        return float(OPTIMIZER_ALL_COINS_PASS1_RANKING_MIN_PROFIT_FACTOR)
    return float(RANKING_MIN_PROFIT_FACTOR)


def _resolve_optimizer_ranking_max_drawdown_pct() -> float:
    if _is_optimizer_all_coins_pass1_mode():
        return float(OPTIMIZER_ALL_COINS_PASS1_RANKING_MAX_DRAWDOWN_PCT)
    return float(RANKING_MAX_DRAWDOWN_PCT)


def _resolve_optimizer_strong_sample_min_trades() -> int:
    if _is_optimizer_all_coins_pass1_mode():
        return int(OPTIMIZER_ALL_COINS_PASS1_STRONG_SAMPLE_MIN_TRADES)
    return int(OPTIMIZER_STRONG_SAMPLE_MIN_TRADES)


def _resolve_optimizer_validation_top_n() -> int:
    if _is_optimizer_all_coins_pass1_mode():
        return int(OPTIMIZER_ALL_COINS_PASS1_VALIDATION_TOP_N)
    with suppress(Exception):
        configured = int(getattr(settings.trading, "optimization_validation_top_n", 0) or 0)
        if configured > 0:
            return configured
    return int(OPTIMIZER_ALL_COINS_PASS1_VALIDATION_TOP_N)


def _resolve_backtest_fee_pct() -> float:
    base_fee_pct = float(settings.trading.taker_fee_pct)
    return max(0.0, base_fee_pct * BACKTEST_FEE_STRESS_MULTIPLIER)


def _resolve_backtest_round_trip_cost_pct() -> float:
    return max(0.0, (_resolve_backtest_fee_pct() * 2.0) + BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE)


def _resolve_optimizer_min_breakeven_activation_pct_for_strategy(
    strategy_name: str | None,
) -> float | None:
    if _is_optimizer_all_coins_pass1_mode():
        return float(OPTIMIZER_ALL_COINS_PASS1_MIN_BREAKEVEN_ACTIVATION_PCT)
    resolved_strategy_name = None
    if strategy_name is not None:
        with suppress(Exception):
            resolved_strategy_name = _validate_strategy_name(str(strategy_name))
    if resolved_strategy_name is None and strategy_name is not None:
        resolved_strategy_name = str(strategy_name).strip().lower() or None
    if resolved_strategy_name in OPTIMIZER_BREAKEVEN_MIN_BY_STRATEGY:
        return OPTIMIZER_BREAKEVEN_MIN_BY_STRATEGY[resolved_strategy_name]
    return float(OPTIMIZER_MIN_BREAKEVEN_ACTIVATION_PCT)


def _describe_optimizer_breakeven_constraint(strategy_name: str | None) -> str:
    minimum_value = _resolve_optimizer_min_breakeven_activation_pct_for_strategy(
        strategy_name
    )
    if minimum_value is None:
        return "breakeven_activation_pct: no minimum"
    return f"breakeven_activation_pct >= {float(minimum_value):.1f}%"


def _profile_meets_breakeven_constraint(
    strategy_profile: OptimizationProfile,
    *,
    strategy_name: str | None = None,
) -> bool:
    resolved_strategy_name = None
    if strategy_name is not None:
        with suppress(Exception):
            resolved_strategy_name = _validate_strategy_name(str(strategy_name))
    if resolved_strategy_name is None:
        with suppress(Exception):
            resolved_strategy_name = _validate_strategy_name(
                str(strategy_profile.get("strategy_name", ""))
            )
    minimum_breakeven_activation_pct = (
        _resolve_optimizer_min_breakeven_activation_pct_for_strategy(
            resolved_strategy_name
        )
    )
    if minimum_breakeven_activation_pct is None:
        return True
    if "breakeven_activation_pct" not in strategy_profile:
        return True
    with suppress(Exception):
        return (
            float(strategy_profile["breakeven_activation_pct"])
            >= float(minimum_breakeven_activation_pct)
        )
    return False


def _resolve_avg_profit_per_trade_net_pct(summary: Mapping[str, object]) -> float:
    total_trades = float(summary.get("total_trades", 0.0) or 0.0)
    if total_trades <= 0.0:
        return 0.0
    start_capital = float(summary.get("start_capital_usd", settings.trading.start_capital) or settings.trading.start_capital)
    if not math.isfinite(start_capital) or start_capital <= 0.0:
        return 0.0
    total_pnl_usd = float(summary.get("total_pnl_usd", 0.0) or 0.0)
    if not math.isfinite(total_pnl_usd):
        return 0.0
    avg_profit_per_trade_net_pct = (
        (total_pnl_usd / start_capital) * 100.0
    ) / total_trades
    if not math.isfinite(avg_profit_per_trade_net_pct):
        return 0.0
    return float(avg_profit_per_trade_net_pct)


def _profile_meets_avg_profit_per_trade_hurdle(summary: Mapping[str, object]) -> bool:
    return _resolve_avg_profit_per_trade_net_pct(summary) >= float(
        _resolve_optimizer_min_avg_profit_per_trade_net_pct()
    )


def _resolve_optimizer_min_total_trades_for_interval(interval: str) -> int:
    if _is_optimizer_all_coins_pass1_mode():
        return int(OPTIMIZER_MIN_TOTAL_TRADES)
    interval_text = str(interval).strip()
    with suppress(Exception):
        interval_text = _validate_interval_name(interval_text)
    with suppress(Exception):
        interval_seconds = int(_interval_total_seconds(interval_text))
        if interval_seconds <= int(_interval_total_seconds("15m")):
            return 20
        if interval_seconds == int(_interval_total_seconds("1h")):
            return 12
        if interval_seconds >= int(_interval_total_seconds("4h")):
            return 8
    return int(OPTIMIZER_MIN_TOTAL_TRADES)


def _apply_strategy_interval_profile_constraints(
    profiles: Sequence[OptimizationProfile],
    *,
    strategy_name: str,
    interval: str,
) -> list[OptimizationProfile]:
    _ = _validate_strategy_name(strategy_name)
    _ = _validate_interval_name(interval)
    return [dict(profile) for profile in profiles]


def _enforce_optimizer_min_confidence_floor(min_confidence_pct: float | None) -> float | None:
    if min_confidence_pct is None:
        return None
    configured_floor = float(settings.trading.min_confidence_pct)
    return max(float(min_confidence_pct), configured_floor)


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

    search_window_candles = _resolve_optimizer_search_window_candles(interval)
    required_candles = _required_candle_count_for_strategy(
        strategy_name,
        use_setup_gate=use_setup_gate,
    )
    hmm_warmup_candles = (
        2500
        if (
            strategy_name == "frama_cross"
            and bool(getattr(settings.trading, "use_hmm_regime_filter", False))
        )
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


def _resolve_optimizer_search_window_candles(interval: str) -> int:
    if _is_optimizer_all_coins_pass1_mode():
        configured_window = int(OPTIMIZER_ALL_COINS_PASS1_SEARCH_WINDOW_CANDLES)
    else:
        configured_window = max(1, int(settings.trading.optimization_search_window_candles))
    interval_key = str(interval).strip().lower()
    override_window = OPTIMIZER_SHORT_INTERVAL_WINDOW_OVERRIDES.get(interval_key)
    if override_window is None:
        return configured_window
    return max(1, min(configured_window, int(override_window)))


def _optimization_risk_profiles() -> list[tuple[float, float, float, float]]:
    # Use universal risk grid when present.
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
        for (
            take_profit_pct,
            stop_loss_pct,
            trailing_activation_pct,
            trailing_distance_pct,
        ) in product(
            tp_opts,
            sl_opts,
            ta_opts,
            td_opts,
        )
        if _passes_common_risk_profile_guards(
            take_profit_pct=float(take_profit_pct),
            stop_loss_pct=float(stop_loss_pct),
            trailing_activation_pct=float(trailing_activation_pct),
            trailing_distance_pct=float(trailing_distance_pct),
        )
    ]


def _passes_common_risk_profile_guards(
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    trailing_activation_pct: float,
    trailing_distance_pct: float,
    breakeven_activation_pct: float | None = None,
    avg_profit_per_trade_net_pct: float | None = None,
) -> bool:
    if not (
        math.isfinite(take_profit_pct)
        and math.isfinite(stop_loss_pct)
        and math.isfinite(trailing_activation_pct)
        and math.isfinite(trailing_distance_pct)
    ):
        return False
    if stop_loss_pct <= 0.0:
        return False
    # Risk-Reward guard: reject non-positive edge and weak RRR combinations.
    if take_profit_pct <= stop_loss_pct:
        return False
    if (take_profit_pct / stop_loss_pct) < 1.5:
        return False
    # Keep trailing distance meaningfully below activation threshold.
    if trailing_distance_pct > (trailing_activation_pct * 0.5):
        return False
    if breakeven_activation_pct is not None:
        if not math.isfinite(float(breakeven_activation_pct)):
            return False
        if trailing_activation_pct < float(breakeven_activation_pct):
            return False
    if avg_profit_per_trade_net_pct is not None:
        if not math.isfinite(float(avg_profit_per_trade_net_pct)):
            return False
        if float(avg_profit_per_trade_net_pct) < float(
            _resolve_optimizer_min_avg_profit_per_trade_net_pct()
        ):
            return False
    return True


def _passes_ema_band_rejection_exit_profile_guards(
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_activation_pct: float,
    breakeven_activation_pct: float,
) -> bool:
    if not (
        math.isfinite(take_profit_pct)
        and math.isfinite(trailing_activation_pct)
        and math.isfinite(breakeven_activation_pct)
    ):
        return False
    # Avoid dead combinations where management cannot logically activate before TP.
    if breakeven_activation_pct >= take_profit_pct:
        return False
    if trailing_activation_pct >= take_profit_pct:
        return False
    if take_profit_pct < (1.5 * stop_loss_pct):
        return False
    return True


def _passes_chandelier_profile_guards(
    *,
    chandelier_period: float,
    chandelier_multiplier: float,
) -> bool:
    if not (
        math.isfinite(chandelier_period)
        and math.isfinite(chandelier_multiplier)
    ):
        return False
    if float(chandelier_period) < 14.0:
        return False
    if float(chandelier_multiplier) < 1.5:
        return False
    return True


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


def generate_ema_optimization_grid(
    *,
    interval: str | None = None,
) -> list[OptimizationProfile]:
    _ = interval
    return _build_capped_strategy_family_profiles(
        strategy_family_name="ema_cross_volume",
        base_options=EMA_CROSS_VOLUME_FIXED_GRID_OPTIONS,
        trim_order=EMA_CROSS_VOLUME_GRID_TRIM_ORDER,
        profile_builder=lambda options: _build_ema_cross_volume_profiles(options),
    )


def generate_ema_band_rejection_optimization_grid(
    *,
    interval: str | None = None,
) -> list[OptimizationProfile]:
    _ = interval
    return _build_capped_strategy_family_profiles(
        strategy_family_name="ema_band_rejection",
        base_options=EMA_BAND_REJECTION_FIXED_GRID_OPTIONS,
        trim_order=EMA_BAND_REJECTION_GRID_TRIM_ORDER,
        profile_builder=lambda options: _build_ema_band_rejection_profiles(options),
    )


def generate_frama_optimization_grid(
    *,
    interval: str | None = None,
) -> list[OptimizationProfile]:
    _ = interval
    return _build_capped_strategy_family_profiles(
        strategy_family_name="frama_cross",
        base_options=FRAMA_FIXED_GRID_OPTIONS,
        trim_order=FRAMA_GRID_TRIM_ORDER,
        profile_builder=lambda options: _build_frama_profiles(options),
    )


def generate_dual_thrust_optimization_grid(
    symbol: str | None = None,
    *,
    interval: str | None = None,
) -> list[OptimizationProfile]:
    _ = symbol
    _ = interval
    return _build_capped_strategy_family_profiles(
        strategy_family_name="dual_thrust",
        base_options=DUAL_THRUST_FIXED_GRID_OPTIONS,
        trim_order=DUAL_THRUST_GRID_TRIM_ORDER,
        profile_builder=lambda options: _build_dual_thrust_profiles(options),
    )


def _trim_outermost_grid_values(
    values: tuple[float | int, ...],
) -> tuple[float | int, ...]:
    if len(values) <= 1:
        return values
    if len(values) == 2:
        return (values[0],)
    return tuple(values[1:-1])


def _build_capped_strategy_family_profiles(
    *,
    strategy_family_name: str,
    base_options: dict[str, tuple[float | int, ...]],
    trim_order: tuple[str, ...],
    profile_builder: Callable[[dict[str, tuple[float | int, ...]]], list[OptimizationProfile]],
) -> list[OptimizationProfile]:
    working_options: dict[str, tuple[float | int, ...]] = {
        str(field_name): tuple(values)
        for field_name, values in base_options.items()
    }
    estimated_profile_count = 1
    for values in working_options.values():
        estimated_profile_count *= max(1, len(values))
    pretrim_limit = int(MAX_STRATEGY_FAMILY_GRID_PROFILES * 64)
    if estimated_profile_count > pretrim_limit:
        while estimated_profile_count > pretrim_limit:
            option_was_trimmed = False
            for field_name in trim_order:
                current_values = tuple(working_options.get(field_name, ()))
                trimmed_values = _trim_outermost_grid_values(current_values)
                if trimmed_values == current_values:
                    continue
                working_options[field_name] = trimmed_values
                option_was_trimmed = True
                estimated_profile_count = 1
                for values in working_options.values():
                    estimated_profile_count *= max(1, len(values))
                if estimated_profile_count <= pretrim_limit:
                    break
            if not option_was_trimmed:
                break
    profiles = profile_builder(working_options)
    if len(profiles) <= MAX_STRATEGY_FAMILY_GRID_PROFILES:
        return profiles

    while len(profiles) > MAX_STRATEGY_FAMILY_GRID_PROFILES:
        option_was_trimmed = False
        for field_name in trim_order:
            current_values = tuple(working_options.get(field_name, ()))
            trimmed_values = _trim_outermost_grid_values(current_values)
            if trimmed_values == current_values:
                continue
            working_options[field_name] = trimmed_values
            option_was_trimmed = True
            profiles = profile_builder(working_options)
            if len(profiles) <= MAX_STRATEGY_FAMILY_GRID_PROFILES:
                break
        if len(profiles) <= MAX_STRATEGY_FAMILY_GRID_PROFILES:
            break
        if not option_was_trimmed:
            raise RuntimeError(
                "Unable to trim strategy grid below limit "
                f"({MAX_STRATEGY_FAMILY_GRID_PROFILES}) for {strategy_family_name}."
            )
    return profiles


def _build_frama_profiles(
    options: dict[str, tuple[float | int, ...]],
) -> list[OptimizationProfile]:
    profiles: list[OptimizationProfile] = []
    optimize_chandelier = (
        "chandelier_period" in options or "chandelier_multiplier" in options
    )
    chandelier_period_values = options.get(
        "chandelier_period",
        (int(settings.trading.chandelier_period),),
    )
    chandelier_multiplier_values = options.get(
        "chandelier_multiplier",
        (float(settings.trading.chandelier_multiplier),),
    )
    min_frama_gap_pct_values = options.get("min_frama_gap_pct", (0.05,))
    min_frama_slope_pct_values = options.get("min_frama_slope_pct", (0.0,))
    post_cross_confirmation_bars_values = options.get("post_cross_confirmation_bars", (0,))
    for (
        frama_fast_period,
        frama_slow_period,
        volume_multiplier,
        chandelier_period,
        chandelier_multiplier,
        min_frama_gap_pct,
        min_frama_slope_pct,
        post_cross_confirmation_bars,
        stop_loss_pct,
        take_profit_pct,
        trailing_activation_pct,
        trailing_distance_pct,
        breakeven_activation_pct,
        breakeven_buffer_pct,
    ) in product(
        options["frama_fast_period"],
        options["frama_slow_period"],
        options["volume_multiplier"],
        chandelier_period_values,
        chandelier_multiplier_values,
        min_frama_gap_pct_values,
        min_frama_slope_pct_values,
        post_cross_confirmation_bars_values,
        options["stop_loss_pct"],
        options["take_profit_pct"],
        options["trailing_activation_pct"],
        options["trailing_distance_pct"],
        options["breakeven_activation_pct"],
        options["breakeven_buffer_pct"],
    ):
        if int(frama_fast_period) >= int(frama_slow_period):
            continue
        resolved_chandelier_period = float(chandelier_period)
        resolved_chandelier_multiplier = float(chandelier_multiplier)
        if optimize_chandelier and not _passes_chandelier_profile_guards(
            chandelier_period=resolved_chandelier_period,
            chandelier_multiplier=resolved_chandelier_multiplier,
        ):
            continue
        resolved_take_profit_pct = float(take_profit_pct)
        resolved_stop_loss_pct = float(stop_loss_pct)
        resolved_trailing_activation_pct = float(trailing_activation_pct)
        resolved_trailing_distance_pct = float(trailing_distance_pct)
        resolved_breakeven_activation_pct = float(breakeven_activation_pct)
        if not _passes_common_risk_profile_guards(
            take_profit_pct=resolved_take_profit_pct,
            stop_loss_pct=resolved_stop_loss_pct,
            trailing_activation_pct=resolved_trailing_activation_pct,
            trailing_distance_pct=resolved_trailing_distance_pct,
            breakeven_activation_pct=resolved_breakeven_activation_pct,
        ):
            continue
        profile: OptimizationProfile = {
            "frama_fast_period": float(frama_fast_period),
            "frama_slow_period": float(frama_slow_period),
            "volume_multiplier": float(volume_multiplier),
            "stop_loss_pct": resolved_stop_loss_pct,
            "take_profit_pct": resolved_take_profit_pct,
            "trailing_activation_pct": resolved_trailing_activation_pct,
            "trailing_distance_pct": resolved_trailing_distance_pct,
            "breakeven_activation_pct": resolved_breakeven_activation_pct,
            "breakeven_buffer_pct": float(breakeven_buffer_pct),
        }
        if "min_frama_gap_pct" in options:
            profile["min_frama_gap_pct"] = float(min_frama_gap_pct)
        if "min_frama_slope_pct" in options:
            profile["min_frama_slope_pct"] = float(min_frama_slope_pct)
        if "post_cross_confirmation_bars" in options:
            profile["post_cross_confirmation_bars"] = float(post_cross_confirmation_bars)
        if optimize_chandelier:
            profile["chandelier_period"] = resolved_chandelier_period
            profile["chandelier_multiplier"] = resolved_chandelier_multiplier
        profiles.append(profile)
    return profiles


def _build_ema_cross_volume_profiles(
    options: dict[str, tuple[float | int, ...]],
) -> list[OptimizationProfile]:
    profiles: list[OptimizationProfile] = []
    min_ema_gap_pct_values = options.get("min_ema_gap_pct", (0.0,))
    cross_confirmation_bars_values = options.get("cross_confirmation_bars", (0,))
    max_price_extension_pct_values = options.get("max_price_extension_pct", (0.0,))
    for (
        ema_fast_period,
        ema_slow_period,
        volume_multiplier,
        min_ema_gap_pct,
        cross_confirmation_bars,
        max_price_extension_pct,
        stop_loss_pct,
        take_profit_pct,
        trailing_activation_pct,
        trailing_distance_pct,
        breakeven_activation_pct,
        breakeven_buffer_pct,
        tight_trailing_activation_pct,
        tight_trailing_distance_pct,
    ) in product(
        options["ema_fast_period"],
        options["ema_slow_period"],
        options["volume_multiplier"],
        min_ema_gap_pct_values,
        cross_confirmation_bars_values,
        max_price_extension_pct_values,
        options["stop_loss_pct"],
        options["take_profit_pct"],
        options["trailing_activation_pct"],
        options["trailing_distance_pct"],
        options["breakeven_activation_pct"],
        options["breakeven_buffer_pct"],
        options["tight_trailing_activation_pct"],
        options["tight_trailing_distance_pct"],
    ):
        if int(ema_fast_period) >= int(ema_slow_period):
            continue
        resolved_take_profit_pct = float(take_profit_pct)
        resolved_stop_loss_pct = float(stop_loss_pct)
        resolved_trailing_activation_pct = float(trailing_activation_pct)
        resolved_trailing_distance_pct = float(trailing_distance_pct)
        resolved_breakeven_activation_pct = float(breakeven_activation_pct)
        resolved_tight_trailing_activation_pct = float(tight_trailing_activation_pct)
        if not _passes_common_risk_profile_guards(
            take_profit_pct=resolved_take_profit_pct,
            stop_loss_pct=resolved_stop_loss_pct,
            trailing_activation_pct=resolved_trailing_activation_pct,
            trailing_distance_pct=resolved_trailing_distance_pct,
            breakeven_activation_pct=resolved_breakeven_activation_pct,
        ):
            continue
        profile: OptimizationProfile = {
            "ema_fast_period": float(ema_fast_period),
            "ema_slow_period": float(ema_slow_period),
            "volume_multiplier": float(volume_multiplier),
            "stop_loss_pct": resolved_stop_loss_pct,
            "take_profit_pct": resolved_take_profit_pct,
            "trailing_activation_pct": resolved_trailing_activation_pct,
            "trailing_distance_pct": resolved_trailing_distance_pct,
            "breakeven_activation_pct": resolved_breakeven_activation_pct,
            "breakeven_buffer_pct": float(breakeven_buffer_pct),
            "tight_trailing_activation_pct": resolved_tight_trailing_activation_pct,
            "tight_trailing_distance_pct": float(tight_trailing_distance_pct),
        }
        if "min_ema_gap_pct" in options:
            profile["min_ema_gap_pct"] = float(min_ema_gap_pct)
        if "cross_confirmation_bars" in options:
            profile["cross_confirmation_bars"] = float(cross_confirmation_bars)
        if "max_price_extension_pct" in options:
            profile["max_price_extension_pct"] = float(max_price_extension_pct)
        profiles.append(profile)
    return profiles


def _build_ema_band_rejection_profiles(
    options: dict[str, tuple[float | int, ...]],
) -> list[OptimizationProfile]:
    profiles: list[OptimizationProfile] = []
    default_rsi_length = int(options["rsi_length"][0])
    default_rsi_midline = float(options["rsi_midline"][0])
    default_use_rsi_cross_filter = int(options["use_rsi_cross_filter"][0])
    default_rsi_midline_margin = float(options["rsi_midline_margin"][0])
    default_volume_ma_length = int(options["volume_ma_length"][0])
    default_volume_multiplier = float(options["volume_multiplier"][0])
    default_atr_length = int(options["atr_length"][0])
    default_atr_stop_buffer_mult = float(options["atr_stop_buffer_mult"][-1])
    default_rejection_wick_min_ratio = float(options["rejection_wick_min_ratio"][0])
    default_rejection_body_min_ratio = float(options["rejection_body_min_ratio"][0])
    trend_persistence_bars_values = options.get("trend_persistence_bars", (1,))
    max_pullback_bars_values = options.get("max_pullback_bars", (0,))
    entry_offset_pct_values = options.get("entry_offset_pct", (0.0,))

    risk_profiles: list[tuple[float, float, float, float, float, float]] = []
    for (
        stop_loss_pct,
        take_profit_pct,
        trailing_activation_pct,
        trailing_distance_pct,
        breakeven_activation_pct,
        breakeven_buffer_pct,
    ) in product(
        options["stop_loss_pct"],
        options["take_profit_pct"],
        options["trailing_activation_pct"],
        options["trailing_distance_pct"],
        options["breakeven_activation_pct"],
        options["breakeven_buffer_pct"],
    ):
        resolved_stop_loss_pct = float(stop_loss_pct)
        resolved_take_profit_pct = float(take_profit_pct)
        resolved_trailing_activation_pct = float(trailing_activation_pct)
        resolved_trailing_distance_pct = float(trailing_distance_pct)
        resolved_breakeven_activation_pct = float(breakeven_activation_pct)
        resolved_breakeven_buffer_pct = float(breakeven_buffer_pct)
        if not _passes_common_risk_profile_guards(
            take_profit_pct=resolved_take_profit_pct,
            stop_loss_pct=resolved_stop_loss_pct,
            trailing_activation_pct=resolved_trailing_activation_pct,
            trailing_distance_pct=resolved_trailing_distance_pct,
            breakeven_activation_pct=resolved_breakeven_activation_pct,
        ):
            continue
        if not _passes_ema_band_rejection_exit_profile_guards(
            stop_loss_pct=resolved_stop_loss_pct,
            take_profit_pct=resolved_take_profit_pct,
            trailing_activation_pct=resolved_trailing_activation_pct,
            breakeven_activation_pct=resolved_breakeven_activation_pct,
        ):
            continue
        risk_profiles.append(
            (
                resolved_stop_loss_pct,
                resolved_take_profit_pct,
                resolved_trailing_activation_pct,
                resolved_trailing_distance_pct,
                resolved_breakeven_activation_pct,
                resolved_breakeven_buffer_pct,
            )
        )

    for ema_fast, ema_mid, ema_slow in product(
        options["ema_fast"],
        options["ema_mid"],
        options["ema_slow"],
    ):
        resolved_ema_fast = int(ema_fast)
        resolved_ema_mid = int(ema_mid)
        resolved_ema_slow = int(ema_slow)
        if not (resolved_ema_fast < resolved_ema_mid < resolved_ema_slow):
            continue
        for slope_lookback, min_ema_spread_pct in product(
            options["slope_lookback"],
            options["min_ema_spread_pct"],
        ):
            resolved_slope_lookback = float(slope_lookback)
            resolved_min_ema_spread_pct = float(min_ema_spread_pct)
            for min_slow_slope_pct in options["min_slow_slope_pct"]:
                resolved_min_slow_slope_pct = float(min_slow_slope_pct)
                for pullback_requires_outer_band_touch in options["pullback_requires_outer_band_touch"]:
                    resolved_pullback_requires_outer_band_touch = (
                        1 if int(pullback_requires_outer_band_touch) > 0 else 0
                    )
                    for use_rejection_quality_filter in options["use_rejection_quality_filter"]:
                        resolved_use_rejection_quality_filter = (
                            1 if int(use_rejection_quality_filter) > 0 else 0
                        )
                        rejection_wick_min_ratio_values = (
                            options["rejection_wick_min_ratio"]
                            if resolved_use_rejection_quality_filter > 0
                            else (default_rejection_wick_min_ratio,)
                        )
                        rejection_body_min_ratio_values = (
                            options["rejection_body_min_ratio"]
                            if resolved_use_rejection_quality_filter > 0
                            else (default_rejection_body_min_ratio,)
                        )
                        for (
                            rejection_wick_min_ratio,
                            rejection_body_min_ratio,
                        ) in product(
                            rejection_wick_min_ratio_values,
                            rejection_body_min_ratio_values,
                        ):
                            resolved_rejection_wick_min_ratio = float(rejection_wick_min_ratio)
                            resolved_rejection_body_min_ratio = float(rejection_body_min_ratio)
                            for use_rsi_filter in options["use_rsi_filter"]:
                                resolved_use_rsi_filter = 1 if int(use_rsi_filter) > 0 else 0
                                rsi_length_values = (
                                    options["rsi_length"]
                                    if resolved_use_rsi_filter > 0
                                    else (default_rsi_length,)
                                )
                                rsi_midline_values = (
                                    options["rsi_midline"]
                                    if resolved_use_rsi_filter > 0
                                    else (default_rsi_midline,)
                                )
                                rsi_cross_filter_values = (
                                    options["use_rsi_cross_filter"]
                                    if resolved_use_rsi_filter > 0
                                    else (default_use_rsi_cross_filter,)
                                )
                                rsi_midline_margin_values = (
                                    options["rsi_midline_margin"]
                                    if resolved_use_rsi_filter > 0
                                    else (default_rsi_midline_margin,)
                                )
                                for (
                                    rsi_length,
                                    rsi_midline,
                                    use_rsi_cross_filter,
                                    rsi_midline_margin,
                                ) in product(
                                    rsi_length_values,
                                    rsi_midline_values,
                                    rsi_cross_filter_values,
                                    rsi_midline_margin_values,
                                ):
                                    resolved_rsi_length = float(rsi_length)
                                    resolved_rsi_midline = float(rsi_midline)
                                    resolved_use_rsi_cross_filter = (
                                        1 if int(use_rsi_cross_filter) > 0 else 0
                                    )
                                    resolved_rsi_midline_margin = float(rsi_midline_margin)
                                    for use_volume_filter in options["use_volume_filter"]:
                                        resolved_use_volume_filter = 1 if int(use_volume_filter) > 0 else 0
                                        volume_ma_length_values = (
                                            options["volume_ma_length"]
                                            if resolved_use_volume_filter > 0
                                            else (default_volume_ma_length,)
                                        )
                                        volume_multiplier_values = (
                                            options["volume_multiplier"]
                                            if resolved_use_volume_filter > 0
                                            else (default_volume_multiplier,)
                                        )
                                        for volume_ma_length, volume_multiplier in product(
                                            volume_ma_length_values,
                                            volume_multiplier_values,
                                        ):
                                            resolved_volume_ma_length = float(volume_ma_length)
                                            resolved_volume_multiplier = float(volume_multiplier)
                                            for use_atr_stop_buffer in options["use_atr_stop_buffer"]:
                                                resolved_use_atr_stop_buffer = 1 if int(use_atr_stop_buffer) > 0 else 0
                                                atr_length_values = (
                                                    options["atr_length"]
                                                    if resolved_use_atr_stop_buffer > 0
                                                    else (default_atr_length,)
                                                )
                                                atr_stop_buffer_mult_values = (
                                                    options["atr_stop_buffer_mult"]
                                                    if resolved_use_atr_stop_buffer > 0
                                                    else (default_atr_stop_buffer_mult,)
                                                )
                                                for atr_length, atr_stop_buffer_mult in product(
                                                    atr_length_values,
                                                    atr_stop_buffer_mult_values,
                                                ):
                                                    resolved_atr_length = float(atr_length)
                                                    resolved_atr_stop_buffer_mult = float(atr_stop_buffer_mult)
                                                    for signal_cooldown_bars in options["signal_cooldown_bars"]:
                                                        resolved_signal_cooldown_bars = float(signal_cooldown_bars)
                                                        for trend_persistence_bars, max_pullback_bars, entry_offset_pct in product(
                                                            trend_persistence_bars_values,
                                                            max_pullback_bars_values,
                                                            entry_offset_pct_values,
                                                        ):
                                                            resolved_trend_persistence_bars = float(trend_persistence_bars)
                                                            resolved_max_pullback_bars = float(max_pullback_bars)
                                                            resolved_entry_offset_pct = float(entry_offset_pct)
                                                            for (
                                                                resolved_stop_loss_pct,
                                                                resolved_take_profit_pct,
                                                                resolved_trailing_activation_pct,
                                                                resolved_trailing_distance_pct,
                                                                resolved_breakeven_activation_pct,
                                                                resolved_breakeven_buffer_pct,
                                                            ) in risk_profiles:
                                                                profile: OptimizationProfile = {
                                                                    "ema_fast": float(resolved_ema_fast),
                                                                    "ema_mid": float(resolved_ema_mid),
                                                                    "ema_slow": float(resolved_ema_slow),
                                                                    "slope_lookback": resolved_slope_lookback,
                                                                    "min_ema_spread_pct": resolved_min_ema_spread_pct,
                                                                    "min_slow_slope_pct": resolved_min_slow_slope_pct,
                                                                    "pullback_requires_outer_band_touch": float(
                                                                        resolved_pullback_requires_outer_band_touch
                                                                    ),
                                                                    "use_rejection_quality_filter": float(
                                                                        resolved_use_rejection_quality_filter
                                                                    ),
                                                                    "rejection_wick_min_ratio": resolved_rejection_wick_min_ratio,
                                                                    "rejection_body_min_ratio": resolved_rejection_body_min_ratio,
                                                                    "use_rsi_filter": float(resolved_use_rsi_filter),
                                                                    "rsi_length": resolved_rsi_length,
                                                                    "rsi_midline": resolved_rsi_midline,
                                                                    "use_rsi_cross_filter": float(
                                                                        resolved_use_rsi_cross_filter
                                                                    ),
                                                                    "rsi_midline_margin": resolved_rsi_midline_margin,
                                                                    "use_volume_filter": float(resolved_use_volume_filter),
                                                                    "volume_ma_length": resolved_volume_ma_length,
                                                                    "volume_multiplier": resolved_volume_multiplier,
                                                                    "use_atr_stop_buffer": float(resolved_use_atr_stop_buffer),
                                                                    "atr_length": resolved_atr_length,
                                                                    "atr_stop_buffer_mult": resolved_atr_stop_buffer_mult,
                                                                    "signal_cooldown_bars": resolved_signal_cooldown_bars,
                                                                    "stop_loss_pct": resolved_stop_loss_pct,
                                                                    "take_profit_pct": resolved_take_profit_pct,
                                                                    "trailing_activation_pct": resolved_trailing_activation_pct,
                                                                    "trailing_distance_pct": resolved_trailing_distance_pct,
                                                                    "breakeven_activation_pct": resolved_breakeven_activation_pct,
                                                                    "breakeven_buffer_pct": resolved_breakeven_buffer_pct,
                                                                }
                                                                if "trend_persistence_bars" in options:
                                                                    profile["trend_persistence_bars"] = resolved_trend_persistence_bars
                                                                if "max_pullback_bars" in options:
                                                                    profile["max_pullback_bars"] = resolved_max_pullback_bars
                                                                if "entry_offset_pct" in options:
                                                                    profile["entry_offset_pct"] = resolved_entry_offset_pct
                                                                profiles.append(profile)
    return profiles


def _build_dual_thrust_profiles(
    options: dict[str, tuple[float | int, ...]],
) -> list[OptimizationProfile]:
    profiles: list[OptimizationProfile] = []
    optimize_chandelier = (
        "chandelier_period" in options or "chandelier_multiplier" in options
    )
    chandelier_period_values = options.get(
        "chandelier_period",
        (int(settings.trading.chandelier_period),),
    )
    chandelier_multiplier_values = options.get(
        "chandelier_multiplier",
        (float(settings.trading.chandelier_multiplier),),
    )
    breakout_buffer_pct_values = options.get("breakout_buffer_pct", (0.0,))
    min_range_pct_values = options.get("min_range_pct", (0.0,))
    cooldown_bars_after_exit_values = options.get("cooldown_bars_after_exit", (0,))
    for (
        dual_thrust_k1,
        dual_thrust_k2,
        dual_thrust_period,
        chandelier_period,
        chandelier_multiplier,
        breakout_buffer_pct,
        min_range_pct,
        cooldown_bars_after_exit,
        stop_loss_pct,
        take_profit_pct,
        trailing_activation_pct,
        trailing_distance_pct,
        breakeven_activation_pct,
        breakeven_buffer_pct,
        tight_trailing_activation_pct,
        tight_trailing_distance_pct,
    ) in product(
        options["dual_thrust_k1"],
        options["dual_thrust_k2"],
        options["dual_thrust_period"],
        chandelier_period_values,
        chandelier_multiplier_values,
        breakout_buffer_pct_values,
        min_range_pct_values,
        cooldown_bars_after_exit_values,
        options["stop_loss_pct"],
        options["take_profit_pct"],
        options["trailing_activation_pct"],
        options["trailing_distance_pct"],
        options["breakeven_activation_pct"],
        options["breakeven_buffer_pct"],
        options["tight_trailing_activation_pct"],
        options["tight_trailing_distance_pct"],
    ):
        resolved_chandelier_period = float(chandelier_period)
        resolved_chandelier_multiplier = float(chandelier_multiplier)
        if optimize_chandelier and not _passes_chandelier_profile_guards(
            chandelier_period=resolved_chandelier_period,
            chandelier_multiplier=resolved_chandelier_multiplier,
        ):
            continue
        resolved_take_profit_pct = float(take_profit_pct)
        resolved_stop_loss_pct = float(stop_loss_pct)
        resolved_trailing_activation_pct = float(trailing_activation_pct)
        resolved_trailing_distance_pct = float(trailing_distance_pct)
        resolved_breakeven_activation_pct = float(breakeven_activation_pct)
        resolved_tight_trailing_activation_pct = float(tight_trailing_activation_pct)
        if not _passes_common_risk_profile_guards(
            take_profit_pct=resolved_take_profit_pct,
            stop_loss_pct=resolved_stop_loss_pct,
            trailing_activation_pct=resolved_trailing_activation_pct,
            trailing_distance_pct=resolved_trailing_distance_pct,
            breakeven_activation_pct=resolved_breakeven_activation_pct,
        ):
            continue
        profile: OptimizationProfile = {
            "dual_thrust_k1": float(dual_thrust_k1),
            "dual_thrust_k2": float(dual_thrust_k2),
            "dual_thrust_period": float(dual_thrust_period),
            "stop_loss_pct": resolved_stop_loss_pct,
            "take_profit_pct": resolved_take_profit_pct,
            "trailing_activation_pct": resolved_trailing_activation_pct,
            "trailing_distance_pct": resolved_trailing_distance_pct,
            "breakeven_activation_pct": resolved_breakeven_activation_pct,
            "breakeven_buffer_pct": float(breakeven_buffer_pct),
            "tight_trailing_activation_pct": resolved_tight_trailing_activation_pct,
            "tight_trailing_distance_pct": float(tight_trailing_distance_pct),
        }
        if "breakout_buffer_pct" in options:
            profile["breakout_buffer_pct"] = float(breakout_buffer_pct)
        if "min_range_pct" in options:
            profile["min_range_pct"] = float(min_range_pct)
        if "cooldown_bars_after_exit" in options:
            profile["cooldown_bars_after_exit"] = float(cooldown_bars_after_exit)
        if optimize_chandelier:
            profile["chandelier_period"] = resolved_chandelier_period
            profile["chandelier_multiplier"] = resolved_chandelier_multiplier
        profiles.append(profile)
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
    interval: str | None = None,
) -> list[OptimizationProfile]:
    resolved_strategy = _coerce_backtest_strategy_name(
        settings.strategy.default_strategy_name
        if strategy_name is None
        else str(strategy_name),
        fallback_strategy_name="frama_cross",
    )
    if resolved_strategy == "ema_cross_volume":
        return generate_ema_optimization_grid(interval=interval)
    if resolved_strategy == "ema_band_rejection":
        return generate_ema_band_rejection_optimization_grid(interval=interval)
    if resolved_strategy == "frama_cross":
        return generate_frama_optimization_grid(interval=interval)
    if resolved_strategy == "dual_thrust":
        return generate_dual_thrust_optimization_grid(symbol=symbol, interval=interval)
    return generate_trade_management_optimization_grid()


def _validate_strategy_name(strategy_name: str) -> str:
    strategy_name = STRATEGY_NAME_ALIASES.get(strategy_name, strategy_name)
    allowed_strategy_names = set(settings.strategy.available_strategies) | set(
        BACKTEST_ONLY_AVAILABLE_STRATEGIES
    )
    if strategy_name not in allowed_strategy_names:
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
    if apply_backtest_optimizer_overrides:
        override_strategy_name = BACKTEST_OPTIMIZER_COIN_STRATEGIES.get(normalized_symbol)
        if override_strategy_name is not None:
            return _coerce_backtest_strategy_name(
                str(override_strategy_name),
                fallback_strategy_name=default_strategy_name,
            )
    fallback_strategy = (
        settings.strategy.default_strategy_name
        if default_strategy_name is None
        else default_strategy_name
    )
    try:
        resolved_strategy = resolve_strategy_for_symbol(
            normalized_symbol,
            fallback_strategy,
            use_coin_override=True,
        )
    except Exception:
        resolved_strategy = str(fallback_strategy)
    return _sanitize_backtest_strategy_name(
        resolved_strategy,
        fallback_strategy_name=default_strategy_name,
    )


def resolve_ema_band_rejection_1h_winner_profile(
    symbol: str,
) -> tuple[str, OptimizationProfile] | None:
    normalized_symbol = str(symbol).strip().upper()
    preset_entry = EMA_BAND_REJECTION_1H_WINNERS.get(normalized_symbol)
    if not isinstance(preset_entry, Mapping):
        return None

    preset_strategy_name = _sanitize_backtest_strategy_name(
        str(preset_entry.get("strategy", "") or "ema_band_rejection"),
        fallback_strategy_name="ema_band_rejection",
    )
    if preset_strategy_name != "ema_band_rejection":
        return None

    preset_interval = _validate_interval_name(
        str(preset_entry.get("interval", "1h") or "1h").strip()
    )
    flattened_profile: OptimizationProfile = {}
    for section_name in ("params", "risk"):
        section_payload = preset_entry.get(section_name)
        if not isinstance(section_payload, Mapping):
            continue
        for field_name, field_value in section_payload.items():
            with suppress(Exception):
                flattened_profile[str(field_name)] = float(field_value)

    return preset_interval, flattened_profile


def resolve_live_candidate_policy(symbol: str) -> dict[str, float | int | str] | None:
    normalized_symbol = str(symbol).strip().upper()
    if normalized_symbol != LIVE_EMA_BAND_REJECTION_CANDIDATE_SYMBOL:
        return None
    production_profile = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol)
    if not isinstance(production_profile, Mapping):
        return None
    strategy_name = _sanitize_backtest_strategy_name(
        str(production_profile.get("strategy_name", "") or ""),
        fallback_strategy_name="",
    )
    interval_text = str(production_profile.get("interval", "") or "").strip()
    if strategy_name != str(LIVE_EMA_BAND_REJECTION_CANDIDATE_POLICY["strategy_name"]):
        return None
    if interval_text != str(LIVE_EMA_BAND_REJECTION_CANDIDATE_POLICY["interval"]):
        return None
    return dict(LIVE_EMA_BAND_REJECTION_CANDIDATE_POLICY)


def _sanitize_backtest_strategy_name(
    strategy_name: str,
    *,
    fallback_strategy_name: str | None = None,
) -> str:
    if strategy_name in BACKTEST_EXECUTION_STRATEGIES:
        return strategy_name
    fallback_name = (
        settings.strategy.default_strategy_name
        if fallback_strategy_name is None
        else fallback_strategy_name
    )
    if fallback_name in BACKTEST_EXECUTION_STRATEGIES:
        return str(fallback_name)
    return "frama_cross"


def _coerce_backtest_strategy_name(
    strategy_name: str,
    *,
    fallback_strategy_name: str | None = None,
) -> str:
    try:
        validated_strategy_name = _validate_strategy_name(str(strategy_name))
    except Exception:
        validated_strategy_name = str(strategy_name)
    return _sanitize_backtest_strategy_name(
        validated_strategy_name,
        fallback_strategy_name=fallback_strategy_name,
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


def _normalize_optional_utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _load_backtest_candles_in_time_range(
    db: Database,
    *,
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime | None = None,
) -> list[CandleRecord]:
    candles = db.fetch_candles_since(
        symbol,
        interval,
        start_time=start_time,
        end_time=end_time,
    )
    if len(candles) <= MAX_BACKTEST_CANDLES:
        return candles
    return candles[-MAX_BACKTEST_CANDLES:]


def _required_candle_count_for_strategy(strategy_name: str, *, use_setup_gate: bool = False) -> int:
    strategy_name = _validate_strategy_name(strategy_name)
    if strategy_name == "ema_cross_volume":
        required_count = max(
            settings.strategy.ema_slow_period + 1,
            settings.strategy.volume_sma_period,
        )
    elif strategy_name == "ema_band_rejection":
        required_count = 64
    elif strategy_name == "frama_cross":
        required_count = max(settings.strategy.frama_slow_period + 1, 6)
    elif strategy_name == "dual_thrust":
        required_count = max(settings.strategy.dual_thrust_period + 2, 6)
    else:
        raise ValueError(f"Unsupported strategy_name for candle count: {strategy_name}")
    required_count = max(required_count, int(settings.trading.chandelier_period) + 2)
    if use_setup_gate:
        required_count = max(
            required_count,
            SmartSetupGate.required_candle_count(),
            SETUP_GATE_SETTLED_WARMUP_CANDLES,
        )
    return required_count


def _required_candle_count_for_profile(
    strategy_name: str,
    strategy_profile: OptimizationProfile,
    *,
    use_setup_gate: bool = False,
) -> int:
    strategy_name = _validate_strategy_name(strategy_name)
    profile_chandelier_period = int(
        float(
            strategy_profile.get(
                "chandelier_period",
                settings.trading.chandelier_period,
            )
            or settings.trading.chandelier_period
        )
    )
    if strategy_name == "ema_cross_volume":
        ema_slow = int(float(strategy_profile.get("ema_slow_period", settings.strategy.ema_slow_period) or settings.strategy.ema_slow_period))
        cross_confirmation_bars = int(
            float(strategy_profile.get("cross_confirmation_bars", 0.0) or 0.0)
        )
        required_count = max(
            ema_slow + cross_confirmation_bars + 2,
            int(settings.strategy.volume_sma_period) + 2,
            profile_chandelier_period + 2,
        )
        if use_setup_gate:
            required_count = max(
                required_count,
                SmartSetupGate.required_candle_count(),
                SETUP_GATE_SETTLED_WARMUP_CANDLES,
            )
        return int(required_count)
    if strategy_name == "ema_band_rejection":
        ema_slow = int(float(strategy_profile.get("ema_slow", 20.0) or 20.0))
        slope_lookback = int(float(strategy_profile.get("slope_lookback", 5.0) or 5.0))
        rsi_length = int(float(strategy_profile.get("rsi_length", 14.0) or 14.0))
        volume_ma_length = int(float(strategy_profile.get("volume_ma_length", 20.0) or 20.0))
        atr_length = int(float(strategy_profile.get("atr_length", 14.0) or 14.0))
        trend_persistence_bars = int(
            float(strategy_profile.get("trend_persistence_bars", 1.0) or 1.0)
        )
        max_pullback_bars = int(float(strategy_profile.get("max_pullback_bars", 0.0) or 0.0))
        required_count = max(
            ema_slow + slope_lookback + trend_persistence_bars + max_pullback_bars + 2,
            rsi_length + 2,
            volume_ma_length + 2,
            atr_length + 2,
            64,
            profile_chandelier_period + 2,
        )
        if use_setup_gate:
            required_count = max(
                required_count,
                SmartSetupGate.required_candle_count(),
                SETUP_GATE_SETTLED_WARMUP_CANDLES,
            )
        return int(required_count)
    if strategy_name == "frama_cross":
        frama_slow = int(
            float(strategy_profile.get("frama_slow_period", settings.strategy.frama_slow_period) or settings.strategy.frama_slow_period)
        )
        post_cross_confirmation_bars = int(
            float(strategy_profile.get("post_cross_confirmation_bars", 0.0) or 0.0)
        )
        required_count = max(
            frama_slow + post_cross_confirmation_bars + 2,
            profile_chandelier_period + 2,
            6,
        )
        if bool(float(strategy_profile.get("use_late_entry_guard", 0.0) or 0.0) >= 0.5):
            required_count = max(required_count, 20)
        if use_setup_gate:
            required_count = max(
                required_count,
                SmartSetupGate.required_candle_count(),
                SETUP_GATE_SETTLED_WARMUP_CANDLES,
            )
        return int(required_count)
    if strategy_name == "dual_thrust":
        dual_thrust_period = int(
            float(strategy_profile.get("dual_thrust_period", settings.strategy.dual_thrust_period) or settings.strategy.dual_thrust_period)
        )
        cooldown_bars_after_exit = int(
            float(strategy_profile.get("cooldown_bars_after_exit", 0.0) or 0.0)
        )
        required_count = max(
            dual_thrust_period + cooldown_bars_after_exit + 2,
            profile_chandelier_period + 2,
            6,
        )
        if bool(float(strategy_profile.get("use_late_entry_guard", 0.0) or 0.0) >= 0.5):
            required_count = max(required_count, 20)
        if use_setup_gate:
            required_count = max(
                required_count,
                SmartSetupGate.required_candle_count(),
                SETUP_GATE_SETTLED_WARMUP_CANDLES,
            )
        return int(required_count)
    with _temporary_strategy_profile(strategy_profile):
        return _required_candle_count_for_strategy(
            strategy_name,
            use_setup_gate=use_setup_gate,
        )

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
        "frama_move_last_1_bar_pct",
        "frama_move_last_2_bars_pct",
        "frama_move_last_3_bars_pct",
        "frama_distance_to_reference_pct",
        "frama_atr_extension_mult",
        "frama_late_entry_guard_enabled",
        "frama_late_entry_guard_blocked",
        "frama_late_entry_guard_block_reason",
        "frama_late_entry_guard_blocked_signals",
        "dual_range",
        "dual_buy_line",
        "dual_sell_line",
        "dual_long_entry",
        "dual_short_entry",
        "dual_long_exit",
        "dual_short_exit",
        "dual_move_last_1_bar_pct",
        "dual_move_last_2_bars_pct",
        "dual_move_last_3_bars_pct",
        "dual_distance_to_reference_pct",
        "dual_atr_extension_mult",
        "dual_breakout_candle_body_pct",
        "dual_breakout_candle_range_atr_mult",
        "dual_late_entry_guard_enabled",
        "dual_late_entry_guard_blocked",
        "dual_late_entry_guard_block_reason",
        "dual_late_entry_guard_blocked_signals",
        "ema_band_ema_fast",
        "ema_band_ema_mid",
        "ema_band_ema_slow",
        "ema_band_ema_slow_slope_pct",
        "ema_band_ema_spread_pct",
        "ema_band_zone_low",
        "ema_band_zone_high",
        "ema_band_rsi",
        "ema_band_volume_ma",
        "ema_band_atr",
        "ema_band_trend_long",
        "ema_band_trend_short",
        "ema_band_pullback_long",
        "ema_band_pullback_short",
        "ema_band_rejection_long",
        "ema_band_rejection_short",
        "ema_band_setup_long",
        "ema_band_setup_short",
        "ema_band_long_entry",
        "ema_band_short_entry",
        "ema_band_long_exit",
        "ema_band_short_exit",
        "ema_band_dynamic_stop_loss_pct",
        "ema_band_signal_direction",
        "ema_band_signal_cooldown_bars",
        "ema_band_cooldown_blocked_signals",
        "ema_band_move_last_1_bar_pct",
        "ema_band_move_last_2_bars_pct",
        "ema_band_move_last_3_bars_pct",
        "ema_band_distance_fast_ref_pct",
        "ema_band_distance_mid_ref_pct",
        "ema_band_distance_to_reference_pct",
        "ema_band_atr_extension_mult",
        "ema_band_late_entry_guard_enabled",
        "ema_band_late_entry_guard_blocked",
        "ema_band_late_entry_guard_block_reason",
        "ema_band_late_entry_guard_blocked_signals",
        "ema_band_pullback_reentry_enabled",
    )
    for column_name in indicator_columns:
        if column_name in candles_dataframe.columns:
            # Use `del` to release indicator arrays immediately without creating drop-copies.
            del candles_dataframe[column_name]
    with suppress(Exception):
        candles_dataframe._item_cache.clear()  # pandas cache, best effort.


def _initialize_optimizer_worker(
    candles_payload: Mapping[str, object],
    regime_mask: list[int] | None = None,
    strategy_signal_cache: dict[object, dict[str, object]] | None = None,
) -> None:
    global _WORKER_CANDLES_DF, _WORKER_CANDLE_ROWS, _WORKER_REGIME_MASK, _WORKER_STRATEGY_SIGNAL_CACHE, _WORKER_OHLCV_ARRAYS
    _WORKER_CANDLES_DF = _worker_payload_to_pandas_dataframe(candles_payload)
    _WORKER_CANDLE_ROWS = PaperTradingEngine._extract_backtest_rows(_WORKER_CANDLES_DF)
    _WORKER_OHLCV_ARRAYS = _extract_ohlcv_numpy_arrays_from_payload(candles_payload)
    _WORKER_REGIME_MASK = None if regime_mask is None else list(regime_mask)
    _WORKER_STRATEGY_SIGNAL_CACHE = (
        {}
        if strategy_signal_cache is None
        else dict(strategy_signal_cache)
    )


def _worker_candles_dataframe(*, copy_deep: bool = True) -> pd.DataFrame:
    if _WORKER_CANDLES_DF is None:
        raise RuntimeError("Optimizer worker candles were not initialized.")
    return _WORKER_CANDLES_DF.copy(deep=copy_deep)


def _worker_candle_rows() -> list[dict[str, object]]:
    if _WORKER_CANDLE_ROWS is None:
        raise RuntimeError("Optimizer worker candle rows were not initialized.")
    return list(_WORKER_CANDLE_ROWS)


def _worker_ohlcv_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if _WORKER_OHLCV_ARRAYS is None:
        return None
    return _WORKER_OHLCV_ARRAYS


def _worker_regime_mask() -> list[int] | None:
    if _WORKER_REGIME_MASK is None:
        return None
    return list(_WORKER_REGIME_MASK)


def _worker_strategy_signal_group(cache_key: object) -> dict[str, object] | None:
    if _WORKER_STRATEGY_SIGNAL_CACHE is None:
        return None
    cached_payload = _WORKER_STRATEGY_SIGNAL_CACHE.get(cache_key)
    if cached_payload is None:
        return None
    return dict(cached_payload)


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


def _memory_safe_worker_count(
    total_profiles: int,
    *,
    worker_cap_override: int | None = None,
) -> int:
    if total_profiles <= 0:
        return 1
    resolved_worker_cap = (
        int(MAX_OPTIMIZATION_WORKERS)
        if worker_cap_override is None
        else int(worker_cap_override)
    )
    requested = max(1, int(resolved_worker_cap))
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
    candles_payload: Mapping[str, object],
    *,
    regime_mask: Sequence[int] | None = None,
    strategy_signal_cache: dict[object, dict[str, object]] | None = None,
) -> Pool:
    return get_context("spawn").Pool(
        processes=worker_count,
        initializer=_initialize_optimizer_worker,
        initargs=(
            dict(candles_payload),
            None if regime_mask is None else list(regime_mask),
            None if strategy_signal_cache is None else dict(strategy_signal_cache),
        ),
        maxtasksperchild=OPTIMIZER_WORKER_MAX_TASKS_PER_CHILD,
    )


def _estimate_strategy_signal_cache_bytes(
    *,
    strategy_name: str,
    candle_count: int,
    variant_count: int,
) -> int:
    if candle_count <= 0 or variant_count <= 0:
        return 0
    # Approximate packed payload footprint per (variant, candle).
    # frama_cross / dual_thrust / ema_cross_volume: signals only (int8).
    # ema_band_rejection: signal + long_exit + short_exit + dynamic_stop(float32).
    series_multiplier = 1
    if strategy_name == "ema_band_rejection":
        series_multiplier = 7
    estimated = int(candle_count) * int(variant_count) * int(series_multiplier)
    return max(0, estimated)


def _finalize_signal_series(
    candles_df: pd.DataFrame,
    raw_signals: Sequence[int],
    *,
    required_candles: int,
    setup_gate: SmartSetupGate | None,
    strategy_name: str,
    regime_mask: Sequence[int] | None = None,
) -> tuple[list[int], int, int, int]:
    signal_array = np.asarray(raw_signals, dtype=np.int8).copy()
    signal_count = int(signal_array.size)
    if signal_count == 0:
        return [], 0, 0, 0

    warmup_cutoff = max(int(required_candles) - 1, 0)
    if warmup_cutoff > 0:
        signal_array[:warmup_cutoff] = 0

    if regime_mask is not None:
        regime_array = np.asarray(regime_mask, dtype=np.int8)
        if regime_array.size > 0:
            valid_count = min(signal_count, int(regime_array.size))
            blocked_mask = regime_array[:valid_count] <= 0
            if blocked_mask.any():
                active_slice = signal_array[:valid_count]
                active_slice[blocked_mask] = 0

    candidate_indices = np.flatnonzero(signal_array)
    total_signals = int(candidate_indices.size)
    if total_signals == 0:
        return signal_array.tolist(), 0, 0, 0

    if setup_gate is None:
        return signal_array.tolist(), total_signals, total_signals, 0

    approved_signals = 0
    blocked_signals = 0
    approved_array = np.zeros(signal_count, dtype=np.int8)
    for index in candidate_indices.tolist():
        signal_direction = int(signal_array[index])
        normalized_direction = 1 if signal_direction > 0 else -1
        is_approved, _score, _reason = setup_gate.evaluate_signal_at_index(
            candles_df,
            int(index),
            normalized_direction,
            strategy_name,
        )
        if not is_approved:
            blocked_signals += 1
            continue
        approved_signals += 1
        approved_array[index] = np.int8(normalized_direction)
    return approved_array.tolist(), total_signals, approved_signals, blocked_signals


def _pack_int8_series(values: Sequence[int]) -> tuple[bytes, int]:
    packed_array = np.asarray(values, dtype=np.int8)
    return packed_array.tobytes(), int(packed_array.size)


def _unpack_int8_series(values_bytes: bytes, values_length: int) -> list[int]:
    if values_length <= 0:
        return []
    values_array = np.frombuffer(values_bytes, dtype=np.int8, count=values_length)
    return values_array.tolist()


def _align_signal_series_to_candle_count(
    signals: Sequence[int],
    *,
    candle_count: int,
) -> list[int]:
    normalized_signals = list(signals)
    if candle_count <= 0:
        return []
    if len(normalized_signals) == candle_count:
        return normalized_signals
    if len(normalized_signals) > candle_count:
        return normalized_signals[:candle_count]
    return normalized_signals + ([0] * (candle_count - len(normalized_signals)))


def _pack_bool_series(values: Sequence[bool]) -> tuple[bytes, int]:
    bool_array = np.asarray(values, dtype=np.bool_)
    packed_bits = np.packbits(bool_array, bitorder="little")
    return packed_bits.tobytes(), int(bool_array.size)


def _unpack_bool_series(values_bytes: bytes, values_length: int) -> list[bool]:
    if values_length <= 0:
        return []
    packed_bits = np.frombuffer(values_bytes, dtype=np.uint8)
    unpacked = np.unpackbits(
        packed_bits,
        bitorder="little",
        count=values_length,
    ).astype(np.bool_)
    return unpacked.tolist()


def _pack_float_series(values: Sequence[float]) -> tuple[bytes, int]:
    float_array = np.asarray(values, dtype=np.float32)
    return float_array.tobytes(), int(float_array.size)


def _unpack_float_series(values_bytes: bytes, values_length: int) -> list[float]:
    if values_length <= 0:
        return []
    values_array = np.frombuffer(values_bytes, dtype=np.float32, count=values_length)
    return values_array.astype(np.float64, copy=False).tolist()


def _normalize_dynamic_pct_series(values: object) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        return None
    try:
        raw_values = list(values)  # type: ignore[arg-type]
    except TypeError:
        return None
    normalized_values: list[float] = []
    for raw_value in raw_values:
        with suppress(Exception):
            normalized_values.append(float(raw_value))
            continue
        normalized_values.append(float("nan"))
    return normalized_values
def _pack_strategy_signal_payload(payload: dict[str, object]) -> dict[str, object]:
    packed_payload: dict[str, object] = {
        "total_signals": int(payload.get("total_signals", 0)),
        "approved_signals": int(payload.get("approved_signals", 0)),
        "blocked_signals": int(payload.get("blocked_signals", 0)),
    }
    signals = list(payload.get("signals", []))
    signals_bytes, signals_length = _pack_int8_series(signals)
    packed_payload["signals_bytes"] = signals_bytes
    packed_payload["signals_len"] = int(signals_length)

    long_exit_flags = payload.get("precomputed_long_exit_flags")
    if long_exit_flags is not None:
        long_exit_bytes, long_exit_length = _pack_bool_series(list(long_exit_flags))
        packed_payload["precomputed_long_exit_flags_bytes"] = long_exit_bytes
        packed_payload["precomputed_long_exit_flags_len"] = int(long_exit_length)

    short_exit_flags = payload.get("precomputed_short_exit_flags")
    if short_exit_flags is not None:
        short_exit_bytes, short_exit_length = _pack_bool_series(list(short_exit_flags))
        packed_payload["precomputed_short_exit_flags_bytes"] = short_exit_bytes
        packed_payload["precomputed_short_exit_flags_len"] = int(short_exit_length)

    dynamic_stop_loss_pcts = payload.get("precomputed_dynamic_stop_loss_pcts")
    if dynamic_stop_loss_pcts is not None:
        dynamic_stop_loss_bytes, dynamic_stop_loss_length = _pack_float_series(
            list(dynamic_stop_loss_pcts)
        )
        packed_payload["precomputed_dynamic_stop_loss_pcts_bytes"] = dynamic_stop_loss_bytes
        packed_payload["precomputed_dynamic_stop_loss_pcts_len"] = int(dynamic_stop_loss_length)

    dynamic_take_profit_pcts = payload.get("precomputed_dynamic_take_profit_pcts")
    if dynamic_take_profit_pcts is not None:
        dynamic_take_profit_bytes, dynamic_take_profit_length = _pack_float_series(
            list(dynamic_take_profit_pcts)
        )
        packed_payload["precomputed_dynamic_take_profit_pcts_bytes"] = dynamic_take_profit_bytes
        packed_payload["precomputed_dynamic_take_profit_pcts_len"] = int(dynamic_take_profit_length)
    return packed_payload


def _build_vectorized_strategy_cache_payload(
    candles_df: pd.DataFrame,
    *,
    strategy_name: str,
    required_candles: int,
    setup_gate: SmartSetupGate | None,
    strategy_profile: OptimizationProfile | None = None,
    regime_mask: Sequence[int] | None = None,
    raw_signal_override: Sequence[int] | None = None,
) -> dict[str, object] | None:
    effective_strategy_profile = strategy_profile
    profile_values = {} if strategy_profile is None else dict(strategy_profile)

    def _profile_int(field_name: str, default_value: int) -> int:
        with suppress(Exception):
            return int(float(profile_values.get(field_name, default_value) or default_value))
        return int(default_value)

    def _profile_float(field_name: str, default_value: float) -> float:
        with suppress(Exception):
            return float(profile_values.get(field_name, default_value) or default_value)
        return float(default_value)

    def _profile_flag(field_name: str, default_value: bool = False) -> bool:
        raw_value = profile_values.get(field_name, default_value)
        if isinstance(raw_value, bool):
            return bool(raw_value)
        with suppress(Exception):
            return bool(float(raw_value) >= 0.5)
        return bool(default_value)

    if strategy_name == "ema_cross_volume":
        with _temporary_strategy_profile(effective_strategy_profile, candles_df):
            working_df = build_ema_cross_volume_signal_frame(
                candles_df,
                min_ema_gap_pct=_profile_float("min_ema_gap_pct", 0.0),
                cross_confirmation_bars=_profile_int("cross_confirmation_bars", 0),
                max_price_extension_pct=_profile_float("max_price_extension_pct", 0.0),
            )
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

    if strategy_name == "ema_band_rejection":
        working_df = build_ema_band_rejection_signal_frame(
            candles_df,
            ema_fast=_profile_int("ema_fast", 5),
            ema_mid=_profile_int("ema_mid", 10),
            ema_slow=_profile_int("ema_slow", 20),
            slope_lookback=_profile_int("slope_lookback", 5),
            min_ema_spread_pct=_profile_float("min_ema_spread_pct", 0.05),
            min_slow_slope_pct=_profile_float("min_slow_slope_pct", 0.0),
            pullback_requires_outer_band_touch=_profile_flag(
                "pullback_requires_outer_band_touch",
                False,
            ),
            use_rejection_quality_filter=_profile_flag(
                "use_rejection_quality_filter",
                False,
            ),
            rejection_wick_min_ratio=_profile_float("rejection_wick_min_ratio", 0.35),
            rejection_body_min_ratio=_profile_float("rejection_body_min_ratio", 0.20),
            use_rsi_filter=_profile_flag("use_rsi_filter", False),
            rsi_length=_profile_int("rsi_length", 14),
            rsi_midline=_profile_float("rsi_midline", 50.0),
            use_rsi_cross_filter=_profile_flag("use_rsi_cross_filter", False),
            rsi_midline_margin=_profile_float("rsi_midline_margin", 0.0),
            use_volume_filter=_profile_flag("use_volume_filter", False),
            volume_ma_length=_profile_int("volume_ma_length", 20),
            volume_multiplier=_profile_float("volume_multiplier", 1.0),
            use_atr_stop_buffer=_profile_flag("use_atr_stop_buffer", False),
            atr_length=_profile_int("atr_length", 14),
            atr_stop_buffer_mult=_profile_float("atr_stop_buffer_mult", 0.5),
            signal_cooldown_bars=_profile_int("signal_cooldown_bars", 0),
            trend_persistence_bars=_profile_int("trend_persistence_bars", 1),
            max_pullback_bars=_profile_int("max_pullback_bars", 0),
            entry_offset_pct=_profile_float("entry_offset_pct", 0.0),
            use_late_entry_guard=_profile_flag(
                "use_late_entry_guard",
                bool(getattr(settings.strategy, "use_late_entry_guard", False)),
            ),
            late_entry_max_move_1_bar_pct=_profile_float(
                "late_entry_max_move_1_bar_pct",
                float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
            ),
            late_entry_max_move_2_bars_pct=_profile_float(
                "late_entry_max_move_2_bars_pct",
                float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
            ),
            late_entry_max_move_3_bars_pct=_profile_float(
                "late_entry_max_move_3_bars_pct",
                float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
            ),
            late_entry_max_distance_ref_pct=_profile_float(
                "late_entry_max_distance_ref_pct",
                float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
            ),
            late_entry_max_distance_fast_ref_pct=_profile_float(
                "late_entry_max_distance_fast_ref_pct",
                float(getattr(settings.strategy, "late_entry_max_distance_fast_ref_pct", 0.0)),
            ),
            late_entry_max_distance_mid_ref_pct=_profile_float(
                "late_entry_max_distance_mid_ref_pct",
                float(getattr(settings.strategy, "late_entry_max_distance_mid_ref_pct", 0.0)),
            ),
            late_entry_max_atr_mult=_profile_float(
                "late_entry_max_atr_mult",
                float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
            ),
            use_pullback_reentry=_profile_flag(
                "use_pullback_reentry",
                bool(getattr(settings.strategy, "use_pullback_reentry", False)),
            ),
            pullback_reentry_min_touch=_profile_float(
                "pullback_reentry_min_touch",
                float(getattr(settings.strategy, "pullback_reentry_min_touch", 0.0)),
            ),
            pullback_reentry_reconfirm_required=_profile_flag(
                "pullback_reentry_reconfirm_required",
                bool(getattr(settings.strategy, "pullback_reentry_reconfirm_required", False)),
            ),
        )

        if raw_signal_override is None:
            raw_signals = pd.to_numeric(
                working_df["ema_band_signal_direction"],
                errors="coerce",
            ).fillna(0).astype("int8").tolist()
        else:
            raw_signals = _align_signal_series_to_candle_count(
                raw_signal_override,
                candle_count=len(working_df),
            )
        long_exit_flags = working_df["ema_band_long_exit"].fillna(False).astype(bool).tolist()
        short_exit_flags = working_df["ema_band_short_exit"].fillna(False).astype(bool).tolist()
        dynamic_stop_loss_pcts = pd.to_numeric(
            working_df["ema_band_dynamic_stop_loss_pct"],
            errors="coerce",
        ).tolist()

        signals, total_signals, approved_signals, blocked_signals = _finalize_signal_series(
            working_df,
            raw_signals,
            required_candles=required_candles,
            setup_gate=setup_gate,
            strategy_name=strategy_name,
            regime_mask=regime_mask,
        )
        aligned_dynamic_stop_loss_pcts: list[float] = []
        for index, signal_value in enumerate(signals):
            raw_value = (
                dynamic_stop_loss_pcts[index]
                if index < len(dynamic_stop_loss_pcts)
                else float("nan")
            )
            with suppress(Exception):
                raw_value = float(raw_value)
            if int(signal_value) == 0:
                aligned_dynamic_stop_loss_pcts.append(float("nan"))
                continue
            if not isinstance(raw_value, float) or not math.isfinite(raw_value) or raw_value <= 0.0:
                aligned_dynamic_stop_loss_pcts.append(float("nan"))
                continue
            aligned_dynamic_stop_loss_pcts.append(float(raw_value))

        latest_signal_direction = int(signals[-1]) if signals else 0
        latest_trend_ok = False
        latest_pullback_ok = False
        latest_rejection_ok = False
        if len(working_df) > 0:
            latest_row = working_df.iloc[-1]
            latest_trend_ok = bool(
                bool(latest_row.get("ema_band_trend_long", False))
                or bool(latest_row.get("ema_band_trend_short", False))
            )
            latest_pullback_ok = bool(
                bool(latest_row.get("ema_band_pullback_long", False))
                or bool(latest_row.get("ema_band_pullback_short", False))
            )
            latest_rejection_ok = bool(
                bool(latest_row.get("ema_band_rejection_long", False))
                or bool(latest_row.get("ema_band_rejection_short", False))
            )

        entry_mask = (
            working_df["ema_band_long_entry"].fillna(False)
            | working_df["ema_band_short_entry"].fillna(False)
        )
        spread_series = pd.to_numeric(
            working_df["ema_band_ema_spread_pct"],
            errors="coerce",
        )
        spread_entry_series = spread_series.where(entry_mask, np.nan)
        dynamic_stop_series = pd.to_numeric(
            working_df["ema_band_dynamic_stop_loss_pct"],
            errors="coerce",
        )
        dynamic_stop_entry_series = dynamic_stop_series.where(entry_mask, np.nan)

        def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:
            numeric_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
            finite_values = numeric_values[np.isfinite(numeric_values)]
            if finite_values.size <= 0:
                return 0.0, 0.0, 0.0, 0
            return (
                float(np.min(finite_values)),
                float(np.max(finite_values)),
                float(np.mean(finite_values)),
                int(finite_values.size),
            )

        spread_min, spread_max, spread_mean, spread_count = _series_stats(spread_series)
        spread_entry_min, spread_entry_max, spread_entry_mean, spread_entry_count = _series_stats(
            spread_entry_series
        )
        dynamic_stop_min, dynamic_stop_max, dynamic_stop_mean, dynamic_stop_count = _series_stats(
            dynamic_stop_entry_series
        )
        dynamic_stop_entry_values = pd.to_numeric(
            dynamic_stop_entry_series,
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=False)
        dynamic_stop_entry_finite = dynamic_stop_entry_values[np.isfinite(dynamic_stop_entry_values)]
        dynamic_stop_clip_low_count = int(np.count_nonzero(dynamic_stop_entry_finite <= 0.0500001))
        dynamic_stop_clip_high_count = int(np.count_nonzero(dynamic_stop_entry_finite >= 94.9999))
        cooldown_blocked_signals = 0
        with suppress(Exception):
            cooldown_blocked_signals = int(
                pd.to_numeric(
                    working_df.get("ema_band_cooldown_blocked_signals"),
                    errors="coerce",
                ).fillna(0.0).max()
            )
        late_entry_blocked_count = 0
        with suppress(Exception):
            late_entry_blocked_count = int(
                pd.to_numeric(
                    working_df.get("ema_band_late_entry_guard_blocked"),
                    errors="coerce",
                ).fillna(0.0).sum()
            )
        latest_blocker_reason = ""
        latest_late_entry_guard_blocked = False
        latest_move_last_1_bar_pct = 0.0
        latest_move_last_2_bars_pct = 0.0
        latest_move_last_3_bars_pct = 0.0
        latest_distance_to_reference_pct = 0.0
        latest_atr_extension_mult = 0.0
        with suppress(Exception):
            latest_blocker_reason = str(
                working_df.get("ema_band_late_entry_guard_block_reason").iloc[-1] or ""
            )
        with suppress(Exception):
            latest_late_entry_guard_blocked = bool(
                working_df.get("ema_band_late_entry_guard_blocked").iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_1_bar_pct = float(
                pd.to_numeric(
                    working_df.get("ema_band_move_last_1_bar_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_2_bars_pct = float(
                pd.to_numeric(
                    working_df.get("ema_band_move_last_2_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_3_bars_pct = float(
                pd.to_numeric(
                    working_df.get("ema_band_move_last_3_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_distance_to_reference_pct = float(
                pd.to_numeric(
                    working_df.get("ema_band_distance_to_reference_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_atr_extension_mult = float(
                pd.to_numeric(
                    working_df.get("ema_band_atr_extension_mult"),
                    errors="coerce",
                ).iloc[-1]
            )

        return {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
            "precomputed_long_exit_flags": long_exit_flags,
            "precomputed_short_exit_flags": short_exit_flags,
            "precomputed_dynamic_stop_loss_pcts": aligned_dynamic_stop_loss_pcts,
            "strategy_diagnostics": {
                "trend_ok_count": int(
                    (
                        working_df["ema_band_trend_long"].fillna(False)
                        | working_df["ema_band_trend_short"].fillna(False)
                    ).sum()
                ),
                "pullback_ok_count": int(
                    (
                        working_df["ema_band_pullback_long"].fillna(False)
                        | working_df["ema_band_pullback_short"].fillna(False)
                    ).sum()
                ),
                "rejection_ok_count": int(
                    (
                        working_df["ema_band_rejection_long"].fillna(False)
                        | working_df["ema_band_rejection_short"].fillna(False)
                    ).sum()
                ),
                "long_entry_count": int(working_df["ema_band_long_entry"].fillna(False).sum()),
                "short_entry_count": int(working_df["ema_band_short_entry"].fillna(False).sum()),
                "latest_trend_ok": int(latest_trend_ok),
                "latest_pullback_ok": int(latest_pullback_ok),
                "latest_rejection_ok": int(latest_rejection_ok),
                "latest_signal_direction": int(latest_signal_direction),
                "ema_fast": _profile_int("ema_fast", 5),
                "ema_mid": _profile_int("ema_mid", 10),
                "ema_slow": _profile_int("ema_slow", 20),
                "slope_lookback": _profile_int("slope_lookback", 5),
                "min_ema_spread_pct": _profile_float("min_ema_spread_pct", 0.05),
                "min_slow_slope_pct": _profile_float("min_slow_slope_pct", 0.0),
                "pullback_requires_outer_band_touch": int(
                    _profile_flag("pullback_requires_outer_band_touch", False)
                ),
                "use_rejection_quality_filter": int(
                    _profile_flag("use_rejection_quality_filter", False)
                ),
                "rejection_wick_min_ratio": _profile_float("rejection_wick_min_ratio", 0.35),
                "rejection_body_min_ratio": _profile_float("rejection_body_min_ratio", 0.20),
                "ema_spread_pct_min": float(spread_min),
                "ema_spread_pct_max": float(spread_max),
                "ema_spread_pct_mean": float(spread_mean),
                "ema_spread_pct_count": int(spread_count),
                "ema_spread_entry_pct_min": float(spread_entry_min),
                "ema_spread_entry_pct_max": float(spread_entry_max),
                "ema_spread_entry_pct_mean": float(spread_entry_mean),
                "ema_spread_entry_pct_count": int(spread_entry_count),
                "use_rsi_filter": int(_profile_flag("use_rsi_filter", False)),
                "rsi_length": _profile_int("rsi_length", 14),
                "rsi_midline": _profile_float("rsi_midline", 50.0),
                "use_rsi_cross_filter": int(_profile_flag("use_rsi_cross_filter", False)),
                "rsi_midline_margin": _profile_float("rsi_midline_margin", 0.0),
                "use_volume_filter": int(_profile_flag("use_volume_filter", False)),
                "volume_ma_length": _profile_int("volume_ma_length", 20),
                "volume_multiplier": _profile_float("volume_multiplier", 1.0),
                "use_atr_stop_buffer": int(_profile_flag("use_atr_stop_buffer", False)),
                "atr_length": _profile_int("atr_length", 14),
                "atr_stop_buffer_mult": _profile_float("atr_stop_buffer_mult", 0.5),
                "signal_cooldown_bars": _profile_int("signal_cooldown_bars", 0),
                "trend_persistence_bars": _profile_int("trend_persistence_bars", 1),
                "max_pullback_bars": _profile_int("max_pullback_bars", 0),
                "entry_offset_pct": _profile_float("entry_offset_pct", 0.0),
                "cooldown_blocked_signals": int(cooldown_blocked_signals),
                "late_entry_guard_enabled": int(
                    _profile_flag(
                        "use_late_entry_guard",
                        bool(getattr(settings.strategy, "use_late_entry_guard", False)),
                    )
                ),
                "late_entry_max_move_1_bar_pct": _profile_float(
                    "late_entry_max_move_1_bar_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
                ),
                "late_entry_max_move_2_bars_pct": _profile_float(
                    "late_entry_max_move_2_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
                ),
                "late_entry_max_move_3_bars_pct": _profile_float(
                    "late_entry_max_move_3_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
                ),
                "late_entry_max_distance_ref_pct": _profile_float(
                    "late_entry_max_distance_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
                ),
                "late_entry_max_distance_fast_ref_pct": _profile_float(
                    "late_entry_max_distance_fast_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_fast_ref_pct", 0.0)),
                ),
                "late_entry_max_distance_mid_ref_pct": _profile_float(
                    "late_entry_max_distance_mid_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_mid_ref_pct", 0.0)),
                ),
                "late_entry_max_atr_mult": _profile_float(
                    "late_entry_max_atr_mult",
                    float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
                ),
                "use_pullback_reentry": int(
                    _profile_flag(
                        "use_pullback_reentry",
                        bool(getattr(settings.strategy, "use_pullback_reentry", False)),
                    )
                ),
                "pullback_reentry_min_touch": _profile_float(
                    "pullback_reentry_min_touch",
                    float(getattr(settings.strategy, "pullback_reentry_min_touch", 0.0)),
                ),
                "pullback_reentry_reconfirm_required": int(
                    _profile_flag(
                        "pullback_reentry_reconfirm_required",
                        bool(getattr(settings.strategy, "pullback_reentry_reconfirm_required", False)),
                    )
                ),
                "late_entry_guard_blocked_count": int(late_entry_blocked_count),
                "latest_late_entry_guard_blocked": int(bool(latest_late_entry_guard_blocked)),
                "latest_blocker_reason": str(latest_blocker_reason),
                "latest_move_last_1_bar_pct": float(latest_move_last_1_bar_pct),
                "latest_move_last_2_bars_pct": float(latest_move_last_2_bars_pct),
                "latest_move_last_3_bars_pct": float(latest_move_last_3_bars_pct),
                "latest_distance_to_reference_pct": float(latest_distance_to_reference_pct),
                "latest_atr_extension_mult": float(latest_atr_extension_mult),
                "dynamic_stop_loss_entry_pct_min": float(dynamic_stop_min),
                "dynamic_stop_loss_entry_pct_max": float(dynamic_stop_max),
                "dynamic_stop_loss_entry_pct_mean": float(dynamic_stop_mean),
                "dynamic_stop_loss_entry_pct_count": int(dynamic_stop_count),
                "dynamic_stop_loss_clip_low_count": int(dynamic_stop_clip_low_count),
                "dynamic_stop_loss_clip_high_count": int(dynamic_stop_clip_high_count),
                "dynamic_stop_loss_reference": "close",
            },
        }

    if strategy_name == "frama_cross":
        with _temporary_strategy_profile(effective_strategy_profile, candles_df):
            working_df = build_frama_cross_signal_frame(
                candles_df,
                min_frama_gap_pct=_profile_float("min_frama_gap_pct", 0.05),
                min_frama_slope_pct=_profile_float("min_frama_slope_pct", 0.0),
                post_cross_confirmation_bars=_profile_int("post_cross_confirmation_bars", 0),
                use_late_entry_guard=_profile_flag(
                    "use_late_entry_guard",
                    bool(getattr(settings.strategy, "use_late_entry_guard", False)),
                ),
                late_entry_max_move_1_bar_pct=_profile_float(
                    "late_entry_max_move_1_bar_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
                ),
                late_entry_max_move_2_bars_pct=_profile_float(
                    "late_entry_max_move_2_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
                ),
                late_entry_max_move_3_bars_pct=_profile_float(
                    "late_entry_max_move_3_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
                ),
                late_entry_max_distance_ref_pct=_profile_float(
                    "late_entry_max_distance_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
                ),
                late_entry_max_atr_mult=_profile_float(
                    "late_entry_max_atr_mult",
                    float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
                ),
            )
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
        latest_blocker_reason = ""
        latest_late_entry_guard_blocked = False
        latest_move_last_1_bar_pct = 0.0
        latest_move_last_2_bars_pct = 0.0
        latest_move_last_3_bars_pct = 0.0
        latest_distance_to_reference_pct = 0.0
        latest_atr_extension_mult = 0.0
        late_entry_blocked_count = 0
        with suppress(Exception):
            latest_blocker_reason = str(
                working_df.get("frama_late_entry_guard_block_reason").iloc[-1] or ""
            )
        with suppress(Exception):
            latest_late_entry_guard_blocked = bool(
                working_df.get("frama_late_entry_guard_blocked").iloc[-1]
            )
        with suppress(Exception):
            late_entry_blocked_count = int(
                pd.to_numeric(
                    working_df.get("frama_late_entry_guard_blocked"),
                    errors="coerce",
                ).fillna(0.0).sum()
            )
        with suppress(Exception):
            latest_move_last_1_bar_pct = float(
                pd.to_numeric(
                    working_df.get("frama_move_last_1_bar_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_2_bars_pct = float(
                pd.to_numeric(
                    working_df.get("frama_move_last_2_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_3_bars_pct = float(
                pd.to_numeric(
                    working_df.get("frama_move_last_3_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_distance_to_reference_pct = float(
                pd.to_numeric(
                    working_df.get("frama_distance_to_reference_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_atr_extension_mult = float(
                pd.to_numeric(
                    working_df.get("frama_atr_extension_mult"),
                    errors="coerce",
                ).iloc[-1]
            )
        return {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
            "strategy_diagnostics": {
                "late_entry_guard_enabled": int(
                    _profile_flag(
                        "use_late_entry_guard",
                        bool(getattr(settings.strategy, "use_late_entry_guard", False)),
                    )
                ),
                "late_entry_max_move_1_bar_pct": _profile_float(
                    "late_entry_max_move_1_bar_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
                ),
                "late_entry_max_move_2_bars_pct": _profile_float(
                    "late_entry_max_move_2_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
                ),
                "late_entry_max_move_3_bars_pct": _profile_float(
                    "late_entry_max_move_3_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
                ),
                "late_entry_max_distance_ref_pct": _profile_float(
                    "late_entry_max_distance_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
                ),
                "late_entry_max_atr_mult": _profile_float(
                    "late_entry_max_atr_mult",
                    float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
                ),
                "late_entry_guard_blocked_count": int(late_entry_blocked_count),
                "latest_late_entry_guard_blocked": int(bool(latest_late_entry_guard_blocked)),
                "latest_blocker_reason": str(latest_blocker_reason),
                "latest_move_last_1_bar_pct": float(latest_move_last_1_bar_pct),
                "latest_move_last_2_bars_pct": float(latest_move_last_2_bars_pct),
                "latest_move_last_3_bars_pct": float(latest_move_last_3_bars_pct),
                "latest_distance_to_reference_pct": float(latest_distance_to_reference_pct),
                "latest_atr_extension_mult": float(latest_atr_extension_mult),
            },
        }

    if strategy_name == "dual_thrust":
        with _temporary_strategy_profile(strategy_profile, candles_df):
            working_df = build_dual_thrust_signal_frame(
                candles_df,
                breakout_buffer_pct=_profile_float("breakout_buffer_pct", 0.0),
                min_range_pct=_profile_float("min_range_pct", 0.0),
                cooldown_bars_after_exit=_profile_int("cooldown_bars_after_exit", 0),
                use_late_entry_guard=_profile_flag(
                    "use_late_entry_guard",
                    bool(getattr(settings.strategy, "use_late_entry_guard", False)),
                ),
                late_entry_max_move_1_bar_pct=_profile_float(
                    "late_entry_max_move_1_bar_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
                ),
                late_entry_max_move_2_bars_pct=_profile_float(
                    "late_entry_max_move_2_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
                ),
                late_entry_max_move_3_bars_pct=_profile_float(
                    "late_entry_max_move_3_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
                ),
                late_entry_max_distance_ref_pct=_profile_float(
                    "late_entry_max_distance_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
                ),
                late_entry_max_atr_mult=_profile_float(
                    "late_entry_max_atr_mult",
                    float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
                ),
                max_breakout_candle_body_pct=_profile_float(
                    "max_breakout_candle_body_pct",
                    float(getattr(settings.strategy, "max_breakout_candle_body_pct", 0.0)),
                ),
                max_breakout_candle_range_atr_mult=_profile_float(
                    "max_breakout_candle_range_atr_mult",
                    float(getattr(settings.strategy, "max_breakout_candle_range_atr_mult", 0.0)),
                ),
            )
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
        latest_blocker_reason = ""
        latest_late_entry_guard_blocked = False
        latest_move_last_1_bar_pct = 0.0
        latest_move_last_2_bars_pct = 0.0
        latest_move_last_3_bars_pct = 0.0
        latest_distance_to_reference_pct = 0.0
        latest_atr_extension_mult = 0.0
        late_entry_blocked_count = 0
        with suppress(Exception):
            latest_blocker_reason = str(
                working_df.get("dual_late_entry_guard_block_reason").iloc[-1] or ""
            )
        with suppress(Exception):
            latest_late_entry_guard_blocked = bool(
                working_df.get("dual_late_entry_guard_blocked").iloc[-1]
            )
        with suppress(Exception):
            late_entry_blocked_count = int(
                pd.to_numeric(
                    working_df.get("dual_late_entry_guard_blocked"),
                    errors="coerce",
                ).fillna(0.0).sum()
            )
        with suppress(Exception):
            latest_move_last_1_bar_pct = float(
                pd.to_numeric(
                    working_df.get("dual_move_last_1_bar_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_2_bars_pct = float(
                pd.to_numeric(
                    working_df.get("dual_move_last_2_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_move_last_3_bars_pct = float(
                pd.to_numeric(
                    working_df.get("dual_move_last_3_bars_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_distance_to_reference_pct = float(
                pd.to_numeric(
                    working_df.get("dual_distance_to_reference_pct"),
                    errors="coerce",
                ).iloc[-1]
            )
        with suppress(Exception):
            latest_atr_extension_mult = float(
                pd.to_numeric(
                    working_df.get("dual_atr_extension_mult"),
                    errors="coerce",
                ).iloc[-1]
            )
        return {
            "signals": signals,
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
            "precomputed_long_exit_flags": long_exit_flags,
            "precomputed_short_exit_flags": short_exit_flags,
            "strategy_diagnostics": {
                "late_entry_guard_enabled": int(
                    _profile_flag(
                        "use_late_entry_guard",
                        bool(getattr(settings.strategy, "use_late_entry_guard", False)),
                    )
                ),
                "late_entry_max_move_1_bar_pct": _profile_float(
                    "late_entry_max_move_1_bar_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
                ),
                "late_entry_max_move_2_bars_pct": _profile_float(
                    "late_entry_max_move_2_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
                ),
                "late_entry_max_move_3_bars_pct": _profile_float(
                    "late_entry_max_move_3_bars_pct",
                    float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
                ),
                "late_entry_max_distance_ref_pct": _profile_float(
                    "late_entry_max_distance_ref_pct",
                    float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
                ),
                "late_entry_max_atr_mult": _profile_float(
                    "late_entry_max_atr_mult",
                    float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
                ),
                "max_breakout_candle_body_pct": _profile_float(
                    "max_breakout_candle_body_pct",
                    float(getattr(settings.strategy, "max_breakout_candle_body_pct", 0.0)),
                ),
                "max_breakout_candle_range_atr_mult": _profile_float(
                    "max_breakout_candle_range_atr_mult",
                    float(getattr(settings.strategy, "max_breakout_candle_range_atr_mult", 0.0)),
                ),
                "late_entry_guard_blocked_count": int(late_entry_blocked_count),
                "latest_late_entry_guard_blocked": int(bool(latest_late_entry_guard_blocked)),
                "latest_blocker_reason": str(latest_blocker_reason),
                "latest_move_last_1_bar_pct": float(latest_move_last_1_bar_pct),
                "latest_move_last_2_bars_pct": float(latest_move_last_2_bars_pct),
                "latest_move_last_3_bars_pct": float(latest_move_last_3_bars_pct),
                "latest_distance_to_reference_pct": float(latest_distance_to_reference_pct),
                "latest_atr_extension_mult": float(latest_atr_extension_mult),
            },
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
        "use_late_entry_guard": settings.strategy.use_late_entry_guard,
        "late_entry_max_move_1_bar_pct": settings.strategy.late_entry_max_move_1_bar_pct,
        "late_entry_max_move_2_bars_pct": settings.strategy.late_entry_max_move_2_bars_pct,
        "late_entry_max_move_3_bars_pct": settings.strategy.late_entry_max_move_3_bars_pct,
        "late_entry_max_distance_ref_pct": settings.strategy.late_entry_max_distance_ref_pct,
        "late_entry_max_distance_fast_ref_pct": settings.strategy.late_entry_max_distance_fast_ref_pct,
        "late_entry_max_distance_mid_ref_pct": settings.strategy.late_entry_max_distance_mid_ref_pct,
        "late_entry_max_atr_mult": settings.strategy.late_entry_max_atr_mult,
        "use_pullback_reentry": settings.strategy.use_pullback_reentry,
        "pullback_reentry_min_touch": settings.strategy.pullback_reentry_min_touch,
        "pullback_reentry_reconfirm_required": settings.strategy.pullback_reentry_reconfirm_required,
        "max_breakout_candle_body_pct": settings.strategy.max_breakout_candle_body_pct,
        "max_breakout_candle_range_atr_mult": settings.strategy.max_breakout_candle_range_atr_mult,
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
            elif isinstance(original_value, str):
                converted_value = str(profile_value)
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
                    elif isinstance(original_value, str):
                        converted_value = str(param_value)
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
    if strategy_name in {
        "ema_cross_volume",
        "ema_band_rejection",
        "frama_cross",
        "dual_thrust",
    }:
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
    breakeven_activation_pct: float | None = None,
    breakeven_buffer_pct: float | None = None,
    tight_trailing_activation_pct: float | None = None,
    tight_trailing_distance_pct: float | None = None,
) -> str:
    base = _format_leverage_log(
        prefix="Runtime profile",
        symbol=symbol,
        effective_leverage=runtime_leverage,
        configured_leverage=configured_leverage,
        leverage_override=leverage_override,
        manual_override_forced=manual_override_forced,
    )
    summary_text = (
        f"{base} "
        f"interval={interval} "
        f"sl={stop_loss_pct:.1f}% tp={take_profit_pct:.1f}% "
        f"trail_on={trailing_activation_pct:.1f}% trail_gap={trailing_distance_pct:.1f}%"
    )
    if (
        breakeven_activation_pct is None
        and breakeven_buffer_pct is None
        and tight_trailing_activation_pct is None
        and tight_trailing_distance_pct is None
    ):
        return summary_text
    return (
        f"{summary_text} "
        f"breakeven_on={float(breakeven_activation_pct or 0.0):.1f}% "
        f"breakeven_buffer={float(breakeven_buffer_pct or 0.0):.2f}% "
        f"tight_on={float(tight_trailing_activation_pct or 0.0):.1f}% "
        f"tight_gap={float(tight_trailing_distance_pct or 0.0):.2f}%"
    )


def _format_hmm_non_convergence_detail(detection: object) -> str:
    def _safe_int(value: object, default: int = -1) -> int:
        if value is None:
            return int(default)
        with suppress(Exception):
            return int(value)
        return int(default)

    events = getattr(detection, "non_convergence_events", None)
    if not isinstance(events, (list, tuple)) or not events:
        return "window_context=n/a"
    latest = events[-1]
    if not isinstance(latest, Mapping):
        return "window_context=n/a"
    window_index = _safe_int(latest.get("window_index", -1))
    train_start = _safe_int(latest.get("train_start", -1))
    train_end = _safe_int(latest.get("train_end", -1))
    apply_start = _safe_int(latest.get("apply_start", -1))
    apply_end = _safe_int(latest.get("apply_end", -1))
    previous_score = latest.get("previous_score")
    current_score = latest.get("current_score")
    delta_score = latest.get("delta_score")
    previous_text = "n/a" if previous_score is None else f"{float(previous_score):.6f}"
    current_text = "n/a" if current_score is None else f"{float(current_score):.6f}"
    delta_text = "n/a" if delta_score is None else f"{float(delta_score):.6f}"
    return (
        f"window={window_index} "
        f"train=[{train_start}:{train_end}) "
        f"apply=[{apply_start}:{apply_end}) "
        f"prev={previous_text} curr={current_text} delta={delta_text}"
    )


def _format_hmm_warning_detail(
    detection: object,
    *,
    warning_kind: str | None = None,
) -> str:
    def _safe_int(value: object, default: int = -1) -> int:
        if value is None:
            return int(default)
        with suppress(Exception):
            return int(value)
        return int(default)

    events = getattr(detection, "warning_events", None)
    if not isinstance(events, (list, tuple)) or not events:
        return "warning_context=n/a"
    filtered_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and (
            warning_kind is None
            or str(event.get("warning_kind", "")).strip().lower()
            == str(warning_kind).strip().lower()
        )
    ]
    if not filtered_events:
        return "warning_context=n/a"
    latest = filtered_events[-1]
    warning_kind_text = str(latest.get("warning_kind", "unknown")).strip() or "unknown"
    window_index = _safe_int(latest.get("window_index", -1))
    train_start = _safe_int(latest.get("train_start", -1))
    train_end = _safe_int(latest.get("train_end", -1))
    apply_start = _safe_int(latest.get("apply_start", -1))
    apply_end = _safe_int(latest.get("apply_end", -1))
    message_text = str(latest.get("warning_message", "") or "").strip()
    if len(message_text) > 220:
        message_text = f"{message_text[:217]}..."
    if not message_text:
        message_text = "n/a"
    return (
        f"kind={warning_kind_text} "
        f"window={window_index} "
        f"train=[{train_start}:{train_end}) "
        f"apply=[{apply_start}:{apply_end}) "
        f"message={message_text}"
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


def _cap_profit_factor_for_sorting(value: float) -> float:
    with suppress(Exception):
        resolved = float(value)
        if math.isnan(resolved) or resolved <= 0.0:
            return 0.0
        if math.isinf(resolved):
            return OPTIMIZER_SORT_PROFIT_FACTOR_CAP
        return min(resolved, OPTIMIZER_SORT_PROFIT_FACTOR_CAP)
    return 0.0


def _resolve_average_win_to_cost_ratio(summary: dict[str, float]) -> float:
    with suppress(Exception):
        ratio = float(summary.get("average_win_to_cost_ratio", 0.0) or 0.0)
        if math.isnan(ratio):
            ratio = 0.0
        if math.isinf(ratio):
            return float("inf") if ratio > 0.0 else 0.0
        if ratio > 0.0:
            return ratio
    average_win_usd = float(summary.get("average_win_usd", 0.0) or 0.0)
    average_trade_cost_usd = float(summary.get("average_trade_cost_usd", 0.0) or 0.0)
    if average_trade_cost_usd <= 0.0:
        return float("inf") if average_win_usd > 0.0 else 0.0
    return max(0.0, average_win_usd / average_trade_cost_usd)


def _calculate_stability_score(summary: dict[str, float]) -> float:
    min_avg_win_to_cost_ratio = _resolve_optimizer_min_average_win_to_cost_ratio()
    ranking_max_drawdown_pct = _resolve_optimizer_ranking_max_drawdown_pct()
    capped_profit_factor = _cap_profit_factor_for_sorting(float(summary.get("profit_factor", 0.0) or 0.0))
    max_dd = float(summary.get("max_drawdown_pct", 0.0) or 0.0)
    if not math.isfinite(max_dd):
        max_dd = 10_000.0
    max_dd = max(0.0, max_dd)
    total_trades = max(0.0, float(summary.get("total_trades", 0.0) or 0.0))
    avg_win_to_cost_ratio = _resolve_average_win_to_cost_ratio(summary)
    if math.isinf(avg_win_to_cost_ratio):
        cost_component = 1.0
    else:
        cost_component = min(
            max(0.0, avg_win_to_cost_ratio) / max(min_avg_win_to_cost_ratio, 0.0001),
            1.0,
        )
    pf_component = min(capped_profit_factor / OPTIMIZER_SORT_PROFIT_FACTOR_CAP, 1.0)
    drawdown_component = max(0.0, 1.0 - (max_dd / max(ranking_max_drawdown_pct, 0.0001)))
    trade_count_component = min(total_trades / 100.0, 1.0)
    score = (
        (0.45 * cost_component)
        + (0.30 * pf_component)
        + (0.15 * drawdown_component)
        + (0.10 * trade_count_component)
    )
    return max(0.0, score)


def _optimization_sort_key(
    summary: dict[str, float],
    *,
    strategy_name: str | None = None,
) -> tuple[float, ...]:
    _ = strategy_name
    min_avg_win_to_cost_ratio = _resolve_optimizer_min_average_win_to_cost_ratio()
    strong_sample_min_trades = _resolve_optimizer_strong_sample_min_trades()
    ranking_min_profit_factor = _resolve_optimizer_ranking_min_profit_factor()
    ranking_max_drawdown_pct = _resolve_optimizer_ranking_max_drawdown_pct()
    profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
    capped_profit_factor = _cap_profit_factor_for_sorting(profit_factor)
    max_dd = float(summary.get("max_drawdown_pct", 0.0) or 0.0)
    if not math.isfinite(max_dd):
        max_dd = 10_000.0
    max_dd = max(0.0, max_dd)
    win_rate = float(summary.get("win_rate_pct", 0.0) or 0.0)
    total_trades = max(0.0, float(summary.get("total_trades", 0.0) or 0.0))
    average_win_to_cost_ratio = _resolve_average_win_to_cost_ratio(summary)
    avg_win_ratio_pass = (
        1.0
        if average_win_to_cost_ratio >= float(min_avg_win_to_cost_ratio)
        else 0.0
    )
    strong_sample_pass = (
        1.0
        if total_trades >= float(strong_sample_min_trades)
        else 0.0
    )
    pf_pass = 1.0 if capped_profit_factor >= float(ranking_min_profit_factor) else 0.0
    dd_pass = 1.0 if max_dd < float(ranking_max_drawdown_pct) else 0.0
    stability_score = float(summary.get("stability_score", _calculate_stability_score(summary)))
    if not math.isfinite(stability_score):
        stability_score = 0.0
    return (
        strong_sample_pass,
        pf_pass,
        avg_win_ratio_pass,
        stability_score,
        dd_pass,
        capped_profit_factor,
        -max_dd,
        win_rate,
        float(summary.get("real_rrr", 0.0)),
        float(summary.get("total_pnl_usd", 0.0)),
    )


OPTIMIZATION_METRIC_FIELDS = {
    "start_capital_usd",
    "total_pnl_usd",
    "win_rate_pct",
    "profit_factor",
    "total_trades",
    "long_trades",
    "short_trades",
    "real_rrr",
    "average_win_usd",
    "average_loss_usd",
    "average_trade_fees_usd",
    "average_trade_slippage_usd",
    "average_trade_cost_usd",
    "average_win_to_cost_ratio",
    "avg_profit_per_trade_net_pct",
    "max_drawdown_pct",
    "longest_consecutive_losses",
    "total_slippage_penalty_usd",
    "slippage_penalty_pct_per_side",
    "slippage_penalty_pct_per_trade",
    "stability_score",
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
        return (
            "ema_fast_period",
            "ema_slow_period",
            "volume_multiplier",
            "min_ema_gap_pct",
            "cross_confirmation_bars",
            "max_price_extension_pct",
        )
    if strategy_name == "ema_band_rejection":
        return (
            "ema_fast",
            "ema_mid",
            "ema_slow",
            "slope_lookback",
            "min_ema_spread_pct",
            "min_slow_slope_pct",
            "pullback_requires_outer_band_touch",
            "use_rejection_quality_filter",
            "rejection_wick_min_ratio",
            "rejection_body_min_ratio",
            "use_rsi_filter",
            "rsi_length",
            "rsi_midline",
            "use_rsi_cross_filter",
            "rsi_midline_margin",
            "use_volume_filter",
            "volume_ma_length",
            "volume_multiplier",
            "use_atr_stop_buffer",
            "atr_length",
            "atr_stop_buffer_mult",
            "signal_cooldown_bars",
            "trend_persistence_bars",
            "max_pullback_bars",
            "entry_offset_pct",
            "use_late_entry_guard",
            "late_entry_max_move_1_bar_pct",
            "late_entry_max_move_2_bars_pct",
            "late_entry_max_move_3_bars_pct",
            "late_entry_max_distance_ref_pct",
            "late_entry_max_distance_fast_ref_pct",
            "late_entry_max_distance_mid_ref_pct",
            "late_entry_max_atr_mult",
            "use_pullback_reentry",
            "pullback_reentry_min_touch",
            "pullback_reentry_reconfirm_required",
        )
    if strategy_name == "frama_cross":
        return (
            "frama_fast_period",
            "frama_slow_period",
            "volume_multiplier",
            "min_frama_gap_pct",
            "min_frama_slope_pct",
            "post_cross_confirmation_bars",
            "use_late_entry_guard",
            "late_entry_max_move_1_bar_pct",
            "late_entry_max_move_2_bars_pct",
            "late_entry_max_move_3_bars_pct",
            "late_entry_max_distance_ref_pct",
            "late_entry_max_atr_mult",
        )
    if strategy_name == "dual_thrust":
        return (
            "dual_thrust_period",
            "dual_thrust_k1",
            "dual_thrust_k2",
            "breakout_buffer_pct",
            "min_range_pct",
            "cooldown_bars_after_exit",
            "use_late_entry_guard",
            "late_entry_max_move_1_bar_pct",
            "late_entry_max_move_2_bars_pct",
            "late_entry_max_move_3_bars_pct",
            "late_entry_max_distance_ref_pct",
            "late_entry_max_atr_mult",
            "max_breakout_candle_body_pct",
            "max_breakout_candle_range_atr_mult",
        )
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


def _resolve_random_search_sample_cap() -> int | None:
    if _is_optimizer_all_coins_pass1_mode():
        return int(
            min(
                int(OPTIMIZER_ALL_COINS_PASS1_MAX_SAMPLE_PROFILES),
                int(OPTIMIZER_ALL_COINS_PASS1_RANDOM_SEARCH_SAMPLES),
            )
        )
    raw_value = getattr(settings.trading, "random_search_samples", None)
    if raw_value is None:
        return None
    with suppress(Exception):
        resolved_value = int(raw_value)
        if resolved_value <= 0:
            return None
        return max(1, resolved_value)
    return None


def _resolve_optimizer_two_stage_strategy_names() -> set[str]:
    if _is_optimizer_all_coins_pass1_mode():
        return set(OPTIMIZER_ALL_COINS_PASS1_TWO_STAGE_STRATEGIES)
    configured = getattr(settings.trading, "optimizer_two_stage_strategies", None)
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        resolved: set[str] = set()
        for item in configured:
            item_text = str(item).strip()
            if not item_text:
                continue
            with suppress(Exception):
                item_text = _validate_strategy_name(item_text)
            resolved.add(item_text)
        if resolved:
            return resolved
    return set(OPTIMIZER_TWO_STAGE_DEFAULT_STRATEGIES)


def _resolve_optimizer_worker_cap_for_strategy(strategy_name: str) -> int | None:
    normalized_strategy_name = _validate_strategy_name(strategy_name)
    worker_cap_mapping: dict[str, int] = dict(OPTIMIZER_WORKER_CAP_BY_STRATEGY_DEFAULT)
    configured_mapping = getattr(settings.trading, "optimizer_worker_cap_by_strategy", None)
    if isinstance(configured_mapping, Mapping):
        for raw_key, raw_value in configured_mapping.items():
            key_text = str(raw_key).strip()
            if not key_text:
                continue
            with suppress(Exception):
                key_text = _validate_strategy_name(key_text)
            with suppress(Exception):
                worker_cap_mapping[key_text] = int(raw_value)
    resolved_cap = worker_cap_mapping.get(normalized_strategy_name)
    if resolved_cap is None:
        return None
    with suppress(Exception):
        cap_value = int(resolved_cap)
        if cap_value > 0:
            return max(1, min(cap_value, int(MAX_OPTIMIZATION_WORKERS)))
    return None


def _resolve_strategy_specific_positive_int(
    *,
    strategy_name: str,
    global_field_name: str,
    strategy_field_name: str,
    default_value: int,
) -> int:
    normalized_strategy_name = _validate_strategy_name(strategy_name)
    resolved_value = int(default_value)
    with suppress(Exception):
        global_value = int(getattr(settings.trading, global_field_name, default_value) or default_value)
        if global_value > 0:
            resolved_value = global_value
    configured_mapping = getattr(settings.trading, strategy_field_name, None)
    if isinstance(configured_mapping, Mapping):
        candidate_value = None
        for raw_key, raw_value in configured_mapping.items():
            key_text = str(raw_key).strip()
            if not key_text:
                continue
            with suppress(Exception):
                key_text = _validate_strategy_name(key_text)
            if key_text == normalized_strategy_name:
                candidate_value = raw_value
                break
        if candidate_value is not None:
            with suppress(Exception):
                mapped_value = int(candidate_value)
                if mapped_value > 0:
                    resolved_value = mapped_value
    return max(1, int(resolved_value))


def _resolve_two_stage_search_window_candles(
    *,
    strategy_name: str,
    interval: str,
    stage_index: int,
) -> int:
    if _is_optimizer_all_coins_pass1_mode():
        base_window = max(1, int(_resolve_optimizer_search_window_candles(interval)))
        if stage_index <= 1:
            return max(
                1,
                min(base_window, int(OPTIMIZER_ALL_COINS_PASS1_STAGE1_SEARCH_WINDOW_CANDLES)),
            )
        return max(1, int(OPTIMIZER_ALL_COINS_PASS1_STAGE2_SEARCH_WINDOW_CANDLES))
    base_window = max(1, int(_resolve_optimizer_search_window_candles(interval)))
    if stage_index <= 1:
        resolved_window = _resolve_strategy_specific_positive_int(
            strategy_name=strategy_name,
            global_field_name="optimizer_stage1_search_window_candles",
            strategy_field_name="optimizer_stage1_search_window_candles_by_strategy",
            default_value=min(base_window, OPTIMIZER_TWO_STAGE_STAGE1_SEARCH_WINDOW_CANDLES_DEFAULT),
        )
        return max(1, min(base_window, resolved_window))
    resolved_window = _resolve_strategy_specific_positive_int(
        strategy_name=strategy_name,
        global_field_name="optimizer_stage2_search_window_candles",
        strategy_field_name="optimizer_stage2_search_window_candles_by_strategy",
        default_value=max(base_window, OPTIMIZER_TWO_STAGE_STAGE2_SEARCH_WINDOW_CANDLES_DEFAULT),
    )
    return max(1, resolved_window)


def _resolve_optimizer_two_stage_policy(
    *,
    strategy_name: str,
    interval: str,
    theoretical_profiles: int,
) -> dict[str, object]:
    normalized_strategy_name = _validate_strategy_name(strategy_name)
    if _is_optimizer_all_coins_pass1_mode():
        sample_cap = int(
            min(
                int(OPTIMIZER_ALL_COINS_PASS1_MAX_SAMPLE_PROFILES),
                int(OPTIMIZER_ALL_COINS_PASS1_RANDOM_SEARCH_SAMPLES),
            )
        )
        stage1_target_profiles = max(
            1,
            min(
                int(theoretical_profiles),
                int(OPTIMIZER_ALL_COINS_PASS1_STAGE1_TARGET_PROFILES),
                int(sample_cap),
            ),
        )
        stage2_top_n = max(
            1,
            min(
                int(OPTIMIZER_ALL_COINS_PASS1_STAGE2_TOP_N),
                int(stage1_target_profiles),
                int(theoretical_profiles),
            ),
        )
        stage1_window_candles = _resolve_two_stage_search_window_candles(
            strategy_name=normalized_strategy_name,
            interval=interval,
            stage_index=1,
        )
        stage2_window_candles = _resolve_two_stage_search_window_candles(
            strategy_name=normalized_strategy_name,
            interval=interval,
            stage_index=2,
        )
        return {
            "two_stage_enabled": bool(
                OPTIMIZER_ALL_COINS_PASS1_TWO_STAGE_ENABLED
                and normalized_strategy_name in OPTIMIZER_ALL_COINS_PASS1_TWO_STAGE_STRATEGIES
                and int(theoretical_profiles) > 1
            ),
            "sample_cap": int(sample_cap),
            "stage1_target_profiles": int(stage1_target_profiles),
            "stage2_top_n": int(stage2_top_n),
            "stage1_search_window_candles": int(stage1_window_candles),
            "stage2_search_window_candles": int(stage2_window_candles),
            "worker_cap": _resolve_optimizer_worker_cap_for_strategy(normalized_strategy_name),
            "force_full_verification_for_winner": bool(
                OPTIMIZER_ALL_COINS_PASS1_FORCE_FULL_VERIFICATION_FOR_WINNER
            ),
        }
    global_two_stage_enabled = bool(
        getattr(
            settings.trading,
            "optimizer_two_stage_enabled",
            OPTIMIZER_TWO_STAGE_DEFAULT_ENABLED,
        )
    )
    two_stage_strategy_names = _resolve_optimizer_two_stage_strategy_names()
    two_stage_enabled = bool(
        global_two_stage_enabled
        and normalized_strategy_name in two_stage_strategy_names
        and int(theoretical_profiles) > 1
    )

    sample_cap = _resolve_random_search_sample_cap()
    stage1_target_profiles = _resolve_strategy_specific_positive_int(
        strategy_name=normalized_strategy_name,
        global_field_name="optimizer_stage1_target_profiles",
        strategy_field_name="optimizer_stage1_target_profiles_by_strategy",
        default_value=OPTIMIZER_TWO_STAGE_STAGE1_TARGET_PROFILES_DEFAULT,
    )
    if sample_cap is not None:
        stage1_target_profiles = min(stage1_target_profiles, int(sample_cap))
    stage1_target_profiles = max(1, min(int(theoretical_profiles), int(stage1_target_profiles)))

    stage2_top_n = _resolve_strategy_specific_positive_int(
        strategy_name=normalized_strategy_name,
        global_field_name="optimizer_stage2_top_n",
        strategy_field_name="optimizer_stage2_top_n_by_strategy",
        default_value=OPTIMIZER_TWO_STAGE_STAGE2_TOP_N_DEFAULT,
    )
    stage2_top_n = max(
        1,
        min(int(stage2_top_n), int(stage1_target_profiles), int(theoretical_profiles)),
    )

    stage1_window_candles = _resolve_two_stage_search_window_candles(
        strategy_name=normalized_strategy_name,
        interval=interval,
        stage_index=1,
    )
    stage2_window_candles = _resolve_two_stage_search_window_candles(
        strategy_name=normalized_strategy_name,
        interval=interval,
        stage_index=2,
    )
    worker_cap = _resolve_optimizer_worker_cap_for_strategy(normalized_strategy_name)
    force_full_verification_for_winner = bool(
        getattr(
            settings.trading,
            "optimizer_force_full_verification_for_winner",
            OPTIMIZER_FORCE_FULL_VERIFICATION_FOR_WINNER_DEFAULT,
        )
    )

    return {
        "two_stage_enabled": bool(two_stage_enabled),
        "sample_cap": sample_cap,
        "stage1_target_profiles": int(stage1_target_profiles),
        "stage2_top_n": int(stage2_top_n),
        "stage1_search_window_candles": int(stage1_window_candles),
        "stage2_search_window_candles": int(stage2_window_candles),
        "worker_cap": worker_cap,
        "force_full_verification_for_winner": bool(force_full_verification_for_winner),
    }


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
        "ema_fast",
        "ema_mid",
        "ema_slow",
        "slope_lookback",
        "rsi_length",
        "volume_ma_length",
        "atr_length",
        "use_rsi_filter",
        "use_rsi_cross_filter",
        "use_volume_filter",
        "use_atr_stop_buffer",
        "pullback_requires_outer_band_touch",
        "use_rejection_quality_filter",
        "signal_cooldown_bars",
        "ema_length",
        "frama_fast_period",
        "frama_slow_period",
        "post_cross_confirmation_bars",
        "cross_confirmation_bars",
        "cooldown_bars_after_exit",
        "trend_persistence_bars",
        "max_pullback_bars",
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
        ema_fast_period_options = EMA_CROSS_VOLUME_FIXED_GRID_OPTIONS["ema_fast_period"]
        ema_slow_period_options = EMA_CROSS_VOLUME_FIXED_GRID_OPTIONS["ema_slow_period"]
        volume_multiplier_options = EMA_CROSS_VOLUME_FIXED_GRID_OPTIONS["volume_multiplier"]
        return [
            {
                "ema_fast_period": float(ema_fast_period),
                "ema_slow_period": float(ema_slow_period),
                "volume_multiplier": float(volume_multiplier),
            }
            for ema_fast_period, ema_slow_period, volume_multiplier in product(
                ema_fast_period_options,
                ema_slow_period_options,
                volume_multiplier_options,
            )
            if int(ema_fast_period) < int(ema_slow_period)
        ]
    if strategy_name == "ema_band_rejection":
        default_rsi_length = int(EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_length"][0])
        default_rsi_midline = float(EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_midline"][0])
        default_use_rsi_cross_filter = int(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["use_rsi_cross_filter"][0]
        )
        default_rsi_midline_margin = float(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_midline_margin"][0]
        )
        default_volume_ma_length = int(EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["volume_ma_length"][0])
        default_volume_multiplier = float(EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["volume_multiplier"][0])
        default_atr_length = int(EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["atr_length"][0])
        default_atr_stop_buffer_mult = float(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["atr_stop_buffer_mult"][-1]
        )
        default_rejection_wick_min_ratio = float(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rejection_wick_min_ratio"][0]
        )
        default_rejection_body_min_ratio = float(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rejection_body_min_ratio"][0]
        )
        profiles: list[OptimizationProfile] = []
        for ema_fast, ema_mid, ema_slow in product(
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["ema_fast"],
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["ema_mid"],
            EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["ema_slow"],
        ):
            resolved_ema_fast = int(ema_fast)
            resolved_ema_mid = int(ema_mid)
            resolved_ema_slow = int(ema_slow)
            if not (resolved_ema_fast < resolved_ema_mid < resolved_ema_slow):
                continue
            for slope_lookback, min_ema_spread_pct in product(
                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["slope_lookback"],
                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["min_ema_spread_pct"],
            ):
                for min_slow_slope_pct in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["min_slow_slope_pct"]:
                    for pullback_requires_outer_band_touch in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS[
                        "pullback_requires_outer_band_touch"
                    ]:
                        resolved_pullback_requires_outer_band_touch = (
                            1 if int(pullback_requires_outer_band_touch) > 0 else 0
                        )
                        for use_rejection_quality_filter in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS[
                            "use_rejection_quality_filter"
                        ]:
                            resolved_use_rejection_quality_filter = (
                                1 if int(use_rejection_quality_filter) > 0 else 0
                            )
                            rejection_wick_min_ratio_values = (
                                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rejection_wick_min_ratio"]
                                if resolved_use_rejection_quality_filter > 0
                                else (default_rejection_wick_min_ratio,)
                            )
                            rejection_body_min_ratio_values = (
                                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rejection_body_min_ratio"]
                                if resolved_use_rejection_quality_filter > 0
                                else (default_rejection_body_min_ratio,)
                            )
                            for rejection_wick_min_ratio, rejection_body_min_ratio in product(
                                rejection_wick_min_ratio_values,
                                rejection_body_min_ratio_values,
                            ):
                                for use_rsi_filter in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["use_rsi_filter"]:
                                    resolved_use_rsi_filter = 1 if int(use_rsi_filter) > 0 else 0
                                    rsi_length_values = (
                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_length"]
                                        if resolved_use_rsi_filter > 0
                                        else (default_rsi_length,)
                                    )
                                    rsi_midline_values = (
                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_midline"]
                                        if resolved_use_rsi_filter > 0
                                        else (default_rsi_midline,)
                                    )
                                    rsi_cross_filter_values = (
                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["use_rsi_cross_filter"]
                                        if resolved_use_rsi_filter > 0
                                        else (default_use_rsi_cross_filter,)
                                    )
                                    rsi_midline_margin_values = (
                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["rsi_midline_margin"]
                                        if resolved_use_rsi_filter > 0
                                        else (default_rsi_midline_margin,)
                                    )
                                    for (
                                        rsi_length,
                                        rsi_midline,
                                        use_rsi_cross_filter,
                                        rsi_midline_margin,
                                    ) in product(
                                        rsi_length_values,
                                        rsi_midline_values,
                                        rsi_cross_filter_values,
                                        rsi_midline_margin_values,
                                    ):
                                        for use_volume_filter in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["use_volume_filter"]:
                                            resolved_use_volume_filter = 1 if int(use_volume_filter) > 0 else 0
                                            volume_ma_length_values = (
                                                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["volume_ma_length"]
                                                if resolved_use_volume_filter > 0
                                                else (default_volume_ma_length,)
                                            )
                                            volume_multiplier_values = (
                                                EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["volume_multiplier"]
                                                if resolved_use_volume_filter > 0
                                                else (default_volume_multiplier,)
                                            )
                                            for volume_ma_length, volume_multiplier in product(
                                                volume_ma_length_values,
                                                volume_multiplier_values,
                                            ):
                                                for use_atr_stop_buffer in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["use_atr_stop_buffer"]:
                                                    resolved_use_atr_stop_buffer = 1 if int(use_atr_stop_buffer) > 0 else 0
                                                    atr_length_values = (
                                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["atr_length"]
                                                        if resolved_use_atr_stop_buffer > 0
                                                        else (default_atr_length,)
                                                    )
                                                    atr_stop_buffer_mult_values = (
                                                        EMA_BAND_REJECTION_FIXED_GRID_OPTIONS["atr_stop_buffer_mult"]
                                                        if resolved_use_atr_stop_buffer > 0
                                                        else (default_atr_stop_buffer_mult,)
                                                    )
                                                    for atr_length, atr_stop_buffer_mult in product(
                                                        atr_length_values,
                                                        atr_stop_buffer_mult_values,
                                                    ):
                                                        for signal_cooldown_bars in EMA_BAND_REJECTION_FIXED_GRID_OPTIONS[
                                                            "signal_cooldown_bars"
                                                        ]:
                                                            profiles.append(
                                                                {
                                                                    "ema_fast": float(resolved_ema_fast),
                                                                    "ema_mid": float(resolved_ema_mid),
                                                                    "ema_slow": float(resolved_ema_slow),
                                                                    "slope_lookback": float(slope_lookback),
                                                                    "min_ema_spread_pct": float(min_ema_spread_pct),
                                                                    "min_slow_slope_pct": float(min_slow_slope_pct),
                                                                    "pullback_requires_outer_band_touch": float(
                                                                        resolved_pullback_requires_outer_band_touch
                                                                    ),
                                                                    "use_rejection_quality_filter": float(
                                                                        resolved_use_rejection_quality_filter
                                                                    ),
                                                                    "rejection_wick_min_ratio": float(
                                                                        rejection_wick_min_ratio
                                                                    ),
                                                                    "rejection_body_min_ratio": float(
                                                                        rejection_body_min_ratio
                                                                    ),
                                                                    "use_rsi_filter": float(resolved_use_rsi_filter),
                                                                    "rsi_length": float(rsi_length),
                                                                    "rsi_midline": float(rsi_midline),
                                                                    "use_rsi_cross_filter": float(
                                                                        1 if int(use_rsi_cross_filter) > 0 else 0
                                                                    ),
                                                                    "rsi_midline_margin": float(rsi_midline_margin),
                                                                    "use_volume_filter": float(resolved_use_volume_filter),
                                                                    "volume_ma_length": float(volume_ma_length),
                                                                    "volume_multiplier": float(volume_multiplier),
                                                                    "use_atr_stop_buffer": float(
                                                                        resolved_use_atr_stop_buffer
                                                                    ),
                                                                    "atr_length": float(atr_length),
                                                                    "atr_stop_buffer_mult": float(atr_stop_buffer_mult),
                                                                    "signal_cooldown_bars": float(signal_cooldown_bars),
                                                                }
                                                            )
        return profiles
    if strategy_name == "frama_cross":
        frama_fast_period_options = FRAMA_FIXED_GRID_OPTIONS["frama_fast_period"]
        frama_slow_period_options = FRAMA_FIXED_GRID_OPTIONS["frama_slow_period"]
        volume_multiplier_options = FRAMA_FIXED_GRID_OPTIONS["volume_multiplier"]
        return [
            {
                "frama_fast_period": float(frama_fast_period),
                "frama_slow_period": float(frama_slow_period),
                "volume_multiplier": float(volume_multiplier),
            }
            for frama_fast_period, frama_slow_period, volume_multiplier in product(
                frama_fast_period_options,
                frama_slow_period_options,
                volume_multiplier_options,
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
                DUAL_THRUST_FIXED_GRID_OPTIONS["dual_thrust_period"],
                DUAL_THRUST_FIXED_GRID_OPTIONS["dual_thrust_k1"],
                DUAL_THRUST_FIXED_GRID_OPTIONS["dual_thrust_k2"],
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
            float(strategy_profile.get("min_ema_gap_pct", 0.0)),
            int(strategy_profile.get("cross_confirmation_bars", 0.0)),
            float(strategy_profile.get("max_price_extension_pct", 0.0)),
        )
    if strategy_name == "ema_band_rejection":
        return (
            int(strategy_profile.get("ema_fast", 0.0)),
            int(strategy_profile.get("ema_mid", 0.0)),
            int(strategy_profile.get("ema_slow", 0.0)),
            int(strategy_profile.get("slope_lookback", 0.0)),
            float(strategy_profile.get("min_ema_spread_pct", 0.0)),
            float(strategy_profile.get("min_slow_slope_pct", 0.0)),
            int(round(float(strategy_profile.get("pullback_requires_outer_band_touch", 0.0)))),
            int(round(float(strategy_profile.get("use_rejection_quality_filter", 0.0)))),
            float(strategy_profile.get("rejection_wick_min_ratio", 0.0)),
            float(strategy_profile.get("rejection_body_min_ratio", 0.0)),
            int(round(float(strategy_profile.get("use_rsi_filter", 0.0)))),
            int(strategy_profile.get("rsi_length", 0.0)),
            float(strategy_profile.get("rsi_midline", 0.0)),
            int(round(float(strategy_profile.get("use_rsi_cross_filter", 0.0)))),
            float(strategy_profile.get("rsi_midline_margin", 0.0)),
            int(round(float(strategy_profile.get("use_volume_filter", 0.0)))),
            int(strategy_profile.get("volume_ma_length", 0.0)),
            float(strategy_profile.get("volume_multiplier", 0.0)),
            int(round(float(strategy_profile.get("use_atr_stop_buffer", 0.0)))),
            int(strategy_profile.get("atr_length", 0.0)),
            float(strategy_profile.get("atr_stop_buffer_mult", 0.0)),
            int(strategy_profile.get("signal_cooldown_bars", 0.0)),
            int(strategy_profile.get("trend_persistence_bars", 1.0)),
            int(strategy_profile.get("max_pullback_bars", 0.0)),
            float(strategy_profile.get("entry_offset_pct", 0.0)),
            int(round(float(strategy_profile.get("use_late_entry_guard", 0.0)))),
            float(strategy_profile.get("late_entry_max_move_1_bar_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_2_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_3_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_distance_ref_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_distance_fast_ref_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_distance_mid_ref_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_atr_mult", 0.0)),
            int(round(float(strategy_profile.get("use_pullback_reentry", 0.0)))),
            float(strategy_profile.get("pullback_reentry_min_touch", 0.0)),
            int(round(float(strategy_profile.get("pullback_reentry_reconfirm_required", 0.0)))),
        )
    if strategy_name == "frama_cross":
        return (
            int(strategy_profile.get("frama_fast_period", 0.0)),
            int(strategy_profile.get("frama_slow_period", 0.0)),
            float(strategy_profile.get("volume_multiplier", 0.0)),
            float(strategy_profile.get("min_frama_gap_pct", 0.0)),
            float(strategy_profile.get("min_frama_slope_pct", 0.0)),
            int(strategy_profile.get("post_cross_confirmation_bars", 0.0)),
            int(round(float(strategy_profile.get("use_late_entry_guard", 0.0)))),
            float(strategy_profile.get("late_entry_max_move_1_bar_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_2_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_3_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_distance_ref_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_atr_mult", 0.0)),
        )
    if strategy_name == "dual_thrust":
        return (
            int(strategy_profile.get("dual_thrust_period", 0.0)),
            float(strategy_profile.get("dual_thrust_k1", 0.0)),
            float(strategy_profile.get("dual_thrust_k2", 0.0)),
            float(strategy_profile.get("breakout_buffer_pct", 0.0)),
            float(strategy_profile.get("min_range_pct", 0.0)),
            int(strategy_profile.get("cooldown_bars_after_exit", 0.0)),
            int(round(float(strategy_profile.get("use_late_entry_guard", 0.0)))),
            float(strategy_profile.get("late_entry_max_move_1_bar_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_2_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_move_3_bars_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_distance_ref_pct", 0.0)),
            float(strategy_profile.get("late_entry_max_atr_mult", 0.0)),
            float(strategy_profile.get("max_breakout_candle_body_pct", 0.0)),
            float(strategy_profile.get("max_breakout_candle_range_atr_mult", 0.0)),
        )
    return strategy_name


def _candles_to_dataframe(candles: Sequence[CandleRecord]) -> pd.DataFrame:
    try:
        polars_loader = (
            PolarsDataLoader.from_candle_records(candles)
            .with_setup_gate_ema(period=200)
        )
        return _worker_payload_to_pandas_dataframe(polars_loader.to_worker_payload())
    except Exception:
        # Fallback for environments where polars is not installed yet.
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


def _worker_payload_to_pandas_dataframe(
    candles_payload: Mapping[str, object],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list(candles_payload.get("symbol", [])),
            "interval": list(candles_payload.get("interval", [])),
            "open_time": list(candles_payload.get("open_time", [])),
            "open": np.asarray(candles_payload.get("open", []), dtype=np.float64),
            "high": np.asarray(candles_payload.get("high", []), dtype=np.float64),
            "low": np.asarray(candles_payload.get("low", []), dtype=np.float64),
            "close": np.asarray(candles_payload.get("close", []), dtype=np.float64),
            "volume": np.asarray(candles_payload.get("volume", []), dtype=np.float64),
        }
    )


def _candles_dataframe_to_worker_payload(
    candles_df: pd.DataFrame,
) -> dict[str, object]:
    try:
        return PolarsDataLoader.from_pandas_dataframe(candles_df).to_worker_payload()
    except Exception:
        return {
            "symbol": candles_df["symbol"].astype(str).tolist(),
            "interval": candles_df["interval"].astype(str).tolist(),
            "open_time": candles_df["open_time"].tolist(),
            "open": candles_df["open"].to_numpy(dtype=np.float64, copy=False),
            "high": candles_df["high"].to_numpy(dtype=np.float64, copy=False),
            "low": candles_df["low"].to_numpy(dtype=np.float64, copy=False),
            "close": candles_df["close"].to_numpy(dtype=np.float64, copy=False),
            "volume": candles_df["volume"].to_numpy(dtype=np.float64, copy=False),
        }


class _NoopDatabase:
    pass


def _calculate_trade_metrics_for_closed_trades(closed_trades: Sequence[dict]) -> dict[str, float]:
    wins = [float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) > 0.0]
    losses = [-float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) < 0.0]
    fee_costs = [max(0.0, float(trade.get("total_fees", 0.0) or 0.0)) for trade in closed_trades]
    slippage_costs = [
        max(0.0, float(trade.get("slippage_penalty_usd", 0.0) or 0.0))
        for trade in closed_trades
    ]
    combined_costs = [
        float(fee_costs[index]) + float(slippage_costs[index])
        for index in range(len(closed_trades))
    ]

    average_win_usd = sum(wins) / len(wins) if wins else 0.0
    average_loss_usd = sum(losses) / len(losses) if losses else 0.0
    average_trade_fees_usd = sum(fee_costs) / len(fee_costs) if fee_costs else 0.0
    average_trade_slippage_usd = sum(slippage_costs) / len(slippage_costs) if slippage_costs else 0.0
    average_trade_cost_usd = sum(combined_costs) / len(combined_costs) if combined_costs else 0.0
    if average_loss_usd == 0.0:
        real_rrr = float("inf") if average_win_usd > 0.0 else 0.0
    else:
        real_rrr = average_win_usd / average_loss_usd
    if average_trade_cost_usd <= 0.0:
        average_win_to_cost_ratio = float("inf") if average_win_usd > 0.0 else 0.0
    else:
        average_win_to_cost_ratio = average_win_usd / average_trade_cost_usd

    return {
        "average_win_usd": average_win_usd,
        "average_loss_usd": average_loss_usd,
        "real_rrr": real_rrr,
        "average_trade_fees_usd": average_trade_fees_usd,
        "average_trade_slippage_usd": average_trade_slippage_usd,
        "average_trade_cost_usd": average_trade_cost_usd,
        "average_win_to_cost_ratio": average_win_to_cost_ratio,
    }


def _calculate_trade_direction_counts(closed_trades: Sequence[dict]) -> tuple[int, int]:
    long_trades_count = 0
    short_trades_count = 0
    for trade in closed_trades:
        side = str(trade.get("side", "")).strip().upper()
        if side == "LONG":
            long_trades_count += 1
        elif side == "SHORT":
            short_trades_count += 1
    return long_trades_count, short_trades_count


def _extract_ohlcv_numpy_arrays(
    candles_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_arr = pd.to_numeric(candles_df["open"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    high_arr = pd.to_numeric(candles_df["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    low_arr = pd.to_numeric(candles_df["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    close_arr = pd.to_numeric(candles_df["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    volume_arr = pd.to_numeric(candles_df["volume"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return (
        np.ascontiguousarray(open_arr, dtype=np.float64),
        np.ascontiguousarray(high_arr, dtype=np.float64),
        np.ascontiguousarray(low_arr, dtype=np.float64),
        np.ascontiguousarray(close_arr, dtype=np.float64),
        np.ascontiguousarray(volume_arr, dtype=np.float64),
    )


def _extract_ohlcv_numpy_arrays_from_payload(
    candles_payload: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        open_arr = np.ascontiguousarray(np.asarray(candles_payload.get("open", ()), dtype=np.float64))
        high_arr = np.ascontiguousarray(np.asarray(candles_payload.get("high", ()), dtype=np.float64))
        low_arr = np.ascontiguousarray(np.asarray(candles_payload.get("low", ()), dtype=np.float64))
        close_arr = np.ascontiguousarray(np.asarray(candles_payload.get("close", ()), dtype=np.float64))
        volume_arr = np.ascontiguousarray(np.asarray(candles_payload.get("volume", ()), dtype=np.float64))
    except Exception:
        return None
    sample_count = int(close_arr.size)
    if sample_count <= 0:
        return None
    if (
        int(open_arr.size) != sample_count
        or int(high_arr.size) != sample_count
        or int(low_arr.size) != sample_count
        or int(volume_arr.size) != sample_count
    ):
        return None
    return open_arr, high_arr, low_arr, close_arr, volume_arr


def _task_ohlcv_numpy_arrays(
    task: Mapping[str, object],
    *,
    candles_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candles_payload = task.get("candles_payload")
    if isinstance(candles_payload, Mapping):
        payload_arrays = _extract_ohlcv_numpy_arrays_from_payload(candles_payload)
        if payload_arrays is not None:
            return payload_arrays
    worker_arrays = _worker_ohlcv_arrays()
    if worker_arrays is not None:
        return worker_arrays
    effective_df = candles_df if candles_df is not None else _task_candles_dataframe(dict(task))
    return _extract_ohlcv_numpy_arrays(effective_df)


def _warmup_numba_runtime(log_callback: Callable[[str], None] | None = None) -> None:
    global _NUMBA_WARMUP_DONE
    if _NUMBA_WARMUP_DONE:
        return
    dummy_open = np.ascontiguousarray(np.linspace(100.0, 101.0, 10, dtype=np.float64))
    dummy_close = np.ascontiguousarray(np.linspace(100.1, 101.1, 10, dtype=np.float64))
    dummy_high = np.ascontiguousarray(dummy_close + 0.2, dtype=np.float64)
    dummy_low = np.ascontiguousarray(dummy_open - 0.2, dtype=np.float64)
    dummy_volume = np.ascontiguousarray(np.linspace(1000.0, 1200.0, 10, dtype=np.float64))

    # Trigger strategy JIT signatures.
    for strategy_code in (1, 2, 3):
        _ = generate_strategy_signals(
            int(strategy_code),
            dummy_open,
            dummy_high,
            dummy_low,
            dummy_close,
            dummy_volume,
            float(12.0),
            float(60.0),
            float(2.0),
            int(settings.strategy.volume_sma_period),
        )
    _ = compute_ema_band_rejection_signals(
        dummy_open,
        dummy_high,
        dummy_low,
        dummy_close,
        dummy_volume,
        5,
        10,
        20,
        5,
        0.05,
        0.0,
        0,
        0,
        0.35,
        0.20,
        0,
        14,
        50.0,
        0,
        0.0,
        0,
        20,
        1.0,
        0,
    )

    # Trigger core loop signature.
    dummy_signals = np.zeros(10, dtype=np.int8)
    dummy_signals[5] = np.int8(1)
    _ = run_fast_backtest_loop(
        dummy_open,
        dummy_high,
        dummy_low,
        dummy_close,
        dummy_signals,
        float(1.5),
        float(3.0),
        float(settings.trading.default_leverage),
        float(_resolve_backtest_fee_pct()),
        float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
        float(settings.trading.start_capital),
    )

    _NUMBA_WARMUP_DONE = True
    if log_callback is not None:
        with suppress(Exception):
            log_callback("[INFO] Numba JIT Compilation triggered successfully.")


def _jit_strategy_code(strategy_name: str) -> int:
    normalized_name = _validate_strategy_name(strategy_name)
    if normalized_name == "ema_cross_volume":
        return 1
    if normalized_name == "frama_cross":
        return 2
    if normalized_name == "dual_thrust":
        return 3
    if normalized_name == "ema_band_rejection":
        return 4
    return 0


def _build_jit_signal_array(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_name: str,
    strategy_profile: OptimizationProfile,
) -> np.ndarray | None:
    def _profile_float(field_name: str, default_value: float) -> float:
        with suppress(Exception):
            return float(strategy_profile.get(field_name, default_value))
        return float(default_value)

    def _profile_int(field_name: str, default_value: int) -> int:
        with suppress(Exception):
            return int(float(strategy_profile.get(field_name, default_value)))
        return int(default_value)

    def _profile_flag(field_name: str, default_value: bool = False) -> int:
        raw_value = strategy_profile.get(field_name, 1.0 if default_value else 0.0)
        if isinstance(raw_value, bool):
            return 1 if bool(raw_value) else 0
        with suppress(Exception):
            return 1 if float(raw_value) >= 0.5 else 0
        return 1 if default_value else 0

    strategy_code = _jit_strategy_code(strategy_name)
    if strategy_code == 0:
        return None
    # Late-entry guard and pullback-reentry logic are implemented in the
    # Python/vectorized strategy builders. Force Python path when enabled so
    # JIT and Python stay behaviorally aligned.
    use_late_entry_guard = bool(_profile_flag("use_late_entry_guard", False))
    use_pullback_reentry = bool(_profile_flag("use_pullback_reentry", False))
    if use_late_entry_guard:
        return None
    if strategy_name == "ema_band_rejection" and use_pullback_reentry:
        return None

    if strategy_code == 1:
        param_a = _profile_float("ema_fast_period", settings.strategy.ema_fast_period)
        param_b = _profile_float("ema_slow_period", settings.strategy.ema_slow_period)
        param_c = _profile_float("volume_multiplier", settings.strategy.volume_multiplier)
    elif strategy_code == 2:
        param_a = _profile_float("frama_fast_period", settings.strategy.frama_fast_period)
        param_b = _profile_float("frama_slow_period", settings.strategy.frama_slow_period)
        param_c = _profile_float("volume_multiplier", settings.strategy.volume_multiplier)
    elif strategy_code == 3:
        param_a = _profile_float("dual_thrust_period", settings.strategy.dual_thrust_period)
        param_b = _profile_float("dual_thrust_k1", settings.strategy.dual_thrust_k1)
        param_c = _profile_float("dual_thrust_k2", settings.strategy.dual_thrust_k2)
    else:
        raw_signals = compute_ema_band_rejection_signals(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            _profile_int("ema_fast", 5),
            _profile_int("ema_mid", 10),
            _profile_int("ema_slow", 20),
            _profile_int("slope_lookback", 5),
            _profile_float("min_ema_spread_pct", 0.05),
            _profile_float("min_slow_slope_pct", 0.0),
            _profile_flag("pullback_requires_outer_band_touch", False),
            _profile_flag("use_rejection_quality_filter", False),
            _profile_float("rejection_wick_min_ratio", 0.35),
            _profile_float("rejection_body_min_ratio", 0.20),
            _profile_flag("use_rsi_filter", False),
            _profile_int("rsi_length", 14),
            _profile_float("rsi_midline", 50.0),
            _profile_flag("use_rsi_cross_filter", False),
            _profile_float("rsi_midline_margin", 0.0),
            _profile_flag("use_volume_filter", False),
            _profile_int("volume_ma_length", 20),
            _profile_float("volume_multiplier", 1.0),
            _profile_int("signal_cooldown_bars", 0),
        )
        return np.ascontiguousarray(raw_signals, dtype=np.int8)

    raw_signals = generate_strategy_signals(
        int(strategy_code),
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        float(param_a),
        float(param_b),
        float(param_c),
        int(settings.strategy.volume_sma_period),
    )
    return np.ascontiguousarray(raw_signals, dtype=np.int8)


def _build_python_ema_band_reference_signals(
    candles_df: pd.DataFrame,
    strategy_profile: OptimizationProfile,
) -> np.ndarray:
    def _profile_int(field_name: str, default_value: int) -> int:
        with suppress(Exception):
            return int(float(strategy_profile.get(field_name, default_value)))
        return int(default_value)

    def _profile_float(field_name: str, default_value: float) -> float:
        with suppress(Exception):
            return float(strategy_profile.get(field_name, default_value))
        return float(default_value)

    def _profile_flag(field_name: str, default_value: bool = False) -> bool:
        raw_value = strategy_profile.get(field_name, default_value)
        if isinstance(raw_value, bool):
            return bool(raw_value)
        with suppress(Exception):
            return bool(float(raw_value) >= 0.5)
        return bool(default_value)

    working_df = build_ema_band_rejection_signal_frame(
        candles_df,
        ema_fast=_profile_int("ema_fast", 5),
        ema_mid=_profile_int("ema_mid", 10),
        ema_slow=_profile_int("ema_slow", 20),
        slope_lookback=_profile_int("slope_lookback", 5),
        min_ema_spread_pct=_profile_float("min_ema_spread_pct", 0.05),
        min_slow_slope_pct=_profile_float("min_slow_slope_pct", 0.0),
        pullback_requires_outer_band_touch=_profile_flag(
            "pullback_requires_outer_band_touch",
            False,
        ),
        use_rejection_quality_filter=_profile_flag(
            "use_rejection_quality_filter",
            False,
        ),
        rejection_wick_min_ratio=_profile_float("rejection_wick_min_ratio", 0.35),
        rejection_body_min_ratio=_profile_float("rejection_body_min_ratio", 0.20),
        use_rsi_filter=_profile_flag("use_rsi_filter", False),
        rsi_length=_profile_int("rsi_length", 14),
        rsi_midline=_profile_float("rsi_midline", 50.0),
        use_rsi_cross_filter=_profile_flag("use_rsi_cross_filter", False),
        rsi_midline_margin=_profile_float("rsi_midline_margin", 0.0),
        use_volume_filter=_profile_flag("use_volume_filter", False),
        volume_ma_length=_profile_int("volume_ma_length", 20),
        volume_multiplier=_profile_float("volume_multiplier", 1.0),
        use_atr_stop_buffer=_profile_flag("use_atr_stop_buffer", False),
        atr_length=_profile_int("atr_length", 14),
        atr_stop_buffer_mult=_profile_float("atr_stop_buffer_mult", 0.5),
        signal_cooldown_bars=_profile_int("signal_cooldown_bars", 0),
        trend_persistence_bars=_profile_int("trend_persistence_bars", 1),
        max_pullback_bars=_profile_int("max_pullback_bars", 0),
        entry_offset_pct=_profile_float("entry_offset_pct", 0.0),
        use_late_entry_guard=_profile_flag(
            "use_late_entry_guard",
            bool(getattr(settings.strategy, "use_late_entry_guard", False)),
        ),
        late_entry_max_move_1_bar_pct=_profile_float(
            "late_entry_max_move_1_bar_pct",
            float(getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)),
        ),
        late_entry_max_move_2_bars_pct=_profile_float(
            "late_entry_max_move_2_bars_pct",
            float(getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)),
        ),
        late_entry_max_move_3_bars_pct=_profile_float(
            "late_entry_max_move_3_bars_pct",
            float(getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)),
        ),
        late_entry_max_distance_ref_pct=_profile_float(
            "late_entry_max_distance_ref_pct",
            float(getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)),
        ),
        late_entry_max_distance_fast_ref_pct=_profile_float(
            "late_entry_max_distance_fast_ref_pct",
            float(getattr(settings.strategy, "late_entry_max_distance_fast_ref_pct", 0.0)),
        ),
        late_entry_max_distance_mid_ref_pct=_profile_float(
            "late_entry_max_distance_mid_ref_pct",
            float(getattr(settings.strategy, "late_entry_max_distance_mid_ref_pct", 0.0)),
        ),
        late_entry_max_atr_mult=_profile_float(
            "late_entry_max_atr_mult",
            float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
        ),
        use_pullback_reentry=_profile_flag(
            "use_pullback_reentry",
            bool(getattr(settings.strategy, "use_pullback_reentry", False)),
        ),
        pullback_reentry_min_touch=_profile_float(
            "pullback_reentry_min_touch",
            float(getattr(settings.strategy, "pullback_reentry_min_touch", 0.0)),
        ),
        pullback_reentry_reconfirm_required=_profile_flag(
            "pullback_reentry_reconfirm_required",
            bool(getattr(settings.strategy, "pullback_reentry_reconfirm_required", False)),
        ),
    )
    raw_signals = pd.to_numeric(
        working_df["ema_band_signal_direction"],
        errors="coerce",
    ).fillna(0).astype("int8").to_numpy(copy=False)
    return np.ascontiguousarray(raw_signals, dtype=np.int8)


def _validate_ema_band_jit_alignment(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    strategy_profile: OptimizationProfile,
    *,
    mismatch_tolerance_pct: float = 1.0,
) -> tuple[bool, int, int, float]:
    jit_signals = _build_jit_signal_array(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name="ema_band_rejection",
        strategy_profile=strategy_profile,
    )
    if jit_signals is None:
        return False, 0, 0, 100.0
    py_signals = _build_python_ema_band_reference_signals(candles_df, strategy_profile)
    compare_count = int(min(jit_signals.size, py_signals.size))
    if compare_count <= 0:
        return False, 0, 0, 100.0
    mismatch_count = int(np.count_nonzero(jit_signals[:compare_count] != py_signals[:compare_count]))
    mismatch_pct = (float(mismatch_count) / float(compare_count)) * 100.0
    return mismatch_pct <= float(mismatch_tolerance_pct), mismatch_count, compare_count, mismatch_pct


def _build_strategy_jit_precomputed_payload(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_name: str,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
) -> dict[str, object] | None:
    raw_signal_array = _build_jit_signal_array(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name=strategy_name,
        strategy_profile=strategy_profile,
    )
    if raw_signal_array is None:
        return None
    required_candles = _required_candle_count_for_profile(
        strategy_name,
        strategy_profile,
        use_setup_gate=use_setup_gate,
    )
    setup_gate = SmartSetupGate(min_confidence_pct=min_confidence_pct) if use_setup_gate else None
    return _build_vectorized_strategy_cache_payload(
        candles_df,
        strategy_name=strategy_name,
        required_candles=required_candles,
        setup_gate=setup_gate,
        strategy_profile=strategy_profile,
        regime_mask=regime_mask,
        raw_signal_override=raw_signal_array.tolist(),
    )


def _build_ema_band_rejection_jit_precomputed_payload(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
) -> dict[str, object] | None:
    return _build_strategy_jit_precomputed_payload(
        candles_df,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name="ema_band_rejection",
        strategy_profile=strategy_profile,
        use_setup_gate=use_setup_gate,
        min_confidence_pct=min_confidence_pct,
        regime_mask=regime_mask,
    )


def _compare_optimizer_final_summary_metrics(
    optimizer_summary: Mapping[str, object],
    final_summary: Mapping[str, object],
) -> tuple[bool, list[str]]:
    metric_tolerances: dict[str, tuple[float, float]] = {
        "total_trades": (0.0, 0.0),
        "profit_factor": (0.20, 0.10),
        "total_pnl_usd": (5.0, 0.05),
        "win_rate_pct": (1.5, 0.05),
        "max_drawdown_pct": (1.5, 0.10),
        "real_rrr": (0.30, 0.20),
    }
    mismatches: list[str] = []
    for metric_name, (absolute_tolerance, relative_tolerance) in metric_tolerances.items():
        with suppress(Exception):
            optimizer_value = float(optimizer_summary.get(metric_name, 0.0) or 0.0)
            final_value = float(final_summary.get(metric_name, 0.0) or 0.0)
            if not math.isfinite(optimizer_value) and not math.isfinite(final_value):
                continue
            if not math.isfinite(optimizer_value) or not math.isfinite(final_value):
                mismatches.append(
                    f"{metric_name}: optimizer={optimizer_value} final={final_value}"
                )
                continue
            max_allowed_delta = max(
                float(absolute_tolerance),
                abs(final_value) * float(relative_tolerance),
            )
            actual_delta = abs(optimizer_value - final_value)
            if actual_delta > max_allowed_delta:
                mismatches.append(
                    f"{metric_name}: optimizer={optimizer_value:.4f} final={final_value:.4f} "
                    f"(delta={actual_delta:.4f} > tol={max_allowed_delta:.4f})"
                )
    return len(mismatches) == 0, mismatches


def _run_strategy_consistent_profile_evaluation(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_name: str,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
    leverage_override: int | None,
    symbol: str,
    interval: str,
    apply_optimizer_early_stop: bool,
) -> tuple[dict[str, float], int, int, int] | None:
    payload = _build_strategy_jit_precomputed_payload(
        candles_df,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name=strategy_name,
        strategy_profile=strategy_profile,
        use_setup_gate=use_setup_gate,
        min_confidence_pct=min_confidence_pct,
        regime_mask=regime_mask,
    )
    if payload is None:
        return None
    return _run_python_profile_evaluation(
        candles_df,
        strategy_name=strategy_name,
        strategy_profile=strategy_profile,
        use_setup_gate=use_setup_gate,
        min_confidence_pct=min_confidence_pct,
        regime_mask=regime_mask,
        leverage_override=leverage_override,
        symbol=symbol,
        interval=interval,
        precomputed_payload=payload,
        apply_optimizer_early_stop=bool(apply_optimizer_early_stop),
    )


def _run_ema_band_rejection_consistent_profile_evaluation(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
    leverage_override: int | None,
    symbol: str,
    interval: str,
) -> tuple[dict[str, float], int, int, int] | None:
    return _run_strategy_consistent_profile_evaluation(
        candles_df,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name="ema_band_rejection",
        strategy_profile=strategy_profile,
        use_setup_gate=use_setup_gate,
        min_confidence_pct=min_confidence_pct,
        regime_mask=regime_mask,
        leverage_override=leverage_override,
        symbol=symbol,
        interval=interval,
        apply_optimizer_early_stop=False,
    )


def _run_jit_profile_evaluation(
    candles_df: pd.DataFrame,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    *,
    strategy_name: str,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
    leverage_override: int | None,
) -> tuple[dict[str, float], int, int, int] | None:
    raw_signal_array = _build_jit_signal_array(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name=strategy_name,
        strategy_profile=strategy_profile,
    )
    if raw_signal_array is None:
        return None

    required_candles = _required_candle_count_for_strategy(
        strategy_name,
        use_setup_gate=use_setup_gate,
    )
    setup_gate = SmartSetupGate(min_confidence_pct=min_confidence_pct) if use_setup_gate else None
    signal_list, total_signals, approved_signals, blocked_signals = _finalize_signal_series(
        candles_df,
        raw_signal_array.tolist(),
        required_candles=required_candles,
        setup_gate=setup_gate,
        strategy_name=strategy_name,
        regime_mask=regime_mask,
    )
    if not signal_list:
        return None

    signal_arr = np.ascontiguousarray(signal_list, dtype=np.int8)
    effective_leverage = (
        int(leverage_override)
        if leverage_override is not None
        else int(settings.trading.default_leverage)
    )
    (
        total_pnl_usd,
        win_rate_pct,
        max_drawdown_pct,
        total_trades,
        gross_profit,
        gross_loss,
        win_count,
        loss_count,
        long_trades,
        short_trades,
        longest_consecutive_losses,
        total_fees_usd,
        total_slippage_usd,
    ) = run_fast_backtest_loop_detailed(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        signal_arr,
        float(strategy_profile["stop_loss_pct"]),
        float(strategy_profile["take_profit_pct"]),
        float(effective_leverage),
        float(_resolve_backtest_fee_pct()),
        float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
        float(settings.trading.start_capital),
    )
    if total_trades <= 0:
        return None

    if gross_loss <= 0.0:
        profit_factor = float("inf") if gross_profit > 0.0 else 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)
    average_win_usd = float(gross_profit / float(max(win_count, 1)))
    average_loss_usd = float(gross_loss / float(max(loss_count, 1)))
    if average_loss_usd <= 0.0:
        real_rrr = float("inf") if average_win_usd > 0.0 else 0.0
    else:
        real_rrr = float(average_win_usd / average_loss_usd)
    average_trade_fees_usd = float(total_fees_usd / float(total_trades))
    average_trade_slippage_usd = float(total_slippage_usd / float(total_trades))
    average_trade_cost_usd = float(average_trade_fees_usd + average_trade_slippage_usd)
    if average_trade_cost_usd <= 0.0:
        average_win_to_cost_ratio = float("inf") if average_win_usd > 0.0 else 0.0
    else:
        average_win_to_cost_ratio = float(average_win_usd / average_trade_cost_usd)

    summary = {
        **{key: float(value) for key, value in strategy_profile.items()},
        "start_capital_usd": float(settings.trading.start_capital),
        "total_pnl_usd": float(total_pnl_usd),
        "win_rate_pct": float(win_rate_pct),
        "profit_factor": float(profit_factor),
        "total_trades": float(total_trades),
        "long_trades": float(long_trades),
        "short_trades": float(short_trades),
        "average_win_usd": float(average_win_usd),
        "average_loss_usd": float(average_loss_usd),
        "real_rrr": float(real_rrr),
        "average_trade_fees_usd": float(average_trade_fees_usd),
        "average_trade_slippage_usd": float(average_trade_slippage_usd),
        "average_trade_cost_usd": float(average_trade_cost_usd),
        "average_win_to_cost_ratio": float(average_win_to_cost_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "longest_consecutive_losses": float(longest_consecutive_losses),
        "total_slippage_penalty_usd": float(total_slippage_usd),
        "slippage_penalty_pct_per_side": float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
        "slippage_penalty_pct_per_trade": float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE),
    }
    summary["avg_profit_per_trade_net_pct"] = float(_resolve_avg_profit_per_trade_net_pct(summary))
    summary["stability_score"] = _calculate_stability_score(summary)
    return summary, int(total_signals), int(approved_signals), int(blocked_signals)


def _should_early_stop_profile(
    closed_trades: Sequence[dict],
    *,
    start_capital: float,
    max_trades_to_evaluate: int = 30,
    max_drawdown_pct: float = 40.0,
) -> bool:
    if not closed_trades or max_trades_to_evaluate <= 0:
        return False
    if not math.isfinite(start_capital) or start_capital <= 0.0:
        return False

    equity = float(start_capital)
    peak_equity = float(start_capital)
    trades_checked = 0
    for trade in closed_trades:
        if trades_checked >= max_trades_to_evaluate:
            break
        with suppress(Exception):
            equity += float(trade.get("pnl", 0.0) or 0.0)
            trades_checked += 1
            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0.0:
                current_drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
                if current_drawdown_pct > max_drawdown_pct:
                    return True
    return False


def _generate_signals_for_worker(
    candles_df: pd.DataFrame,
    *,
    strategy_name: str,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    strategy_profile: OptimizationProfile | None = None,
    regime_mask: Sequence[int] | None = None,
) -> tuple[list[int], int, int, int]:
    normalized_regime_mask = (
        None
        if regime_mask is None
        else list(regime_mask)[: len(candles_df)]
    )
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
        regime_mask=normalized_regime_mask,
    )
    if vectorized_payload is not None:
        aligned_signals = _align_signal_series_to_candle_count(
            vectorized_payload["signals"],
            candle_count=len(candles_df),
        )
        return (
            aligned_signals,
            int(vectorized_payload["total_signals"]),
            int(vectorized_payload["approved_signals"]),
            int(vectorized_payload["blocked_signals"]),
        )

    effective_signal_profile = strategy_profile
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
            effective_signal_profile,
        )
        if (
            signal_direction != 0
            and normalized_regime_mask is not None
            and index < len(normalized_regime_mask)
            and int(normalized_regime_mask[index]) <= 0
        ):
            signal_direction = 0
        if signal_direction != 0 and setup_gate is not None:
            total_signals += 1
            normalized_direction = 1 if signal_direction > 0 else -1
            is_approved, _score, _reason = setup_gate.evaluate_signal_at_index(
                candles_df,
                index,
                normalized_direction,
                strategy_name,
            )
            if not is_approved:
                blocked_signals += 1
                signal_direction = 0
            else:
                signal_direction = normalized_direction
                approved_signals += 1
        elif signal_direction != 0:
            total_signals += 1
            approved_signals += 1
        signals.append(signal_direction)
    return (
        _align_signal_series_to_candle_count(signals, candle_count=len(candles_df)),
        total_signals,
        approved_signals,
        blocked_signals,
    )


def _task_candles_dataframe(task: dict[str, object], *, copy_deep: bool = False) -> pd.DataFrame:
    candles_payload = task.get("candles_payload")
    if isinstance(candles_payload, Mapping):
        df = _worker_payload_to_pandas_dataframe(candles_payload)
        return df.copy(deep=copy_deep) if copy_deep else df
    candles_records = task.get("candles_records")
    if candles_records is not None:
        # avoid unnecessary deep copies inside worker processes
        df = pd.DataFrame(candles_records)
        return df.copy(deep=copy_deep) if copy_deep else df
    return _worker_candles_dataframe(copy_deep=copy_deep)


def _task_candle_rows(task: dict[str, object]) -> list[dict[str, object]]:
    candles_payload = task.get("candles_payload")
    if isinstance(candles_payload, Mapping):
        candles_df = _worker_payload_to_pandas_dataframe(candles_payload)
        return PaperTradingEngine._extract_backtest_rows(candles_df)
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


def _build_zero_fitness_summary(strategy_profile: OptimizationProfile) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, value in strategy_profile.items():
        with suppress(Exception):
            summary[str(key)] = float(value)
    summary.update(
        {
            "start_capital_usd": float(settings.trading.start_capital),
            "total_pnl_usd": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0.0,
            "long_trades": 0.0,
            "short_trades": 0.0,
            "average_win_usd": 0.0,
            "average_loss_usd": 0.0,
            "real_rrr": 0.0,
            "average_trade_fees_usd": 0.0,
            "average_trade_slippage_usd": 0.0,
            "average_trade_cost_usd": 0.0,
            "average_win_to_cost_ratio": 0.0,
            "avg_profit_per_trade_net_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_consecutive_losses": 0.0,
            "total_slippage_penalty_usd": 0.0,
            "slippage_penalty_pct_per_side": BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE,
            "slippage_penalty_pct_per_trade": BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE,
            "stability_score": 0.0,
        }
    )
    return summary


def _unpack_strategy_signal_payload(packed_payload: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "total_signals": int(packed_payload.get("total_signals", 0) or 0),
        "approved_signals": int(packed_payload.get("approved_signals", 0) or 0),
        "blocked_signals": int(packed_payload.get("blocked_signals", 0) or 0),
    }
    if "signals_bytes" in packed_payload and "signals_len" in packed_payload:
        payload["signals"] = _unpack_int8_series(
            bytes(packed_payload["signals_bytes"]),
            int(packed_payload["signals_len"]),
        )
    else:
        payload["signals"] = list(packed_payload.get("signals", []))

    if (
        "precomputed_long_exit_flags_bytes" in packed_payload
        and "precomputed_long_exit_flags_len" in packed_payload
    ):
        payload["precomputed_long_exit_flags"] = _unpack_bool_series(
            bytes(packed_payload["precomputed_long_exit_flags_bytes"]),
            int(packed_payload["precomputed_long_exit_flags_len"]),
        )
    elif "precomputed_long_exit_flags" in packed_payload:
        payload["precomputed_long_exit_flags"] = list(
            packed_payload.get("precomputed_long_exit_flags", [])
        )

    if (
        "precomputed_short_exit_flags_bytes" in packed_payload
        and "precomputed_short_exit_flags_len" in packed_payload
    ):
        payload["precomputed_short_exit_flags"] = _unpack_bool_series(
            bytes(packed_payload["precomputed_short_exit_flags_bytes"]),
            int(packed_payload["precomputed_short_exit_flags_len"]),
        )
    elif "precomputed_short_exit_flags" in packed_payload:
        payload["precomputed_short_exit_flags"] = list(
            packed_payload.get("precomputed_short_exit_flags", [])
        )

    if (
        "precomputed_dynamic_stop_loss_pcts_bytes" in packed_payload
        and "precomputed_dynamic_stop_loss_pcts_len" in packed_payload
    ):
        payload["precomputed_dynamic_stop_loss_pcts"] = _unpack_float_series(
            bytes(packed_payload["precomputed_dynamic_stop_loss_pcts_bytes"]),
            int(packed_payload["precomputed_dynamic_stop_loss_pcts_len"]),
        )
    elif "precomputed_dynamic_stop_loss_pcts" in packed_payload:
        payload["precomputed_dynamic_stop_loss_pcts"] = list(
            packed_payload.get("precomputed_dynamic_stop_loss_pcts", [])
        )

    if (
        "precomputed_dynamic_take_profit_pcts_bytes" in packed_payload
        and "precomputed_dynamic_take_profit_pcts_len" in packed_payload
    ):
        payload["precomputed_dynamic_take_profit_pcts"] = _unpack_float_series(
            bytes(packed_payload["precomputed_dynamic_take_profit_pcts_bytes"]),
            int(packed_payload["precomputed_dynamic_take_profit_pcts_len"]),
        )
    elif "precomputed_dynamic_take_profit_pcts" in packed_payload:
        payload["precomputed_dynamic_take_profit_pcts"] = list(
            packed_payload.get("precomputed_dynamic_take_profit_pcts", [])
        )
    return payload


def _task_precomputed_signal_payload(task: Mapping[str, object]) -> dict[str, object] | None:
    cache_key = task.get("precomputed_cache_key")
    if cache_key is not None:
        cached_payload = _worker_strategy_signal_group(cache_key)
        if cached_payload is not None:
            return _unpack_strategy_signal_payload(cached_payload)

    precomputed_signals = task.get("precomputed_signals")
    if precomputed_signals is None:
        return None

    return {
        "signals": list(precomputed_signals),
        "total_signals": int(task.get("total_signals", 0) or 0),
        "approved_signals": int(task.get("approved_signals", 0) or 0),
        "blocked_signals": int(task.get("blocked_signals", 0) or 0),
        "precomputed_long_exit_flags": list(task.get("precomputed_long_exit_flags") or []),
        "precomputed_short_exit_flags": list(task.get("precomputed_short_exit_flags") or []),
        "precomputed_dynamic_stop_loss_pcts": list(task.get("precomputed_dynamic_stop_loss_pcts") or []),
        "precomputed_dynamic_take_profit_pcts": list(task.get("precomputed_dynamic_take_profit_pcts") or []),
    }


def _build_worker_backtest_engine(
    *,
    symbol: str,
    interval: str,
    strategy_profile: OptimizationProfile,
    leverage_override: int | None,
) -> PaperTradingEngine:
    def _profile_or_default(field_name: str, default_value: float) -> float:
        raw_value = strategy_profile.get(field_name)
        if raw_value is None:
            return float(default_value)
        with suppress(Exception):
            return float(raw_value)
        return float(default_value)

    breakeven_activation_default = float(LIVE_BREAKEVEN_ACTIVATION_PCT)
    breakeven_buffer_default = float(LIVE_BREAKEVEN_BUFFER_PCT)
    tight_trailing_activation_default = 8.0
    tight_trailing_distance_default = 0.3

    return PaperTradingEngine(
        _NoopDatabase(),
        symbol=symbol,
        interval=interval,
        leverage=leverage_override,
        take_profit_pct=_profile_or_default("take_profit_pct", settings.trading.take_profit_pct),
        stop_loss_pct=_profile_or_default("stop_loss_pct", settings.trading.stop_loss_pct),
        trailing_activation_pct=_profile_or_default(
            "trailing_activation_pct",
            settings.trading.trailing_activation_pct,
        ),
        trailing_distance_pct=_profile_or_default(
            "trailing_distance_pct",
            settings.trading.trailing_distance_pct,
        ),
        breakeven_activation_pct=_profile_or_default(
            "breakeven_activation_pct",
            breakeven_activation_default,
        ),
        breakeven_buffer_pct=_profile_or_default(
            "breakeven_buffer_pct",
            breakeven_buffer_default,
        ),
        tight_trailing_activation_pct=_profile_or_default(
            "tight_trailing_activation_pct",
            tight_trailing_activation_default,
        ),
        tight_trailing_distance_pct=_profile_or_default(
            "tight_trailing_distance_pct",
            tight_trailing_distance_default,
        ),
        fee_pct_override=_resolve_backtest_fee_pct(),
        enable_persistence=False,
    )


def _run_python_profile_evaluation(
    candles_df: pd.DataFrame,
    *,
    strategy_name: str,
    strategy_profile: OptimizationProfile,
    use_setup_gate: bool,
    min_confidence_pct: float | None,
    regime_mask: Sequence[int] | None,
    leverage_override: int | None,
    symbol: str,
    interval: str,
    precomputed_payload: Mapping[str, object] | None = None,
    apply_optimizer_early_stop: bool = True,
) -> tuple[dict[str, float], int, int, int] | None:
    if precomputed_payload is None:
        required_candles = _required_candle_count_for_profile(
            strategy_name,
            strategy_profile,
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
                "signals": signals,
                "total_signals": total_signals,
                "approved_signals": approved_signals,
                "blocked_signals": blocked_signals,
            }
        else:
            payload = {
                "signals": list(vectorized_payload.get("signals", [])),
                "total_signals": int(vectorized_payload.get("total_signals", 0) or 0),
                "approved_signals": int(vectorized_payload.get("approved_signals", 0) or 0),
                "blocked_signals": int(vectorized_payload.get("blocked_signals", 0) or 0),
            }
            for optional_field_name in (
                "precomputed_long_exit_flags",
                "precomputed_short_exit_flags",
                "precomputed_dynamic_stop_loss_pcts",
                "precomputed_dynamic_take_profit_pcts",
            ):
                if optional_field_name in vectorized_payload:
                    payload[optional_field_name] = list(vectorized_payload[optional_field_name])  # type: ignore[index]
    else:
        payload = dict(precomputed_payload)

    aligned_signals = _align_signal_series_to_candle_count(
        payload.get("signals", []),
        candle_count=len(candles_df),
    )
    if not any(int(value) != 0 for value in aligned_signals):
        return None

    candle_rows = PaperTradingEngine._extract_backtest_rows(candles_df)
    precomputed_exit_rule = _build_precomputed_exit_rule(
        candle_rows,
        long_exit_flags=payload.get("precomputed_long_exit_flags"),  # type: ignore[arg-type]
        short_exit_flags=payload.get("precomputed_short_exit_flags"),  # type: ignore[arg-type]
    )
    normalized_dynamic_stop_loss_pcts = _normalize_dynamic_pct_series(
        payload.get("precomputed_dynamic_stop_loss_pcts")
    )
    normalized_dynamic_take_profit_pcts = _normalize_dynamic_pct_series(
        payload.get("precomputed_dynamic_take_profit_pcts")
    )
    resolved_chandelier_period = int(
        float(
            strategy_profile.get(
                "chandelier_period",
                settings.trading.chandelier_period,
            )
            or settings.trading.chandelier_period
        )
    )
    resolved_chandelier_multiplier = float(
        strategy_profile.get(
            "chandelier_multiplier",
            strategy_profile.get(
                "chandelier_mult",
                settings.trading.chandelier_multiplier,
            ),
        )
        or settings.trading.chandelier_multiplier
    )

    engine = _build_worker_backtest_engine(
        symbol=symbol,
        interval=interval,
        strategy_profile=strategy_profile,
        leverage_override=leverage_override,
    )
    result = engine.run_historical_backtest(
        candles_df,
        aligned_signals,
        strategy_exit_rule=precomputed_exit_rule,
        dynamic_stop_loss_pcts=normalized_dynamic_stop_loss_pcts,
        dynamic_take_profit_pcts=normalized_dynamic_take_profit_pcts,
        enable_chandelier_exit=True,
        chandelier_period=resolved_chandelier_period,
        chandelier_multiplier=resolved_chandelier_multiplier,
        strategy_exit_pre_take_profit=False,
        strategy_name=strategy_name,
        early_stop_max_trades=(30 if apply_optimizer_early_stop else None),
        early_stop_max_drawdown_pct=(40.0 if apply_optimizer_early_stop else None),
    )

    total_trades = int(result.get("total_trades", 0) or 0)
    if total_trades <= 0:
        return None

    closed_trades = list(result.get("closed_trades", []))
    trade_metrics = _calculate_trade_metrics_for_closed_trades(closed_trades)
    long_trades, short_trades = _calculate_trade_direction_counts(closed_trades)
    summary: dict[str, float] = {
        **{key: float(value) for key, value in strategy_profile.items()},
        "start_capital_usd": float(settings.trading.start_capital),
        "total_pnl_usd": float(result.get("total_pnl_usd", 0.0) or 0.0),
        "win_rate_pct": float(result.get("win_rate_pct", 0.0) or 0.0),
        "profit_factor": float(result.get("profit_factor", 0.0) or 0.0),
        "total_trades": float(total_trades),
        "long_trades": float(long_trades),
        "short_trades": float(short_trades),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0) or 0.0),
        "longest_consecutive_losses": float(result.get("longest_consecutive_losses", 0) or 0),
        "total_slippage_penalty_usd": float(result.get("total_slippage_penalty_usd", 0.0) or 0.0),
        "slippage_penalty_pct_per_side": float(
            result.get("slippage_penalty_pct_per_side", BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE) or 0.0
        ),
        "slippage_penalty_pct_per_trade": float(
            result.get("slippage_penalty_pct_per_trade", BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE) or 0.0
        ),
        **{key: float(value) for key, value in trade_metrics.items()},
    }
    summary["avg_profit_per_trade_net_pct"] = float(_resolve_avg_profit_per_trade_net_pct(summary))
    summary["stability_score"] = _calculate_stability_score(summary)
    return (
        summary,
        int(payload.get("total_signals", 0) or 0),
        int(payload.get("approved_signals", 0) or 0),
        int(payload.get("blocked_signals", 0) or 0),
    )


def _run_optimization_profile_worker(task: dict[str, object]) -> dict[str, object] | None:
    strategy_name = str(task["strategy_name"])
    force_python_path = bool(task.get("force_python_path", False))
    strategy_profile = dict(task["strategy_profile"])
    if "breakeven_activation_pct" not in strategy_profile:
        minimum_breakeven_activation_pct = (
            _resolve_optimizer_min_breakeven_activation_pct_for_strategy(strategy_name)
        )
        if minimum_breakeven_activation_pct is not None:
            strategy_profile["breakeven_activation_pct"] = float(
                minimum_breakeven_activation_pct
            )

    def _finalize_profile_result(
        summary: dict[str, float],
        *,
        total_signals: int,
        approved_signals: int,
        blocked_signals: int,
    ) -> dict[str, object]:
        guard_pass = True
        with suppress(Exception):
            guard_pass = _passes_common_risk_profile_guards(
                take_profit_pct=float(
                    summary.get("take_profit_pct", strategy_profile.get("take_profit_pct", 0.0))
                    or 0.0
                ),
                stop_loss_pct=float(
                    summary.get("stop_loss_pct", strategy_profile.get("stop_loss_pct", 0.0))
                    or 0.0
                ),
                trailing_activation_pct=float(
                    summary.get(
                        "trailing_activation_pct",
                        strategy_profile.get("trailing_activation_pct", 0.0),
                    )
                    or 0.0
                ),
                trailing_distance_pct=float(
                    summary.get(
                        "trailing_distance_pct",
                        strategy_profile.get("trailing_distance_pct", 0.0),
                    )
                    or 0.0
                ),
                breakeven_activation_pct=float(
                    summary.get(
                        "breakeven_activation_pct",
                        strategy_profile.get("breakeven_activation_pct", 0.0),
                    )
                    or 0.0
                ),
                avg_profit_per_trade_net_pct=float(
                    summary.get("avg_profit_per_trade_net_pct", 0.0) or 0.0
                ),
            )
        normalized_summary = summary if guard_pass else _build_zero_fitness_summary(strategy_profile)
        return {
            "summary": normalized_summary,
            "total_signals": int(total_signals),
            "approved_signals": int(approved_signals),
            "blocked_signals": int(blocked_signals),
        }

    total_signals = int(task.get("total_signals", 0))
    approved_signals = int(task.get("approved_signals", 0))
    blocked_signals = int(task.get("blocked_signals", 0))
    if not _profile_meets_breakeven_constraint(
        strategy_profile,
        strategy_name=strategy_name,
    ):
        return {
            "summary": _build_zero_fitness_summary(strategy_profile),
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
        }

    use_setup_gate = bool(task["use_setup_gate"])
    min_confidence_pct = (
        None if task["min_confidence_pct"] is None else float(task["min_confidence_pct"])
    )
    if use_setup_gate:
        min_confidence_pct = _enforce_optimizer_min_confidence_floor(min_confidence_pct)
    regime_mask = _task_regime_mask(task)
    leverage_override = (
        None if task["leverage_override"] is None else int(task["leverage_override"])
    )
    symbol = str(task.get("symbol", "") or "").strip().upper()
    interval = str(task.get("interval", settings.trading.interval) or settings.trading.interval).strip()

    if (
        strategy_name in OPTIMIZER_JIT_STRATEGY_CONSISTENT_BACKTEST
        and not force_python_path
        and _jit_strategy_code(strategy_name) != 0
    ):
        candles_df = _task_candles_dataframe(task)
        open_arr, high_arr, low_arr, close_arr, volume_arr = _task_ohlcv_numpy_arrays(
            task,
            candles_df=candles_df,
        )
        consistent_result = _run_strategy_consistent_profile_evaluation(
            candles_df,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            strategy_name=strategy_name,
            strategy_profile=strategy_profile,
            use_setup_gate=use_setup_gate,
            min_confidence_pct=min_confidence_pct,
            regime_mask=regime_mask,
            leverage_override=leverage_override,
            symbol=symbol or "BACKTEST",
            interval=interval or str(settings.trading.interval),
            apply_optimizer_early_stop=False,
        )
        if consistent_result is None:
            return None
        summary, total_signals, approved_signals, blocked_signals = consistent_result
        return _finalize_profile_result(
            summary,
            total_signals=int(total_signals),
            approved_signals=int(approved_signals),
            blocked_signals=int(blocked_signals),
        )

    if force_python_path or _jit_strategy_code(strategy_name) == 0:
        python_result = _run_python_profile_evaluation(
            _task_candles_dataframe(task),
            strategy_name=strategy_name,
            strategy_profile=strategy_profile,
            use_setup_gate=use_setup_gate,
            min_confidence_pct=min_confidence_pct,
            regime_mask=regime_mask,
            leverage_override=leverage_override,
            symbol=symbol or "BACKTEST",
            interval=interval or str(settings.trading.interval),
            precomputed_payload=_task_precomputed_signal_payload(task),
            apply_optimizer_early_stop=(strategy_name != "ema_band_rejection"),
        )
        if python_result is None:
            return None
        py_summary, py_total_signals, py_approved_signals, py_blocked_signals = python_result
        return _finalize_profile_result(
            py_summary,
            total_signals=int(py_total_signals),
            approved_signals=int(py_approved_signals),
            blocked_signals=int(py_blocked_signals),
        )

    candles_df = _task_candles_dataframe(task)
    open_arr, high_arr, low_arr, close_arr, volume_arr = _task_ohlcv_numpy_arrays(
        task,
        candles_df=candles_df,
    )

    jit_result = _run_jit_profile_evaluation(
        candles_df,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        strategy_name=strategy_name,
        strategy_profile=strategy_profile,
        use_setup_gate=use_setup_gate,
        min_confidence_pct=min_confidence_pct,
        regime_mask=regime_mask,
        leverage_override=leverage_override,
    )
    if jit_result is None:
        return None
    jit_summary, jit_total_signals, jit_approved_signals, jit_blocked_signals = jit_result
    return _finalize_profile_result(
        jit_summary,
        total_signals=int(jit_total_signals),
        approved_signals=int(jit_approved_signals),
        blocked_signals=int(jit_blocked_signals),
    )


def _run_optimization_profile_chunk_worker(task: dict[str, object]) -> dict[str, object]:
    profile_tasks_raw = task.get("profile_tasks", [])
    if not isinstance(profile_tasks_raw, Sequence):
        return {"chunk_results": [], "submitted_profiles": 0}
    chunk_results: list[dict[str, object]] = []
    submitted_profiles = 0
    for raw_profile_task in profile_tasks_raw:
        if not isinstance(raw_profile_task, Mapping):
            continue
        submitted_profiles += 1
        profile_result = _run_optimization_profile_worker(dict(raw_profile_task))
        if profile_result is not None:
            chunk_results.append(profile_result)
    return {
        "chunk_results": chunk_results,
        "submitted_profiles": int(submitted_profiles),
    }


def _precompute_signal_cache_worker(task: dict[str, object]) -> dict[str, object]:
    candles_df = _task_candles_dataframe(task)
    strategy_name = str(task["strategy_name"])
    strategy_profile = dict(task["strategy_profile"])
    use_setup_gate = bool(task["use_setup_gate"])
    min_confidence_pct = (
        None if task["min_confidence_pct"] is None else float(task["min_confidence_pct"])
    )
    if use_setup_gate:
        min_confidence_pct = _enforce_optimizer_min_confidence_floor(min_confidence_pct)
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
    runtime_profile_update = pyqtSignal(dict)

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
        self._live_candidate_policies: dict[str, dict[str, float | int | str]] = {}
        self._live_candidate_status: dict[str, str] = {}
        self._live_candidate_reason: dict[str, str] = {}
        self._live_candidate_entry_slippage_pct: dict[str, deque[float]] = {}
        self._runtime_regime_cache: dict[tuple[str, str], dict[str, object]] = {}
        self._runtime_hmm_detectors: dict[str, HMMRegimeDetector] = {}
        self._runtime_hmm_non_convergence_warned_at: dict[tuple[str, str], datetime] = {}
        self._runtime_hmm_structural_warned_at: dict[tuple[str, str, str], datetime] = {}
        self._meta_policy_cache: dict[tuple[str, str, str], dict[str, object]] = {}
        self._meta_cooldown_until: dict[tuple[str, str, str], datetime] = {}
        self._meta_last_state_by_key: dict[tuple[str, str, str], str] = {}
        self._meta_fail_safe_active = False
        self._meta_fail_safe_reason = ""
        self._meta_fail_safe_activated_at: datetime | None = None
        self._meta_operational_errors: deque[tuple[datetime, str, str, str, str]] = deque(maxlen=600)
        self._meta_operational_error_last_seen: dict[str, datetime] = {}
        self._meta_operational_error_source_counts: Counter[str] = Counter()
        self._meta_operational_error_signature_counts: Counter[str] = Counter()
        self._meta_operational_error_last_overview_at: datetime | None = None
        self._meta_last_daily_report_date: datetime.date | None = None
        self._meta_last_weekly_report_iso: tuple[int, int] | None = None
        self._meta_service_active = True
        self._meta_global_risk_multiplier = 1.0
        self._symbol_reconcile_required: dict[str, dict[str, object]] = {}
        self._meta_service_degraded: dict[tuple[str, str, str, str], dict[str, object]] = {}
        self._runtime_warning_once_keys: set[str] = set()
        for symbol_name in self._symbols:
            candidate_policy = resolve_live_candidate_policy(symbol_name)
            if candidate_policy is None:
                continue
            self._live_candidate_policies[symbol_name] = dict(candidate_policy)
            self._live_candidate_status[symbol_name] = "ACTIVE"
            self._live_candidate_reason[symbol_name] = ""
            slippage_window_size = max(
                1,
                int(float(candidate_policy.get("fill_slippage_window_size", 20) or 20)),
            )
            self._live_candidate_entry_slippage_pct[symbol_name] = deque(
                maxlen=slippage_window_size
            )

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as exc:
            stacktrace_text = self._emit_exception_context_log(
                component="runtime_thread",
                action="run",
                symbol=(self._symbols[0] if self._symbols else None),
                strategy_name=self._strategy_name,
                interval=(self._intervals[0] if self._intervals else None),
                exc=exc,
                severity="CRITICAL",
            )
            if self._symbols:
                first_symbol = self._symbols[0]
                with suppress(Exception):
                    self._register_operational_error(
                        symbol=first_symbol,
                        interval=self._target_interval_for_symbol(first_symbol),
                        reason="bot_thread_run_exception",
                        exception_text=str(exc),
                        exception_stacktrace=stacktrace_text,
                        source="runtime_exception",
                    )
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
        effective_live_fee_pct = float(LIVE_TAKER_FEE_PCT_STRESS)
        self._paper_engine = PaperTradingEngine(
            self._db,
            symbol=self._symbols[0] if len(self._symbols) == 1 else None,
            interval=self._symbol_intervals.get(self._symbols[0]) if len(self._symbols) == 1 else None,
            leverage=self._leverage_override,
            breakeven_activation_pct=float(LIVE_BREAKEVEN_ACTIVATION_PCT),
            breakeven_buffer_pct=float(LIVE_BREAKEVEN_BUFFER_PCT),
            fee_pct_override=effective_live_fee_pct,
            on_warning=self._on_paper_engine_warning,
            on_critical_state=self._on_paper_engine_critical_state,
        )
        self._initialize_meta_services()
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
        self.log_message.emit(
            "Live fee stress active: "
            f"taker_fee={effective_live_fee_pct:.4f}% per side "
            f"(base={float(settings.trading.taker_fee_pct):.4f}%)."
        )
        self.log_message.emit(
            "Live breakeven profile active: "
            f"activation={float(LIVE_BREAKEVEN_ACTIVATION_PCT):.1f}% "
            f"buffer={float(LIVE_BREAKEVEN_BUFFER_PCT):.2f}%."
        )
        for symbol in self._symbols:
            resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            runtime_settings = self._paper_engine.get_runtime_settings(symbol)
            self._emit_runtime_profile_for_symbol(symbol)
            if symbol in PRODUCTION_PROFILE_REGISTRY:
                profile_strategy = str(PRODUCTION_PROFILE_REGISTRY[symbol].get("strategy_name", ""))
                profile_strategy = STRATEGY_NAME_ALIASES.get(profile_strategy, profile_strategy)
                self.log_message.emit(
                    f"[PRODUCTION] Loaded optimized profile for {symbol} "
                    f"({profile_strategy} + HMM enabled)."
                )
            if self._is_live_candidate_symbol(symbol):
                policy = self._live_candidate_policies.get(symbol, {})
                self.log_message.emit(
                    f"[LIVE_CANDIDATE] {symbol} enabled: "
                    f"strategy={resolved_strategy_name} interval={runtime_settings.interval} "
                    f"target_leverage={int(float(policy.get('target_leverage', runtime_settings.leverage) or runtime_settings.leverage))}x "
                    f"risk_multiplier={float(policy.get('risk_multiplier', 1.0) or 1.0):.2f}."
                )
                candidate_metrics = self._sync_live_candidate_state(
                    symbol,
                    context="startup",
                )
                self._log_live_candidate_snapshot(
                    symbol,
                    event_label="startup",
                    metrics=candidate_metrics,
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
                    breakeven_activation_pct=runtime_settings.breakeven_activation_pct,
                    breakeven_buffer_pct=runtime_settings.breakeven_buffer_pct,
                    tight_trailing_activation_pct=runtime_settings.tight_trailing_activation_pct,
                    tight_trailing_distance_pct=runtime_settings.tight_trailing_distance_pct,
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

        try:
            closed_trade_id = self._paper_engine.close_position_at_price(
                symbol,
                exit_price,
                status="MANUAL_CLOSE",
            )
        except Exception as exc:
            interval = self._target_interval_for_symbol(symbol)
            strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            stacktrace_text = self._emit_exception_context_log(
                component="paper_engine",
                action="close_position_manually",
                symbol=symbol,
                strategy_name=strategy_name,
                interval=interval,
                exc=exc,
                severity="CRITICAL",
            )
            self._set_symbol_reconcile_required(
                symbol=symbol,
                reason="exit_persist_failed",
                strategy_name=strategy_name,
                interval=interval,
                payload={
                    "state": "position_state_unknown",
                    "action": "close_position_manually",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            self._register_operational_error(
                symbol=symbol,
                interval=interval,
                reason="exit_persist_failed",
                exception_text=str(exc),
                exception_stacktrace=stacktrace_text,
                source="runtime_exception",
            )
            return
        if closed_trade_id is None:
            self.log_message.emit(f"Manual close skipped: no open position for {symbol}.")
            return

        self._review_and_recompute_after_close(closed_trade_id)
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
            strategy_profile = self._resolve_live_strategy_profile(
                symbol,
                resolved_strategy_name,
            )
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

            try:
                closed_trade_ids = self._paper_engine.update_positions(
                    candle.close,
                    symbol=symbol,
                    intrabar=False,
                )
            except Exception as exc:
                stacktrace_text = self._emit_exception_context_log(
                    component="paper_engine",
                    action="update_positions",
                    symbol=symbol,
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    exc=exc,
                    severity="CRITICAL",
                )
                self._set_symbol_reconcile_required(
                    symbol=symbol,
                    reason="exit_persist_failed",
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    payload={
                        "state": "position_state_unknown",
                        "action": "update_positions",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                self._register_operational_error(
                    symbol=symbol,
                    interval=candle.interval,
                    reason="exit_persist_failed",
                    exception_text=str(exc),
                    exception_stacktrace=stacktrace_text,
                    source="runtime_exception",
                )
                return
            for trade_id in closed_trade_ids:
                self._review_and_recompute_after_close(trade_id)
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
                candidate_metrics_after_close = self._sync_live_candidate_state(
                    symbol,
                    context="post_close",
                )
                self._log_live_candidate_snapshot(
                    symbol,
                    event_label="post_close",
                    metrics=candidate_metrics_after_close,
                )

            required_candle_count = _required_candle_count_for_strategy(
                resolved_strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
            if isinstance(strategy_profile, Mapping) and strategy_profile:
                with suppress(Exception):
                    required_candle_count = max(
                        int(required_candle_count),
                        int(
                            _required_candle_count_for_profile(
                                resolved_strategy_name,
                                dict(strategy_profile),
                                use_setup_gate=self._use_setup_gate,
                            )
                        ),
                    )
            analysis_window = _live_analysis_window_for_strategy(
                resolved_strategy_name,
                use_setup_gate=self._use_setup_gate,
            )
            analysis_window = max(int(analysis_window), int(required_candle_count) + 3)
            regime_window = max(int(analysis_window), 240)
            recent_candles = self._db.fetch_recent_candles(
                symbol,
                candle.interval,
                limit=regime_window,
            )
            if not recent_candles:
                return

            candles_dataframe_full = _candles_to_dataframe(recent_candles)
            regime_payload = self.resolve_market_regime(
                symbol=symbol,
                timeframe=candle.interval,
                candles_dataframe=candles_dataframe_full,
                observed_time=candle.open_time,
            )
            if len(candles_dataframe_full) < required_candle_count:
                return
            candles_dataframe = candles_dataframe_full.tail(analysis_window).copy(deep=True)
            open_trades = self._db.fetch_open_trades(symbol=symbol)
            if open_trades:
                open_trade = open_trades[0]
                if _should_exit_strategy_position(
                    resolved_strategy_name,
                    candles_dataframe,
                    side=open_trade.side,
                ):
                    try:
                        closed_trade_id = self._paper_engine.close_position_at_price(
                            symbol,
                            candle.close,
                            status="STRATEGY_EXIT",
                        )
                    except Exception as exc:
                        stacktrace_text = self._emit_exception_context_log(
                            component="paper_engine",
                            action="close_position_at_price",
                            symbol=symbol,
                            strategy_name=resolved_strategy_name,
                            interval=candle.interval,
                            exc=exc,
                            severity="CRITICAL",
                        )
                        self._set_symbol_reconcile_required(
                            symbol=symbol,
                            reason="exit_persist_failed",
                            strategy_name=resolved_strategy_name,
                            interval=candle.interval,
                            payload={
                                "state": "position_state_unknown",
                                "action": "close_position_at_price",
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                            },
                        )
                        self._register_operational_error(
                            symbol=symbol,
                            interval=candle.interval,
                            reason="exit_persist_failed",
                            exception_text=str(exc),
                            exception_stacktrace=stacktrace_text,
                            source="runtime_exception",
                        )
                        return
                    if closed_trade_id is not None:
                        self._review_and_recompute_after_close(closed_trade_id)
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

            candidate_state = self._live_candidate_status.get(symbol, "ACTIVE")
            if self._is_live_candidate_symbol(symbol):
                candidate_metrics_pre_signal = self._sync_live_candidate_state(
                    symbol,
                    context="pre_signal",
                )
                candidate_state = self._live_candidate_status.get(symbol, "ACTIVE")
                if candidate_state != "ACTIVE":
                    self._log_live_candidate_snapshot(
                        symbol,
                        event_label="entries_blocked",
                        metrics=candidate_metrics_pre_signal,
                    )
                    return

            execution_price = float(self._latest_market_prices.get(symbol, candle.close))
            (
                signal_direction,
                dynamic_stop_loss_pct,
                dynamic_take_profit_pct,
                signal_diagnostics,
            ) = self._evaluate_live_signal_direction(
                symbol=symbol,
                strategy_name=resolved_strategy_name,
                interval=candle.interval,
                candles_dataframe=candles_dataframe,
                strategy_profile=strategy_profile,
                required_candle_count=required_candle_count,
            )
            signal_name = "LONG" if signal_direction > 0 else "SHORT" if signal_direction < 0 else "NEUTRAL"
            dynamic_risk_parts: list[str] = []
            if dynamic_stop_loss_pct is not None:
                dynamic_risk_parts.append(f"dyn_sl={dynamic_stop_loss_pct:.4f}%")
            if dynamic_take_profit_pct is not None:
                dynamic_risk_parts.append(f"dyn_tp={dynamic_take_profit_pct:.4f}%")
            dynamic_risk_text = (
                " | " + " ".join(dynamic_risk_parts)
                if dynamic_risk_parts
                else ""
            )
            self.log_message.emit(
                f"Signal evaluation: {signal_name} on {symbol} {candle.interval} via {resolved_strategy_name}"
                f"{dynamic_risk_text}"
            )
            latest_blocker_reason = str(signal_diagnostics.get("latest_blocker_reason", "") or "")
            latest_guard_blocked = bool(
                float(signal_diagnostics.get("latest_late_entry_guard_blocked", 0.0) or 0.0)
                >= 0.5
            )
            if signal_direction == 0 and (latest_guard_blocked or latest_blocker_reason):
                latest_move_1 = float(signal_diagnostics.get("latest_move_last_1_bar_pct", 0.0) or 0.0)
                latest_move_2 = float(signal_diagnostics.get("latest_move_last_2_bars_pct", 0.0) or 0.0)
                latest_move_3 = float(signal_diagnostics.get("latest_move_last_3_bars_pct", 0.0) or 0.0)
                latest_distance_ref = float(
                    signal_diagnostics.get("latest_distance_to_reference_pct", 0.0) or 0.0
                )
                latest_atr_extension = float(
                    signal_diagnostics.get("latest_atr_extension_mult", 0.0) or 0.0
                )
                self.log_message.emit(
                    "Signal blocked by late-entry guard: "
                    f"symbol={symbol} strategy={resolved_strategy_name} interval={candle.interval} "
                    f"reason={latest_blocker_reason or 'overextended_from_reference'} "
                    f"move1={latest_move_1:.4f}% move2={latest_move_2:.4f}% move3={latest_move_3:.4f}% "
                    f"dist_ref={latest_distance_ref:.4f}% atr_ext={latest_atr_extension:.4f}x"
                )
            if self._is_live_candidate_symbol(symbol):
                self.log_message.emit(
                    f"[LIVE_CANDIDATE] {symbol} signal={signal_name} strategy={resolved_strategy_name} interval={candle.interval}"
                )
            setup_gate_score: float | None = None
            setup_gate_reason = ""
            if signal_direction != 0 and self._setup_gate is not None:
                is_approved, score, reason = self._setup_gate.evaluate_signal_at_index(
                    candles_dataframe,
                    len(candles_dataframe) - 1,
                    signal_direction,
                    resolved_strategy_name,
                )
                setup_gate_score = float(score)
                setup_gate_reason = str(reason or "")
                if not is_approved:
                    self.log_message.emit(
                        f"Signal filtered by Setup Gate (Score: {score:.1f}%): {reason}"
                    )
                    return
            if signal_direction == 0:
                return
            meta_policy = self.evaluate_meta_policy(
                symbol=symbol,
                strategy_name=resolved_strategy_name,
                interval=candle.interval,
                current_context={
                    "regime_payload": dict(regime_payload),
                    "signal_diagnostics": dict(signal_diagnostics),
                    "candidate_state": candidate_state,
                },
            )
            if not bool(meta_policy.get("allow_trade", True)):
                self.log_message.emit(
                    "[META] entry blocked: "
                    f"symbol={symbol} strategy={resolved_strategy_name} interval={candle.interval} "
                    f"state={meta_policy.get('state')} "
                    f"reason={meta_policy.get('block_reason', 'blocked_by_meta_policy')}"
                )
                self.build_learning_log_entry(
                    change_type="Regime-Veto aktiv"
                    if "regime_veto" in str(meta_policy.get("block_reason", ""))
                    else "Symbol pausiert",
                    symbol=symbol,
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    details={
                        "meta_policy": dict(meta_policy),
                    },
                )
                return
            meta_warning_reason = str(meta_policy.get("warning_reason", "") or "")
            if meta_warning_reason:
                self.log_message.emit(
                    f"[META] warning: {symbol} {resolved_strategy_name} {candle.interval} -> {meta_warning_reason}"
                )
            profile_version = self._resolve_profile_version(
                symbol=symbol,
                strategy_name=resolved_strategy_name,
                timeframe=candle.interval,
            )
            leverage_scale, risk_multiplier = self._candidate_trade_controls(symbol)
            meta_risk_multiplier = float(meta_policy.get("risk_multiplier", 1.0) or 1.0)
            combined_risk_multiplier = max(0.0, float(risk_multiplier) * max(0.0, meta_risk_multiplier))
            if combined_risk_multiplier <= 0.0:
                self.log_message.emit(
                    f"[META] entry blocked due to zero combined risk multiplier: {symbol} {resolved_strategy_name}"
                )
                return
            runtime_settings = self._paper_engine.get_runtime_settings(symbol)
            effective_leverage = max(
                1,
                int(round(float(runtime_settings.leverage) * float(leverage_scale))),
            )
            entry_snapshot = self.build_entry_snapshot(
                symbol=symbol,
                strategy_name=resolved_strategy_name,
                timeframe=candle.interval,
                side=("LONG" if signal_direction > 0 else "SHORT"),
                entry_time=datetime.now(tz=UTC).replace(tzinfo=None),
                entry_price=execution_price,
                leverage_scale=float(effective_leverage),
                profile_version=profile_version,
                regime_payload=regime_payload,
                signal_direction=signal_direction,
                signal_diagnostics=signal_diagnostics,
                candles_dataframe=candles_dataframe,
                setup_gate_score=setup_gate_score,
                setup_gate_reason=setup_gate_reason,
                candidate_state=candidate_state,
            )
            entry_snapshot["meta_flags"] = list(meta_policy.get("meta_flags", []))
            entry_snapshot["meta_policy"] = dict(meta_policy.get("effective_policy_json", {}))
            entry_snapshot["risk_multiplier_effective"] = float(combined_risk_multiplier)
            try:
                trade_id = self._paper_engine.process_signal(
                    symbol=symbol,
                    current_price=execution_price,
                    signal_direction=signal_direction,
                    strategy_name=resolved_strategy_name,
                    leverage_scale=leverage_scale,
                    risk_multiplier=combined_risk_multiplier,
                    dynamic_stop_loss_pct=dynamic_stop_loss_pct,
                    dynamic_take_profit_pct=dynamic_take_profit_pct,
                    timeframe=candle.interval,
                    regime_label_at_entry=str(entry_snapshot.get("regime_label_at_entry", "")),
                    regime_confidence=float(entry_snapshot.get("regime_confidence", 0.0) or 0.0),
                    session_label=str(entry_snapshot.get("session_label", "")),
                    signal_strength=float(entry_snapshot.get("signal_strength", 0.0) or 0.0),
                    confidence_score=float(entry_snapshot.get("confidence_score", 0.0) or 0.0),
                    atr_pct_at_entry=float(entry_snapshot.get("atr_pct_at_entry", 0.0) or 0.0),
                    volume_ratio_at_entry=float(entry_snapshot.get("volume_ratio_at_entry", 0.0) or 0.0),
                    spread_estimate=float(entry_snapshot.get("spread_estimate", 0.0) or 0.0),
                    move_already_extended_pct=float(entry_snapshot.get("move_already_extended_pct", 0.0) or 0.0),
                    entry_snapshot_json=entry_snapshot,
                    lifecycle_snapshot_json=None,
                    profile_version=profile_version,
                    review_status="PENDING",
                )
            except Exception as exc:
                stacktrace_text = self._emit_exception_context_log(
                    component="paper_engine",
                    action="process_signal",
                    symbol=symbol,
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    exc=exc,
                    severity="CRITICAL",
                )
                self._set_symbol_reconcile_required(
                    symbol=symbol,
                    reason="order_submit_failed",
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    payload={
                        "state": "position_state_unknown",
                        "action": "process_signal",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                self._register_operational_error(
                    symbol=symbol,
                    interval=candle.interval,
                    reason="order_submit_failed",
                    exception_text=str(exc),
                    exception_stacktrace=stacktrace_text,
                    source="runtime_exception",
                )
                return
            if trade_id is None:
                return

            trade = self._db.fetch_trade_by_id(trade_id)
            if trade is None:
                self.log_message.emit(f"Trade opened but could not be reloaded: id={trade_id}")
                self._set_symbol_reconcile_required(
                    symbol=symbol,
                    reason="position_state_unknown",
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    payload={
                        "state": "position_state_unknown",
                        "action": "fetch_trade_after_open",
                        "trade_id": int(trade_id),
                    },
                )
                return

            self.trade_opened.emit(self._trade_to_payload(trade))
            self._emit_positions_snapshot()
            self.log_message.emit(
                "Trade opened: "
                f"id={trade.id} symbol={trade.symbol} side={trade.side} leverage={trade.leverage}x "
                f"entry_price={trade.entry_price:.4f} signal_close={candle.close:.4f} "
                f"qty={trade.qty:.6f} entry_fee={trade.total_fees:.4f} high_water_mark={trade.high_water_mark:.4f}"
            )
            if self._is_live_candidate_symbol(symbol):
                close_reference = max(float(candle.close), 1e-9)
                entry_slippage_pct = abs(float(trade.entry_price) - float(candle.close)) / close_reference * 100.0
                self._live_candidate_entry_slippage_pct.setdefault(symbol, deque(maxlen=20)).append(
                    float(entry_slippage_pct)
                )
                candidate_metrics_post_entry = self._sync_live_candidate_state(
                    symbol,
                    context="post_entry",
                )
                self._log_live_candidate_snapshot(
                    symbol,
                    event_label="post_entry",
                    metrics=candidate_metrics_post_entry,
                )
        except Exception as exc:
            stacktrace_text = self._emit_exception_context_log(
                component="runtime",
                action="on_candle_closed",
                symbol=symbol,
                strategy_name=resolved_strategy_name if "resolved_strategy_name" in locals() else self._strategy_name,
                interval=candle.interval,
                exc=exc,
                severity="ERROR",
            )
            self._register_operational_error(
                symbol=symbol,
                interval=candle.interval,
                reason="candle_closed_runtime_error",
                exception_text=str(exc),
                exception_stacktrace=stacktrace_text,
                source="runtime_exception",
            )
            self.log_message.emit(
                f"[META_FAILSAFE_CHECK] runtime error on candle processing for {symbol} {candle.interval}: {exc}"
            )
        finally:
            self._last_closed_candle_times[(symbol, candle.interval)] = candle.open_time

    async def _on_candle_update(self, symbol: str, candle: CandleRecord) -> None:
        if self._db is None or self._paper_engine is None:
            return
        target_interval = self._target_interval_for_symbol(symbol)
        if candle.interval != target_interval:
            return
        self._latest_market_prices[symbol] = float(candle.close)
        self.price_update.emit(symbol, candle.close)
        try:
            try:
                closed_trade_ids = self._paper_engine.update_positions(
                    candle.close,
                    symbol=symbol,
                    intrabar=True,
                )
            except Exception as exc:
                resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
                stacktrace_text = self._emit_exception_context_log(
                    component="paper_engine",
                    action="update_positions_intrabar",
                    symbol=symbol,
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    exc=exc,
                    severity="CRITICAL",
                )
                self._set_symbol_reconcile_required(
                    symbol=symbol,
                    reason="exit_persist_failed",
                    strategy_name=resolved_strategy_name,
                    interval=candle.interval,
                    payload={
                        "state": "position_state_unknown",
                        "action": "update_positions_intrabar",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                self._register_operational_error(
                    symbol=symbol,
                    interval=candle.interval,
                    reason="exit_persist_failed",
                    exception_text=str(exc),
                    exception_stacktrace=stacktrace_text,
                    source="runtime_exception",
                )
                return
            for trade_id in closed_trade_ids:
                self._review_and_recompute_after_close(trade_id)
                closed_trade = self._db.fetch_trade_by_id(trade_id)
                if closed_trade is None:
                    self.log_message.emit(
                        f"[INTRABAR_EXIT] Trade closed: id={trade_id} symbol={symbol} price={candle.close:.4f}"
                    )
                    continue
                self.log_message.emit(
                    "[INTRABAR_EXIT] Trade closed: "
                    f"id={closed_trade.id} symbol={closed_trade.symbol} price={candle.close:.4f} "
                    f"reason={closed_trade.status} net_pnl={closed_trade.pnl:.4f} fees={closed_trade.total_fees:.4f}"
                )
            if closed_trade_ids:
                self._emit_positions_snapshot()
                if self._is_live_candidate_symbol(symbol):
                    candidate_metrics_after_close = self._sync_live_candidate_state(
                        symbol,
                        context="intrabar_close",
                    )
                    self._log_live_candidate_snapshot(
                        symbol,
                        event_label="intrabar_close",
                        metrics=candidate_metrics_after_close,
                    )
        except Exception as exc:
            stacktrace_text = self._emit_exception_context_log(
                component="runtime",
                action="on_candle_update",
                symbol=symbol,
                strategy_name=resolve_strategy_for_symbol(symbol, self._strategy_name),
                interval=candle.interval,
                exc=exc,
                severity="ERROR",
            )
            self._register_operational_error(
                symbol=symbol,
                interval=candle.interval,
                reason="candle_update_runtime_error",
                exception_text=str(exc),
                exception_stacktrace=stacktrace_text,
                source="runtime_exception",
            )
            self.log_message.emit(
                f"[META_FAILSAFE_CHECK] runtime error on candle update for {symbol} {candle.interval}: {exc}"
            )

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
        try:
            self._maybe_generate_periodic_meta_reports()
            self._clear_meta_service_degraded(
                service_name="meta_report_heartbeat",
                symbol="GLOBAL",
                strategy_name="meta_bootstrap",
                interval="global",
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="meta_report_heartbeat",
                symbol="GLOBAL",
                strategy_name="meta_bootstrap",
                interval="global",
                exc=exc,
            )

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
            resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            meta_policy = self.evaluate_meta_policy(
                symbol=symbol,
                strategy_name=resolved_strategy_name,
                interval=interval,
                current_context={},
            )

            status_payload[symbol] = {
                "status": symbol_status,
                "latest_open_time": None if latest_time is None else latest_time.isoformat(),
                "latest_interval": latest_interval,
                "lag_seconds": worst_lag_seconds,
                "interval_statuses": interval_statuses,
                "meta_state": str(meta_policy.get("state", "unknown")),
                "meta_allow_trade": bool(meta_policy.get("allow_trade", True)),
                "meta_risk_multiplier": float(meta_policy.get("risk_multiplier", 1.0) or 1.0),
                "meta_block_reason": str(meta_policy.get("block_reason", "") or ""),
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
                resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
                self._emit_runtime_profile_for_symbol(symbol)
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
                        breakeven_activation_pct=runtime_settings.breakeven_activation_pct,
                        breakeven_buffer_pct=runtime_settings.breakeven_buffer_pct,
                        tight_trailing_activation_pct=runtime_settings.tight_trailing_activation_pct,
                        tight_trailing_distance_pct=runtime_settings.tight_trailing_distance_pct,
                    )
                )
                if self._is_live_candidate_symbol(symbol):
                    candidate_metrics = self._sync_live_candidate_state(
                        symbol,
                        context="leverage_update",
                    )
                    self._log_live_candidate_snapshot(
                        symbol,
                        event_label="leverage_update",
                        metrics=candidate_metrics,
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

    def _on_paper_engine_warning(self, warning_code: str, payload: Mapping[str, object]) -> None:
        warning_payload = dict(payload or {})
        symbol = str(warning_payload.get("symbol", "") or "").strip().upper()
        interval = str(
            warning_payload.get("interval", "")
            or (self._target_interval_for_symbol(symbol) if symbol else "")
        ).strip()
        message = str(warning_payload.get("message", "") or "")
        self.log_message.emit(
            "[WARN] paper_engine warning: "
            f"code={str(warning_code).strip()} "
            f"symbol={symbol or '-'} interval={interval or '-'} "
            f"message={message or '-'}"
        )
        stacktrace_text = str(warning_payload.get("exception_stacktrace", "") or "").strip()
        if stacktrace_text:
            self.log_message.emit(
                f"[WARN] paper_engine warning traceback:\n{stacktrace_text}"
            )
        if symbol and interval:
            self._register_operational_error(
                symbol=symbol,
                interval=interval,
                reason=f"paper_engine_warning:{str(warning_code).strip()}",
                exception_text=message or str(warning_code),
                exception_stacktrace=(stacktrace_text if stacktrace_text else None),
                source="paper_engine_warning",
            )

    def _on_paper_engine_critical_state(self, state_code: str, payload: Mapping[str, object]) -> None:
        state_payload = dict(payload or {})
        symbol = str(state_payload.get("symbol", "") or "").strip().upper()
        if not symbol:
            return
        strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
        interval = str(
            state_payload.get("interval", "")
            or self._target_interval_for_symbol(symbol)
        ).strip()
        reason = str(state_payload.get("reason", "") or str(state_code).strip() or "critical_state")
        self._set_symbol_reconcile_required(
            symbol=symbol,
            reason=reason,
            strategy_name=strategy_name,
            interval=interval,
            payload={
                "state_code": str(state_code),
                **state_payload,
            },
        )
        message = str(state_payload.get("message", "") or reason)
        stacktrace_text = str(state_payload.get("exception_stacktrace", "") or "").strip()
        self._register_operational_error(
            symbol=symbol,
            interval=interval,
            reason=reason,
            exception_text=message,
            exception_stacktrace=(stacktrace_text if stacktrace_text else None),
            source="paper_engine",
        )

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
                            should_abort=lambda: self.isInterruptionRequested(),
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

    def _is_live_candidate_symbol(self, symbol: str) -> bool:
        return str(symbol).strip().upper() in self._live_candidate_policies

    def _resolve_live_strategy_profile(
        self,
        symbol: str,
        strategy_name: str,
    ) -> OptimizationProfile | None:
        normalized_symbol = str(symbol).strip().upper()
        strategy_params = settings.strategy.coin_strategy_params.get(normalized_symbol)
        if not isinstance(strategy_params, Mapping) or not strategy_params:
            return None
        if _sanitize_backtest_strategy_name(strategy_name) == "ema_band_rejection":
            return {
                str(field_name): float(field_value)
                for field_name, field_value in strategy_params.items()
            }
        return {
            str(field_name): float(field_value)
            for field_name, field_value in strategy_params.items()
        }

    def _candidate_trade_controls(self, symbol: str) -> tuple[float, float]:
        normalized_symbol = str(symbol).strip().upper()
        policy = self._live_candidate_policies.get(normalized_symbol)
        if policy is None or self._paper_engine is None:
            return 1.0, 1.0
        runtime_settings = self._paper_engine.get_runtime_settings(normalized_symbol)
        base_leverage = max(1, int(runtime_settings.leverage))
        target_leverage = max(
            1,
            int(float(policy.get("target_leverage", base_leverage) or base_leverage)),
        )
        leverage_scale = float(target_leverage) / float(base_leverage)
        risk_multiplier = float(policy.get("risk_multiplier", 1.0) or 1.0)
        if not math.isfinite(leverage_scale) or leverage_scale <= 0.0:
            leverage_scale = 1.0
        if not math.isfinite(risk_multiplier) or risk_multiplier <= 0.0:
            risk_multiplier = 1.0
        return leverage_scale, risk_multiplier

    def _fetch_recent_closed_trades(
        self,
        symbol: str,
        *,
        limit: int,
    ) -> list[PaperTrade]:
        if self._db is None:
            return []
        normalized_symbol = str(symbol).strip().upper()
        resolved_strategy_name = resolve_strategy_for_symbol(
            normalized_symbol,
            self._strategy_name,
        )
        resolved_interval = self._target_interval_for_symbol(normalized_symbol)
        try:
            rows = self._db.fetch_recent_closed_trades(normalized_symbol, limit=limit)
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="closed_trade_fetch",
                symbol=normalized_symbol,
                strategy_name=resolved_strategy_name,
                interval=resolved_interval,
                exc=exc,
            )
            return []
        self._clear_meta_service_degraded(
            service_name="closed_trade_fetch",
            symbol=normalized_symbol,
            strategy_name=resolved_strategy_name,
            interval=resolved_interval,
        )
        return rows

    def _compute_live_candidate_metrics(
        self,
        symbol: str,
    ) -> dict[str, float]:
        normalized_symbol = str(symbol).strip().upper()
        policy = self._live_candidate_policies.get(normalized_symbol)
        if policy is None:
            return {
                "trade_count": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor_window": 0.0,
                "loss_streak": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_fill_slippage_pct": 0.0,
                "window_trades": 0.0,
            }

        pf_window_size = max(1, int(float(policy.get("pf_window_size", 20) or 20)))
        closed_trades_desc = self._fetch_recent_closed_trades(
            normalized_symbol,
            limit=max(250, pf_window_size),
        )
        trade_count = int(len(closed_trades_desc))
        win_count = 0
        for trade in closed_trades_desc:
            trade_pnl = float(trade.pnl or 0.0)
            if trade_pnl > 0.0:
                win_count += 1
        win_rate_pct = (float(win_count) / float(trade_count) * 100.0) if trade_count > 0 else 0.0

        window_trades = min(pf_window_size, trade_count)
        window_slice = closed_trades_desc[:window_trades]
        gross_profit_window = 0.0
        gross_loss_window = 0.0
        for trade in window_slice:
            trade_pnl = float(trade.pnl or 0.0)
            if trade_pnl > 0.0:
                gross_profit_window += trade_pnl
            elif trade_pnl < 0.0:
                gross_loss_window += abs(trade_pnl)
        if gross_loss_window <= 0.0:
            profit_factor_window = float("inf") if gross_profit_window > 0.0 else 0.0
        else:
            profit_factor_window = gross_profit_window / gross_loss_window

        loss_streak = 0
        for trade in closed_trades_desc:
            trade_pnl = float(trade.pnl or 0.0)
            if trade_pnl < 0.0:
                loss_streak += 1
            else:
                break

        equity = float(settings.trading.start_capital)
        peak_equity = float(equity)
        max_drawdown_pct = 0.0
        for trade in reversed(closed_trades_desc):
            equity += float(trade.pnl or 0.0)
            if equity > peak_equity:
                peak_equity = float(equity)
            elif peak_equity > 0.0:
                drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
                if drawdown_pct > max_drawdown_pct:
                    max_drawdown_pct = float(drawdown_pct)

        slippage_values = list(
            self._live_candidate_entry_slippage_pct.get(normalized_symbol, deque())
        )
        avg_fill_slippage_pct = (
            float(sum(slippage_values) / len(slippage_values))
            if slippage_values
            else 0.0
        )
        return {
            "trade_count": float(trade_count),
            "win_rate_pct": float(win_rate_pct),
            "profit_factor_window": float(profit_factor_window),
            "loss_streak": float(loss_streak),
            "max_drawdown_pct": float(max_drawdown_pct),
            "avg_fill_slippage_pct": float(avg_fill_slippage_pct),
            "window_trades": float(window_trades),
        }

    def _evaluate_live_candidate_status(
        self,
        symbol: str,
    ) -> tuple[str, str, dict[str, float]]:
        normalized_symbol = str(symbol).strip().upper()
        policy = self._live_candidate_policies.get(normalized_symbol)
        metrics = self._compute_live_candidate_metrics(normalized_symbol)
        if policy is None:
            return "ACTIVE", "", metrics

        status = "ACTIVE"
        reason = ""
        max_loss_streak = int(float(policy.get("max_loss_streak", 4) or 4))
        max_drawdown_pct = float(policy.get("max_drawdown_pct", 6.0) or 6.0)
        pf_window_size = int(float(policy.get("pf_window_size", 20) or 20))
        pf_minimum = float(policy.get("pf_minimum", 1.0) or 1.0)
        max_avg_fill_slippage_pct = float(
            policy.get("max_avg_fill_slippage_pct", 0.15) or 0.15
        )
        fill_slippage_window_size = int(
            float(policy.get("fill_slippage_window_size", 20) or 20)
        )

        if int(metrics["loss_streak"]) >= max_loss_streak:
            status = "PAUSED"
            reason = (
                f"loss_streak={int(metrics['loss_streak'])} reached limit {max_loss_streak}"
            )
        elif float(metrics["max_drawdown_pct"]) > max_drawdown_pct:
            status = "PAUSED"
            reason = (
                f"profile_drawdown={float(metrics['max_drawdown_pct']):.2f}% exceeded {max_drawdown_pct:.2f}%"
            )
        elif (
            int(metrics["window_trades"]) >= pf_window_size
            and math.isfinite(float(metrics["profit_factor_window"]))
            and float(metrics["profit_factor_window"]) < pf_minimum
        ):
            status = "KILL_SWITCHED"
            reason = (
                f"pf_window={float(metrics['profit_factor_window']):.2f} below {pf_minimum:.2f} "
                f"for last {pf_window_size} trades"
            )
        elif (
            len(self._live_candidate_entry_slippage_pct.get(normalized_symbol, deque()))
            >= fill_slippage_window_size
            and float(metrics["avg_fill_slippage_pct"]) > max_avg_fill_slippage_pct
        ):
            status = "KILL_SWITCHED"
            reason = (
                f"avg_fill_slippage={float(metrics['avg_fill_slippage_pct']):.3f}% exceeded "
                f"{max_avg_fill_slippage_pct:.3f}%"
            )

        previous_status = self._live_candidate_status.get(normalized_symbol, "ACTIVE")
        if previous_status == "KILL_SWITCHED":
            status = "KILL_SWITCHED"
            if not reason:
                reason = self._live_candidate_reason.get(normalized_symbol, "")
        return status, reason, metrics

    def _emit_runtime_profile_for_symbol(self, symbol: str) -> None:
        if self._paper_engine is None:
            return
        resolved_strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
        runtime_settings = self._paper_engine.get_runtime_settings(symbol)
        candidate_status = self._live_candidate_status.get(symbol)
        is_candidate = bool(self._is_live_candidate_symbol(symbol))
        candidate_policy = self._live_candidate_policies.get(symbol, {})
        payload_effective_leverage = int(runtime_settings.leverage)
        candidate_target_leverage = None
        if is_candidate:
            candidate_target_leverage = max(
                1,
                int(float(candidate_policy.get("target_leverage", payload_effective_leverage) or payload_effective_leverage)),
            )
            payload_effective_leverage = int(candidate_target_leverage)
        meta_key = self._meta_key(symbol, resolved_strategy_name, runtime_settings.interval)
        meta_policy = self._meta_policy_cache.get(meta_key, {})
        payload = {
            "symbol": symbol,
            "resolved_strategy_name": resolved_strategy_name,
            "resolved_interval": runtime_settings.interval,
            "effective_leverage": int(payload_effective_leverage),
            "configured_leverage": int(self._configured_leverage),
            "production_profile_loaded": bool(symbol in PRODUCTION_PROFILE_REGISTRY),
            "live_candidate": bool(is_candidate),
            "live_candidate_status": (
                str(candidate_status) if candidate_status else None
            ),
            "live_candidate_reason": (
                str(self._live_candidate_reason.get(symbol, "") or "")
            ),
            "live_candidate_target_leverage": candidate_target_leverage,
            "meta_service_active": bool(self._meta_service_active),
            "meta_state": str(meta_policy.get("state", "unknown") or "unknown"),
            "meta_allow_trade": bool(meta_policy.get("allow_trade", True)),
            "meta_risk_multiplier": float(meta_policy.get("risk_multiplier", 1.0) or 1.0),
        }
        self.runtime_profile_update.emit(payload)

    def _sync_live_candidate_state(
        self,
        symbol: str,
        *,
        context: str,
    ) -> dict[str, float]:
        normalized_symbol = str(symbol).strip().upper()
        if not self._is_live_candidate_symbol(normalized_symbol):
            return {}
        status, reason, metrics = self._evaluate_live_candidate_status(normalized_symbol)
        previous_status = self._live_candidate_status.get(normalized_symbol, "ACTIVE")
        previous_reason = self._live_candidate_reason.get(normalized_symbol, "")
        self._live_candidate_status[normalized_symbol] = status
        self._live_candidate_reason[normalized_symbol] = reason
        if status != previous_status or reason != previous_reason:
            if status in {"PAUSED", "KILL_SWITCHED"}:
                self.log_message.emit(
                    f"[LIVE_CANDIDATE] {normalized_symbol} {status}: {reason}"
                )
            else:
                self.log_message.emit(
                    f"[LIVE_CANDIDATE] {normalized_symbol} ACTIVE: constraints restored ({context})."
                )
        self._emit_runtime_profile_for_symbol(normalized_symbol)
        return metrics

    def _log_live_candidate_snapshot(
        self,
        symbol: str,
        *,
        event_label: str,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not self._is_live_candidate_symbol(normalized_symbol):
            return
        resolved_metrics = dict(metrics or self._compute_live_candidate_metrics(normalized_symbol))
        status = self._live_candidate_status.get(normalized_symbol, "ACTIVE")
        self.log_message.emit(
            f"[LIVE_CANDIDATE] {normalized_symbol} {event_label} | "
            f"status={status} | "
            f"trades={int(resolved_metrics.get('trade_count', 0.0) or 0.0)} | "
            f"win_rate={float(resolved_metrics.get('win_rate_pct', 0.0) or 0.0):.2f}% | "
            f"pf20={self._format_profit_factor(float(resolved_metrics.get('profit_factor_window', 0.0) or 0.0))} | "
            f"loss_streak={int(resolved_metrics.get('loss_streak', 0.0) or 0.0)} | "
            f"drawdown={float(resolved_metrics.get('max_drawdown_pct', 0.0) or 0.0):.2f}% | "
            f"avg_fill_slip={float(resolved_metrics.get('avg_fill_slippage_pct', 0.0) or 0.0):.3f}%"
        )

    @staticmethod
    def _format_profit_factor(value: float) -> str:
        if value == float("inf"):
            return "inf"
        return f"{float(value):.2f}"

    def _evaluate_live_signal_direction(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str,
        interval: str | None = None,
        candles_dataframe: pd.DataFrame,
        strategy_profile: OptimizationProfile | None,
        required_candle_count: int,
    ) -> tuple[int, float | None, float | None, dict[str, object]]:
        normalized_symbol = (
            str(symbol).strip().upper()
            if symbol is not None and str(symbol).strip()
            else None
        )
        normalized_interval = (
            str(interval).strip()
            if interval is not None and str(interval).strip()
            else None
        )
        payload = _build_vectorized_strategy_cache_payload(
            candles_dataframe,
            strategy_name=strategy_name,
            required_candles=required_candle_count,
            setup_gate=None,
            strategy_profile=strategy_profile,
            regime_mask=None,
        )
        if isinstance(payload, Mapping):
            diagnostics_payload = payload.get("strategy_diagnostics")
            strategy_diagnostics = (
                dict(diagnostics_payload)
                if isinstance(diagnostics_payload, Mapping)
                else {}
            )
            signals_payload = payload.get("signals")
            if isinstance(signals_payload, Sequence) and len(signals_payload) > 0:
                try:
                    signal_direction = int(signals_payload[-1])
                except Exception as exc:
                    self._emit_exception_context_log(
                        component="signal_payload",
                        action="parse_signal_direction",
                        symbol=normalized_symbol,
                        strategy_name=strategy_name,
                        interval=normalized_interval,
                        exc=exc,
                        severity="WARN",
                    )
                    strategy_diagnostics["signal_payload_parse_failed"] = True
                else:
                    dynamic_stop_loss_pct: float | None = None
                    dynamic_take_profit_pct: float | None = None
                    if signal_direction != 0:
                        dynamic_stop_loss_series = payload.get("precomputed_dynamic_stop_loss_pcts")
                        if (
                            isinstance(dynamic_stop_loss_series, Sequence)
                            and len(dynamic_stop_loss_series) > 0
                        ):
                            try:
                                candidate_dynamic_stop_loss_pct = float(
                                    dynamic_stop_loss_series[-1]
                                )
                            except Exception as exc:
                                self._emit_exception_context_log(
                                    component="signal_payload",
                                    action="parse_dynamic_stop_loss_pct",
                                    symbol=normalized_symbol,
                                    strategy_name=strategy_name,
                                    interval=normalized_interval,
                                    exc=exc,
                                    severity="WARN",
                                )
                                strategy_diagnostics["dynamic_stop_loss_parse_failed"] = True
                            else:
                                if (
                                    math.isfinite(candidate_dynamic_stop_loss_pct)
                                    and candidate_dynamic_stop_loss_pct > 0.0
                                ):
                                    dynamic_stop_loss_pct = float(
                                        candidate_dynamic_stop_loss_pct
                                    )
                        dynamic_take_profit_series = payload.get("precomputed_dynamic_take_profit_pcts")
                        if (
                            isinstance(dynamic_take_profit_series, Sequence)
                            and len(dynamic_take_profit_series) > 0
                        ):
                            try:
                                candidate_dynamic_take_profit_pct = float(
                                    dynamic_take_profit_series[-1]
                                )
                            except Exception as exc:
                                self._emit_exception_context_log(
                                    component="signal_payload",
                                    action="parse_dynamic_take_profit_pct",
                                    symbol=normalized_symbol,
                                    strategy_name=strategy_name,
                                    interval=normalized_interval,
                                    exc=exc,
                                    severity="WARN",
                                )
                                strategy_diagnostics["dynamic_take_profit_parse_failed"] = True
                            else:
                                if (
                                    math.isfinite(candidate_dynamic_take_profit_pct)
                                    and candidate_dynamic_take_profit_pct > 0.0
                                ):
                                    dynamic_take_profit_pct = float(
                                        candidate_dynamic_take_profit_pct
                                    )
                    return (
                        signal_direction,
                        dynamic_stop_loss_pct,
                        dynamic_take_profit_pct,
                        strategy_diagnostics,
                    )
        return (
            int(
                _evaluate_strategy_signal(
                    strategy_name,
                    candles_dataframe,
                    strategy_profile=strategy_profile,
                )
            ),
            None,
            None,
            {},
        )

    def _review_and_recompute_after_close(self, trade_id: int) -> None:
        if self._db is None:
            return
        closed_trade = self._db.fetch_trade_by_id(trade_id)
        if closed_trade is None:
            return
        strategy_name = (
            str(closed_trade.strategy_name).strip()
            if closed_trade.strategy_name is not None and str(closed_trade.strategy_name).strip()
            else resolve_strategy_for_symbol(closed_trade.symbol, self._strategy_name)
        )
        timeframe = (
            str(closed_trade.timeframe).strip()
            if closed_trade.timeframe is not None and str(closed_trade.timeframe).strip()
            else self._target_interval_for_symbol(closed_trade.symbol)
        )
        try:
            self.review_closed_trade(trade_id)
            self._clear_meta_service_degraded(
                service_name="trade_review",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="trade_review",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
                exc=exc,
            )
        try:
            self.recompute_strategy_health(
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                timeframe=timeframe,
            )
            self._clear_meta_service_degraded(
                service_name="strategy_health",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="strategy_health",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
                exc=exc,
            )
        try:
            self.evaluate_meta_policy(
                closed_trade.symbol,
                strategy_name,
                timeframe,
                current_context={},
            )
            self._clear_meta_service_degraded(
                service_name="meta_policy",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="meta_policy",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
                exc=exc,
            )
        try:
            self._maybe_generate_periodic_meta_reports()
            self._clear_meta_service_degraded(
                service_name="meta_reports",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="meta_reports",
                symbol=closed_trade.symbol,
                strategy_name=strategy_name,
                interval=timeframe,
                exc=exc,
            )

    def _resolve_profile_version(
        self,
        *,
        symbol: str,
        strategy_name: str,
        timeframe: str,
    ) -> str:
        normalized_symbol = str(symbol).strip().upper()
        normalized_strategy = str(strategy_name).strip()
        normalized_timeframe = str(timeframe).strip()
        profile_payload = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol, {})
        digest_payload = {
            "symbol": normalized_symbol,
            "strategy": normalized_strategy,
            "timeframe": normalized_timeframe,
            "profile": profile_payload,
        }
        payload_text = json.dumps(
            digest_payload,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        digest = hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:12]
        return f"{normalized_strategy}:{normalized_timeframe}:{digest}"

    @staticmethod
    def _resolve_session_label(timestamp: datetime) -> str:
        if timestamp.weekday() >= 5:
            return "weekend"
        hour = int(timestamp.hour)
        if 0 <= hour < 7:
            return "asia"
        if 7 <= hour < 12:
            return "asia_europe_overlap"
        if 12 <= hour < 17:
            return "europe"
        if 17 <= hour < 22:
            return "us"
        return "us_asia_overlap"

    @staticmethod
    def _parse_json_payload(payload: object) -> dict[str, object]:
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, str):
            normalized = payload.strip()
            if not normalized:
                return {}
            with suppress(Exception):
                parsed = json.loads(normalized)
                if isinstance(parsed, Mapping):
                    return dict(parsed)
        return {}

    @staticmethod
    def _meta_key(symbol: str, strategy_name: str, interval: str) -> tuple[str, str, str]:
        return (
            str(symbol).strip().upper(),
            _sanitize_backtest_strategy_name(strategy_name),
            str(interval).strip(),
        )

    @staticmethod
    def _exception_stacktrace_text(exc: BaseException) -> str:
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).strip()

    def _emit_exception_context_log(
        self,
        *,
        component: str,
        action: str,
        symbol: str | None,
        strategy_name: str | None,
        interval: str | None,
        exc: BaseException,
        severity: str = "ERROR",
    ) -> str:
        stacktrace_text = self._exception_stacktrace_text(exc)
        self.log_message.emit(
            f"[{severity}] component={component} action={action} "
            f"symbol={str(symbol or '-').strip().upper() or '-'} "
            f"strategy={str(strategy_name or '-').strip() or '-'} "
            f"interval={str(interval or '-').strip() or '-'} "
            f"exception={type(exc).__name__}: {exc}"
        )
        if stacktrace_text:
            self.log_message.emit(
                f"[{severity}] traceback {component}.{action}:\n{stacktrace_text}"
            )
        return stacktrace_text

    def _emit_warning_once(self, *, key: str, message: str) -> None:
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            self.log_message.emit(f"[WARN] {message}")
            return
        if normalized_key in self._runtime_warning_once_keys:
            return
        self._runtime_warning_once_keys.add(normalized_key)
        self.log_message.emit(f"[WARN] {message}")

    @staticmethod
    def _meta_service_key(
        service_name: str,
        symbol: str,
        strategy_name: str,
        interval: str,
    ) -> tuple[str, str, str, str]:
        return (
            str(service_name).strip().lower(),
            str(symbol).strip().upper(),
            _sanitize_backtest_strategy_name(strategy_name),
            str(interval).strip(),
        )

    def _mark_meta_service_degraded(
        self,
        *,
        service_name: str,
        symbol: str,
        strategy_name: str,
        interval: str,
        exc: BaseException,
    ) -> None:
        key = self._meta_service_key(service_name, symbol, strategy_name, interval)
        stacktrace_text = self._emit_exception_context_log(
            component="meta_service",
            action=f"{service_name}_failed",
            symbol=symbol,
            strategy_name=strategy_name,
            interval=interval,
            exc=exc,
            severity="WARN",
        )
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        previous_payload = self._meta_service_degraded.get(key)
        self._meta_service_degraded[key] = {
            "service_name": key[0],
            "symbol": key[1],
            "strategy_name": key[2],
            "interval": key[3],
            "state": "degraded",
            "updated_at": now.isoformat(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "stacktrace": stacktrace_text,
        }
        if previous_payload is None:
            self._insert_adaptation_event(
                event_type="meta_service_degraded",
                symbol=key[1],
                strategy_name=key[2],
                interval=key[3],
                payload=dict(self._meta_service_degraded[key]),
                source="meta_bot",
            )

    def _clear_meta_service_degraded(
        self,
        *,
        service_name: str,
        symbol: str,
        strategy_name: str,
        interval: str,
    ) -> None:
        key = self._meta_service_key(service_name, symbol, strategy_name, interval)
        payload = self._meta_service_degraded.pop(key, None)
        if payload is None:
            return
        self._insert_adaptation_event(
            event_type="meta_service_recovered",
            symbol=key[1],
            strategy_name=key[2],
            interval=key[3],
            payload={
                "service_name": key[0],
                "state": "healthy",
                "recovered_at": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
                "previous_updated_at": payload.get("updated_at"),
            },
            source="meta_bot",
        )
        self.log_message.emit(
            "[META] service recovered: "
            f"service={key[0]} symbol={key[1]} strategy={key[2]} interval={key[3]}"
        )

    def _collect_meta_service_degraded(
        self,
        *,
        symbol: str,
        strategy_name: str,
        interval: str,
    ) -> list[str]:
        normalized_symbol, normalized_strategy, normalized_interval = self._meta_key(
            symbol,
            strategy_name,
            interval,
        )
        degraded_services: list[str] = []
        for (
            service_name,
            service_symbol,
            service_strategy,
            service_interval,
        ), payload in self._meta_service_degraded.items():
            if service_symbol != normalized_symbol:
                continue
            if service_strategy != normalized_strategy:
                continue
            if service_interval != normalized_interval:
                continue
            if str(payload.get("state", "")).strip().lower() != "degraded":
                continue
            degraded_services.append(service_name)
        return sorted(set(degraded_services))

    def _set_symbol_reconcile_required(
        self,
        *,
        symbol: str,
        reason: str,
        strategy_name: str | None,
        interval: str | None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return
        normalized_reason = str(reason).strip() or "reconcile_required"
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        previous_payload = self._symbol_reconcile_required.get(normalized_symbol)
        next_payload = {
            "symbol": normalized_symbol,
            "state": "position_state_unknown",
            "reason": normalized_reason,
            "updated_at": now.isoformat(),
            "payload": dict(payload or {}),
        }
        self._symbol_reconcile_required[normalized_symbol] = next_payload
        if previous_payload is not None and previous_payload.get("reason") == normalized_reason:
            return
        self.log_message.emit(
            "[CRITICAL] reconcile_required: "
            f"symbol={normalized_symbol} reason={normalized_reason}"
        )
        self._insert_adaptation_event(
            event_type="reconcile_required",
            symbol=normalized_symbol,
            strategy_name=strategy_name,
            interval=interval,
            payload=next_payload,
            source="risk_guard",
        )
        self._emit_meta_warning(
            warning_code="reconcile_required",
            message=f"Symbol {normalized_symbol} requires reconciliation ({normalized_reason}).",
            payload=next_payload,
        )

    def _clear_symbol_reconcile_required(
        self,
        *,
        symbol: str,
        strategy_name: str | None,
        interval: str | None,
        resolution: str,
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return
        previous_payload = self._symbol_reconcile_required.pop(normalized_symbol, None)
        if previous_payload is None:
            return
        self.log_message.emit(
            "[RECOVERY] reconciliation cleared: "
            f"symbol={normalized_symbol} resolution={resolution}"
        )
        self._insert_adaptation_event(
            event_type="reconcile_resolved",
            symbol=normalized_symbol,
            strategy_name=strategy_name,
            interval=interval,
            payload={
                "resolution": str(resolution),
                "resolved_at": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
                "previous_state": dict(previous_payload),
            },
            source="risk_guard",
        )

    def _attempt_symbol_reconciliation(
        self,
        *,
        symbol: str,
        strategy_name: str,
        interval: str,
    ) -> tuple[bool, str]:
        if self._db is None:
            return False, "db_unavailable"
        normalized_symbol = str(symbol).strip().upper()
        normalized_interval = str(interval).strip()
        open_trades_symbol = self._db.fetch_open_trades(symbol=normalized_symbol)
        for open_trade in open_trades_symbol:
            high_water_mark = float(open_trade.high_water_mark or 0.0)
            if not math.isfinite(high_water_mark) or high_water_mark <= 0.0:
                return False, "missing_exit_control"
            mark_price = float(self._latest_market_prices.get(normalized_symbol, 0.0) or 0.0)
            if mark_price > 0.0 and math.isfinite(mark_price):
                continue
            trade_interval = (
                str(open_trade.timeframe).strip()
                if open_trade.timeframe is not None and str(open_trade.timeframe).strip()
                else normalized_interval
            )
            candles = self._db.fetch_recent_candles(
                normalized_symbol,
                trade_interval,
                limit=1,
            )
            if not candles:
                return False, "mark_price_unavailable"
            last_close = float(candles[-1].close)
            if not math.isfinite(last_close) or last_close <= 0.0:
                return False, "mark_price_invalid"

        if self._paper_engine is not None:
            runtime_trade_ids: set[int] = set()
            for runtime_trade in self._paper_engine.active_trades:
                runtime_symbol = str(getattr(runtime_trade, "symbol", "") or "").strip().upper()
                if runtime_symbol != normalized_symbol:
                    continue
                candidate_id = getattr(runtime_trade, "id", runtime_trade)
                with suppress(TypeError, ValueError):
                    runtime_trade_ids.add(int(candidate_id))
            db_trade_ids = {int(trade.id) for trade in open_trades_symbol}
            if runtime_trade_ids != db_trade_ids:
                return False, "runtime_db_symbol_mismatch"

        return True, "ok"

    def _emit_operational_error_overview(self, *, now: datetime, force: bool = False) -> None:
        if not force:
            last_overview_at = self._meta_operational_error_last_overview_at
            if (
                isinstance(last_overview_at, datetime)
                and now < (last_overview_at + timedelta(minutes=5))
            ):
                return
        self._meta_operational_error_last_overview_at = now
        source_summary = ", ".join(
            f"{source}={count}"
            for source, count in self._meta_operational_error_source_counts.most_common(5)
        ) or "-"
        signature_summary = ", ".join(
            f"{signature} x{count}"
            for signature, count in self._meta_operational_error_signature_counts.most_common(3)
        ) or "-"
        self.log_message.emit(
            "[META] operational error overview: "
            f"sources[{source_summary}] top[{signature_summary}]"
        )

    def _insert_adaptation_event(
        self,
        *,
        event_type: str,
        symbol: str,
        strategy_name: str | None,
        interval: str | None,
        payload: Mapping[str, object] | None = None,
        source: str = "meta_bot",
    ) -> None:
        if self._db is None:
            return
        try:
            self._db.insert_adaptation_log(
                AdaptationLogCreate(
                    created_at=datetime.now(tz=UTC).replace(tzinfo=None),
                    symbol=str(symbol).strip().upper(),
                    strategy_name=(
                        None
                        if strategy_name is None
                        else _sanitize_backtest_strategy_name(strategy_name)
                    ),
                    timeframe=(
                        None
                        if interval is None
                        else str(interval).strip()
                    ),
                    event_type=str(event_type).strip(),
                    payload_json=dict(payload or {}),
                    source=str(source).strip(),
                )
            )
        except Exception as exc:
            stacktrace_text = self._exception_stacktrace_text(exc)
            self.log_message.emit(
                "[WARN] adaptation_log_write_failed: "
                f"event_type={event_type} symbol={str(symbol).strip().upper()} "
                f"strategy={str(strategy_name or '-').strip() or '-'} "
                f"interval={str(interval or '-').strip() or '-'} "
                f"source={str(source).strip()} exception={type(exc).__name__}: {exc}"
            )
            if stacktrace_text:
                self.log_message.emit(
                    f"[WARN] adaptation_log_write_failed traceback:\n{stacktrace_text}"
                )

    def _emit_meta_warning(
        self,
        *,
        warning_code: str,
        message: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        warning_payload = {
            "warning_code": str(warning_code),
            "message": str(message),
            "payload": dict(payload or {}),
            "emitted_at": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
        }
        self.log_message.emit(
            f"[META_WARNING] {warning_payload['warning_code']}: {warning_payload['message']}"
        )
        for channel in META_WARN_HOOK_CHANNELS:
            if channel == "log":
                continue
            self._dispatch_warning_stub(channel=channel, payload=warning_payload)

    def _dispatch_warning_stub(self, *, channel: str, payload: Mapping[str, object]) -> None:
        self.log_message.emit(
            f"[META_WARNING_STUB] channel={channel} payload={json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)}"
        )

    def _activate_fail_safe_mode(self, *, reason: str, payload: Mapping[str, object] | None = None) -> None:
        normalized_reason = str(reason).strip() or "unspecified_fail_safe_reason"
        if self._meta_fail_safe_active and normalized_reason == self._meta_fail_safe_reason:
            return
        activated_at = datetime.now(tz=UTC).replace(tzinfo=None)
        self._meta_fail_safe_active = True
        self._meta_fail_safe_reason = normalized_reason
        self._meta_fail_safe_activated_at = activated_at
        self._emit_meta_warning(
            warning_code="system_paused",
            message=f"Fail-safe mode active: {normalized_reason}",
            payload=payload,
        )
        self._insert_adaptation_event(
            event_type="fail_safe_activated",
            symbol="GLOBAL",
            strategy_name=None,
            interval=None,
            payload={
                "reason": normalized_reason,
                "payload": dict(payload or {}),
                "activated_at": activated_at.isoformat(),
            },
            source="meta_bot",
        )

    def _maybe_recover_fail_safe_mode(self, *, now: datetime) -> None:
        if not self._meta_fail_safe_active:
            return
        if self._meta_fail_safe_reason != "repeated_operational_error":
            return
        activated_at = self._meta_fail_safe_activated_at
        if activated_at is None:
            return
        if now < (activated_at + timedelta(minutes=META_FAILSAFE_AUTO_RECOVER_MINUTES)):
            return
        window_start = now - timedelta(minutes=META_OPERATIONAL_ERROR_WINDOW_MINUTES)
        recent_hard_errors = [
            item
            for item in self._meta_operational_errors
            if item[0] >= window_start and item[4] in {"runtime_exception", "runtime"}
        ]
        if recent_hard_errors:
            return
        previous_reason = str(self._meta_fail_safe_reason)
        self._meta_fail_safe_active = False
        self._meta_fail_safe_reason = ""
        self._meta_fail_safe_activated_at = None
        self.log_message.emit(
            "[META] fail-safe auto-recovered after operational error cooldown."
        )
        self._insert_adaptation_event(
            event_type="fail_safe_recovered",
            symbol="GLOBAL",
            strategy_name=None,
            interval=None,
            payload={
                "previous_reason": previous_reason,
                "recovered_at": now.isoformat(),
                "cooldown_minutes": META_FAILSAFE_AUTO_RECOVER_MINUTES,
                "window_minutes": META_OPERATIONAL_ERROR_WINDOW_MINUTES,
            },
            source="meta_bot",
        )

    def _register_operational_error(
        self,
        *,
        symbol: str,
        interval: str,
        reason: str,
        exception_text: str,
        exception_stacktrace: str | None = None,
        source: str = "runtime_exception",
    ) -> None:
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        normalized_symbol = str(symbol).strip().upper()
        normalized_interval = str(interval).strip()
        normalized_reason = str(reason).strip() or "unknown_operational_error"
        normalized_source = str(source).strip().lower() or "runtime_exception"
        exception_head = str(exception_text).strip().splitlines()[0][:240]
        signature = (
            f"{normalized_source}|{normalized_symbol}|{normalized_interval}|"
            f"{normalized_reason}|{exception_head}"
        )
        self._meta_operational_error_source_counts[normalized_source] += 1
        self._meta_operational_error_signature_counts[
            f"{normalized_source}:{normalized_reason}"
        ] += 1
        dedup_since = self._meta_operational_error_last_seen.get(signature)
        if (
            isinstance(dedup_since, datetime)
            and (now - dedup_since).total_seconds() < float(META_OPERATIONAL_ERROR_DEDUP_SECONDS)
        ):
            self.log_message.emit(
                "[META] operational error deduped: "
                f"source={normalized_source} symbol={normalized_symbol} "
                f"interval={normalized_interval} reason={normalized_reason}"
            )
            return
        self._meta_operational_error_last_seen[signature] = now
        self._meta_operational_errors.append(
            (
                now,
                normalized_symbol,
                normalized_interval,
                normalized_reason,
                normalized_source,
            )
        )
        window_start = now - timedelta(minutes=META_OPERATIONAL_ERROR_WINDOW_MINUTES)
        while self._meta_operational_errors and self._meta_operational_errors[0][0] < window_start:
            self._meta_operational_errors.popleft()
        for cached_signature, cached_time in list(self._meta_operational_error_last_seen.items()):
            if (now - cached_time).total_seconds() > float(META_OPERATIONAL_ERROR_WINDOW_MINUTES * 60):
                self._meta_operational_error_last_seen.pop(cached_signature, None)

        current_window_entries = [
            entry
            for entry in self._meta_operational_errors
            if entry[0] >= window_start
        ]
        current_count = len(current_window_entries)
        self._insert_adaptation_event(
            event_type="operational_error",
            symbol=normalized_symbol,
            strategy_name=None,
            interval=normalized_interval,
            payload={
                "reason": normalized_reason,
                "exception_text": exception_head,
                "exception_stacktrace": (
                    str(exception_stacktrace)[:8000]
                    if exception_stacktrace is not None and str(exception_stacktrace).strip()
                    else None
                ),
                "source": normalized_source,
                "window_minutes": META_OPERATIONAL_ERROR_WINDOW_MINUTES,
                "window_error_count": current_count,
            },
            source=normalized_source,
        )

        # Heartbeat / sync hints remain warnings only and are never escalated to fail-safe.
        if normalized_source in {"heartbeat", "heartbeat_warning"}:
            self.log_message.emit(
                "[META] heartbeat warning observed (non-escalating): "
                f"symbol={normalized_symbol} interval={normalized_interval} "
                f"reason={normalized_reason} detail={exception_head}"
            )
            return

        hard_entries = [
            entry
            for entry in current_window_entries
            if entry[4] in {"runtime_exception", "runtime"}
        ]
        hard_count = len(hard_entries)
        distinct_symbols = {entry[1] for entry in hard_entries}
        distinct_signatures = {(entry[1], entry[2], entry[3]) for entry in hard_entries}
        enough_volume = hard_count >= META_OPERATIONAL_ERROR_FAILSAFE_THRESHOLD
        enough_breadth = (
            len(distinct_symbols) >= META_OPERATIONAL_ERROR_MIN_DISTINCT_SYMBOLS
            or len(distinct_signatures) >= META_OPERATIONAL_ERROR_MIN_DISTINCT_SIGNATURES
        )
        if enough_volume and enough_breadth:
            self._emit_meta_warning(
                warning_code="repeated_operational_error",
                message=(
                    f"Operational runtime exceptions reached {hard_count} within "
                    f"{META_OPERATIONAL_ERROR_WINDOW_MINUTES} minutes."
                ),
                payload={
                    "symbol": normalized_symbol,
                    "interval": normalized_interval,
                    "window_error_count": hard_count,
                    "distinct_symbols": sorted(distinct_symbols),
                    "distinct_signatures": len(distinct_signatures),
                },
            )
            self._activate_fail_safe_mode(
                reason="repeated_operational_error",
                payload={
                    "window_minutes": META_OPERATIONAL_ERROR_WINDOW_MINUTES,
                    "window_error_count": hard_count,
                    "distinct_symbols": sorted(distinct_symbols),
                    "distinct_signatures": len(distinct_signatures),
                },
            )
            return

        self.log_message.emit(
            "[META] operational error recorded without fail-safe escalation: "
            f"hard_count={hard_count} distinct_symbols={len(distinct_symbols)} "
            f"distinct_signatures={len(distinct_signatures)} threshold={META_OPERATIONAL_ERROR_FAILSAFE_THRESHOLD}"
        )
        self._emit_operational_error_overview(now=now)

    def _initialize_meta_services(self) -> None:
        META_REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self._meta_service_active = True
        self._meta_global_risk_multiplier = 1.0
        self.log_message.emit(
            "[META] services initialized: "
            "mode=active_observe_and_guard | parameter_overwrite=disabled | "
            f"reports_dir={META_REPORTS_DIRECTORY.as_posix()}"
        )
        self.log_message.emit(
            "[META] guards: "
            f"max_daily_loss={META_MAX_DAILY_LOSS_USD:.2f} | "
            f"max_symbol_loss_streak={META_MAX_SYMBOL_LOSS_STREAK} | "
            f"max_strategy_drawdown_pct={META_MAX_STRATEGY_DRAWDOWN_PCT:.2f}% | "
            f"max_open_positions={META_MAX_OPEN_POSITIONS_GUARD} | "
            f"max_correlated_risk={META_MAX_CORRELATED_RISK}."
        )
        recompute_targets: set[tuple[str, str, str]] = set()
        for symbol in self._symbols:
            strategy_name = resolve_strategy_for_symbol(symbol, self._strategy_name)
            interval = self._target_interval_for_symbol(symbol)
            recompute_targets.add(
                self._meta_key(symbol, strategy_name, interval)
            )
        if self._db is not None:
            try:
                for row in self._db.fetch_strategy_health_rows(limit=5000):
                    recompute_targets.add(
                        self._meta_key(row.symbol, row.strategy_name, row.timeframe)
                    )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="meta_init_health_fetch",
                    symbol="GLOBAL",
                    strategy_name="meta_bootstrap",
                    interval="global",
                    exc=exc,
                )
            else:
                self._clear_meta_service_degraded(
                    service_name="meta_init_health_fetch",
                    symbol="GLOBAL",
                    strategy_name="meta_bootstrap",
                    interval="global",
                )
        for normalized_symbol, normalized_strategy, normalized_interval in sorted(recompute_targets):
            try:
                self.recompute_strategy_health(
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    timeframe=normalized_interval,
                )
                self._clear_meta_service_degraded(
                    service_name="meta_init_strategy_health",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="meta_init_strategy_health",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    exc=exc,
                )
            try:
                self.evaluate_meta_policy(
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    current_context={},
                )
                self._clear_meta_service_degraded(
                    service_name="meta_init_policy_eval",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="meta_init_policy_eval",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    exc=exc,
                )
        try:
            self._maybe_generate_periodic_meta_reports()
            self._clear_meta_service_degraded(
                service_name="meta_init_reports",
                symbol="GLOBAL",
                strategy_name="meta_bootstrap",
                interval="global",
            )
        except Exception as exc:
            self._mark_meta_service_degraded(
                service_name="meta_init_reports",
                symbol="GLOBAL",
                strategy_name="meta_bootstrap",
                interval="global",
                exc=exc,
            )

    def resolve_market_regime(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles_dataframe: pd.DataFrame,
        observed_time: datetime | None = None,
    ) -> dict[str, object]:
        normalized_symbol = str(symbol).strip().upper()
        normalized_timeframe = str(timeframe).strip()
        normalized_strategy_name = resolve_strategy_for_symbol(
            normalized_symbol,
            self._strategy_name,
        )
        if candles_dataframe.empty:
            observed_at = observed_time or datetime.now(tz=UTC).replace(tzinfo=None)
            return {
                "regime_label": "range_balanced",
                "regime_confidence": 0.2,
                "trend_bias": "neutral",
                "volatility_state": "normal",
                "expansion_state": "balanced",
                "liquidity_state": "normal",
                "session_label": self._resolve_session_label(observed_at),
                "regime_features_json": {
                    "engine_version": REGIME_ENGINE_VERSION,
                    "reason": "empty_candles",
                },
            }

        working_df = candles_dataframe.tail(720).copy(deep=True)
        observed_at = observed_time
        if observed_at is None:
            with suppress(Exception):
                candidate = working_df.iloc[-1].get("open_time")
                if isinstance(candidate, datetime):
                    observed_at = candidate.replace(tzinfo=None)
        if observed_at is None:
            observed_at = datetime.now(tz=UTC).replace(tzinfo=None)

        cache_key = (normalized_symbol, normalized_timeframe)
        cache_entry = self._runtime_regime_cache.get(cache_key)
        stable_payload: dict[str, object] | None = None
        if isinstance(cache_entry, dict):
            cached_time = cache_entry.get("observed_at")
            cached_payload = cache_entry.get("payload")
            if isinstance(cached_payload, Mapping):
                stable_payload = dict(cached_payload)
            if (
                isinstance(cached_time, datetime)
                and cached_time == observed_at
                and isinstance(cached_payload, Mapping)
            ):
                return dict(cached_payload)

        close_series = pd.to_numeric(working_df["close"], errors="coerce").ffill().bfill().fillna(0.0)
        high_series = pd.to_numeric(working_df["high"], errors="coerce").ffill().bfill().fillna(close_series)
        low_series = pd.to_numeric(working_df["low"], errors="coerce").ffill().bfill().fillna(close_series)
        volume_series = pd.to_numeric(working_df["volume"], errors="coerce").ffill().bfill().fillna(0.0)
        close_safe = close_series.clip(lower=1e-9)
        returns_pct = close_safe.pct_change().fillna(0.0) * 100.0
        prev_close = close_safe.shift(1).fillna(close_safe)
        true_range = pd.concat(
            [
                (high_series - low_series).abs(),
                (high_series - prev_close).abs(),
                (low_series - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct_series = (true_range / close_safe) * 100.0
        range_pct_series = ((high_series - low_series).abs() / close_safe) * 100.0

        close_now = float(close_safe.iloc[-1])
        close_20_ref = float(close_safe.iloc[-min(len(close_safe), 20)])
        close_50_ref = float(close_safe.iloc[-min(len(close_safe), 50)])
        trend_pct_20 = ((close_now / max(close_20_ref, 1e-9)) - 1.0) * 100.0
        trend_pct_50 = ((close_now / max(close_50_ref, 1e-9)) - 1.0) * 100.0
        atr_pct_latest = float(atr_pct_series.tail(14).mean()) if not atr_pct_series.empty else 0.0
        atr_pct_baseline = float(atr_pct_series.tail(80).mean()) if not atr_pct_series.empty else atr_pct_latest
        vol_ratio = (
            float(volume_series.iloc[-1]) / max(float(volume_series.tail(20).mean()), 1e-9)
            if not volume_series.empty
            else 1.0
        )
        range_pct_latest = float(range_pct_series.iloc[-1]) if not range_pct_series.empty else 0.0
        range_pct_baseline = float(range_pct_series.tail(20).mean()) if not range_pct_series.empty else range_pct_latest
        expansion_ratio = range_pct_latest / max(range_pct_baseline, 1e-9)
        ret_std_20 = float(returns_pct.tail(20).std(ddof=0)) if len(returns_pct) > 1 else 0.0
        move_3_bars_pct = 0.0
        if len(close_safe) >= 4:
            close_3_ref = float(close_safe.iloc[-4])
            move_3_bars_pct = abs((close_now / max(close_3_ref, 1e-9)) - 1.0) * 100.0

        hmm_label = HMMRegimeDetector.UNKNOWN
        hmm_allowed_ratio = 1.0
        hmm_converged = True
        hmm_non_converged_windows = 0
        hmm_window_count = 0
        hmm_non_convergence_detail = "window_context=n/a"
        if len(working_df) >= 80:
            detector = self._runtime_hmm_detectors.get(normalized_timeframe)
            if detector is None:
                detector = HMMRegimeDetector(
                    n_components=4,
                    train_window_candles=1200,
                    apply_window_candles=240,
                    warmup_candles=120,
                    stability_candles=3,
                    allowed_regimes=(
                        HMMRegimeDetector.BULL_TREND,
                        HMMRegimeDetector.BEAR_TREND,
                        HMMRegimeDetector.HIGH_VOL_RANGE,
                        HMMRegimeDetector.LOW_VOL_RANGE,
                        HMMRegimeDetector.UNKNOWN,
                        HMMRegimeDetector.WARMUP,
                    ),
                )
                self._runtime_hmm_detectors[normalized_timeframe] = detector
            try:
                detection = detector.detect(working_df)
                hmm_converged = bool(getattr(detection, "converged", True))
                hmm_non_converged_windows = int(getattr(detection, "non_converged_windows", 0) or 0)
                hmm_window_count = int(getattr(detection, "window_count", 0) or 0)
                if not bool(hmm_converged):
                    hmm_non_convergence_detail = _format_hmm_non_convergence_detail(detection)
                if detection.regime_labels:
                    hmm_label = str(detection.regime_labels[-1])
                hmm_allowed_ratio = float(detection.allowed_ratio)
                transmat_warning_events = [
                    event
                    for event in tuple(getattr(detection, "warning_events", tuple()) or tuple())
                    if isinstance(event, Mapping)
                    and str(event.get("warning_kind", "")).strip().lower()
                    == "transmat_zero_sum_no_transition"
                ]
                if transmat_warning_events:
                    now = datetime.now(tz=UTC).replace(tzinfo=None)
                    structural_warn_key = (
                        normalized_symbol,
                        normalized_timeframe,
                        "transmat_zero_sum_no_transition",
                    )
                    last_warned_at = self._runtime_hmm_structural_warned_at.get(structural_warn_key)
                    if (
                        not isinstance(last_warned_at, datetime)
                        or now
                        >= (
                            last_warned_at
                            + timedelta(minutes=REGIME_HMM_NON_CONVERGENCE_WARN_INTERVAL_MINUTES)
                        )
                    ):
                        self._runtime_hmm_structural_warned_at[structural_warn_key] = now
                        warning_detail = _format_hmm_warning_detail(
                            detection,
                            warning_kind="transmat_zero_sum_no_transition",
                        )
                        self.log_message.emit(
                            "[REGIME] HMM structural warning: "
                            f"{normalized_symbol} {normalized_timeframe} "
                            f"{warning_detail}."
                        )
                self._clear_meta_service_degraded(
                    service_name="regime_detection",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy_name,
                    interval=normalized_timeframe,
                )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="regime_detection",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy_name,
                    interval=normalized_timeframe,
                    exc=exc,
                )
                hmm_converged = False
                hmm_non_converged_windows = max(1, int(hmm_non_converged_windows or 0))
                hmm_window_count = max(1, int(hmm_window_count or 0))
                hmm_non_convergence_detail = (
                    f"hmm_detection_exception={type(exc).__name__}:{str(exc).strip()}"
                )

        if (not bool(hmm_converged)) and stable_payload is not None:
            now = datetime.now(tz=UTC).replace(tzinfo=None)
            last_warned_at = self._runtime_hmm_non_convergence_warned_at.get(cache_key)
            if (
                not isinstance(last_warned_at, datetime)
                or now >= (last_warned_at + timedelta(minutes=REGIME_HMM_NON_CONVERGENCE_WARN_INTERVAL_MINUTES))
            ):
                self._runtime_hmm_non_convergence_warned_at[cache_key] = now
                self.log_message.emit(
                    "[REGIME] HMM non-converged; reusing last stable regime: "
                    f"{normalized_symbol} {normalized_timeframe} "
                    f"(non_converged_windows={hmm_non_converged_windows}, windows={hmm_window_count}) "
                    f"{hmm_non_convergence_detail}."
                )
            fallback_payload = dict(stable_payload)
            fallback_features = self._parse_json_payload(fallback_payload.get("regime_features_json"))
            fallback_features.update(
                {
                    "engine_version": REGIME_ENGINE_VERSION,
                    "hmm_converged": False,
                    "hmm_non_converged_windows": int(hmm_non_converged_windows),
                    "hmm_window_count": int(hmm_window_count),
                    "fallback_reason": "hmm_non_converged_reuse_last_stable",
                    "fallback_observed_at": observed_at.isoformat(),
                }
            )
            fallback_payload["session_label"] = self._resolve_session_label(observed_at)
            fallback_payload["regime_features_json"] = fallback_features
            self._runtime_regime_cache[cache_key] = {
                "observed_at": observed_at,
                "payload": dict(fallback_payload),
            }
            return fallback_payload
        if not bool(hmm_converged):
            now = datetime.now(tz=UTC).replace(tzinfo=None)
            last_warned_at = self._runtime_hmm_non_convergence_warned_at.get(cache_key)
            if (
                not isinstance(last_warned_at, datetime)
                or now >= (last_warned_at + timedelta(minutes=REGIME_HMM_NON_CONVERGENCE_WARN_INTERVAL_MINUTES))
            ):
                self._runtime_hmm_non_convergence_warned_at[cache_key] = now
                self.log_message.emit(
                    "[REGIME] HMM non-converged without stable cache; "
                    f"using heuristic regime fallback for {normalized_symbol} {normalized_timeframe}. "
                    f"{hmm_non_convergence_detail}."
                )

        if trend_pct_20 >= 0.6 and trend_pct_50 >= 0.1:
            trend_bias = "bullish"
        elif trend_pct_20 <= -0.6 and trend_pct_50 <= -0.1:
            trend_bias = "bearish"
        else:
            trend_bias = "neutral"

        volatility_state = "normal"
        if atr_pct_latest >= 2.0 or ret_std_20 >= 1.6:
            volatility_state = "extreme"
        elif atr_pct_latest >= 1.0 or ret_std_20 >= 0.9:
            volatility_state = "high"
        elif atr_pct_latest <= 0.3:
            volatility_state = "low"

        expansion_state = "balanced"
        if expansion_ratio >= 2.2 or move_3_bars_pct >= 4.0:
            expansion_state = "panic_expansion"
        elif expansion_ratio >= 1.35:
            expansion_state = "expanding"
        elif expansion_ratio <= 0.7:
            expansion_state = "compressed"

        liquidity_state = "normal"
        if vol_ratio < 0.45:
            liquidity_state = "illiquid"
        elif vol_ratio < 0.8:
            liquidity_state = "thin"
        elif vol_ratio > 1.7:
            liquidity_state = "thick"

        regime_label = "range_balanced"
        if liquidity_state in {"illiquid", "thin"} and expansion_state in {"balanced", "compressed"}:
            regime_label = "illiquid_noise"
        elif expansion_state == "panic_expansion":
            regime_label = "panic_expansion"
        elif expansion_state == "compressed" and abs(trend_pct_20) < 0.8:
            regime_label = "compression_pre_breakout"
        elif abs(trend_pct_20) >= 2.2 and expansion_ratio >= 1.35:
            regime_label = "trend_exhausted"
        elif trend_bias == "bullish" or hmm_label == HMMRegimeDetector.BULL_TREND:
            regime_label = "trend_clean_up"
        elif trend_bias == "bearish" or hmm_label == HMMRegimeDetector.BEAR_TREND:
            regime_label = "trend_clean_down"
        elif expansion_ratio >= 1.3 and abs(trend_pct_20) >= 0.7:
            regime_label = "breakout_transition"
        elif volatility_state in {"high", "extreme"} or hmm_label == HMMRegimeDetector.HIGH_VOL_RANGE:
            regime_label = "range_volatile"

        if regime_label not in REGIME_LABEL_UNIVERSE:
            regime_label = "range_balanced"

        trend_conf = min(1.0, abs(trend_pct_20) / 3.0)
        vol_conf = min(1.0, abs(expansion_ratio - 1.0) / 1.4)
        confidence = 0.38
        if regime_label.startswith("trend_clean"):
            confidence += 0.32 * trend_conf + 0.18 * vol_conf
        elif regime_label == "trend_exhausted":
            confidence += 0.22 * trend_conf + 0.25 * vol_conf
        elif regime_label in {"range_balanced", "compression_pre_breakout", "illiquid_noise"}:
            confidence += 0.20 * (1.0 - trend_conf) + 0.15 * (1.0 - min(vol_conf, 1.0))
        else:
            confidence += 0.28 * vol_conf + 0.14 * trend_conf
        if hmm_label in {
            HMMRegimeDetector.BULL_TREND,
            HMMRegimeDetector.BEAR_TREND,
            HMMRegimeDetector.HIGH_VOL_RANGE,
            HMMRegimeDetector.LOW_VOL_RANGE,
        }:
            confidence += 0.08
        elif hmm_label in {HMMRegimeDetector.UNKNOWN, HMMRegimeDetector.WARMUP}:
            confidence -= 0.05
        regime_confidence = float(max(0.05, min(confidence, 0.99)))

        session_label = self._resolve_session_label(observed_at)
        regime_features = {
            "engine_version": REGIME_ENGINE_VERSION,
            "hmm_label": str(hmm_label),
            "hmm_allowed_ratio": float(hmm_allowed_ratio),
            "hmm_converged": bool(hmm_converged),
            "hmm_non_converged_windows": int(hmm_non_converged_windows),
            "hmm_window_count": int(hmm_window_count),
            "trend_pct_20": float(trend_pct_20),
            "trend_pct_50": float(trend_pct_50),
            "atr_pct_latest": float(atr_pct_latest),
            "atr_pct_baseline": float(atr_pct_baseline),
            "range_pct_latest": float(range_pct_latest),
            "range_pct_baseline": float(range_pct_baseline),
            "ret_std_20": float(ret_std_20),
            "expansion_ratio": float(expansion_ratio),
            "volume_ratio": float(vol_ratio),
            "move_3_bars_pct": float(move_3_bars_pct),
            "trend_bias": str(trend_bias),
            "volatility_state": str(volatility_state),
            "expansion_state": str(expansion_state),
            "liquidity_state": str(liquidity_state),
        }
        regime_payload = {
            "regime_label": regime_label,
            "regime_confidence": regime_confidence,
            "trend_bias": trend_bias,
            "volatility_state": volatility_state,
            "expansion_state": expansion_state,
            "liquidity_state": liquidity_state,
            "session_label": session_label,
            "regime_features_json": regime_features,
        }

        if self._db is not None:
            try:
                self._db.insert_regime_observation(
                    RegimeObservationCreate(
                        observed_at=observed_at,
                        symbol=normalized_symbol,
                        timeframe=normalized_timeframe,
                        regime_label=regime_label,
                        regime_confidence=regime_confidence,
                        trend_bias=trend_bias,
                        volatility_state=volatility_state,
                        expansion_state=expansion_state,
                        liquidity_state=liquidity_state,
                        session_label=session_label,
                        regime_features_json=regime_features,
                        source=REGIME_ENGINE_VERSION,
                    )
                )
                self._clear_meta_service_degraded(
                    service_name="regime_observation_persist",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy_name,
                    interval=normalized_timeframe,
                )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="regime_observation_persist",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy_name,
                    interval=normalized_timeframe,
                    exc=exc,
                )

        self._runtime_regime_cache[cache_key] = {
            "observed_at": observed_at,
            "payload": dict(regime_payload),
        }
        return regime_payload

    def build_entry_snapshot(
        self,
        *,
        symbol: str,
        strategy_name: str,
        timeframe: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        leverage_scale: float,
        profile_version: str,
        regime_payload: Mapping[str, object],
        signal_direction: int,
        signal_diagnostics: Mapping[str, object] | None,
        candles_dataframe: pd.DataFrame,
        setup_gate_score: float | None,
        setup_gate_reason: str,
        candidate_state: str,
    ) -> dict[str, object]:
        diagnostics = dict(signal_diagnostics or {})
        safe_entry_price = float(entry_price) if math.isfinite(float(entry_price)) else 0.0
        if safe_entry_price <= 0.0:
            safe_entry_price = 1e-9
        close_series = pd.to_numeric(candles_dataframe["close"], errors="coerce").ffill().bfill()
        high_series = pd.to_numeric(candles_dataframe["high"], errors="coerce").ffill().bfill()
        low_series = pd.to_numeric(candles_dataframe["low"], errors="coerce").ffill().bfill()
        volume_series = pd.to_numeric(candles_dataframe["volume"], errors="coerce").ffill().bfill()
        close_safe = close_series.clip(lower=1e-9)
        prev_close = close_safe.shift(1).fillna(close_safe)
        true_range = pd.concat(
            [
                (high_series - low_series).abs(),
                (high_series - prev_close).abs(),
                (low_series - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct_at_entry = float(((true_range / close_safe) * 100.0).tail(14).mean()) if len(true_range) > 0 else 0.0
        volume_ratio_at_entry = (
            float(volume_series.iloc[-1]) / max(float(volume_series.tail(20).mean()), 1e-9)
            if len(volume_series) > 0
            else 1.0
        )
        spread_estimate = (
            float((high_series.iloc[-1] - low_series.iloc[-1]) / max(close_safe.iloc[-1], 1e-9) * 100.0)
            if len(close_safe) > 0
            else 0.0
        )
        move_last_1 = float(diagnostics.get("latest_move_last_1_bar_pct", 0.0) or 0.0)
        move_last_2 = float(diagnostics.get("latest_move_last_2_bars_pct", 0.0) or 0.0)
        move_last_3 = float(diagnostics.get("latest_move_last_3_bars_pct", 0.0) or 0.0)
        distance_to_reference = float(diagnostics.get("latest_distance_to_reference_pct", 0.0) or 0.0)
        atr_extension_mult = float(diagnostics.get("latest_atr_extension_mult", 0.0) or 0.0)
        move_already_extended_pct = max(abs(move_last_1), abs(move_last_2), abs(move_last_3), abs(distance_to_reference))
        base_signal_strength = 1.0 if abs(int(signal_direction)) >= 1 else 0.0
        expansion_component = min(1.0, abs(move_last_3) / 3.5)
        guard_component = 0.0
        with suppress(Exception):
            if float(diagnostics.get("latest_late_entry_guard_blocked", 0.0) or 0.0) < 0.5:
                guard_component = 1.0
        signal_strength = float(
            max(
                0.05,
                min(
                    1.0,
                    (0.55 * base_signal_strength)
                    + (0.25 * expansion_component)
                    + (0.20 * guard_component),
                ),
            )
        )
        regime_confidence = float(regime_payload.get("regime_confidence", 0.0) or 0.0)
        if setup_gate_score is not None and math.isfinite(float(setup_gate_score)):
            confidence_score = float(max(0.0, min(1.0, float(setup_gate_score) / 100.0)))
        else:
            confidence_score = float(
                max(
                    0.05,
                    min(
                        0.99,
                        0.30 + (0.45 * signal_strength) + (0.25 * regime_confidence),
                    ),
                )
            )
        setup_reason_code = str(setup_gate_reason).strip()
        if not setup_reason_code:
            setup_reason_code = str(diagnostics.get("latest_blocker_reason", "") or "").strip()
        if not setup_reason_code:
            setup_reason_code = "signal_confirmed"

        regime_features_json = regime_payload.get("regime_features_json")
        if not isinstance(regime_features_json, Mapping):
            regime_features_json = {}
        return {
            "symbol": str(symbol).strip().upper(),
            "strategy_name": str(strategy_name).strip(),
            "timeframe": str(timeframe).strip(),
            "side": str(side).strip().upper(),
            "entry_time": entry_time.isoformat(),
            "entry_price": float(safe_entry_price),
            "leverage": float(max(1.0, leverage_scale)),
            "profile_version": str(profile_version),
            "regime_label_at_entry": str(regime_payload.get("regime_label", "range_balanced")),
            "regime_confidence": float(max(0.0, min(regime_confidence, 1.0))),
            "trend_bias": str(regime_payload.get("trend_bias", "neutral")),
            "volatility_state": str(regime_payload.get("volatility_state", "normal")),
            "expansion_state": str(regime_payload.get("expansion_state", "balanced")),
            "liquidity_state": str(regime_payload.get("liquidity_state", "normal")),
            "session_label": str(regime_payload.get("session_label", self._resolve_session_label(entry_time))),
            "signal_strength": float(signal_strength),
            "confidence_score": float(confidence_score),
            "atr_pct_at_entry": float(atr_pct_at_entry),
            "volume_ratio_at_entry": float(volume_ratio_at_entry),
            "spread_estimate": float(spread_estimate),
            "move_already_extended_pct": float(move_already_extended_pct),
            "ema_distance_metrics": {
                "distance_to_reference_pct": float(distance_to_reference),
                "atr_extension_mult": float(atr_extension_mult),
                "ema_spread_entry_pct_mean": float(diagnostics.get("ema_spread_entry_pct_mean", 0.0) or 0.0),
                "ema_spread_entry_pct_max": float(diagnostics.get("ema_spread_entry_pct_max", 0.0) or 0.0),
                "ema_spread_pct_mean": float(diagnostics.get("ema_spread_pct_mean", 0.0) or 0.0),
            },
            "recent_range_metrics": {
                "move_last_1_bar_pct": float(move_last_1),
                "move_last_2_bars_pct": float(move_last_2),
                "move_last_3_bars_pct": float(move_last_3),
                "atr_extension_mult": float(atr_extension_mult),
                "range_pct_last_bar": float(spread_estimate),
            },
            "setup_reason_code": setup_reason_code,
            "veto_flags": {
                "setup_gate_enabled": bool(self._setup_gate is not None),
                "setup_gate_reason": str(setup_gate_reason or ""),
                "live_candidate_state": str(candidate_state or ""),
                "late_entry_guard_blocked": bool(
                    float(diagnostics.get("latest_late_entry_guard_blocked", 0.0) or 0.0) >= 0.5
                ),
                "latest_blocker_reason": str(diagnostics.get("latest_blocker_reason", "") or ""),
            },
            "regime_features_json": dict(regime_features_json),
            "feature_snapshot_version": ENTRY_SNAPSHOT_SCHEMA_VERSION,
        }

    def _resolve_configured_take_profit_pct(self, symbol: str) -> float:
        normalized_symbol = str(symbol).strip().upper()
        production_profile = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol)
        if isinstance(production_profile, Mapping):
            raw_tp = production_profile.get("take_profit_pct", 0.0)
            try:
                resolved_tp = float(raw_tp or 0.0)
            except Exception as exc:
                self._emit_exception_context_log(
                    component="config_profile",
                    action="parse_take_profit_pct_production",
                    symbol=normalized_symbol,
                    strategy_name=resolve_strategy_for_symbol(normalized_symbol, self._strategy_name),
                    interval=self._target_interval_for_symbol(normalized_symbol),
                    exc=exc,
                    severity="WARN",
                )
            else:
                if math.isfinite(resolved_tp) and resolved_tp > 0.0:
                    return resolved_tp
                if raw_tp not in (None, "", 0, 0.0, "0", "0.0"):
                    self._emit_warning_once(
                        key=f"invalid_take_profit_pct:production:{normalized_symbol}:{raw_tp}",
                        message=(
                            "Invalid non-positive production take_profit_pct detected; "
                            f"symbol={normalized_symbol} value={raw_tp!r}. Falling back to coin/default settings."
                        ),
                    )
        coin_profile = settings.trading.coin_profiles.get(normalized_symbol)
        if coin_profile is not None and coin_profile.take_profit_pct is not None:
            raw_tp = coin_profile.take_profit_pct
            try:
                resolved_tp = float(raw_tp)
            except Exception as exc:
                self._emit_exception_context_log(
                    component="config_profile",
                    action="parse_take_profit_pct_coin_profile",
                    symbol=normalized_symbol,
                    strategy_name=resolve_strategy_for_symbol(normalized_symbol, self._strategy_name),
                    interval=self._target_interval_for_symbol(normalized_symbol),
                    exc=exc,
                    severity="WARN",
                )
            else:
                if math.isfinite(resolved_tp) and resolved_tp > 0.0:
                    return resolved_tp
                self._emit_warning_once(
                    key=f"invalid_take_profit_pct:coin_profile:{normalized_symbol}:{raw_tp}",
                    message=(
                        "Invalid non-positive coin_profile take_profit_pct detected; "
                        f"symbol={normalized_symbol} value={raw_tp!r}. Falling back to global default."
                    ),
                )
        return float(settings.trading.take_profit_pct)

    def _resolve_configured_stop_loss_pct(self, symbol: str) -> float:
        normalized_symbol = str(symbol).strip().upper()
        production_profile = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol)
        if isinstance(production_profile, Mapping):
            raw_sl = production_profile.get("stop_loss_pct", 0.0)
            try:
                resolved_sl = float(raw_sl or 0.0)
            except Exception as exc:
                self._emit_exception_context_log(
                    component="config_profile",
                    action="parse_stop_loss_pct_production",
                    symbol=normalized_symbol,
                    strategy_name=resolve_strategy_for_symbol(normalized_symbol, self._strategy_name),
                    interval=self._target_interval_for_symbol(normalized_symbol),
                    exc=exc,
                    severity="WARN",
                )
            else:
                if math.isfinite(resolved_sl) and resolved_sl > 0.0:
                    return resolved_sl
                if raw_sl not in (None, "", 0, 0.0, "0", "0.0"):
                    self._emit_warning_once(
                        key=f"invalid_stop_loss_pct:production:{normalized_symbol}:{raw_sl}",
                        message=(
                            "Invalid non-positive production stop_loss_pct detected; "
                            f"symbol={normalized_symbol} value={raw_sl!r}. Falling back to coin/default settings."
                        ),
                    )
        coin_profile = settings.trading.coin_profiles.get(normalized_symbol)
        if coin_profile is not None and coin_profile.stop_loss_pct is not None:
            raw_sl = coin_profile.stop_loss_pct
            try:
                resolved_sl = float(raw_sl)
            except Exception as exc:
                self._emit_exception_context_log(
                    component="config_profile",
                    action="parse_stop_loss_pct_coin_profile",
                    symbol=normalized_symbol,
                    strategy_name=resolve_strategy_for_symbol(normalized_symbol, self._strategy_name),
                    interval=self._target_interval_for_symbol(normalized_symbol),
                    exc=exc,
                    severity="WARN",
                )
            else:
                if math.isfinite(resolved_sl) and resolved_sl > 0.0:
                    return resolved_sl
                self._emit_warning_once(
                    key=f"invalid_stop_loss_pct:coin_profile:{normalized_symbol}:{raw_sl}",
                    message=(
                        "Invalid non-positive coin_profile stop_loss_pct detected; "
                        f"symbol={normalized_symbol} value={raw_sl!r}. Falling back to global default."
                    ),
                )
        return float(settings.trading.stop_loss_pct)

    def _build_trade_lifecycle_snapshot(self, trade: PaperTrade) -> dict[str, object]:
        if self._db is None:
            return {
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
                "bars_in_trade": 0,
                "exit_trigger_category": "unknown",
                "did_reach_partial_tp": False,
                "best_unrealized_pnl": 0.0,
                "worst_unrealized_pnl": 0.0,
            }
        if trade.exit_time is None:
            return {
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
                "bars_in_trade": 0,
                "exit_trigger_category": "open",
                "did_reach_partial_tp": False,
                "best_unrealized_pnl": 0.0,
                "worst_unrealized_pnl": 0.0,
            }

        timeframe = (
            str(trade.timeframe).strip()
            if trade.timeframe is not None and str(trade.timeframe).strip()
            else self._target_interval_for_symbol(trade.symbol)
        )
        interval_seconds = max(_interval_total_seconds(timeframe), 60)
        start_time = trade.entry_time - timedelta(seconds=interval_seconds)
        end_time = trade.exit_time + timedelta(seconds=interval_seconds)
        candles = self._db.fetch_candles_since(
            trade.symbol,
            timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=6000,
        )
        side = str(trade.side).strip().upper()
        entry_price = max(float(trade.entry_price), 1e-9)
        max_favorable_pct = 0.0
        max_adverse_pct = 0.0
        best_unrealized_pnl = 0.0
        worst_unrealized_pnl = 0.0
        for candle in candles:
            if side == "LONG":
                favorable_pct = ((float(candle.high) / entry_price) - 1.0) * 100.0
                adverse_pct = ((float(candle.low) / entry_price) - 1.0) * 100.0
            else:
                favorable_pct = ((entry_price / max(float(candle.low), 1e-9)) - 1.0) * 100.0
                adverse_pct = ((entry_price / max(float(candle.high), 1e-9)) - 1.0) * 100.0
                adverse_pct = min(0.0, adverse_pct)
            max_favorable_pct = max(max_favorable_pct, float(favorable_pct))
            adverse_abs = abs(min(0.0, float(adverse_pct)))
            max_adverse_pct = max(max_adverse_pct, adverse_abs)
            best_unrealized_pnl = max(
                best_unrealized_pnl,
                float(favorable_pct) / 100.0 * entry_price * float(trade.qty),
            )
            worst_unrealized_pnl = min(
                worst_unrealized_pnl,
                -adverse_abs / 100.0 * entry_price * float(trade.qty),
            )

        exit_status = str(trade.status or "").strip().upper()
        if exit_status in {
            "CLOSED_SL",
            "BREAKEVEN_STOP",
            "TRAILING_STOP",
            "TIGHT_TRAILING_STOP",
            "INTRABAR_SL",
            "INTRABAR_TRAILING",
            "INTRABAR_BREAKEVEN",
        }:
            exit_trigger_category = "stop"
        elif exit_status in {"CLOSED_TP", "INTRABAR_TP"}:
            exit_trigger_category = "take_profit"
        elif exit_status in {"STRATEGY_EXIT", "MANUAL_CLOSE"}:
            exit_trigger_category = "manual_or_strategy"
        else:
            exit_trigger_category = "other"

        configured_tp_pct = self._resolve_configured_take_profit_pct(trade.symbol)
        did_reach_partial_tp = bool(max_favorable_pct >= max(0.25, configured_tp_pct * 0.5))
        return {
            "max_favorable_excursion": float(max_favorable_pct),
            "max_adverse_excursion": float(max_adverse_pct),
            "bars_in_trade": int(len(candles)),
            "exit_trigger_category": str(exit_trigger_category),
            "did_reach_partial_tp": bool(did_reach_partial_tp),
            "best_unrealized_pnl": float(best_unrealized_pnl),
            "worst_unrealized_pnl": float(worst_unrealized_pnl),
            "configured_take_profit_pct": float(configured_tp_pct),
            "configured_stop_loss_pct": float(self._resolve_configured_stop_loss_pct(trade.symbol)),
        }

    def _ensure_lifecycle_snapshot(self, trade: PaperTrade) -> dict[str, object]:
        existing_payload = self._parse_json_payload(trade.lifecycle_snapshot_json)
        required_fields = {
            "max_favorable_excursion",
            "max_adverse_excursion",
            "bars_in_trade",
            "exit_trigger_category",
            "did_reach_partial_tp",
            "best_unrealized_pnl",
            "worst_unrealized_pnl",
        }
        if required_fields.issubset(existing_payload.keys()):
            return existing_payload
        lifecycle_snapshot = self._build_trade_lifecycle_snapshot(trade)
        resolved_strategy_name = (
            str(trade.strategy_name).strip()
            if trade.strategy_name is not None and str(trade.strategy_name).strip()
            else resolve_strategy_for_symbol(trade.symbol, self._strategy_name)
        )
        resolved_interval = (
            str(trade.timeframe).strip()
            if trade.timeframe is not None and str(trade.timeframe).strip()
            else self._target_interval_for_symbol(trade.symbol)
        )
        if self._db is not None:
            try:
                self._db.update_trade(
                    trade.id,
                    PaperTradeUpdate(lifecycle_snapshot_json=lifecycle_snapshot),
                )
                self._clear_meta_service_degraded(
                    service_name="lifecycle_snapshot_persist",
                    symbol=trade.symbol,
                    strategy_name=resolved_strategy_name,
                    interval=resolved_interval,
                )
            except Exception as exc:
                self._mark_meta_service_degraded(
                    service_name="lifecycle_snapshot_persist",
                    symbol=trade.symbol,
                    strategy_name=resolved_strategy_name,
                    interval=resolved_interval,
                    exc=exc,
                )
        return lifecycle_snapshot

    def review_closed_trade(self, trade_id: int) -> dict[str, object] | None:
        if self._db is None:
            return None
        trade = self._db.fetch_trade_by_id(trade_id)
        if trade is None or str(trade.status).upper() == "OPEN":
            return None

        lifecycle_snapshot = self._ensure_lifecycle_snapshot(trade)
        entry_snapshot = self._parse_json_payload(trade.entry_snapshot_json)
        recent_range_metrics = self._parse_json_payload(entry_snapshot.get("recent_range_metrics"))

        strategy_name = (
            str(trade.strategy_name).strip()
            if trade.strategy_name is not None and str(trade.strategy_name).strip()
            else resolve_strategy_for_symbol(trade.symbol, self._strategy_name)
        )
        timeframe = (
            str(trade.timeframe).strip()
            if trade.timeframe is not None and str(trade.timeframe).strip()
            else self._target_interval_for_symbol(trade.symbol)
        )
        regime_label = (
            str(trade.regime_label_at_entry).strip()
            if trade.regime_label_at_entry is not None and str(trade.regime_label_at_entry).strip()
            else str(entry_snapshot.get("regime_label_at_entry", "range_balanced"))
        )
        trend_bias = str(entry_snapshot.get("trend_bias", "neutral") or "neutral")
        signal_strength = float(
            trade.signal_strength
            if trade.signal_strength is not None
            else float(entry_snapshot.get("signal_strength", 0.0) or 0.0)
        )
        confidence_score = float(
            trade.confidence_score
            if trade.confidence_score is not None
            else float(entry_snapshot.get("confidence_score", 0.0) or 0.0)
        )
        move_already_extended_pct = float(
            trade.move_already_extended_pct
            if trade.move_already_extended_pct is not None
            else float(entry_snapshot.get("move_already_extended_pct", 0.0) or 0.0)
        )
        max_favorable_excursion = float(lifecycle_snapshot.get("max_favorable_excursion", 0.0) or 0.0)
        max_adverse_excursion = float(lifecycle_snapshot.get("max_adverse_excursion", 0.0) or 0.0)
        best_unrealized_pnl = float(lifecycle_snapshot.get("best_unrealized_pnl", 0.0) or 0.0)
        worst_unrealized_pnl = float(lifecycle_snapshot.get("worst_unrealized_pnl", 0.0) or 0.0)
        did_reach_partial_tp = bool(lifecycle_snapshot.get("did_reach_partial_tp", False))
        trade_pnl = float(trade.pnl or 0.0)
        stop_loss_pct = float(lifecycle_snapshot.get("configured_stop_loss_pct", self._resolve_configured_stop_loss_pct(trade.symbol)) or 0.0)
        take_profit_pct = float(lifecycle_snapshot.get("configured_take_profit_pct", self._resolve_configured_take_profit_pct(trade.symbol)) or 0.0)

        side = str(trade.side).strip().upper()
        entry_against_regime = bool(
            (side == "LONG" and (regime_label == "trend_clean_down" or trend_bias == "bearish"))
            or (side == "SHORT" and (regime_label == "trend_clean_up" or trend_bias == "bullish"))
        )
        strategy_regime_mismatch = not self._strategy_regime_matches(strategy_name, regime_label)
        regime_mismatch_flag = bool(entry_against_regime or strategy_regime_mismatch)
        overextended_entry_flag = bool(
            move_already_extended_pct >= max(1.2, stop_loss_pct * 0.8)
            or regime_label in {"trend_exhausted", "panic_expansion"}
        )
        late_entry_flag = bool(
            overextended_entry_flag
            or float(recent_range_metrics.get("move_last_1_bar_pct", 0.0) or 0.0) >= 1.0
            or float(recent_range_metrics.get("move_last_2_bars_pct", 0.0) or 0.0) >= 1.6
        )
        insufficient_confirmation = bool(confidence_score < 0.45 or signal_strength < 0.45)
        strategy_coin_mismatch = False
        profile = PRODUCTION_PROFILE_REGISTRY.get(str(trade.symbol).strip().upper())
        if isinstance(profile, Mapping):
            profile_strategy = str(profile.get("strategy_name", "") or "").strip()
            profile_strategy = STRATEGY_NAME_ALIASES.get(profile_strategy, profile_strategy)
            strategy_coin_mismatch = bool(profile_strategy and profile_strategy != strategy_name)

        status_upper = str(trade.status or "").strip().upper()
        stop_exit_status = status_upper in {
            "CLOSED_SL",
            "BREAKEVEN_STOP",
            "TRAILING_STOP",
            "TIGHT_TRAILING_STOP",
            "INTRABAR_SL",
            "INTRABAR_TRAILING",
            "INTRABAR_BREAKEVEN",
        }
        sl_too_tight_flag = bool(
            stop_exit_status
            and max_favorable_excursion >= max(0.6, stop_loss_pct * 0.75)
            and max_adverse_excursion <= max(0.6, stop_loss_pct * 0.9)
        )
        sl_too_wide_flag = bool(
            trade_pnl < 0.0
            and max_adverse_excursion > max(0.75, stop_loss_pct * 1.35)
        )
        tp_too_conservative = bool(
            trade_pnl > 0.0
            and best_unrealized_pnl > trade_pnl * 1.7
            and max_favorable_excursion >= max(0.5, take_profit_pct * 0.5)
        )
        tp_too_ambitious = bool(
            trade_pnl < 0.0
            and stop_exit_status
            and max_favorable_excursion < max(0.25, take_profit_pct * 0.25)
            and not did_reach_partial_tp
        )
        better_no_trade_flag = bool(
            trade_pnl < 0.0
            and confidence_score < 0.6
            and (
                regime_mismatch_flag
                or late_entry_flag
                or overextended_entry_flag
                or insufficient_confirmation
                or strategy_coin_mismatch
            )
        )
        avoidable_loss_flag = bool(
            trade_pnl < 0.0
            and (
                better_no_trade_flag
                or late_entry_flag
                or regime_mismatch_flag
                or overextended_entry_flag
                or insufficient_confirmation
                or sl_too_tight_flag
            )
        )
        better_exit_possible_flag = bool(tp_too_conservative or tp_too_ambitious or sl_too_tight_flag or sl_too_wide_flag)

        error_hits: list[str] = []
        if better_no_trade_flag:
            error_hits.append("should_not_have_traded")
        if strategy_coin_mismatch:
            error_hits.append("strategy_coin_mismatch")
        if strategy_regime_mismatch:
            error_hits.append("strategy_regime_mismatch")
        if entry_against_regime:
            error_hits.append("entry_against_regime")
        if regime_label in {"trend_exhausted", "panic_expansion"}:
            error_hits.append("entry_into_exhaustion")
        if late_entry_flag:
            error_hits.append("late_entry_after_expansion")
        if insufficient_confirmation:
            error_hits.append("entry_without_sufficient_confirmation")
        if sl_too_tight_flag:
            error_hits.append("stop_too_tight")
        if sl_too_wide_flag:
            error_hits.append("stop_too_wide")
        if tp_too_conservative:
            error_hits.append("tp_too_conservative")
        if tp_too_ambitious:
            error_hits.append("tp_too_ambitious")
        error_hits = [error for error in error_hits if error in REVIEW_ERROR_CATALOG]
        error_type_primary = error_hits[0] if error_hits else None
        error_type_secondary = error_hits[1] if len(error_hits) > 1 else None

        trade_quality_score = 55.0
        if trade_pnl > 0.0:
            trade_quality_score += 18.0
        elif trade_pnl < 0.0:
            trade_quality_score -= 22.0
        if confidence_score >= 0.65:
            trade_quality_score += 6.0
        if signal_strength >= 0.65:
            trade_quality_score += 4.0
        if late_entry_flag:
            trade_quality_score -= 10.0
        if regime_mismatch_flag:
            trade_quality_score -= 12.0
        if overextended_entry_flag:
            trade_quality_score -= 8.0
        if insufficient_confirmation:
            trade_quality_score -= 8.0
        if sl_too_tight_flag:
            trade_quality_score -= 7.0
        if sl_too_wide_flag:
            trade_quality_score -= 7.0
        if better_no_trade_flag:
            trade_quality_score -= 10.0
        if avoidable_loss_flag:
            trade_quality_score -= 8.0
        trade_quality_score = float(max(0.0, min(100.0, trade_quality_score)))

        notes_auto = {
            "engine_version": TRADE_REVIEW_ENGINE_VERSION,
            "error_catalog": list(REVIEW_ERROR_CATALOG),
            "triggered_errors": list(error_hits),
            "regime_label": regime_label,
            "trend_bias": trend_bias,
            "strategy_name": strategy_name,
            "timeframe": timeframe,
            "confidence_score": float(confidence_score),
            "signal_strength": float(signal_strength),
            "move_already_extended_pct": float(move_already_extended_pct),
            "max_favorable_excursion": float(max_favorable_excursion),
            "max_adverse_excursion": float(max_adverse_excursion),
            "best_unrealized_pnl": float(best_unrealized_pnl),
            "worst_unrealized_pnl": float(worst_unrealized_pnl),
            "trade_pnl": float(trade_pnl),
        }
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        review_create = TradeReviewCreate(
            trade_id=int(trade.id),
            created_at=now,
            updated_at=now,
            symbol=str(trade.symbol).strip().upper(),
            strategy_name=strategy_name,
            timeframe=timeframe,
            trade_quality_score=float(trade_quality_score),
            error_type_primary=error_type_primary,
            error_type_secondary=error_type_secondary,
            better_no_trade_flag=bool(better_no_trade_flag),
            better_exit_possible_flag=bool(better_exit_possible_flag),
            late_entry_flag=bool(late_entry_flag),
            regime_mismatch_flag=bool(regime_mismatch_flag),
            overextended_entry_flag=bool(overextended_entry_flag),
            sl_too_tight_flag=bool(sl_too_tight_flag),
            sl_too_wide_flag=bool(sl_too_wide_flag),
            avoidable_loss_flag=bool(avoidable_loss_flag),
            notes_auto_json=notes_auto,
            review_engine_version=TRADE_REVIEW_ENGINE_VERSION,
        )
        self._db.upsert_trade_review(review_create)
        self._db.update_trade(
            trade.id,
            PaperTradeUpdate(
                strategy_name=strategy_name,
                timeframe=timeframe,
                lifecycle_snapshot_json=lifecycle_snapshot,
                review_status="REVIEWED_AUTO",
            ),
        )
        return {
            "trade_id": int(trade.id),
            "trade_quality_score": float(trade_quality_score),
            "error_type_primary": error_type_primary,
            "error_type_secondary": error_type_secondary,
            "avoidable_loss_flag": bool(avoidable_loss_flag),
        }

    @staticmethod
    def _strategy_regime_matches(strategy_name: str, regime_label: str) -> bool:
        normalized_strategy = _sanitize_backtest_strategy_name(strategy_name)
        normalized_regime = str(regime_label).strip()
        if normalized_regime not in REGIME_LABEL_UNIVERSE:
            return True
        expected_map: dict[str, set[str]] = {
            "frama_cross": {
                "trend_clean_up",
                "trend_clean_down",
                "breakout_transition",
                "trend_exhausted",
            },
            "dual_thrust": {
                "breakout_transition",
                "compression_pre_breakout",
                "range_volatile",
                "panic_expansion",
            },
            "ema_band_rejection": {
                "range_balanced",
                "range_volatile",
                "compression_pre_breakout",
                "trend_exhausted",
            },
            "ema_cross_volume": {
                "trend_clean_up",
                "trend_clean_down",
                "breakout_transition",
                "range_volatile",
            },
        }
        expected = expected_map.get(normalized_strategy)
        if expected is None:
            return True
        return normalized_regime in expected

    @staticmethod
    def _max_loss_streak(trades_desc: Sequence[PaperTrade]) -> int:
        max_streak = 0
        running_streak = 0
        for trade in reversed(list(trades_desc)):
            trade_pnl = float(trade.pnl or 0.0)
            if trade_pnl < 0.0:
                running_streak += 1
                max_streak = max(max_streak, running_streak)
            else:
                running_streak = 0
        return int(max_streak)

    def recompute_strategy_health(
        self,
        *,
        symbol: str,
        strategy_name: str,
        timeframe: str,
    ) -> StrategyHealthSnapshot | None:
        if self._db is None:
            return None
        normalized_symbol = str(symbol).strip().upper()
        normalized_strategy = _sanitize_backtest_strategy_name(strategy_name)
        normalized_timeframe = str(timeframe).strip()

        production_profile = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol)
        configured_window = 60
        if isinstance(production_profile, Mapping):
            with suppress(Exception):
                configured_window = int(float(production_profile.get("health_window_size", configured_window) or configured_window))
        window_size = max(20, min(320, configured_window))

        closed_trades_desc = self._db.fetch_closed_trades(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            timeframe=normalized_timeframe,
            limit=window_size,
        )
        trade_count = int(len(closed_trades_desc))
        trade_ids = [int(trade.id) for trade in closed_trades_desc]
        reviews = self._db.fetch_trade_reviews_for_trade_ids(trade_ids)
        review_by_trade_id = {int(review.trade_id): review for review in reviews}
        review_count = int(len(reviews))
        regime_observations = self._db.fetch_recent_regime_observations(
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            limit=max(window_size * 3, 120),
        )

        pnl_sum = float(sum(float(trade.pnl or 0.0) for trade in closed_trades_desc))
        wins = int(sum(1 for trade in closed_trades_desc if float(trade.pnl or 0.0) > 0.0))
        winrate = float((wins / trade_count) * 100.0) if trade_count > 0 else 0.0
        avg_pnl = float(pnl_sum / trade_count) if trade_count > 0 else 0.0
        avg_fees = (
            float(sum(float(trade.total_fees or 0.0) for trade in closed_trades_desc) / trade_count)
            if trade_count > 0
            else 0.0
        )
        late_entry_count = int(
            sum(1 for trade_id, review in review_by_trade_id.items() if review.late_entry_flag and trade_id in trade_ids)
        )
        regime_mismatch_count = int(
            sum(1 for trade_id, review in review_by_trade_id.items() if review.regime_mismatch_flag and trade_id in trade_ids)
        )
        avoidable_loss_count = int(
            sum(1 for trade_id, review in review_by_trade_id.items() if review.avoidable_loss_flag and trade_id in trade_ids)
        )
        error_count = int(
            sum(
                1
                for trade_id, review in review_by_trade_id.items()
                if review.error_type_primary and trade_id in trade_ids
            )
        )
        late_entry_rate = float(late_entry_count / trade_count) if trade_count > 0 else 0.0
        regime_mismatch_rate = float(regime_mismatch_count / trade_count) if trade_count > 0 else 0.0
        avoidable_loss_rate = float(avoidable_loss_count / trade_count) if trade_count > 0 else 0.0
        error_rate = float(error_count / trade_count) if trade_count > 0 else 0.0
        max_loss_streak = self._max_loss_streak(closed_trades_desc)

        regime_fit_trials = 0
        regime_fit_hits = 0
        for trade in closed_trades_desc:
            if trade.regime_label_at_entry is None:
                continue
            regime_fit_trials += 1
            if self._strategy_regime_matches(normalized_strategy, str(trade.regime_label_at_entry)):
                regime_fit_hits += 1
        regime_fit_rate = (
            float(regime_fit_hits / regime_fit_trials)
            if regime_fit_trials > 0
            else 0.5
        )
        dominant_regime = "unknown"
        if regime_observations:
            regime_counts: dict[str, int] = {}
            for observation in regime_observations:
                regime_counts[observation.regime_label] = regime_counts.get(observation.regime_label, 0) + 1
            if regime_counts:
                dominant_regime = max(regime_counts.items(), key=lambda item: item[1])[0]

        sample_soft_floor = 8.0
        sample_full_weight = 40.0
        sample_factor = float(
            max(
                0.0,
                min(
                    1.0,
                    (float(trade_count) - sample_soft_floor) / max(sample_full_weight - sample_soft_floor, 1.0),
                ),
            )
        )
        profitability_component = 50.0
        profitability_component += max(-22.0, min(22.0, (winrate - 50.0) * 0.55))
        profitability_component += max(
            -20.0,
            min(
                20.0,
                (avg_pnl / max(abs(avg_pnl) + 5.0, 5.0)) * 20.0,
            ),
        )
        profitability_component += max(
            -16.0,
            min(
                16.0,
                (pnl_sum / max(float(settings.trading.start_capital), 1.0)) * 140.0,
            ),
        )
        error_penalty = (
            (late_entry_rate * 12.0)
            + (regime_mismatch_rate * 18.0)
            + (avoidable_loss_rate * 22.0)
            + (error_rate * 16.0)
            + (float(max_loss_streak) * 2.5)
        )
        regime_bonus = (regime_fit_rate - 0.5) * 18.0
        raw_health_score = profitability_component + regime_bonus - error_penalty
        health_score = 50.0 + (raw_health_score - 50.0) * (0.20 + (0.80 * sample_factor))
        health_score = float(max(0.0, min(100.0, health_score)))

        # Mode B: low-sample data should stay neutral/healthy unless we have solid evidence.
        low_sample_observe_only = trade_count < META_MIN_SAMPLE_SIZE_FOR_PROMOTION
        state = "healthy"
        if low_sample_observe_only:
            state = "healthy"
        elif health_score < 40.0 or max_loss_streak >= 7:
            state = "degraded"
        elif health_score < 55.0:
            state = "watchlist"
        elif health_score < 68.0:
            state = "degraded"

        risk_multiplier = 1.0
        if state == "healthy":
            if low_sample_observe_only:
                risk_multiplier = 1.0
            else:
                risk_multiplier = float(max(0.95, min(1.10, 0.98 + ((health_score - 68.0) / 32.0) * 0.12)))
        elif state == "degraded":
            risk_multiplier = 0.90
        elif state == "watchlist":
            risk_multiplier = 1.00

        last_review_at: datetime | None = None
        for review in reviews:
            if last_review_at is None or review.updated_at > last_review_at:
                last_review_at = review.updated_at
        profile_version = self._resolve_profile_version(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            timeframe=normalized_timeframe,
        )
        health_features = {
            "engine_version": STRATEGY_HEALTH_ENGINE_VERSION,
            "sample_factor": float(sample_factor),
            "review_count": int(review_count),
            "review_coverage": float(review_count / trade_count) if trade_count > 0 else 0.0,
            "regime_fit_rate": float(regime_fit_rate),
            "regime_fit_trials": int(regime_fit_trials),
            "dominant_recent_regime": str(dominant_regime),
            "regime_observation_count": int(len(regime_observations)),
            "profitability_component": float(profitability_component),
            "error_penalty": float(error_penalty),
            "regime_bonus": float(regime_bonus),
            "raw_health_score": float(raw_health_score),
            "low_sample_observe_only": bool(low_sample_observe_only),
        }
        snapshot = StrategyHealthSnapshot(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            timeframe=normalized_timeframe,
            computed_at=datetime.now(tz=UTC).replace(tzinfo=None),
            trades_count=trade_count,
            pnl_sum=float(pnl_sum),
            winrate=float(winrate),
            avg_pnl=float(avg_pnl),
            avg_fees=float(avg_fees),
            late_entry_rate=float(late_entry_rate),
            regime_mismatch_rate=float(regime_mismatch_rate),
            avoidable_loss_rate=float(avoidable_loss_rate),
            error_rate=float(error_rate),
            max_loss_streak=int(max_loss_streak),
            health_score=float(health_score),
            risk_multiplier=float(risk_multiplier),
            state=str(state),
            last_review_at=last_review_at,
            window_size=int(window_size),
            health_features_json=health_features,
            profile_version=profile_version,
        )
        previous_snapshot = self._db.fetch_strategy_health(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            timeframe=normalized_timeframe,
        )
        self._db.upsert_strategy_health(snapshot)
        previous_state = None if previous_snapshot is None else str(previous_snapshot.state)
        if previous_state is not None and previous_state != state:
            with suppress(Exception):
                self._db.insert_adaptation_log(
                    AdaptationLogCreate(
                        created_at=datetime.now(tz=UTC).replace(tzinfo=None),
                        symbol=normalized_symbol,
                        strategy_name=normalized_strategy,
                        timeframe=normalized_timeframe,
                        event_type="health_state_transition",
                        payload_json={
                            "previous_state": previous_state,
                            "new_state": state,
                            "previous_risk_multiplier": float(previous_snapshot.risk_multiplier),
                            "new_risk_multiplier": float(risk_multiplier),
                            "health_score": float(health_score),
                            "trade_count": int(trade_count),
                        },
                        source=STRATEGY_HEALTH_ENGINE_VERSION,
                    )
                )
        return snapshot

    def _compute_strategy_drawdown_pct(
        self,
        *,
        strategy_name: str,
        interval: str,
        max_trades: int = 200,
    ) -> float:
        if self._db is None:
            return 0.0
        closed_trades_desc = self._db.fetch_closed_trades(
            strategy_name=_sanitize_backtest_strategy_name(strategy_name),
            timeframe=str(interval).strip(),
            limit=max(20, int(max_trades)),
        )
        if not closed_trades_desc:
            return 0.0
        equity = float(settings.trading.start_capital)
        peak = float(equity)
        max_drawdown_pct = 0.0
        for trade in reversed(closed_trades_desc):
            equity += float(trade.pnl or 0.0)
            if equity > peak:
                peak = float(equity)
            elif peak > 0.0:
                drawdown_pct = ((peak - equity) / peak) * 100.0
                if drawdown_pct > max_drawdown_pct:
                    max_drawdown_pct = float(drawdown_pct)
        return float(max_drawdown_pct)

    def _evaluate_global_meta_guards(
        self,
        *,
        symbol: str,
        strategy_name: str,
        interval: str,
    ) -> dict[str, object]:
        if self._db is None:
            return {"allow_trade": True, "risk_cap": 1.0, "block_reason": "", "warning_reason": "", "flags": []}
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        day_start = datetime(now.year, now.month, now.day)
        flags: list[str] = []
        allow_trade = True
        risk_cap = 1.0
        block_reason = ""
        warning_reason = ""
        normalized_symbol = str(symbol).strip().upper()
        normalized_strategy = _sanitize_backtest_strategy_name(strategy_name)
        normalized_interval = str(interval).strip()
        meta_key = self._meta_key(normalized_symbol, normalized_strategy, normalized_interval)

        reconcile_payload = self._symbol_reconcile_required.get(normalized_symbol)
        reconcile_reason = (
            str(reconcile_payload.get("reason", "") or "reconcile_required")
            if isinstance(reconcile_payload, Mapping)
            else ""
        )
        if reconcile_reason:
            reconciled, reconcile_resolution = self._attempt_symbol_reconciliation(
                symbol=normalized_symbol,
                strategy_name=normalized_strategy,
                interval=normalized_interval,
            )
            if reconciled:
                self._clear_symbol_reconcile_required(
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    resolution=f"auto_reconciled:{reconcile_resolution}",
                )
                reconcile_reason = ""
            else:
                return {
                    "allow_trade": False,
                    "risk_cap": 0.0,
                    "block_reason": f"reconcile_required:{reconcile_reason}:{reconcile_resolution}",
                    "warning_reason": "",
                    "flags": ["reconcile_required", "position_state_unknown"],
                }

        self._maybe_recover_fail_safe_mode(now=now)
        if self._meta_fail_safe_active:
            return {
                "allow_trade": False,
                "risk_cap": 0.0,
                "block_reason": f"fail_safe_active:{self._meta_fail_safe_reason}",
                "warning_reason": "",
                "flags": ["fail_safe_active"],
            }

        daily_realized_pnl = float(self._db.fetch_realized_pnl_since(day_start))
        if daily_realized_pnl <= -abs(META_MAX_DAILY_LOSS_USD):
            allow_trade = False
            flags.append("max_daily_loss")
            block_reason = f"max_daily_loss:{daily_realized_pnl:.2f}"
            self._emit_meta_warning(
                warning_code="max_daily_loss",
                message=f"Daily loss limit reached ({daily_realized_pnl:.2f} <= {-abs(META_MAX_DAILY_LOSS_USD):.2f}).",
                payload={"daily_realized_pnl": daily_realized_pnl},
            )

        if allow_trade:
            symbol_recent = self._db.fetch_recent_closed_trades(
                normalized_symbol,
                limit=max(20, META_MAX_SYMBOL_LOSS_STREAK + 2),
            )
            symbol_loss_streak = 0
            for trade in symbol_recent:
                if float(trade.pnl or 0.0) < 0.0:
                    symbol_loss_streak += 1
                else:
                    break
            if symbol_loss_streak >= META_MAX_SYMBOL_LOSS_STREAK:
                allow_trade = False
                flags.append("max_symbol_loss_streak")
                block_reason = f"max_symbol_loss_streak:{symbol_loss_streak}"

        strategy_drawdown_pct = self._compute_strategy_drawdown_pct(
            strategy_name=normalized_strategy,
            interval=normalized_interval,
        )
        if allow_trade and strategy_drawdown_pct > META_MAX_STRATEGY_DRAWDOWN_PCT:
            allow_trade = False
            flags.append("max_strategy_drawdown")
            block_reason = f"max_strategy_drawdown:{strategy_drawdown_pct:.2f}%"

        open_trades = self._db.fetch_open_trades()
        runtime_trade_ids: set[int] = set()
        if self._paper_engine is not None:
            for runtime_trade in self._paper_engine.active_trades:
                candidate_id = getattr(runtime_trade, "id", runtime_trade)
                with suppress(TypeError, ValueError):
                    runtime_trade_ids.add(int(candidate_id))
            db_trade_ids = {int(trade.id) for trade in open_trades}
            if runtime_trade_ids != db_trade_ids:
                self._activate_fail_safe_mode(
                    reason="positions_desync",
                    payload={
                        "runtime_trade_ids": sorted(runtime_trade_ids),
                        "db_trade_ids": sorted(db_trade_ids),
                    },
                )
                return {
                    "allow_trade": False,
                    "risk_cap": 0.0,
                    "block_reason": "positions_desync",
                    "warning_reason": "",
                    "flags": ["positions_desync"],
                    "daily_realized_pnl": float(daily_realized_pnl),
                    "strategy_drawdown_pct": float(strategy_drawdown_pct),
                    "open_positions": int(len(open_trades)),
                }
        for open_trade in open_trades:
            high_water_mark = float(open_trade.high_water_mark or 0.0)
            if not math.isfinite(high_water_mark) or high_water_mark <= 0.0:
                self._activate_fail_safe_mode(
                    reason="missing_exit_control",
                    payload={
                        "trade_id": int(open_trade.id),
                        "symbol": str(open_trade.symbol),
                        "high_water_mark": float(high_water_mark),
                    },
                )
                return {
                    "allow_trade": False,
                    "risk_cap": 0.0,
                    "block_reason": f"missing_exit_control:trade={int(open_trade.id)}",
                    "warning_reason": "",
                    "flags": ["missing_exit_control"],
                    "daily_realized_pnl": float(daily_realized_pnl),
                    "strategy_drawdown_pct": float(strategy_drawdown_pct),
                    "open_positions": int(len(open_trades)),
                }
        open_by_symbol: Counter[str] = Counter(str(trade.symbol).strip().upper() for trade in open_trades)
        duplicate_open_symbols = [sym for sym, count in open_by_symbol.items() if count > 1]
        if duplicate_open_symbols:
            self._activate_fail_safe_mode(
                reason="duplicate_entries_detected",
                payload={"symbols": duplicate_open_symbols},
            )
            return {
                "allow_trade": False,
                "risk_cap": 0.0,
                "block_reason": f"duplicate_entries:{','.join(duplicate_open_symbols)}",
                "warning_reason": "",
                "flags": ["duplicate_entries_detected"],
                "daily_realized_pnl": float(daily_realized_pnl),
                "strategy_drawdown_pct": float(strategy_drawdown_pct),
                "open_positions": int(len(open_trades)),
            }
        if allow_trade and len(open_trades) >= META_MAX_OPEN_POSITIONS_GUARD:
            allow_trade = False
            flags.append("max_open_positions")
            block_reason = f"max_open_positions:{len(open_trades)}"

        side_counter: Counter[str] = Counter(str(trade.side).strip().upper() for trade in open_trades)
        correlated_open = max(int(side_counter.get("LONG", 0)), int(side_counter.get("SHORT", 0)))
        if correlated_open >= META_MAX_CORRELATED_RISK:
            risk_cap = min(risk_cap, 0.5)
            flags.append("correlated_risk")
            if not warning_reason:
                warning_reason = f"correlated_risk:{correlated_open}"

        pause_events_today = self._db.fetch_adaptation_logs(
            event_type="fail_safe_activated",
            since=day_start,
            limit=1000,
        )
        pause_event_count = 0
        for entry in pause_events_today:
            payload = self._parse_json_payload(entry.payload_json)
            reason = str(payload.get("reason", "") or "")
            if reason:
                pause_event_count += 1
        if allow_trade and pause_event_count >= META_MAX_PAUSE_EVENTS_PER_DAY:
            allow_trade = False
            flags.append("max_pause_events_per_day")
            block_reason = f"max_pause_events_per_day:{pause_event_count}"

        cooldown_until = self._meta_cooldown_until.get(meta_key)
        if allow_trade and isinstance(cooldown_until, datetime) and now < cooldown_until:
            allow_trade = False
            flags.append("cooldown_active")
            block_reason = f"cooldown_active_until:{cooldown_until.isoformat()}"

        return {
            "allow_trade": bool(allow_trade),
            "risk_cap": float(max(0.0, min(risk_cap, 1.0))),
            "block_reason": str(block_reason),
            "warning_reason": str(warning_reason),
            "flags": flags,
            "daily_realized_pnl": float(daily_realized_pnl),
            "strategy_drawdown_pct": float(strategy_drawdown_pct),
            "open_positions": int(len(open_trades)),
        }

    def evaluate_meta_policy(
        self,
        symbol: str,
        strategy_name: str,
        interval: str,
        current_context: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_symbol, normalized_strategy, normalized_interval = self._meta_key(
            symbol,
            strategy_name,
            interval,
        )
        context = dict(current_context or {})
        regime_payload = self._parse_json_payload(context.get("regime_payload"))
        regime_label = str(regime_payload.get("regime_label", "range_balanced") or "range_balanced")
        regime_confidence = float(regime_payload.get("regime_confidence", 0.0) or 0.0)
        health_snapshot = None if self._db is None else self._db.fetch_strategy_health(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            timeframe=normalized_interval,
        )
        trades_count = int(health_snapshot.trades_count) if health_snapshot is not None else 0
        low_sample_observe_only = trades_count < META_MIN_SAMPLE_SIZE_FOR_PROMOTION
        state = "healthy"
        risk_multiplier = 1.0
        allow_trade = True
        block_reason = ""
        warning_reason = ""
        meta_flags: list[str] = []
        negative_signals: set[str] = set()
        hard_guard_block = False

        if health_snapshot is not None:
            snapshot_state = str(health_snapshot.state).strip().lower() or "healthy"
            if snapshot_state == "paper_only":
                snapshot_state = "watchlist"
                meta_flags.append("paper_only_mapped_to_watchlist")
            if snapshot_state == "paused":
                snapshot_state = "watchlist"
                meta_flags.append("paused_health_mapped_to_watchlist")
            state = snapshot_state
            risk_multiplier = float(health_snapshot.risk_multiplier or 1.0)
            if state == "healthy":
                allow_trade = True
                risk_multiplier = max(0.95, min(risk_multiplier, 1.10))
            elif state == "degraded":
                allow_trade = True
                risk_multiplier = max(0.80, min(risk_multiplier, 0.95))
            elif state == "watchlist":
                allow_trade = True
                risk_multiplier = max(0.95, min(risk_multiplier, 1.05))
            else:
                state = "healthy"
                allow_trade = True
                risk_multiplier = 1.0
        else:
            meta_flags.append("health_missing")
            state = "healthy"
            allow_trade = True
            risk_multiplier = 1.0

        if low_sample_observe_only:
            meta_flags.append("low_sample_observe_only")

        degraded_services = self._collect_meta_service_degraded(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            interval=normalized_interval,
        )
        if degraded_services:
            meta_flags.append("meta_services_degraded")
            for service_name in degraded_services:
                meta_flags.append(f"meta_service_degraded:{service_name}")
            if state == "healthy":
                state = "degraded"
            risk_multiplier = min(risk_multiplier, 0.95)
            if not warning_reason:
                warning_reason = "meta_services_degraded:" + ",".join(degraded_services)

        if health_snapshot is not None:
            late_rate = float(health_snapshot.late_entry_rate)
            mismatch_rate = float(health_snapshot.regime_mismatch_rate)
            avoidable_rate = float(health_snapshot.avoidable_loss_rate)
            if mismatch_rate >= 0.55 or late_rate >= 0.55 or avoidable_rate >= 0.55:
                negative_signals.add("error_rates_high")
                if state == "healthy":
                    state = "degraded"
                risk_multiplier = min(risk_multiplier, 0.90)
                warning_reason = (
                    f"error_rates_high:late={late_rate:.2f}|regime={mismatch_rate:.2f}|avoidable={avoidable_rate:.2f}"
                )
                meta_flags.append("error_rates_high_observe")
            elif mismatch_rate >= 0.35 or late_rate >= 0.35 or avoidable_rate >= 0.35:
                negative_signals.add("error_rates_warn")
                risk_multiplier = min(risk_multiplier, 0.95)
                if state == "healthy":
                    state = "degraded"
                warning_reason = (
                    f"error_rates_warn:late={late_rate:.2f}|regime={mismatch_rate:.2f}|avoidable={avoidable_rate:.2f}"
                )
                meta_flags.append("error_rates_warn")
            if int(health_snapshot.max_loss_streak or 0) >= 4:
                negative_signals.add("loss_streak_alert")
                if state == "healthy":
                    state = "watchlist"
                risk_multiplier = min(risk_multiplier, 0.95)
                if not warning_reason:
                    warning_reason = f"loss_streak_alert:{int(health_snapshot.max_loss_streak or 0)}"
                meta_flags.append("loss_streak_alert")

        if regime_confidence >= 0.60 and not self._strategy_regime_matches(normalized_strategy, regime_label):
            negative_signals.add("regime_mismatch")
            risk_multiplier = min(risk_multiplier, 0.95)
            if state == "healthy":
                state = "watchlist"
            warning_reason = f"regime_mismatch:{regime_label}"
            meta_flags.append("regime_veto_warn_soft")
        if regime_confidence >= 0.55 and regime_label == "illiquid_noise":
            negative_signals.add("illiquid_noise")
            risk_multiplier = min(risk_multiplier, 0.95)
            if state == "healthy":
                state = "watchlist"
            if not warning_reason:
                warning_reason = "illiquid_noise"
            meta_flags.append("illiquid_noise_warn_soft")

        if low_sample_observe_only:
            if negative_signals:
                if state == "healthy":
                    state = "watchlist"
                risk_multiplier = min(risk_multiplier, 0.95)
                meta_flags.append("low_sample_with_negative_evidence")
            else:
                # Neutral low-sample mode: keep normal papertrading behavior and visible healthy state.
                state = "healthy"
                allow_trade = True
                risk_multiplier = 1.0
                if warning_reason == "health_snapshot_missing":
                    warning_reason = ""

        guard_payload = self._evaluate_global_meta_guards(
            symbol=normalized_symbol,
            strategy_name=normalized_strategy,
            interval=normalized_interval,
        )
        risk_multiplier = min(risk_multiplier, float(guard_payload.get("risk_cap", 1.0) or 1.0))
        if not bool(guard_payload.get("allow_trade", True)):
            hard_guard_block = True
            allow_trade = False
            state = "paused"
            risk_multiplier = 0.0
            block_reason = str(guard_payload.get("block_reason", "") or "hard_global_guard")
            meta_flags.append("hard_guard_block")
        guard_warning = str(guard_payload.get("warning_reason", "") or "")
        if guard_warning and not warning_reason:
            warning_reason = guard_warning
        guard_flags = guard_payload.get("flags")
        if isinstance(guard_flags, list):
            meta_flags.extend(str(item) for item in guard_flags)

        if allow_trade:
            block_reason = ""
            if state == "paused":
                state = "watchlist"
                meta_flags.append("paused_deescalated_mode_b")
        risk_multiplier = max(0.0, min(float(risk_multiplier), 1.25))

        unique_meta_flags = sorted(set(str(item) for item in meta_flags))
        is_low_sample_info_only = (
            state == "healthy"
            and bool(allow_trade)
            and not block_reason
            and not warning_reason
            and abs(float(risk_multiplier) - 1.0) <= 1e-9
            and unique_meta_flags in (["low_sample_observe_only"], ["health_missing", "low_sample_observe_only"])
        )
        guard_signature = ""
        if (meta_flags or block_reason or warning_reason) and not is_low_sample_info_only:
            guard_signature = "|".join(
                [
                    str(state),
                    str(block_reason),
                    str(warning_reason),
                    ",".join(unique_meta_flags),
                ]
            )
        guard_key = (normalized_symbol, normalized_strategy, normalized_interval)
        previous_guard_signature = self._meta_last_state_by_key.get(guard_key)
        if guard_signature:
            if guard_signature != previous_guard_signature:
                self._meta_last_state_by_key[guard_key] = guard_signature
                self._insert_adaptation_event(
                    event_type="meta_guard_triggered",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    payload={
                        "state": str(state),
                        "block_reason": str(block_reason),
                        "warning_reason": str(warning_reason),
                        "meta_flags": sorted(set(meta_flags)),
                        "allow_trade": bool(allow_trade),
                        "risk_multiplier": float(risk_multiplier),
                    },
                    source="meta_bot",
                )
        elif previous_guard_signature is not None:
            self._meta_last_state_by_key.pop(guard_key, None)

        effective_policy_json = {
            "symbol": normalized_symbol,
            "strategy_name": normalized_strategy,
            "interval": normalized_interval,
            "allow_trade": bool(allow_trade),
            "risk_multiplier": float(risk_multiplier),
            "state": str(state),
            "block_reason": str(block_reason),
            "warning_reason": str(warning_reason),
            "meta_flags": sorted(set(meta_flags)),
            "health_snapshot_available": bool(health_snapshot is not None),
            "health_score": (
                None
                if health_snapshot is None
                else float(health_snapshot.health_score)
            ),
            "trades_count": int(trades_count),
            "regime_label": regime_label,
            "regime_confidence": float(regime_confidence),
            "global_guards": dict(guard_payload),
            "timestamp": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
        }
        policy = {
            "allow_trade": bool(allow_trade),
            "risk_multiplier": float(risk_multiplier),
            "state": str(state),
            "block_reason": str(block_reason),
            "warning_reason": str(warning_reason),
            "meta_flags": sorted(set(meta_flags)),
            "effective_policy_json": effective_policy_json,
        }
        previous_policy = self._meta_policy_cache.get((normalized_symbol, normalized_strategy, normalized_interval))
        self._meta_policy_cache[(normalized_symbol, normalized_strategy, normalized_interval)] = dict(policy)
        previous_state = (
            None
            if previous_policy is None
            else str(previous_policy.get("state", ""))
        )
        previous_risk = (
            None
            if previous_policy is None
            else float(previous_policy.get("risk_multiplier", 1.0) or 1.0)
        )
        if previous_state != str(state) or previous_risk is None or abs(previous_risk - risk_multiplier) > 1e-9:
            self._insert_adaptation_event(
                event_type="meta_state_change",
                symbol=normalized_symbol,
                strategy_name=normalized_strategy,
                interval=normalized_interval,
                payload={
                    "previous_state": previous_state,
                    "new_state": str(state),
                    "previous_risk_multiplier": previous_risk,
                    "new_risk_multiplier": float(risk_multiplier),
                    "allow_trade": bool(allow_trade),
                    "block_reason": str(block_reason),
                    "warning_reason": str(warning_reason),
                    "meta_flags": sorted(set(meta_flags)),
                },
                source="meta_bot",
            )
            self.log_message.emit(
                "[META] policy update: "
                f"{normalized_symbol} {normalized_strategy} {normalized_interval} "
                f"state={state} allow={int(bool(allow_trade))} risk={risk_multiplier:.2f} "
                f"block={block_reason or '-'} warn={warning_reason or '-'}"
            )
            if previous_risk is not None and float(risk_multiplier) < float(previous_risk):
                self.build_learning_log_entry(
                    change_type="Risk reduziert",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    details={
                        "previous_risk_multiplier": float(previous_risk),
                        "new_risk_multiplier": float(risk_multiplier),
                        "state": str(state),
                    },
                )
            if str(state) == "paused":
                self._emit_meta_warning(
                    warning_code=f"health_state_{state}",
                    message=(
                        f"Meta state changed to {state} for "
                        f"{normalized_symbol} {normalized_strategy} {normalized_interval}."
                    ),
                    payload={"policy": dict(effective_policy_json)},
                )
                self.build_learning_log_entry(
                    change_type="Symbol pausiert",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    details={"state": str(state), "block_reason": str(block_reason)},
                )
            if "low_sample_observe_only" in meta_flags:
                if "low_sample_with_negative_evidence" in meta_flags:
                    self._emit_meta_warning(
                        warning_code="low_sample_observe_only",
                        message=(
                            f"Low sample + negative evidence for "
                            f"{normalized_symbol} {normalized_strategy} {normalized_interval}."
                        ),
                        payload={
                            "trades_count": trades_count,
                            "state": str(state),
                            "warning_reason": str(warning_reason),
                        },
                    )
                else:
                    self.log_message.emit(
                        "[META] low-sample observe-only: "
                        f"{normalized_symbol} {normalized_strategy} {normalized_interval} "
                        f"trades={trades_count}"
                    )
            if "regime_veto_warn_soft" in meta_flags:
                self.build_learning_log_entry(
                    change_type="Regime-Veto aktiv",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    details={
                        "regime_label": regime_label,
                        "regime_confidence": regime_confidence,
                        "meta_flags": sorted(set(meta_flags)),
                    },
                )
            if state == "paused" and (not allow_trade) and hard_guard_block:
                cooldown_until = datetime.now(tz=UTC).replace(tzinfo=None) + timedelta(
                    minutes=META_COOLDOWN_AFTER_PAUSE_MINUTES
                )
                self._meta_cooldown_until[(normalized_symbol, normalized_strategy, normalized_interval)] = cooldown_until
                self._insert_adaptation_event(
                    event_type="meta_cooldown_started",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    payload={"cooldown_until": cooldown_until.isoformat()},
                    source="meta_bot",
                )
                self.build_learning_log_entry(
                    change_type="Cooldown gestartet",
                    symbol=normalized_symbol,
                    strategy_name=normalized_strategy,
                    interval=normalized_interval,
                    details={"cooldown_until": cooldown_until.isoformat()},
                )
        return policy

    def build_learning_log_entry(
        self,
        *,
        change_type: str,
        symbol: str,
        strategy_name: str,
        interval: str,
        details: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        entry = {
            "timestamp": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
            "change_type": str(change_type),
            "symbol": str(symbol).strip().upper(),
            "strategy_name": _sanitize_backtest_strategy_name(strategy_name),
            "interval": str(interval).strip(),
            "details": dict(details or {}),
        }
        self._insert_adaptation_event(
            event_type="learning_log_entry",
            symbol=str(symbol).strip().upper(),
            strategy_name=_sanitize_backtest_strategy_name(strategy_name),
            interval=str(interval).strip(),
            payload=entry,
            source="meta_reports",
        )
        return entry

    def _render_meta_report_markdown(self, report: Mapping[str, object], *, title: str) -> str:
        lines = [f"# {title}", ""]
        lines.append(f"- Generated At: {report.get('generated_at')}")
        lines.append(f"- Report Type: {report.get('report_type')}")
        lines.append("")
        overview = report.get("overview")
        if isinstance(overview, Mapping):
            lines.append("## Overview")
            for key, value in overview.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        rows = report.get("rows")
        if isinstance(rows, list) and rows:
            header = [
                "symbol",
                "strategy_name",
                "interval",
                "trade_count",
                "pnl_sum",
                "winrate",
                "late_entry_rate",
                "regime_mismatch_rate",
                "avoidable_loss_rate",
                "health_state",
                "risk_multiplier",
            ]
            lines.append("## Rows")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join("---" for _ in header) + "|")
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                values = [str(row.get(column, "")) for column in header]
                lines.append("| " + " | ".join(values) + " |")
            lines.append("")
        highlights = report.get("highlights")
        if isinstance(highlights, list) and highlights:
            lines.append("## Highlights")
            for item in highlights:
                lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _persist_meta_report_files(
        self,
        *,
        report: Mapping[str, object],
        report_prefix: str,
        markdown_title: str,
    ) -> dict[str, str]:
        META_REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        generated_at = str(report.get("generated_at", datetime.now(tz=UTC).replace(tzinfo=None).isoformat()))
        safe_stamp = generated_at.replace(":", "").replace("-", "").replace("T", "_").replace(".", "")
        json_path = META_REPORTS_DIRECTORY / f"{report_prefix}_{safe_stamp}.json"
        md_path = META_REPORTS_DIRECTORY / f"{report_prefix}_{safe_stamp}.md"
        txt_path = META_REPORTS_DIRECTORY / f"{report_prefix}_{safe_stamp}.txt"
        json_path.write_text(
            json.dumps(dict(report), ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        markdown_text = self._render_meta_report_markdown(report, title=markdown_title)
        md_path.write_text(markdown_text, encoding="utf-8")
        txt_path.write_text(markdown_text, encoding="utf-8")
        return {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "text": txt_path.as_posix(),
        }

    def build_daily_meta_report(self, *, report_date: datetime | None = None) -> dict[str, object]:
        if self._db is None:
            return {}
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        if report_date is None:
            report_date = now
        day_start = datetime(report_date.year, report_date.month, report_date.day)
        day_end = day_start + timedelta(days=1)
        closed_trades = self._db.fetch_closed_trades_since(
            start_time=day_start,
            end_time=day_end,
            limit=5000,
        )
        grouped: dict[tuple[str, str, str], list[PaperTrade]] = {}
        for trade in closed_trades:
            strategy_name = (
                str(trade.strategy_name).strip()
                if trade.strategy_name is not None and str(trade.strategy_name).strip()
                else resolve_strategy_for_symbol(trade.symbol, self._strategy_name)
            )
            interval = (
                str(trade.timeframe).strip()
                if trade.timeframe is not None and str(trade.timeframe).strip()
                else self._target_interval_for_symbol(trade.symbol)
            )
            key = self._meta_key(trade.symbol, strategy_name, interval)
            grouped.setdefault(key, []).append(trade)

        rows: list[dict[str, object]] = []
        total_pnl = 0.0
        for key, trades in grouped.items():
            symbol, strategy_name, interval = key
            pnl_sum = float(sum(float(trade.pnl or 0.0) for trade in trades))
            total_pnl += pnl_sum
            wins = int(sum(1 for trade in trades if float(trade.pnl or 0.0) > 0.0))
            trade_count = int(len(trades))
            winrate = float((wins / trade_count) * 100.0) if trade_count > 0 else 0.0
            trade_reviews = self._db.fetch_recent_trade_reviews(
                symbol=symbol,
                strategy_name=strategy_name,
                timeframe=interval,
                limit=max(50, trade_count * 4),
            )
            review_for_ids = [item for item in trade_reviews if int(item.get("trade_id", -1)) in {int(t.id) for t in trades}]
            error_counter: Counter[str] = Counter()
            late_count = 0
            mismatch_count = 0
            avoidable_count = 0
            for review in review_for_ids:
                primary = str(review.get("error_type_primary", "") or "").strip()
                if primary:
                    error_counter[primary] += 1
                if bool(review.get("late_entry_flag", False)):
                    late_count += 1
                if bool(review.get("regime_mismatch_flag", False)):
                    mismatch_count += 1
                if bool(review.get("avoidable_loss_flag", False)):
                    avoidable_count += 1
            health = self._db.fetch_strategy_health(
                symbol=symbol,
                strategy_name=strategy_name,
                timeframe=interval,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "interval": interval,
                    "trade_count": trade_count,
                    "pnl_sum": round(pnl_sum, 6),
                    "winrate": round(winrate, 4),
                    "error_type_distribution": dict(error_counter),
                    "late_entry_rate": round(float(late_count / trade_count) if trade_count > 0 else 0.0, 6),
                    "regime_mismatch_rate": round(float(mismatch_count / trade_count) if trade_count > 0 else 0.0, 6),
                    "avoidable_loss_rate": round(float(avoidable_count / trade_count) if trade_count > 0 else 0.0, 6),
                    "health_state": None if health is None else str(health.state),
                    "risk_multiplier": None if health is None else float(health.risk_multiplier),
                }
            )
        rows.sort(key=lambda row: float(row.get("pnl_sum", 0.0) or 0.0))
        overview_counts = self._db.fetch_meta_overview_counts()
        paper_only_count_raw = int(overview_counts.get("paper_only", 0))
        report = {
            "report_type": "daily_meta_report",
            "generated_at": now.isoformat(),
            "date": day_start.date().isoformat(),
            "overview": {
                "daily_meta_pnl": round(total_pnl, 6),
                "closed_trades": int(len(closed_trades)),
                "healthy_count": int(overview_counts.get("healthy", 0)),
                "degraded_count": int(overview_counts.get("degraded", 0)),
                "watchlist_count": int(overview_counts.get("watchlist", 0)) + paper_only_count_raw,
                "paper_only_count": 0,
                "paused_count": int(overview_counts.get("paused", 0)),
            },
            "rows": rows,
            "highlights": [
                f"Worst row pnl: {rows[0]['symbol']} {rows[0]['strategy_name']} {rows[0]['interval']} -> {rows[0]['pnl_sum']:.4f}"
                if rows
                else "No rows available.",
                f"Best row pnl: {rows[-1]['symbol']} {rows[-1]['strategy_name']} {rows[-1]['interval']} -> {rows[-1]['pnl_sum']:.4f}"
                if rows
                else "No rows available.",
            ],
        }
        report_paths = self._persist_meta_report_files(
            report=report,
            report_prefix="daily_meta_report",
            markdown_title="Daily Meta Report",
        )
        report["files"] = report_paths
        return report

    def build_weekly_meta_report(self, *, end_time: datetime | None = None) -> dict[str, object]:
        if self._db is None:
            return {}
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        if end_time is None:
            end_time = now
        start_time = end_time - timedelta(days=7)
        closed_trades = self._db.fetch_closed_trades_since(
            start_time=start_time,
            end_time=end_time,
            limit=20_000,
        )
        reviews = self._db.fetch_recent_trade_reviews(limit=5000)
        review_trade_ids = {int(item.get("trade_id", -1)) for item in reviews}
        weekly_trade_ids = {int(trade.id) for trade in closed_trades}
        relevant_reviews = [item for item in reviews if int(item.get("trade_id", -1)) in weekly_trade_ids]
        error_counter: Counter[str] = Counter()
        for review in relevant_reviews:
            primary = str(review.get("error_type_primary", "") or "").strip()
            if primary:
                error_counter[primary] += 1
        health_rows = self._db.fetch_strategy_health_rows(limit=2000)
        paused_rows = [
            row for row in health_rows if str(row.state).strip().lower() == "paused"
        ]
        paper_only_rows: list[StrategyHealthSnapshot] = []
        adaptation_rows = self._db.fetch_adaptation_logs(since=start_time, limit=5000)
        deterioration = [
            entry
            for entry in adaptation_rows
            if str(entry.event_type).strip().lower() in {"meta_state_change", "fail_safe_activated"}
        ]
        improvement = [
            entry
            for entry in adaptation_rows
            if str(entry.event_type).strip().lower() in {"learning_log_entry"}
        ]
        weekly_pnl = float(sum(float(trade.pnl or 0.0) for trade in closed_trades))
        challenger_hypotheses: list[str] = []
        for error_name, count in error_counter.most_common(3):
            if error_name in {"late_entry_after_expansion", "entry_into_exhaustion"}:
                challenger_hypotheses.append(
                    f"Tighten late-entry veto thresholds for regime transitions ({error_name}, count={count})."
                )
            elif error_name in {"entry_against_regime", "strategy_regime_mismatch"}:
                challenger_hypotheses.append(
                    f"Re-evaluate regime fit mapping and challenger fallback for mismatch-heavy pairs ({error_name}, count={count})."
                )
            else:
                challenger_hypotheses.append(
                    f"Run focused challenger sweep for recurring error '{error_name}' (count={count})."
                )
        report = {
            "report_type": "weekly_meta_report",
            "generated_at": now.isoformat(),
            "window_start": start_time.isoformat(),
            "window_end": end_time.isoformat(),
            "overview": {
                "weekly_meta_pnl": round(weekly_pnl, 6),
                "closed_trades": int(len(closed_trades)),
                "review_coverage": round(float(len(relevant_reviews) / max(len(weekly_trade_ids), 1)), 6),
                "paused_symbols": int(len(paused_rows)),
                "paper_only_symbols": 0,
                "deterioration_events": int(len(deterioration)),
                "improvement_events": int(len(improvement)),
            },
            "top_error_types": dict(error_counter.most_common(12)),
            "paused_rows": [
                {
                    "symbol": row.symbol,
                    "strategy_name": row.strategy_name,
                    "interval": row.timeframe,
                    "health_score": row.health_score,
                    "risk_multiplier": row.risk_multiplier,
                }
                for row in paused_rows
            ],
            "paper_only_rows": [
                {
                    "symbol": row.symbol,
                    "strategy_name": row.strategy_name,
                    "interval": row.timeframe,
                    "health_score": row.health_score,
                    "risk_multiplier": row.risk_multiplier,
                }
                for row in paper_only_rows
            ],
            "challenger_hypotheses": challenger_hypotheses,
            "highlights": [
                f"Weekly pnl: {weekly_pnl:.2f}",
                f"Top error: {next(iter(error_counter.keys()), 'n/a')}",
                f"Paused rows: {len(paused_rows)}, paper_only rows: 0",
            ],
        }
        report_paths = self._persist_meta_report_files(
            report=report,
            report_prefix="weekly_meta_report",
            markdown_title="Weekly Meta Report",
        )
        report["files"] = report_paths
        return report

    def _maybe_generate_periodic_meta_reports(self) -> None:
        if self._db is None:
            return
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        today = now.date()
        iso_week = now.isocalendar()[:2]
        if self._meta_last_daily_report_date != today:
            daily_report = self.build_daily_meta_report(report_date=now)
            self._meta_last_daily_report_date = today
            self.log_message.emit(
                f"[META_REPORT] daily generated: {daily_report.get('files', {}).get('markdown', 'n/a')}"
            )
        if self._meta_last_weekly_report_iso != iso_week:
            weekly_report = self.build_weekly_meta_report(end_time=now)
            self._meta_last_weekly_report_iso = iso_week
            self.log_message.emit(
                f"[META_REPORT] weekly generated: {weekly_report.get('files', {}).get('markdown', 'n/a')}"
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
    _LOG_MODE_QUIET = "quiet"
    _LOG_MODE_STANDARD = "standard"
    _LOG_MODE_DEBUG = "debug"
    _HISTORY_PROGRESS_LOG_PREFIX = "[HISTORY_PROGRESS]"
    _SIGNAL_CACHE_PROGRESS_LOG_PREFIX = "[SIGNAL_CACHE_PROGRESS]"
    _PRIMARY_DB_PERSIST_MAX_ATTEMPTS = 120
    _PRIMARY_DB_PERSIST_RETRY_SECONDS = 0.5
    _PRIMARY_DB_PERSIST_LOG_EVERY_ATTEMPTS = 10

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        strategy_name: str,
        auto_from_config: bool = False,
        allow_auto_interval_override: bool = True,
        enforce_interval_fairness: bool = False,
        leverage: int | None = None,
        min_confidence_pct: float | None = None,
        optimize_profile: bool = False,
        isolated_db: bool = False,
        db_path: str | Path = "data/paper_trading.duckdb",
        history_requested_start_utc: datetime | None = None,
        history_end_utc: datetime | None = None,
        log_mode: str = _LOG_MODE_STANDARD,
        batch_history_cache: dict[str, dict[str, object]] | None = None,
        batch_history_cache_lock: RLock | None = None,
    ) -> None:
        super().__init__()
        self._symbol = symbol
        resolved_interval = _validate_interval_name(interval)
        self._optimize_profile = optimize_profile
        self._interval = resolved_interval
        self._strategy_name = _coerce_backtest_strategy_name(
            strategy_name,
            fallback_strategy_name="frama_cross",
        )
        self._auto_from_config = bool(auto_from_config)
        self._allow_auto_interval_override = bool(allow_auto_interval_override)
        self._enforce_interval_fairness = bool(enforce_interval_fairness)
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
        if self._use_setup_gate and self._optimize_profile:
            self._min_confidence_pct = _enforce_optimizer_min_confidence_floor(
                self._min_confidence_pct
            )
        self._setup_gate = (
            SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
            if self._use_setup_gate
            else None
        )
        self._optimization_grid: list[OptimizationProfile] = []
        self._db_path = Path(db_path)
        self._isolated_db = isolated_db
        self._history_requested_start_utc = _normalize_optional_utc_naive(
            history_requested_start_utc
        )
        self._history_end_utc = _normalize_optional_utc_naive(history_end_utc)
        self._active_pool: object | None = None
        self._stop_requested = False
        self._last_precomputed_exit_cache: dict[str, object] | None = None
        self._hmm_regime_mask: list[int] | None = None
        self._hmm_regime_labels: list[str] | None = None
        self._last_runtime_trace: dict[str, object] | None = None
        self._last_strategy_diagnostics: dict[str, object] | None = None
        self._checkpoint_backtest_run_id: int | None = None
        self._history_progress_last_emit_monotonic = 0.0
        self._history_progress_last_percent = -1
        self._history_progress_last_stage = ""
        self._log_mode = self._normalize_log_mode(log_mode)
        self._batch_history_cache = (
            batch_history_cache
            if isinstance(batch_history_cache, dict)
            else None
        )
        self._batch_history_cache_lock = batch_history_cache_lock

    @classmethod
    def _normalize_log_mode(cls, raw_mode: str | None) -> str:
        normalized = str(raw_mode or "").strip().lower()
        if normalized in {cls._LOG_MODE_QUIET, cls._LOG_MODE_STANDARD, cls._LOG_MODE_DEBUG}:
            return normalized
        return cls._LOG_MODE_STANDARD

    def _is_quiet_mode(self) -> bool:
        return self._log_mode == self._LOG_MODE_QUIET

    def _is_debug_mode(self) -> bool:
        return self._log_mode == self._LOG_MODE_DEBUG

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

    def _resolve_backtest_interval_for_symbol(self, symbol: str) -> tuple[str, str]:
        if self._enforce_interval_fairness:
            return self._interval, "live_vs_challenger fairness lock"
        if not self._auto_from_config:
            return self._interval, "configured interval"
        if not self._allow_auto_interval_override:
            return self._interval, "batch-selected interval (auto interval override disabled)"

        normalized_symbol = str(symbol).strip().upper()
        production_profile = PRODUCTION_PROFILE_REGISTRY.get(normalized_symbol)
        if isinstance(production_profile, dict):
            profile_interval = str(production_profile.get("interval", "") or "").strip()
            if profile_interval:
                return _validate_interval_name(profile_interval), "production profile (Auto from Config)"

        # Fallback to coin profile resolution if production registry entry is missing/incomplete.
        return (
            resolve_interval_for_symbol(
                normalized_symbol,
                self._interval,
                use_coin_override=True,
            ),
            "coin profile fallback (Auto from Config)",
        )

    def _emit_trace_event(self, event_name: str, payload: dict[str, object]) -> None:
        if not self._is_debug_mode():
            return
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

    def _resolve_requested_history_window(self) -> tuple[datetime, datetime | None]:
        requested_start = (
            _history_start_for_mode(optimize_profile=self._optimize_profile)
            if self._history_requested_start_utc is None
            else self._history_requested_start_utc
        )
        requested_end = self._history_end_utc
        if requested_end is not None and requested_end <= requested_start:
            interval_seconds = max(_interval_total_seconds(self._interval), 60)
            requested_end = requested_start + timedelta(seconds=interval_seconds)
        return requested_start, requested_end

    def _resolve_history_sync_window(
        self,
        *,
        requested_start_utc: datetime,
        requested_end_utc: datetime | None,
        required_warmup_candles: int,
    ) -> tuple[datetime, datetime | None]:
        sync_start = requested_start_utc
        if required_warmup_candles > 0:
            interval_seconds = max(_interval_total_seconds(self._interval), 60)
            sync_start = requested_start_utc - timedelta(
                seconds=required_warmup_candles * interval_seconds
            )
        return sync_start, requested_end_utc

    def _apply_history_warmup_correction(
        self,
        db: Database,
        *,
        requested_start_utc: datetime,
        requested_end_utc: datetime | None,
        required_warmup_candles: int,
    ) -> tuple[datetime, int]:
        if required_warmup_candles <= 0:
            return requested_start_utc, 0
        _count, oldest_time, _newest_time = db.get_candle_stats(self._symbol, self._interval)
        if oldest_time is None:
            return requested_start_utc, 0

        interval_seconds = max(_interval_total_seconds(self._interval), 60)
        corrected_start = requested_start_utc
        shifted_candles = 0
        for _ in range(4):
            available_warmup = db.count_candles_in_range(
                self._symbol,
                self._interval,
                start_time=oldest_time,
                end_time=corrected_start,
            )
            missing_candles = max(0, required_warmup_candles - int(available_warmup))
            if missing_candles <= 0:
                break
            corrected_start += timedelta(seconds=missing_candles * interval_seconds)
            shifted_candles += missing_candles
            if requested_end_utc is not None and corrected_start >= requested_end_utc:
                corrected_start = max(
                    requested_start_utc,
                    requested_end_utc - timedelta(seconds=interval_seconds),
                )
                break
        return corrected_start, shifted_candles

    @contextmanager
    def _batch_history_cache_guard(self):
        lock = self._batch_history_cache_lock
        if lock is None:
            yield
            return
        with lock:
            yield

    def _build_batch_history_cache_key(
        self,
        *,
        requested_start_utc: datetime,
        requested_end_utc: datetime | None,
        explicit_history_window: bool,
    ) -> str | None:
        if self._batch_history_cache is None:
            return None
        start_token = requested_start_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_token = (
            requested_end_utc.strftime("%Y-%m-%d %H:%M:%S")
            if requested_end_utc is not None
            else "latest"
        )
        window_mode_token = "date_range" if explicit_history_window else "tail_window"
        setup_gate_token = "setup_gate_on" if self._use_setup_gate else "setup_gate_off"
        optimize_token = "optimizer" if self._optimize_profile else "backtest"
        return (
            f"{self._symbol}|{self._interval}|{window_mode_token}|"
            f"{start_token}|{end_token}|{setup_gate_token}|{optimize_token}"
        )

    def _load_batch_history_cache_entry(
        self,
        cache_key: str | None,
    ) -> dict[str, object] | None:
        if cache_key is None or self._batch_history_cache is None:
            return None
        with self._batch_history_cache_guard():
            cache_entry = self._batch_history_cache.get(cache_key)
            if not isinstance(cache_entry, dict):
                return None
            return cache_entry

    def _store_batch_history_cache_entry(
        self,
        cache_key: str | None,
        cache_entry: dict[str, object],
    ) -> None:
        if cache_key is None or self._batch_history_cache is None:
            return
        with self._batch_history_cache_guard():
            self._batch_history_cache[cache_key] = cache_entry

    def _load_candles_dataframe_from_batch_history_cache(
        self,
        *,
        cache_key: str | None,
        required_candles: int,
        allow_insufficient: bool = False,
    ) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
        cache_entry = self._load_batch_history_cache_entry(cache_key)
        if cache_entry is None:
            return None, None
        cached_candles_df = cache_entry.get("candles_df")
        if not isinstance(cached_candles_df, pd.DataFrame):
            return None, None
        if cached_candles_df.empty:
            return None, None
        if len(cached_candles_df) < required_candles and not allow_insufficient:
            return None, None
        cached_metadata = cache_entry.get("metadata")
        if not isinstance(cached_metadata, dict):
            return None, None
        return cached_candles_df.copy(deep=True), dict(cached_metadata)

    def _cache_history_and_prepare_run_dataframe(
        self,
        *,
        cache_key: str | None,
        candles: Sequence[CandleRecord],
        metadata: dict[str, object],
    ) -> pd.DataFrame:
        base_candles_df = _candles_to_dataframe(candles).copy(deep=True)
        if cache_key is not None and self._batch_history_cache is not None:
            cache_entry: dict[str, object] = {
                "candles_df": base_candles_df,
                "metadata": dict(metadata),
                "regime_masks": {},
                "regime_labels": {},
            }
            self._store_batch_history_cache_entry(cache_key, cache_entry)
        return base_candles_df.copy(deep=True)

    def _resolve_hmm_regime_mask_with_batch_cache(
        self,
        *,
        cache_key: str | None,
        candles_df: pd.DataFrame,
        strategy_name: str,
    ) -> list[int] | None:
        self._hmm_regime_labels = None
        cache_entry = self._load_batch_history_cache_entry(cache_key)
        if cache_entry is not None:
            cached_masks = cache_entry.get("regime_masks")
            if not isinstance(cached_masks, dict):
                cached_masks = {}
                cache_entry["regime_masks"] = cached_masks
            if strategy_name in cached_masks:
                cached_mask = cached_masks.get(strategy_name)
                cached_labels_map = cache_entry.get("regime_labels")
                cached_labels = None
                if isinstance(cached_labels_map, dict):
                    cached_labels = cached_labels_map.get(strategy_name)
                if isinstance(cached_mask, list) and len(cached_mask) == len(candles_df):
                    if isinstance(cached_labels, list) and len(cached_labels) == len(candles_df):
                        self._hmm_regime_labels = list(cached_labels)
                    self.log_message.emit(
                        "Batch history cache hit: "
                        f"reusing regime payload for {self._symbol} {self._interval} ({strategy_name})."
                    )
                    return list(cached_mask)
                if cached_mask is None:
                    self.log_message.emit(
                        "Batch history cache hit: "
                        f"reusing neutral regime payload for {self._symbol} {self._interval} ({strategy_name})."
                    )
                    return None

        resolved_mask = self._prepare_hmm_regime_mask(
            candles_df,
            strategy_name=strategy_name,
        )
        if cache_entry is not None:
            cached_masks = cache_entry.get("regime_masks")
            if not isinstance(cached_masks, dict):
                cached_masks = {}
                cache_entry["regime_masks"] = cached_masks
            cached_masks[strategy_name] = (
                list(resolved_mask)
                if isinstance(resolved_mask, list)
                else None
            )
            cached_labels_map = cache_entry.get("regime_labels")
            if not isinstance(cached_labels_map, dict):
                cached_labels_map = {}
                cache_entry["regime_labels"] = cached_labels_map
            cached_labels_map[strategy_name] = (
                list(self._hmm_regime_labels)
                if isinstance(self._hmm_regime_labels, list)
                else None
            )
        return resolved_mask

    def _resolve_strategy_diagnostics_payload(
        self,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None = None,
        diagnostics_source: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        _ = strategy_name, strategy_profile
        if isinstance(diagnostics_source, dict):
            return dict(diagnostics_source)
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
                                        self._reset_history_download_progress()
                                        history_manager.sync_recent_candles(
                                            symbols=[symbol],
                                            interval=interval,
                                            candles_per_symbol=MAX_BACKTEST_CANDLES,
                                            should_abort=self._should_stop,
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
                            required_candles = _required_candle_count_for_strategy(
                                strategy_name,
                                use_setup_gate=self._use_setup_gate,
                            )
                            if len(candles_df) < required_candles:
                                self.log_message.emit(
                                    f"Not enough candles for {symbol} {strategy_name}: have {len(candles_df)}, need {required_candles}."
                                )
                                continue

                            # Configure runtime state for this run
                            self._symbol = symbol
                            self._interval = interval
                            self._strategy_name = _coerce_backtest_strategy_name(
                                strategy_name,
                                fallback_strategy_name=self._strategy_name,
                            )
                            if self._use_setup_gate:
                                self._min_confidence_pct = settings.trading.min_confidence_pct
                                self._setup_gate = SmartSetupGate(
                                    min_confidence_pct=self._min_confidence_pct
                                )
                            else:
                                self._min_confidence_pct = None
                                self._setup_gate = None
                            self._hmm_regime_mask = self._prepare_hmm_regime_mask(
                                candles_df,
                                strategy_name=self._strategy_name,
                            )

                            # Build and possibly cap optimization grid
                            self._optimization_grid = generate_optimization_grid(
                                self._strategy_name,
                                symbol=symbol,
                                interval=interval,
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
        self._checkpoint_backtest_run_id = None
        try:
            working_db_path = self._db_path
            if self._isolated_db:
                temp_workspace = TemporaryDirectory(prefix="backtest_")
                working_db_path = Path(temp_workspace.name) / "isolated_backtest.duckdb"
                self.log_message.emit(
                    "Live bot active: running backtest on isolated temp DB to avoid lock contention."
                )

            with Database(working_db_path) as db:
                if self._use_setup_gate:
                    self._setup_gate = SmartSetupGate(
                        min_confidence_pct=self._min_confidence_pct
                    )
                resolved_interval, interval_source = self._resolve_backtest_interval_for_symbol(
                    self._symbol
                )
                if self._enforce_interval_fairness:
                    self.log_message.emit(
                        "Live vs Challenger interval fairness lock active: "
                        f"{self._symbol} fixed to {resolved_interval} "
                        f"(source={interval_source})."
                    )
                elif resolved_interval != self._interval:
                    self.log_message.emit(
                        "Auto interval routing active: "
                        f"{self._symbol} -> {resolved_interval} "
                        f"(source={interval_source}); GUI interval ignored."
                    )
                elif self._auto_from_config:
                    self.log_message.emit(
                        "Auto interval routing active: "
                        f"{self._symbol} -> {resolved_interval} (source={interval_source})."
                    )
                self._interval = resolved_interval
                resolved_strategy_name = _sanitize_backtest_strategy_name(
                    resolve_strategy_for_symbol(
                        self._symbol,
                        self._strategy_name,
                        use_coin_override=False,
                    ),
                    fallback_strategy_name=self._strategy_name,
                )
                selected_strategy_profile: OptimizationProfile | None = None
                if (
                    not self._optimize_profile
                    and resolved_strategy_name == "ema_band_rejection"
                    and not self._enforce_interval_fairness
                ):
                    resolved_winner_profile = resolve_ema_band_rejection_1h_winner_profile(
                        self._symbol
                    )
                    if resolved_winner_profile is not None:
                        winner_interval, winner_profile = resolved_winner_profile
                        selected_strategy_profile = dict(winner_profile)
                        if self._interval != winner_interval:
                            self.log_message.emit(
                                "EMA Band Rejection winner preset active: "
                                f"{self._symbol} forced to {winner_interval} "
                                "(1h winner default; non-optimization run)."
                            )
                        else:
                            self.log_message.emit(
                                "EMA Band Rejection winner preset active: "
                                f"{self._symbol} using fixed {winner_interval} profile."
                            )
                        self._interval = winner_interval
                    elif self._symbol in EMA_BAND_REJECTION_1H_EXCLUDED_COINS:
                        self.log_message.emit(
                            "EMA Band Rejection winner preset skipped: "
                            f"{self._symbol} is excluded from the fixed 1h winner list."
                        )
                elif (
                    not self._optimize_profile
                    and resolved_strategy_name == "ema_band_rejection"
                    and self._enforce_interval_fairness
                ):
                    self.log_message.emit(
                        "Live vs Challenger interval fairness lock active: "
                        "EMA Band Rejection fixed winner preset disabled for this run."
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
                requested_history_start_time, requested_history_end_time = (
                    self._resolve_requested_history_window()
                )
                explicit_history_window = (
                    self._history_requested_start_utc is not None
                    or self._history_end_utc is not None
                )
                required_warmup_candles, required_candles = self._resolve_required_runtime_candles(
                    strategy_name=resolved_strategy_name,
                    strategy_profile=selected_strategy_profile,
                )
                effective_history_start_time = requested_history_start_time
                effective_history_end_time = requested_history_end_time
                history_loader_mode = "latest_tail_window"
                warmup_shifted_candles = 0
                self._optimization_grid = generate_optimization_grid(
                    resolved_strategy_name,
                    symbol=self._symbol,
                    interval=self._interval,
                )
                self.log_message.emit(
                    f"Backtest strategy selected: {self._symbol} -> {resolved_strategy_name}"
                )
                if self._optimize_profile:
                    requested_end_text = (
                        f"{requested_history_end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        if requested_history_end_time is not None
                        else "latest available"
                    )
                    self.log_message.emit(
                        "Backtest history window base: "
                        f"{requested_history_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC -> {requested_end_text} "
                        "(optimizer uses incremental recent-window sync when cache exists)."
                    )
                else:
                    requested_end_text = (
                        f"{requested_history_end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        if requested_history_end_time is not None
                        else "latest available"
                    )
                    if explicit_history_window:
                        self.log_message.emit(
                            "Backtest history window fixed: "
                            f"{requested_history_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC -> {requested_end_text}."
                        )
                    else:
                        self.log_message.emit(
                            "Backtest history base from config: "
                            f"{requested_history_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC -> {requested_end_text} "
                            "(runtime may still use incremental tail sync when no explicit date-range is requested)."
                        )
                if self._use_setup_gate:
                    if self._optimize_profile:
                        self.log_message.emit(
                            "Phase 19 active: enforcing a 500-candle settled warmup before Setup Gate approval "
                            f"with optimizer history base start at {requested_history_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
                        )
                    else:
                        self.log_message.emit(
                            "Phase 19 active: enforcing a 500-candle settled warmup before Setup Gate approval "
                            f"with fixed backtest history start at {requested_history_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
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
                        "history_requested_start_utc": _trace_datetime_text(requested_history_start_time),
                        "history_requested_end_utc": _trace_datetime_text(requested_history_end_time),
                        "optimization_mode": bool(self._optimize_profile),
                    },
                )
                self.progress_update.emit(10, "Downloading history...")
                now_utc_naive = datetime.now(tz=UTC).replace(tzinfo=None)
                requested_history_sync_end_time = (
                    now_utc_naive
                    if requested_history_end_time is None
                    else min(requested_history_end_time, now_utc_naive)
                )
                if requested_history_sync_end_time <= requested_history_start_time:
                    requested_history_sync_end_time = requested_history_start_time + timedelta(
                        seconds=max(_interval_total_seconds(self._interval), 60)
                    )
                full_history_seed_required = False
                full_history_seed_reason = "not-applicable"
                if explicit_history_window:
                    (
                        full_history_seed_required,
                        full_history_seed_reason,
                    ) = _is_full_history_seed_required(
                        symbol=self._symbol,
                        interval=self._interval,
                        requested_start_utc=requested_history_start_time,
                        requested_end_utc=requested_history_sync_end_time,
                    )
                history_cache_key = self._build_batch_history_cache_key(
                    requested_start_utc=requested_history_start_time,
                    requested_end_utc=requested_history_sync_end_time,
                    explicit_history_window=explicit_history_window,
                )
                candles_df: pd.DataFrame | None = None
                candles: list[CandleRecord] = []
                total_candles = 0
                oldest_time: datetime | None = None
                newest_time: datetime | None = None
                last_history_error: Exception | None = None
                full_history_seed_performed = False
                cached_candles_df, cached_history_metadata = (
                    self._load_candles_dataframe_from_batch_history_cache(
                        cache_key=history_cache_key,
                        required_candles=required_candles,
                        allow_insufficient=True,
                    )
                )
                if cached_candles_df is not None and cached_history_metadata is not None:
                    candles_df = cached_candles_df
                    total_candles = int(
                        cached_history_metadata.get("history_candles", len(cached_candles_df))
                        or len(cached_candles_df)
                    )
                    oldest_time = cached_history_metadata.get("history_start_utc")
                    newest_time = cached_history_metadata.get("history_end_utc")
                    effective_history_start_time = (
                        cached_history_metadata.get("history_effective_start_utc")
                        if isinstance(
                            cached_history_metadata.get("history_effective_start_utc"),
                            datetime,
                        )
                        else requested_history_start_time
                    )
                    effective_history_end_time = (
                        cached_history_metadata.get("history_effective_end_utc")
                        if isinstance(
                            cached_history_metadata.get("history_effective_end_utc"),
                            datetime,
                        )
                        else requested_history_end_time
                    )
                    history_loader_mode = str(
                        cached_history_metadata.get("history_loader_mode", "session_cache")
                        or "session_cache"
                    )
                    warmup_shifted_candles = int(
                        cached_history_metadata.get("history_warmup_shifted_candles", 0)
                        or 0
                    )
                    self.log_message.emit(
                        "Batch history cache hit: "
                        f"{self._symbol} {self._interval} "
                        f"({total_candles} candles, loader={history_loader_mode})."
                    )
                    history_loader_mode = f"{history_loader_mode}|session_cache"
                else:
                    try:
                        for source_index, (history_db, source_label) in enumerate(history_sources):
                            try:
                                self.log_message.emit(
                                    f"Downloading historical data for backtest via {source_label}..."
                                )
                                with BitunixClient(max_retries=6, backoff_factor=1.0) as client:
                                    with HistoryManager(history_db, client) as history_manager:
                                        self._reset_history_download_progress()
                                        if explicit_history_window:
                                            if full_history_seed_required:
                                                sync_start_utc, sync_end_utc = self._resolve_history_sync_window(
                                                    requested_start_utc=requested_history_start_time,
                                                    requested_end_utc=requested_history_sync_end_time,
                                                    required_warmup_candles=required_warmup_candles,
                                                )
                                                sync_end_label = (
                                                    sync_end_utc.strftime("%Y-%m-%d %H:%M:%S")
                                                    if sync_end_utc is not None
                                                    else "latest"
                                                )
                                                self.log_message.emit(
                                                    "Date-range history sync active: "
                                                    f"{sync_start_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC -> {sync_end_label} UTC."
                                                )
                                                history_manager.sync_candles_since(
                                                    symbols=[self._symbol],
                                                    interval=self._interval,
                                                    start_time=sync_start_utc,
                                                    end_time=sync_end_utc,
                                                    on_progress=self._emit_history_download_progress,
                                                    on_log=self.log_message.emit,
                                                    should_abort=self._should_stop,
                                                )
                                                full_history_seed_performed = True
                                            else:
                                                self.log_message.emit(
                                                    "Date-range baseline already seeded: "
                                                    f"{self._symbol} {self._interval} "
                                                    f"(reason={full_history_seed_reason}). "
                                                    "Running incremental tail refresh only."
                                                )
                                                history_manager.sync_recent_candles(
                                                    symbols=[self._symbol],
                                                    interval=self._interval,
                                                    candles_per_symbol=max(
                                                        int(OPTIMIZER_INCREMENTAL_SYNC_MIN_CANDLES),
                                                        5_000,
                                                    ),
                                                    should_abort=self._should_stop,
                                                    on_progress=self._emit_history_download_progress,
                                                )
                                        else:
                                            self.log_message.emit(
                                                "Incremental history sync active: "
                                                f"refreshing latest tail window ({MAX_BACKTEST_CANDLES} candles)."
                                            )
                                            history_manager.sync_recent_candles(
                                                symbols=[self._symbol],
                                                interval=self._interval,
                                                candles_per_symbol=MAX_BACKTEST_CANDLES,
                                                should_abort=self._should_stop,
                                                on_progress=self._emit_history_download_progress,
                                            )
                                if self._should_stop():
                                    return
                                if explicit_history_window:
                                    effective_history_start_time, warmup_shifted_candles = (
                                        self._apply_history_warmup_correction(
                                            history_db,
                                            requested_start_utc=requested_history_start_time,
                                            requested_end_utc=requested_history_sync_end_time,
                                            required_warmup_candles=required_warmup_candles,
                                        )
                                    )
                                    effective_history_end_time = requested_history_sync_end_time
                                    candles = _load_backtest_candles_in_time_range(
                                        history_db,
                                        symbol=self._symbol,
                                        interval=self._interval,
                                        start_time=effective_history_start_time,
                                        end_time=effective_history_end_time,
                                    )
                                    history_loader_mode = "date_range_window"
                                else:
                                    candles = _load_latest_backtest_candles(
                                        history_db,
                                        symbol=self._symbol,
                                        interval=self._interval,
                                        limit=MAX_BACKTEST_CANDLES,
                                    )
                                    effective_history_start_time = requested_history_start_time
                                    effective_history_end_time = requested_history_sync_end_time
                                    history_loader_mode = "latest_tail_window"
                                total_candles = len(candles)
                                if self._should_stop():
                                    return
                                if candles:
                                    oldest_time = candles[0].open_time
                                    newest_time = candles[-1].open_time
                                    if explicit_history_window and full_history_seed_performed:
                                        _mark_history_seed_completed(
                                            symbol=self._symbol,
                                            interval=self._interval,
                                            seeded_start_utc=requested_history_start_time,
                                            seeded_end_utc=max(
                                                newest_time,
                                                requested_history_sync_end_time,
                                            ),
                                            mode="date_range",
                                        )
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
                    cache_metadata: dict[str, object] = {
                        "history_effective_start_utc": effective_history_start_time,
                        "history_effective_end_utc": effective_history_end_time,
                        "history_start_utc": oldest_time,
                        "history_end_utc": newest_time,
                        "history_candles": int(total_candles),
                        "history_loader_mode": history_loader_mode,
                        "history_warmup_shifted_candles": int(warmup_shifted_candles),
                    }
                    candles_df = self._cache_history_and_prepare_run_dataframe(
                        cache_key=history_cache_key,
                        candles=candles,
                        metadata=cache_metadata,
                    )
                    if history_cache_key is not None:
                        self.log_message.emit(
                            "Batch history cache store: "
                            f"{self._symbol} {self._interval} "
                            f"({total_candles} candles, loader={history_loader_mode})."
                        )
                if candles_df is None:
                    raise RuntimeError(
                        "History load failed: candles dataframe could not be prepared."
                    )
                if shared_history_db is not None:
                    with suppress(Exception):
                        shared_history_db.close()
                    shared_history_db = None

                if total_candles > 0 and oldest_time is not None and newest_time is not None:
                    self.log_message.emit(
                        "Historical data range loaded: "
                        f"{oldest_time.strftime('%Y-%m-%d %H:%M:%S')} -> "
                        f"{newest_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({total_candles} candles, interval={self._interval})"
                    )
                    if str(history_loader_mode).startswith("date_range_window"):
                        self.log_message.emit(
                            "Backtest loader mode: "
                            f"using configured date-range window ({total_candles} candles)."
                        )
                    else:
                        self.log_message.emit(
                            "Backtest loader mode: "
                            f"using latest tail window ({min(total_candles, MAX_BACKTEST_CANDLES)}/{MAX_BACKTEST_CANDLES} candles)."
                        )
                if total_candles > 0 and oldest_time is not None and newest_time is not None:
                    self.log_message.emit(
                        "Backtest window in use: "
                        f"{oldest_time.strftime('%Y-%m-%d %H:%M:%S')} -> "
                        f"{newest_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({total_candles} candles)"
                    )
                self._emit_trace_event(
                    "history_loaded",
                    {
                        "symbol": self._symbol,
                        "resolved_strategy_name": resolved_strategy_name,
                        "resolved_interval": self._interval,
                        "history_requested_start_utc": _trace_datetime_text(
                            requested_history_start_time
                        ),
                        "history_requested_end_utc": _trace_datetime_text(
                            requested_history_end_time
                        ),
                        "history_effective_start_utc": _trace_datetime_text(
                            effective_history_start_time
                        ),
                        "history_effective_end_utc": _trace_datetime_text(
                            effective_history_end_time
                        ),
                        "history_start_utc": _trace_datetime_text(oldest_time),
                        "history_end_utc": _trace_datetime_text(newest_time),
                        "history_candles": int(total_candles),
                        "history_loader_mode": history_loader_mode,
                        "history_warmup_shifted_candles": int(warmup_shifted_candles),
                        "required_warmup_candles": int(required_warmup_candles),
                        "required_candles": int(required_candles),
                    },
                )
                if len(candles_df) < required_candles:
                    loaded_range_text = "unknown range"
                    if oldest_time is not None and newest_time is not None:
                        loaded_range_text = (
                            f"{oldest_time.strftime('%Y-%m-%d %H:%M:%S')} -> "
                            f"{newest_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    missing_candles = max(0, int(required_candles - len(candles_df)))
                    raise RuntimeError(
                        "Not enough candles for backtest: "
                        f"have {len(candles_df)}, need {required_candles} "
                        f"(missing {missing_candles}) for {self._symbol} {self._interval}. "
                        f"Loaded range: {loaded_range_text}. "
                        "The symbol is likely newly listed or has limited exchange history for this timeframe."
                    )

                self._hmm_regime_mask = self._resolve_hmm_regime_mask_with_batch_cache(
                    cache_key=history_cache_key,
                    candles_df=candles_df,
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
                        strategy_profile=selected_strategy_profile,
                        regime_mask=self._hmm_regime_mask,
                    )
                    if self._should_stop():
                        return
                    if not self._is_quiet_mode():
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
                        strategy_profile=selected_strategy_profile,
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
                        "history_requested_start_utc": requested_history_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "history_requested_end_utc": (
                            requested_history_end_time.strftime("%Y-%m-%d %H:%M:%S")
                            if requested_history_end_time is not None
                            else None
                        ),
                        "history_effective_start_utc": effective_history_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "history_effective_end_utc": (
                            effective_history_end_time.strftime("%Y-%m-%d %H:%M:%S")
                            if effective_history_end_time is not None
                            else None
                        ),
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
                        "history_loader_mode": history_loader_mode,
                        "history_warmup_shifted_candles": int(warmup_shifted_candles),
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
                        "breakeven_activation_pct": result.get("breakeven_activation_pct"),
                        "breakeven_buffer_pct": result.get("breakeven_buffer_pct"),
                        "tight_trailing_activation_pct": result.get("tight_trailing_activation_pct"),
                        "tight_trailing_distance_pct": result.get("tight_trailing_distance_pct"),
                        "history_requested_start_utc": result.get("history_requested_start_utc"),
                        "history_requested_end_utc": result.get("history_requested_end_utc"),
                        "history_effective_start_utc": result.get("history_effective_start_utc"),
                        "history_effective_end_utc": result.get("history_effective_end_utc"),
                        "history_start_utc": result.get("history_start_utc"),
                        "history_end_utc": result.get("history_end_utc"),
                        "history_candles": result.get("history_candles"),
                        "history_loader_mode": result.get("history_loader_mode"),
                        "history_warmup_shifted_candles": result.get("history_warmup_shifted_candles"),
                        "required_warmup_candles": result.get("required_warmup_candles"),
                        "required_candles": result.get("required_candles"),
                        "total_signals": result.get("total_signals"),
                        "approved_signals": result.get("approved_signals"),
                        "blocked_signals": result.get("blocked_signals"),
                        "final_pnl": result.get("total_pnl_usd"),
                        "win_rate": result.get("win_rate_pct"),
                        "profit_factor": result.get("profit_factor"),
                        "total_trades": result.get("total_trades"),
                        "long_trades": result.get("long_trades"),
                        "short_trades": result.get("short_trades"),
                        "real_leverage_avg": result.get("real_leverage_avg"),
                        "average_win": result.get("average_win_usd"),
                        "average_loss": result.get("average_loss_usd"),
                        "real_rrr": result.get("real_rrr"),
                        "max_drawdown": result.get("max_drawdown_pct"),
                        "optimizer_quality_status": result.get("optimizer_quality_status"),
                    },
                )
                if self._isolated_db:
                    if (
                        self._optimize_profile
                        and self._checkpoint_backtest_run_id is not None
                    ):
                        self._update_backtest_result_in_primary_db_with_retry(
                            self._checkpoint_backtest_run_id,
                            result,
                            strategy_name=resolved_strategy_name,
                            stage_label="final_result_update",
                        )
                    else:
                        self._persist_backtest_result_to_primary_db_with_retry(
                            result,
                            strategy_name=resolved_strategy_name,
                            stage_label="final_result",
                        )
                else:
                    self._persist_backtest_result(
                        db,
                        result,
                        strategy_name=resolved_strategy_name,
                    )
                self.log_message.emit(
                    f"Backtest result saved to DB for {self._symbol} {self._interval} ({resolved_strategy_name})."
                )
                self._checkpoint_backtest_run_id = None
        except Exception as exc:
            if not self._should_stop():
                self.backtest_error.emit(str(exc))
            return
        finally:
            self._hmm_regime_mask = None
            self._checkpoint_backtest_run_id = None
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
        profile_breakeven_activation = (
            float(strategy_profile["breakeven_activation_pct"])
            if isinstance(strategy_profile, dict) and "breakeven_activation_pct" in strategy_profile
            else None
        )
        profile_breakeven_buffer = (
            float(strategy_profile["breakeven_buffer_pct"])
            if isinstance(strategy_profile, dict) and "breakeven_buffer_pct" in strategy_profile
            else None
        )
        profile_tight_trailing_activation = (
            float(strategy_profile["tight_trailing_activation_pct"])
            if isinstance(strategy_profile, dict) and "tight_trailing_activation_pct" in strategy_profile
            else None
        )
        profile_tight_trailing_distance = (
            float(strategy_profile["tight_trailing_distance_pct"])
            if isinstance(strategy_profile, dict) and "tight_trailing_distance_pct" in strategy_profile
            else None
        )
        profile_chandelier_period = (
            int(float(strategy_profile["chandelier_period"]))
            if isinstance(strategy_profile, dict) and "chandelier_period" in strategy_profile
            else int(settings.trading.chandelier_period)
        )
        profile_chandelier_multiplier = (
            float(strategy_profile["chandelier_multiplier"])
            if isinstance(strategy_profile, dict) and "chandelier_multiplier" in strategy_profile
            else float(
                strategy_profile.get("chandelier_mult", settings.trading.chandelier_multiplier)
            )
            if isinstance(strategy_profile, dict)
            else float(settings.trading.chandelier_multiplier)
        )
        engine = self._build_backtest_engine(
            db,
            take_profit_pct=profile_tp,
            stop_loss_pct=profile_sl,
            trailing_activation_pct=profile_trailing_activation,
            trailing_distance_pct=profile_trailing_distance,
            breakeven_activation_pct=profile_breakeven_activation,
            breakeven_buffer_pct=profile_breakeven_buffer,
            tight_trailing_activation_pct=profile_tight_trailing_activation,
            tight_trailing_distance_pct=profile_tight_trailing_distance,
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
        precomputed_dynamic_stop_loss_pcts = (
            None
            if self._last_precomputed_exit_cache is None
            else self._last_precomputed_exit_cache.get("precomputed_dynamic_stop_loss_pcts")
        )
        precomputed_dynamic_take_profit_pcts = (
            None
            if self._last_precomputed_exit_cache is None
            else self._last_precomputed_exit_cache.get("precomputed_dynamic_take_profit_pcts")
        )
        normalized_dynamic_stop_loss_pcts = _normalize_dynamic_pct_series(
            precomputed_dynamic_stop_loss_pcts
        )
        normalized_dynamic_take_profit_pcts = _normalize_dynamic_pct_series(
            precomputed_dynamic_take_profit_pcts
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
            dynamic_stop_loss_pcts=normalized_dynamic_stop_loss_pcts,
            dynamic_take_profit_pcts=normalized_dynamic_take_profit_pcts,
            enable_chandelier_exit=True,
            chandelier_period=profile_chandelier_period,
            chandelier_multiplier=profile_chandelier_multiplier,
            strategy_exit_pre_take_profit=False,
            strategy_name=strategy_name,
        )
        trailing_level_1_hits = int(result.get("breakeven_level_hits", 0) or 0)
        trailing_level_2_hits = int(result.get("normal_trailing_level_hits", 0) or 0)
        trailing_level_3_hits = int(result.get("tight_trailing_level_hits", 0) or 0)
        if not self._is_quiet_mode() and trailing_level_1_hits > 0:
            self.log_message.emit(
                f"Trailing Level 1 (Breakeven) hit: {trailing_level_1_hits} trade(s)."
            )
        if not self._is_quiet_mode() and trailing_level_2_hits > 0:
            self.log_message.emit(
                f"Trailing Level 2 (Normal Trail) hit: {trailing_level_2_hits} trade(s)."
            )
        if not self._is_quiet_mode() and trailing_level_3_hits > 0:
            self.log_message.emit(
                f"Trailing Level 3 (Tight Trail) hit: {trailing_level_3_hits} trade(s)."
            )
        if not self._is_quiet_mode():
            self.log_message.emit(
                "Tiered trailing summary: "
                f"level1_breakeven={trailing_level_1_hits} "
                f"level2_normal={trailing_level_2_hits} "
                f"level3_tight={trailing_level_3_hits}"
            )
        result.update(self._calculate_trade_metrics(result["closed_trades"]))
        long_trades_count, short_trades_count = _calculate_trade_direction_counts(result["closed_trades"])
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
                "breakeven_activation_pct": runtime_trace.get("breakeven_activation_pct"),
                "breakeven_buffer_pct": runtime_trace.get("breakeven_buffer_pct"),
                "tight_trailing_activation_pct": runtime_trace.get("tight_trailing_activation_pct"),
                "tight_trailing_distance_pct": runtime_trace.get("tight_trailing_distance_pct"),
                "effective_taker_fee_pct": runtime_trace.get("effective_taker_fee_pct"),
                "slippage_penalty_pct_per_side": runtime_trace.get("slippage_penalty_pct_per_side"),
                "slippage_penalty_pct_per_trade": runtime_trace.get("slippage_penalty_pct_per_trade"),
                "estimated_round_trip_cost_pct": runtime_trace.get("estimated_round_trip_cost_pct"),
                "min_confidence_pct": (
                    float(self._min_confidence_pct)
                    if self._min_confidence_pct is not None
                    else None
                ),
                "required_warmup_candles": int(required_warmup_candles),
                "required_candles": int(required_candles),
                "breakeven_level_hits": trailing_level_1_hits,
                "normal_trailing_level_hits": trailing_level_2_hits,
                "tight_trailing_level_hits": trailing_level_3_hits,
                "long_trades": int(long_trades_count),
                "short_trades": int(short_trades_count),
            }
        )
        active_strategy_profile = self._resolve_active_strategy_profile(
            candles_df,
            strategy_name=strategy_name,
            strategy_profile=strategy_profile,
        )
        equity_curve_events = self._build_equity_curve_events(
            candles_df,
            closed_trades=(
                result.get("closed_trades")
                if isinstance(result.get("closed_trades"), list)
                else []
            ),
        )
        regime_pnl = self._build_regime_pnl_payload(
            candles_df,
            closed_trades=(
                result.get("closed_trades")
                if isinstance(result.get("closed_trades"), list)
                else []
            ),
        )
        result["equity_curve_events"] = equity_curve_events
        result["regime_pnl"] = regime_pnl
        if active_strategy_profile:
            result["best_profile"] = dict(active_strategy_profile)
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

    def _resolve_active_strategy_profile(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None,
    ) -> OptimizationProfile:
        variant_fields = _strategy_variant_field_names(strategy_name)
        if not variant_fields:
            return {}
        if strategy_profile is not None:
            resolved_profile: OptimizationProfile = {}
            for field_name in variant_fields:
                if field_name not in strategy_profile:
                    continue
                resolved_profile[field_name] = float(strategy_profile[field_name])
            return resolved_profile
        with _temporary_strategy_profile(None, candles_df):
            return {
                field_name: float(getattr(settings.strategy, field_name))
                for field_name in variant_fields
                if hasattr(settings.strategy, field_name)
            }

    def _build_equity_curve_events(
        self,
        candles_df: pd.DataFrame,
        *,
        closed_trades: list[dict[str, object]],
    ) -> list[dict[str, float]]:
        if not closed_trades or candles_df.empty:
            return []

        open_times = pd.to_datetime(candles_df["open_time"], utc=False, errors="coerce")
        interval_text = str(candles_df["interval"].iloc[0]) if "interval" in candles_df.columns else self._interval
        interval_seconds = _interval_total_seconds(interval_text)
        close_times = open_times + pd.to_timedelta(interval_seconds, unit="s")
        close_times_values = close_times.to_numpy(dtype="datetime64[ns]")
        if close_times_values.size == 0:
            return []

        exit_events: list[tuple[np.datetime64, float]] = []
        for trade in closed_trades:
            try:
                exit_time_raw = trade.get("exit_time")
                if exit_time_raw is None:
                    continue
                exit_time = pd.to_datetime(exit_time_raw, utc=False, errors="coerce")
                if pd.isna(exit_time):
                    continue
                pnl_value = float(trade.get("pnl", 0.0) or 0.0)
                exit_events.append((np.datetime64(exit_time.to_datetime64()), pnl_value))
            except Exception:
                continue
        if not exit_events:
            return []

        exit_events.sort(key=lambda item: item[0])
        cumulative_pnl = 0.0
        close_index = 0
        event_index = 0
        curve_events: list[dict[str, float]] = []
        while close_index < close_times_values.size and event_index < len(exit_events):
            close_time_value = close_times_values[close_index]
            updated = False
            while event_index < len(exit_events) and exit_events[event_index][0] <= close_time_value:
                cumulative_pnl += float(exit_events[event_index][1])
                event_index += 1
                updated = True
            if updated:
                curve_events.append(
                    {
                        "candle_index": float(close_index + 1),
                        "cum_pnl": float(cumulative_pnl),
                    }
                )
            close_index += 1

        # Ensure final state always exists for right-edge plotting.
        if not curve_events:
            return []
        last_candle_index = int(close_times_values.size)
        if int(curve_events[-1]["candle_index"]) < last_candle_index:
            curve_events.append(
                {
                    "candle_index": float(last_candle_index),
                    "cum_pnl": float(cumulative_pnl),
                }
            )
        return curve_events

    def _ensure_hmm_regime_labels_for_analytics(
        self,
        candles_df: pd.DataFrame,
    ) -> list[str] | None:
        if (
            isinstance(self._hmm_regime_labels, list)
            and len(self._hmm_regime_labels) == len(candles_df)
        ):
            return list(self._hmm_regime_labels)
        if candles_df.empty:
            return None

        try:
            detector = HMMRegimeDetector(
                n_components=4,
                train_window_candles=8000,
                apply_window_candles=2000,
                warmup_candles=2000,
                stability_candles=5,
                allowed_regimes=(
                    HMMRegimeDetector.BULL_TREND,
                    HMMRegimeDetector.BEAR_TREND,
                    HMMRegimeDetector.HIGH_VOL_RANGE,
                    HMMRegimeDetector.LOW_VOL_RANGE,
                    HMMRegimeDetector.UNKNOWN,
                    HMMRegimeDetector.WARMUP,
                ),
            )
            detection = detector.detect(candles_df)
            transmat_warning_events = [
                event
                for event in tuple(getattr(detection, "warning_events", tuple()) or tuple())
                if isinstance(event, Mapping)
                and str(event.get("warning_kind", "")).strip().lower()
                == "transmat_zero_sum_no_transition"
            ]
            if transmat_warning_events:
                warning_detail = _format_hmm_warning_detail(
                    detection,
                    warning_kind="transmat_zero_sum_no_transition",
                )
                self.log_message.emit(
                    "HMM regime analytics detector structural warning. "
                    f"symbol={self._symbol} interval={self._interval} {warning_detail}."
                )
            if not bool(getattr(detection, "converged", True)):
                non_convergence_detail = _format_hmm_non_convergence_detail(detection)
                self.log_message.emit(
                    "HMM regime analytics detector non-converged. "
                    f"symbol={self._symbol} interval={self._interval} "
                    f"{non_convergence_detail}. Falling back to neutral regime view."
                )
                self._hmm_regime_labels = None
                return None
            labels = list(detection.regime_labels)
            if len(labels) == len(candles_df):
                self._hmm_regime_labels = labels
                return labels
        except Exception as exc:
            self.log_message.emit(
                f"HMM regime analytics detector failed ({exc}). Falling back to neutral regime view."
            )
        self._hmm_regime_labels = None
        return None

    def _build_regime_pnl_payload(
        self,
        candles_df: pd.DataFrame,
        *,
        closed_trades: list[dict[str, object]],
    ) -> dict[str, float]:
        regime_payload = {"Bull": 0.0, "Bear": 0.0, "Range": 0.0}
        if not closed_trades:
            return regime_payload

        regime_labels = self._ensure_hmm_regime_labels_for_analytics(candles_df)
        if not regime_labels:
            for trade in closed_trades:
                try:
                    pnl_value = float(trade.get("pnl", 0.0) or 0.0)
                except Exception:
                    continue
                side = str(trade.get("side", "")).strip().upper()
                if side == "LONG":
                    regime_payload["Bull"] += pnl_value
                elif side == "SHORT":
                    regime_payload["Bear"] += pnl_value
                else:
                    regime_payload["Range"] += pnl_value
            return regime_payload

        open_times = pd.to_datetime(candles_df["open_time"], utc=False, errors="coerce")
        interval_text = str(candles_df["interval"].iloc[0]) if "interval" in candles_df.columns else self._interval
        interval_seconds = _interval_total_seconds(interval_text)
        close_times = open_times + pd.to_timedelta(interval_seconds, unit="s")
        close_times_values = close_times.to_numpy(dtype="datetime64[ns]")
        if close_times_values.size == 0:
            return regime_payload

        for trade in closed_trades:
            try:
                exit_time_raw = trade.get("exit_time")
                if exit_time_raw is None:
                    continue
                exit_time = pd.to_datetime(exit_time_raw, utc=False, errors="coerce")
                if pd.isna(exit_time):
                    continue
                pnl_value = float(trade.get("pnl", 0.0) or 0.0)
            except Exception:
                continue

            exit_time_value = np.datetime64(exit_time.to_datetime64())
            candle_index = int(np.searchsorted(close_times_values, exit_time_value, side="left"))
            candle_index = max(0, min(candle_index, len(regime_labels) - 1))
            regime_label = str(regime_labels[candle_index]).lower()
            if "bull" in regime_label:
                regime_payload["Bull"] += pnl_value
            elif "bear" in regime_label:
                regime_payload["Bear"] += pnl_value
            else:
                regime_payload["Range"] += pnl_value
        return regime_payload

    def _run_micro_stability_scan(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        strategy_profile: OptimizationProfile | None,
        regime_mask: Sequence[int] | None,
    ) -> list[dict[str, float]]:
        _ = candles_df, strategy_name, strategy_profile, regime_mask
        return []

    def _slice_optimizer_window(
        self,
        candles_df: pd.DataFrame,
        *,
        window_candles: int,
        regime_mask: Sequence[int] | None,
    ) -> tuple[pd.DataFrame, list[int] | None, bool]:
        resolved_window = max(1, int(window_candles))
        if 0 < resolved_window < len(candles_df):
            sliced_candles_df = candles_df.iloc[-resolved_window:].copy(deep=True)
            uses_full_history = False
        else:
            sliced_candles_df = candles_df
            uses_full_history = True

        sliced_regime_mask: list[int] | None = None
        if regime_mask is not None:
            if len(regime_mask) == len(candles_df):
                sliced_regime_mask = list(regime_mask)[-len(sliced_candles_df):]
            else:
                self.log_message.emit(
                    "Warning: regime mask length mismatch; disabling regime filter for optimizer window."
                )
        return sliced_candles_df, sliced_regime_mask, uses_full_history

    def _analyze_optimization_phase_summaries(
        self,
        *,
        strategy_name: str,
        stage_label: str,
        phase_summaries: Sequence[dict[str, float]],
        min_required_trades: int,
    ) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, int]]:
        fail_min_trades = 0
        fail_breakeven = 0
        fail_avg_profit = 0
        fail_multi = 0
        eligible_summaries: list[dict[str, float]] = []
        diagnostics_rows: list[tuple[dict[str, float], bool, bool, bool, int]] = []

        for raw_summary in phase_summaries:
            profile_result = dict(raw_summary)
            profile_result["avg_profit_per_trade_net_pct"] = float(
                _resolve_avg_profit_per_trade_net_pct(profile_result)
            )
            passes_min_trades = (
                float(profile_result.get("total_trades", 0.0) or 0.0)
                >= float(min_required_trades)
            )
            passes_breakeven = _profile_meets_breakeven_constraint(
                profile_result,
                strategy_name=strategy_name,
            )
            passes_avg_profit = _profile_meets_avg_profit_per_trade_hurdle(profile_result)
            fail_count = int((not passes_min_trades) + (not passes_breakeven) + (not passes_avg_profit))
            if not passes_min_trades:
                fail_min_trades += 1
            if not passes_breakeven:
                fail_breakeven += 1
            if not passes_avg_profit:
                fail_avg_profit += 1
            if fail_count >= 2:
                fail_multi += 1
            diagnostics_rows.append(
                (
                    profile_result,
                    passes_min_trades,
                    passes_breakeven,
                    passes_avg_profit,
                    fail_count,
                )
            )
            if fail_count == 0:
                eligible_summaries.append(profile_result)

        self.log_message.emit(
            f"{stage_label} fail-reason summary: "
            f"total={len(phase_summaries)} | "
            f"fail_min_trades={fail_min_trades} | "
            f"fail_breakeven={fail_breakeven} | "
            f"fail_avg_profit={fail_avg_profit} | "
            f"fail_multi={fail_multi} | "
            f"eligible={len(eligible_summaries)}"
        )

        diagnostics_rows_sorted = sorted(
            diagnostics_rows,
            key=lambda row: _optimization_sort_key(row[0], strategy_name=strategy_name),
            reverse=True,
        )
        if self._is_debug_mode():
            top_diagnostics = diagnostics_rows_sorted[:10]
            if top_diagnostics:
                self.log_message.emit(
                    f"{stage_label} top profile diagnostics (top 10 by optimization ranking):"
                )
            for rank, row in enumerate(top_diagnostics, start=1):
                summary_row, passes_min_trades, passes_breakeven, passes_avg_profit, _fail_count = row
                profile_preview = _format_profile_dict(_extract_profile_from_summary(summary_row))
                self.log_message.emit(
                    f"{stage_label} #{rank} profile={profile_preview} | "
                    f"trades={int(float(summary_row.get('total_trades', 0.0) or 0.0))} | "
                    f"pf={self._format_profit_factor(float(summary_row.get('profit_factor', 0.0) or 0.0))} | "
                    f"pnl={float(summary_row.get('total_pnl_usd', 0.0) or 0.0):.2f} | "
                    f"avg_net={float(summary_row.get('avg_profit_per_trade_net_pct', 0.0) or 0.0):.4f}% | "
                    f"pass_min_trades={'yes' if passes_min_trades else 'no'} | "
                    f"pass_breakeven={'yes' if passes_breakeven else 'no'} | "
                    f"pass_avg_profit={'yes' if passes_avg_profit else 'no'}"
                )

        ranked_all = [
            dict(row[0])
            for row in diagnostics_rows_sorted
        ]
        ranked_eligible = sorted(
            eligible_summaries,
            key=lambda summary: _optimization_sort_key(summary, strategy_name=strategy_name),
            reverse=True,
        )
        return ranked_eligible, ranked_all, {
            "total": int(len(phase_summaries)),
            "fail_min_trades": int(fail_min_trades),
            "fail_breakeven": int(fail_breakeven),
            "fail_avg_profit": int(fail_avg_profit),
            "fail_multi": int(fail_multi),
            "eligible": int(len(eligible_summaries)),
        }

    @staticmethod
    def _audit_ema_band_dynamic_stop_alignment(
        candles_df: pd.DataFrame,
        *,
        strategy_profile: OptimizationProfile,
        aligned_signals: Sequence[int],
        aligned_dynamic_stop_loss_pcts: Sequence[float] | None,
    ) -> tuple[int, float, float]:
        if aligned_dynamic_stop_loss_pcts is None:
            return 0, 0.0, 0.0
        if len(candles_df) <= 1:
            return 0, 0.0, 0.0

        def _profile_int(field_name: str, default_value: int) -> int:
            with suppress(Exception):
                return int(float(strategy_profile.get(field_name, default_value) or default_value))
            return int(default_value)

        def _profile_float(field_name: str, default_value: float) -> float:
            with suppress(Exception):
                return float(strategy_profile.get(field_name, default_value) or default_value)
            return float(default_value)

        def _profile_flag(field_name: str, default_value: bool = False) -> bool:
            raw_value = strategy_profile.get(field_name, default_value)
            if isinstance(raw_value, bool):
                return bool(raw_value)
            with suppress(Exception):
                return bool(float(raw_value) >= 0.5)
            return bool(default_value)

        working_df = build_ema_band_rejection_signal_frame(
            candles_df,
            ema_fast=_profile_int("ema_fast", 5),
            ema_mid=_profile_int("ema_mid", 10),
            ema_slow=_profile_int("ema_slow", 20),
            slope_lookback=_profile_int("slope_lookback", 5),
            min_ema_spread_pct=_profile_float("min_ema_spread_pct", 0.05),
            min_slow_slope_pct=_profile_float("min_slow_slope_pct", 0.0),
            pullback_requires_outer_band_touch=_profile_flag(
                "pullback_requires_outer_band_touch",
                False,
            ),
            use_rejection_quality_filter=_profile_flag(
                "use_rejection_quality_filter",
                False,
            ),
            rejection_wick_min_ratio=_profile_float("rejection_wick_min_ratio", 0.35),
            rejection_body_min_ratio=_profile_float("rejection_body_min_ratio", 0.20),
            use_rsi_filter=_profile_flag("use_rsi_filter", False),
            rsi_length=_profile_int("rsi_length", 14),
            rsi_midline=_profile_float("rsi_midline", 50.0),
            use_rsi_cross_filter=_profile_flag("use_rsi_cross_filter", False),
            rsi_midline_margin=_profile_float("rsi_midline_margin", 0.0),
            use_volume_filter=_profile_flag("use_volume_filter", False),
            volume_ma_length=_profile_int("volume_ma_length", 20),
            volume_multiplier=_profile_float("volume_multiplier", 1.0),
            use_atr_stop_buffer=_profile_flag("use_atr_stop_buffer", False),
            atr_length=_profile_int("atr_length", 14),
            atr_stop_buffer_mult=_profile_float("atr_stop_buffer_mult", 0.5),
            signal_cooldown_bars=_profile_int("signal_cooldown_bars", 0),
        )

        close_values = pd.to_numeric(working_df["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        high_values = pd.to_numeric(working_df["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        low_values = pd.to_numeric(working_df["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        atr_values = pd.to_numeric(working_df["ema_band_atr"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        use_atr_buffer = _profile_flag("use_atr_stop_buffer", False)
        atr_buffer_mult = _profile_float("atr_stop_buffer_mult", 0.5)

        diffs: list[float] = []
        compare_count = min(
            len(aligned_signals),
            len(aligned_dynamic_stop_loss_pcts),
            len(close_values),
            len(high_values),
            len(low_values),
            len(atr_values),
        )
        for index in range(compare_count):
            signal_value = int(aligned_signals[index] or 0)
            if signal_value == 0 or index <= 0:
                continue
            with suppress(Exception):
                observed_pct = float(aligned_dynamic_stop_loss_pcts[index])
            if not math.isfinite(observed_pct) or observed_pct <= 0.0:
                continue
            entry_price = float(close_values[index])
            if not math.isfinite(entry_price) or entry_price <= 0.0:
                continue

            atr_buffer = 0.0
            if use_atr_buffer:
                atr_value = float(atr_values[index - 1])
                if math.isfinite(atr_value) and atr_value > 0.0:
                    atr_buffer = atr_value * float(atr_buffer_mult)

            if signal_value > 0:
                stop_price = float(low_values[index - 1]) - atr_buffer
                expected_pct = ((entry_price - stop_price) / entry_price) * 100.0
            else:
                stop_price = float(high_values[index - 1]) + atr_buffer
                expected_pct = ((stop_price - entry_price) / entry_price) * 100.0
            if not math.isfinite(expected_pct) or expected_pct <= 0.0:
                continue
            expected_pct = min(95.0, max(0.05, float(expected_pct)))
            diffs.append(abs(float(observed_pct) - float(expected_pct)))

        if not diffs:
            return 0, 0.0, 0.0
        return int(len(diffs)), float(sum(diffs) / len(diffs)), float(max(diffs))

    def _log_ema_band_stage2_forensic_audit(
        self,
        *,
        candles_df: pd.DataFrame,
        stage2_ranked_all: Sequence[dict[str, float]],
        regime_mask: Sequence[int] | None,
    ) -> None:
        if not stage2_ranked_all:
            self.log_message.emit("EMA Band Rejection audit: no Stage-2 candidates available.")
            return
        resolved_leverage = (
            int(self._leverage_override)
            if self._leverage_override is not None
            else int(settings.trading.default_leverage)
        )

        self.log_message.emit(
            "EMA Band Rejection audit: evaluating top Stage-2 candidates for unit/formula consistency."
        )
        self.log_message.emit(
            "EMA Band Rejection audit fix applied: dynamic_stop_loss_pct now uses entry close price (not open) "
            "to match backtest execution."
        )
        self.log_message.emit(
            "EMA Band Rejection spread unit check: spread_pct=(|EMAfast-EMAslow|/close)*100, "
            "so threshold 0.03 equals 0.03% (not 3%)."
        )

        probe_min_confidence_pct = self._min_confidence_pct
        if self._use_setup_gate:
            probe_min_confidence_pct = _enforce_optimizer_min_confidence_floor(
                probe_min_confidence_pct
            )
        required_candles_default = _required_candle_count_for_strategy(
            "ema_band_rejection",
            use_setup_gate=self._use_setup_gate,
        )
        setup_gate = (
            SmartSetupGate(min_confidence_pct=probe_min_confidence_pct)
            if self._use_setup_gate
            else None
        )

        formula_or_unit_mismatch = False
        for rank, summary in enumerate(stage2_ranked_all[:3], start=1):
            summary_row = dict(summary)
            profile = _extract_profile_from_summary(summary_row)
            total_pnl_usd = float(summary_row.get("total_pnl_usd", 0.0) or 0.0)
            start_capital_usd = float(
                summary_row.get("start_capital_usd", settings.trading.start_capital)
                or settings.trading.start_capital
            )
            total_trades = int(float(summary_row.get("total_trades", 0.0) or 0.0))
            stored_avg_net_pct = float(summary_row.get("avg_profit_per_trade_net_pct", 0.0) or 0.0)
            recalculated_avg_net_pct = float(
                _resolve_avg_profit_per_trade_net_pct(summary_row)
            )
            avg_net_diff_pct = abs(stored_avg_net_pct - recalculated_avg_net_pct)
            if avg_net_diff_pct > 1e-9:
                formula_or_unit_mismatch = True

            required_candles = max(
                int(required_candles_default),
                int(_required_candle_count_for_profile(
                    "ema_band_rejection",
                    profile,
                    use_setup_gate=self._use_setup_gate,
                )),
            )
            payload = _build_vectorized_strategy_cache_payload(
                candles_df,
                strategy_name="ema_band_rejection",
                required_candles=required_candles,
                setup_gate=setup_gate,
                strategy_profile=profile,
                regime_mask=regime_mask,
            )
            diagnostics = (
                dict(payload.get("strategy_diagnostics", {}))
                if isinstance(payload, Mapping)
                and isinstance(payload.get("strategy_diagnostics"), Mapping)
                else {}
            )
            aligned_signals = (
                list(payload.get("signals", []))
                if isinstance(payload, Mapping)
                else []
            )
            aligned_dynamic_stop_loss_pcts = (
                list(payload.get("precomputed_dynamic_stop_loss_pcts", []))
                if isinstance(payload, Mapping)
                else []
            )
            compare_count, dyn_sl_diff_mean, dyn_sl_diff_max = self._audit_ema_band_dynamic_stop_alignment(
                candles_df,
                strategy_profile=profile,
                aligned_signals=aligned_signals,
                aligned_dynamic_stop_loss_pcts=aligned_dynamic_stop_loss_pcts,
            )
            if compare_count > 0 and dyn_sl_diff_max > 1e-6:
                formula_or_unit_mismatch = True

            spread_threshold_pct = float(diagnostics.get("min_ema_spread_pct", 0.0) or 0.0)
            spread_entry_mean = float(diagnostics.get("ema_spread_entry_pct_mean", 0.0) or 0.0)
            spread_entry_min = float(diagnostics.get("ema_spread_entry_pct_min", 0.0) or 0.0)
            spread_entry_max = float(diagnostics.get("ema_spread_entry_pct_max", 0.0) or 0.0)
            if spread_threshold_pct > 2.0:
                formula_or_unit_mismatch = True

            self.log_message.emit(
                f"EMA Band Rejection audit #{rank}: "
                f"pnl={total_pnl_usd:.2f} | start_capital={start_capital_usd:.2f} | trades={total_trades} | "
                f"avg_net_stored={stored_avg_net_pct:.6f}% | "
                f"avg_net_calc=(( {total_pnl_usd:.6f} / {start_capital_usd:.2f} )*100)/{max(total_trades, 1)}={recalculated_avg_net_pct:.6f}% | "
                f"avg_net_diff={avg_net_diff_pct:.9f} | "
                f"leverage={resolved_leverage}x | "
                f"avg_fees={float(summary_row.get('average_trade_fees_usd', 0.0) or 0.0):.4f} | "
                f"avg_slippage={float(summary_row.get('average_trade_slippage_usd', 0.0) or 0.0):.4f} | "
                f"avg_cost={float(summary_row.get('average_trade_cost_usd', 0.0) or 0.0):.4f} | "
                f"avg_win_cost_ratio={self._format_profit_factor(float(summary_row.get('average_win_to_cost_ratio', 0.0) or 0.0))} | "
                f"pf={self._format_profit_factor(float(summary_row.get('profit_factor', 0.0) or 0.0))} | "
                f"win_rate={float(summary_row.get('win_rate_pct', 0.0) or 0.0):.2f}% | "
                f"max_dd={float(summary_row.get('max_drawdown_pct', 0.0) or 0.0):.2f}%"
            )
            self.log_message.emit(
                f"EMA Band Rejection audit #{rank} spread/dynSL: "
                f"threshold={spread_threshold_pct:.4f}% | "
                f"observed_spread_entry[min/mean/max]={spread_entry_min:.4f}/{spread_entry_mean:.4f}/{spread_entry_max:.4f}% | "
                f"dyn_sl_alignment[n={compare_count},mean_abs_diff={dyn_sl_diff_mean:.6f},max_abs_diff={dyn_sl_diff_max:.6f}] | "
                f"dyn_sl_clip_low={int(diagnostics.get('dynamic_stop_loss_clip_low_count', 0) or 0)} | "
                f"dyn_sl_clip_high={int(diagnostics.get('dynamic_stop_loss_clip_high_count', 0) or 0)}"
            )

        if formula_or_unit_mismatch:
            self.log_message.emit(
                "EMA Band Rejection audit result: detected formula/unit mismatch; corrective patch applied and diagnostics printed."
            )
        else:
            self.log_message.emit(
                "EMA Band Rejection audit result: calculations verified, strategy currently fails due to insufficient net edge per trade."
            )

    def _run_profile_optimization(
        self,
        db: Database,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
        regime_mask: Sequence[int] | None = None,
    ) -> dict:
        if not self._use_setup_gate and not self._is_quiet_mode():
            self.log_message.emit(
                "Backtest scalper mode active: SmartSetupGate is disabled for optimizer workers."
            )

        strategy_label = strategy_name.replace("_", " ").title()
        raw_grid_profiles = [dict(profile) for profile in self._optimization_grid]
        full_grid_profiles = _apply_strategy_interval_profile_constraints(
            raw_grid_profiles,
            strategy_name=strategy_name,
            interval=self._interval,
        )
        raw_grid_profile_count = int(len(raw_grid_profiles))
        constrained_grid_profile_count = int(len(full_grid_profiles))
        if len(full_grid_profiles) < len(raw_grid_profiles) and not self._is_quiet_mode():
            self.log_message.emit(
                "Optimizer strategy/interval constraints active: "
                f"{len(raw_grid_profiles)} -> {len(full_grid_profiles)} profiles "
                f"for {self._symbol} {strategy_label} {self._interval}."
            )
        theoretical_profiles = len(full_grid_profiles)
        if theoretical_profiles == 0:
            raise RuntimeError(
                "Optimization grid is empty after strategy/interval constraints."
            )
        optimizer_profile_mode = resolve_optimizer_profile_mode_name()
        if not self._is_quiet_mode():
            self.log_message.emit(
                f"{strategy_label} optimizer profile mode: {optimizer_profile_mode}"
            )

        min_required_trades = _resolve_optimizer_min_total_trades_for_interval(self._interval)
        breakeven_constraint_text = _describe_optimizer_breakeven_constraint(strategy_name)
        if self._is_debug_mode():
            self.log_message.emit(
                f"Optimizer breakeven stability rule ({strategy_label}): {breakeven_constraint_text}."
            )

        optimizer_policy = _resolve_optimizer_two_stage_policy(
            strategy_name=strategy_name,
            interval=self._interval,
            theoretical_profiles=theoretical_profiles,
        )
        two_stage_enabled = bool(optimizer_policy.get("two_stage_enabled", False))
        force_full_verification_for_winner = bool(
            optimizer_policy.get("force_full_verification_for_winner", True)
        )
        configured_sample_cap = optimizer_policy.get("sample_cap")
        worker_cap = optimizer_policy.get("worker_cap")
        worker_cap_value = (
            int(worker_cap)
            if isinstance(worker_cap, int) and worker_cap > 0
            else None
        )
        if worker_cap_value is None:
            worker_cap_value = int(OPTIMIZER_GUI_DEFAULT_MAX_WORKERS)
        configured_search_window = int(_resolve_optimizer_search_window_candles(self._interval))
        stage1_evaluated_profiles = 0
        stage2_evaluated_profiles = 0
        stage1_worker_count = 0
        stage2_worker_count = 0
        stage1_search_window_candles = int(configured_search_window)
        stage2_search_window_candles = int(configured_search_window)
        optimizer_mode_label = "single_stage"
        optimizer_quality_status = "STRICT_PASS"
        optimizer_low_edge_fallback_used = False
        optimizer_low_edge_reason = ""
        stage1_gate_stats: dict[str, int] = {
            "total": 0,
            "fail_min_trades": 0,
            "fail_breakeven": 0,
            "fail_avg_profit": 0,
            "fail_multi": 0,
            "eligible": 0,
        }
        stage2_gate_stats: dict[str, int] = {
            "total": 0,
            "fail_min_trades": 0,
            "fail_breakeven": 0,
            "fail_avg_profit": 0,
            "fail_multi": 0,
            "eligible": 0,
        }
        effective_grid_profiles_for_edge: list[OptimizationProfile] = []

        if worker_cap_value is not None and not self._is_quiet_mode():
            self.log_message.emit(
                "Optimizer worker policy active: "
                f"strategy cap={worker_cap_value}, global max={MAX_OPTIMIZATION_WORKERS}."
            )

        if self._use_setup_gate and not self._is_quiet_mode():
            self.log_message.emit(
                "Phase 19 optimizer guarantee active: every worker instantiates its own SmartSetupGate "
                f"and validates signals only after {SETUP_GATE_SETTLED_WARMUP_CANDLES} settled candles."
            )

        if two_stage_enabled:
            optimizer_mode_label = "two_stage"
            self.log_message.emit(
                f"{strategy_label} optimizer mode: 2-stage"
            )
            if self._is_debug_mode():
                self.log_message.emit(
                    "Stage 2 winner policy: "
                    + (
                        "force full final verification enabled."
                        if force_full_verification_for_winner
                        else "force full final verification disabled."
                    )
                )
            stage1_target_profiles = max(
                1,
                int(optimizer_policy.get("stage1_target_profiles", 1) or 1),
            )
            stage2_top_n = max(
                1,
                int(optimizer_policy.get("stage2_top_n", 1) or 1),
            )
            stage1_search_window_candles = max(
                1,
                int(optimizer_policy.get("stage1_search_window_candles", configured_search_window) or configured_search_window),
            )
            stage2_search_window_candles = max(
                1,
                int(optimizer_policy.get("stage2_search_window_candles", configured_search_window) or configured_search_window),
            )
            configured_search_window = int(stage2_search_window_candles)

            stage1_candles_df, stage1_regime_mask, stage1_uses_full_history = self._slice_optimizer_window(
                candles_df,
                window_candles=stage1_search_window_candles,
                regime_mask=regime_mask,
            )
            optimization_candles_df, optimization_regime_mask, optimization_uses_full_history = self._slice_optimizer_window(
                candles_df,
                window_candles=stage2_search_window_candles,
                regime_mask=regime_mask,
            )

            max_sample_profiles = min(theoretical_profiles, stage1_target_profiles)
            force_full_scan = max_sample_profiles >= theoretical_profiles
            sampled_grid_profiles = _sample_optimization_profiles(
                full_grid_profiles,
                max_profiles=max_sample_profiles,
                symbol=self._symbol,
                strategy_name=strategy_name,
                interval=self._interval,
                random_seed=OPTIMIZER_SAMPLE_RANDOM_SEED,
            )
            stage1_evaluated_profiles = len(sampled_grid_profiles)
            scan_profile_count = int(stage1_evaluated_profiles)
            is_sample_scan = stage1_evaluated_profiles < theoretical_profiles
            coverage_ratio = stage1_evaluated_profiles / float(theoretical_profiles)
            stage1_worker_count = _memory_safe_worker_count(
                stage1_evaluated_profiles,
                worker_cap_override=worker_cap_value,
            )
            stage2_worker_count = _memory_safe_worker_count(
                min(stage2_top_n, max(1, stage1_evaluated_profiles)),
                worker_cap_override=worker_cap_value,
            )
            initial_worker_count = int(stage2_worker_count)

            self.log_message.emit(
                f"Stage 1: evaluating {stage1_evaluated_profiles} profiles on {stage1_worker_count} workers "
                f"over {len(stage1_candles_df)} candles."
            )
            if stage1_uses_full_history and not self._is_quiet_mode():
                self.log_message.emit(
                    "Stage 1 window mode: full history."
                )
            elif not self._is_quiet_mode():
                self.log_message.emit(
                    f"Stage 1 window mode: last {len(stage1_candles_df)}/{len(candles_df)} candles."
                )
            try:
                stage1_phase_summaries = self._evaluate_optimization_phase(
                    stage1_candles_df,
                    strategy_name=strategy_name,
                    profiles=sampled_grid_profiles,
                    phase_label=f"Stage 1 {strategy_label} exploration",
                    cache_progress_start=20,
                    cache_progress_span=15,
                    optimization_progress_start=46,
                    optimization_progress_end=74,
                    regime_mask=stage1_regime_mask,
                    worker_cap_override=worker_cap_value,
                )
            except Exception as exc:
                self.log_message.emit(f"Optimization for {strategy_label} failed in Stage 1: {exc}")
                self._clear_signal_caches_and_gc()
                return {}
            if self._should_stop():
                return {}
            if not stage1_phase_summaries:
                raise RuntimeError("Stage 1 did not produce any optimization result.")

            stage1_ranked_eligible, stage1_ranked_all, stage1_gate_stats = self._analyze_optimization_phase_summaries(
                strategy_name=strategy_name,
                stage_label="Stage 1",
                phase_summaries=stage1_phase_summaries,
                min_required_trades=min_required_trades,
            )
            stage1_candidate_source = stage1_ranked_eligible
            if not stage1_candidate_source:
                self.log_message.emit(
                    "Stage 1 produced no eligible profiles; forwarding top raw candidates into Stage 2 verification."
                )
                stage1_candidate_source = stage1_ranked_all
            if not stage1_candidate_source:
                raise RuntimeError("Stage 1 produced no candidates for Stage 2 verification.")
            stage1_best_candidate = dict(stage1_candidate_source[0])
            if not self._is_quiet_mode():
                self.log_message.emit(
                    "Stage 1 Best Candidate: "
                    f"profile={_format_profile_dict(_extract_profile_from_summary(stage1_best_candidate))} | "
                    f"pf={self._format_profit_factor(float(stage1_best_candidate.get('profit_factor', 0.0) or 0.0))} | "
                    f"pnl={float(stage1_best_candidate.get('total_pnl_usd', 0.0) or 0.0):.2f} | "
                    f"win_rate={float(stage1_best_candidate.get('win_rate_pct', 0.0) or 0.0):.2f}%"
                )

            stage2_candidate_profiles: list[OptimizationProfile] = []
            stage2_seen_keys: set[object] = set()
            for summary in stage1_candidate_source[:stage2_top_n]:
                candidate_profile = _extract_profile_from_summary(summary)
                if not candidate_profile:
                    continue
                candidate_key = _strategy_cache_key(strategy_name, candidate_profile)
                if candidate_key in stage2_seen_keys:
                    continue
                stage2_seen_keys.add(candidate_key)
                stage2_candidate_profiles.append(candidate_profile)
            if not stage2_candidate_profiles:
                raise RuntimeError("Stage 2 candidate set is empty after Stage 1 ranking.")
            effective_grid_profiles_for_edge = [
                dict(profile)
                for profile in stage2_candidate_profiles
            ]

            stage2_evaluated_profiles = len(stage2_candidate_profiles)
            scan_profile_count = int(stage1_evaluated_profiles + stage2_evaluated_profiles)
            stage2_worker_count = _memory_safe_worker_count(
                stage2_evaluated_profiles,
                worker_cap_override=worker_cap_value,
            )
            initial_worker_count = int(stage2_worker_count)
            self.log_message.emit(
                f"Stage 2: verifying top {stage2_evaluated_profiles} profiles on {stage2_worker_count} workers "
                f"over {len(optimization_candles_df)} candles."
            )
            if optimization_uses_full_history and not self._is_quiet_mode():
                self.log_message.emit(
                    "Stage 2 window mode: full history."
                )
            elif not self._is_quiet_mode():
                self.log_message.emit(
                    f"Stage 2 window mode: last {len(optimization_candles_df)}/{len(candles_df)} candles."
                )
            try:
                stage2_phase_summaries = self._evaluate_optimization_phase(
                    optimization_candles_df,
                    strategy_name=strategy_name,
                    profiles=stage2_candidate_profiles,
                    phase_label=f"Stage 2 {strategy_label} verification",
                    cache_progress_start=35,
                    cache_progress_span=15,
                    optimization_progress_start=75,
                    optimization_progress_end=99,
                    regime_mask=optimization_regime_mask,
                    worker_cap_override=worker_cap_value,
                )
            except Exception as exc:
                self.log_message.emit(f"Optimization for {strategy_label} failed in Stage 2: {exc}")
                self._clear_signal_caches_and_gc()
                return {}
            if self._should_stop():
                return {}
            if not stage2_phase_summaries:
                raise RuntimeError("Stage 2 did not produce any optimization result.")

            stage2_ranked_eligible, stage2_ranked_all, stage2_gate_stats = self._analyze_optimization_phase_summaries(
                strategy_name=strategy_name,
                stage_label="Stage 2",
                phase_summaries=stage2_phase_summaries,
                min_required_trades=min_required_trades,
            )
            if strategy_name == "ema_band_rejection":
                self._log_ema_band_stage2_forensic_audit(
                    candles_df=optimization_candles_df,
                    stage2_ranked_all=stage2_ranked_all,
                    regime_mask=optimization_regime_mask,
            )
            if not stage2_ranked_eligible:
                if strategy_name == "ema_band_rejection":
                    low_edge_candidates = [
                        dict(summary)
                        for summary in stage2_ranked_all
                        if float(summary.get("total_trades", 0.0) or 0.0) > 0.0
                    ]
                    if low_edge_candidates:
                        optimizer_quality_status = "LOW_EDGE"
                        optimizer_low_edge_fallback_used = True
                        optimizer_low_edge_reason = (
                            "avg_profit_per_trade_net_pct below strict gate"
                        )
                        self.log_message.emit(
                            "No eligible profile under strict avg_profit gate; returning best verified low-edge candidate for inspection."
                        )
                        ranked_optimization_summaries = low_edge_candidates
                        self.log_message.emit(
                            "Stage 2 Low-Edge Candidate: "
                            f"profile={_format_profile_dict(_extract_profile_from_summary(ranked_optimization_summaries[0]))} | "
                            f"pf={self._format_profit_factor(float(ranked_optimization_summaries[0].get('profit_factor', 0.0) or 0.0))} | "
                            f"pnl={float(ranked_optimization_summaries[0].get('total_pnl_usd', 0.0) or 0.0):.2f} | "
                            f"win_rate={float(ranked_optimization_summaries[0].get('win_rate_pct', 0.0) or 0.0):.2f}%"
                        )
                    else:
                        self._clear_signal_caches_and_gc()
                        min_avg_profit_per_trade_net_pct = _resolve_optimizer_min_avg_profit_per_trade_net_pct()
                        constraint_parts = [
                            f">= {min_required_trades} trades",
                            breakeven_constraint_text,
                            f"avg_profit_per_trade_net_pct >= {min_avg_profit_per_trade_net_pct:.2f}%",
                        ]
                        raise RuntimeError(
                            "FAIL: Stage 2 verification produced no executable low-edge candidate: "
                            + " | ".join(constraint_parts)
                            + "."
                        )
                else:
                    self._clear_signal_caches_and_gc()
                    min_avg_profit_per_trade_net_pct = _resolve_optimizer_min_avg_profit_per_trade_net_pct()
                    constraint_parts = [
                        f">= {min_required_trades} trades",
                        breakeven_constraint_text,
                        f"avg_profit_per_trade_net_pct >= {min_avg_profit_per_trade_net_pct:.2f}%",
                    ]
                    raise RuntimeError(
                        "Stage 2 verification produced no eligible profile: "
                        + " | ".join(constraint_parts)
                        + "."
                    )
            else:
                ranked_optimization_summaries = stage2_ranked_eligible
                self.log_message.emit(
                    "Stage 2 Verified Candidate: "
                    f"profile={_format_profile_dict(_extract_profile_from_summary(ranked_optimization_summaries[0]))} | "
                    f"pf={self._format_profit_factor(float(ranked_optimization_summaries[0].get('profit_factor', 0.0) or 0.0))} | "
                    f"pnl={float(ranked_optimization_summaries[0].get('total_pnl_usd', 0.0) or 0.0):.2f} | "
                    f"win_rate={float(ranked_optimization_summaries[0].get('win_rate_pct', 0.0) or 0.0):.2f}%"
                )
        else:
            optimizer_mode_label = "single_stage"
            self.log_message.emit(
                f"{strategy_label} optimizer mode: 1-stage"
            )
            optimization_candles_df, optimization_regime_mask, optimization_uses_full_history = self._slice_optimizer_window(
                candles_df,
                window_candles=configured_search_window,
                regime_mask=regime_mask,
            )
            if configured_sample_cap is None:
                max_sample_profiles = int(theoretical_profiles)
            else:
                max_sample_profiles = min(int(theoretical_profiles), int(configured_sample_cap))
            max_sample_profiles = max(1, int(max_sample_profiles))
            force_full_scan = max_sample_profiles >= theoretical_profiles
            sampled_grid_profiles = _sample_optimization_profiles(
                full_grid_profiles,
                max_profiles=max_sample_profiles,
                symbol=self._symbol,
                strategy_name=strategy_name,
                interval=self._interval,
                random_seed=OPTIMIZER_SAMPLE_RANDOM_SEED,
            )
            effective_grid_profiles_for_edge = [
                dict(profile)
                for profile in sampled_grid_profiles
            ]
            scan_profile_count = len(sampled_grid_profiles)
            is_sample_scan = scan_profile_count < theoretical_profiles
            coverage_ratio = scan_profile_count / float(theoretical_profiles)
            initial_worker_count = _memory_safe_worker_count(
                scan_profile_count,
                worker_cap_override=worker_cap_value,
            )
            stage1_evaluated_profiles = int(scan_profile_count)
            stage1_worker_count = int(initial_worker_count)

            if is_sample_scan:
                self.log_message.emit(
                    "Sampling mode active: "
                    f"evaluating {scan_profile_count} of {theoretical_profiles} profiles "
                    f"(~{coverage_ratio * 100.0:.1f}% coverage)."
                )
            else:
                self.log_message.emit(
                    f"Full Scan active for {self._symbol} {strategy_label}: "
                    f"{theoretical_profiles}/{theoretical_profiles} profiles "
                    f"(100.00% coverage), random_seed={OPTIMIZER_SAMPLE_RANDOM_SEED}."
                )
            profiles_per_worker = (
                scan_profile_count / float(initial_worker_count)
                if initial_worker_count > 0
                else float(scan_profile_count)
            )
            if not self._is_quiet_mode():
                self.log_message.emit(
                    "Optimizer workload split: "
                    f"~{profiles_per_worker:.0f} profiles per worker "
                    f"across {initial_worker_count} worker processes."
                )
                self.log_message.emit(
                    f"Optimization mode active for {self._symbol}: evaluating {scan_profile_count} profiles "
                    f"(grid_total={theoretical_profiles}) on {initial_worker_count} worker processes."
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
                    worker_cap_override=worker_cap_value,
                )
            except Exception as exc:
                self.log_message.emit(f"Optimization for {strategy_label} failed: {exc}")
                self._clear_signal_caches_and_gc()
                return {}
            if self._should_stop():
                return {}
            if not optimization_phase_summaries:
                raise RuntimeError("Optimization did not produce any result.")
            ranked_eligible, _ranked_all, stage1_gate_stats = self._analyze_optimization_phase_summaries(
                strategy_name=strategy_name,
                stage_label="Stage 1",
                phase_summaries=optimization_phase_summaries,
                min_required_trades=min_required_trades,
            )
            if not ranked_eligible:
                if strategy_name == "ema_band_rejection":
                    low_edge_candidates = [
                        dict(summary)
                        for summary in _ranked_all
                        if float(summary.get("total_trades", 0.0) or 0.0) > 0.0
                    ]
                    if low_edge_candidates:
                        optimizer_quality_status = "LOW_EDGE"
                        optimizer_low_edge_fallback_used = True
                        optimizer_low_edge_reason = (
                            "avg_profit_per_trade_net_pct below strict gate"
                        )
                        self.log_message.emit(
                            "No eligible profile under strict avg_profit gate; returning best verified low-edge candidate for inspection."
                        )
                        ranked_optimization_summaries = low_edge_candidates
                    else:
                        self._clear_signal_caches_and_gc()
                        min_avg_profit_per_trade_net_pct = _resolve_optimizer_min_avg_profit_per_trade_net_pct()
                        constraint_parts = [
                            f">= {min_required_trades} trades",
                            breakeven_constraint_text,
                            f"avg_profit_per_trade_net_pct >= {min_avg_profit_per_trade_net_pct:.2f}%",
                        ]
                        raise RuntimeError(
                            "FAIL: No optimization profile passed stability constraints and no low-edge candidate had executable trades: "
                            + " | ".join(constraint_parts)
                            + "."
                        )
                else:
                    self._clear_signal_caches_and_gc()
                    min_avg_profit_per_trade_net_pct = _resolve_optimizer_min_avg_profit_per_trade_net_pct()
                    constraint_parts = [
                        f">= {min_required_trades} trades",
                        breakeven_constraint_text,
                        f"avg_profit_per_trade_net_pct >= {min_avg_profit_per_trade_net_pct:.2f}%",
                    ]
                    raise RuntimeError(
                        "No optimization profile passed stability constraints: "
                        + " | ".join(constraint_parts)
                        + "."
                    )
            else:
                ranked_optimization_summaries = ranked_eligible

        best_summary = dict(ranked_optimization_summaries[0])
        best_profile = _extract_profile_from_summary(best_summary)
        if strategy_name in OPTIMIZER_JIT_STRATEGY_CONSISTENT_BACKTEST:
            try:
                probe_min_confidence_pct = self._min_confidence_pct
                if self._use_setup_gate:
                    probe_min_confidence_pct = _enforce_optimizer_min_confidence_floor(
                        probe_min_confidence_pct
                    )
                (
                    best_open_arr,
                    best_high_arr,
                    best_low_arr,
                    best_close_arr,
                    best_volume_arr,
                ) = _extract_ohlcv_numpy_arrays(optimization_candles_df)
                consistent_payload = _build_strategy_jit_precomputed_payload(
                    optimization_candles_df,
                    best_open_arr,
                    best_high_arr,
                    best_low_arr,
                    best_close_arr,
                    best_volume_arr,
                    strategy_name=strategy_name,
                    strategy_profile=best_profile,
                    use_setup_gate=self._use_setup_gate,
                    min_confidence_pct=probe_min_confidence_pct,
                    regime_mask=optimization_regime_mask,
                )
            except Exception:
                consistent_payload = None
            if isinstance(consistent_payload, Mapping):
                best_signals = _align_signal_series_to_candle_count(
                    consistent_payload.get("signals", []),
                    candle_count=len(optimization_candles_df),
                )
                best_total_signals = int(consistent_payload.get("total_signals", 0) or 0)
                best_approved_signals = int(consistent_payload.get("approved_signals", 0) or 0)
                best_blocked_signals = int(consistent_payload.get("blocked_signals", 0) or 0)
                self._last_precomputed_exit_cache = {
                    key: value
                    for key, value in consistent_payload.items()
                    if key in {
                        "precomputed_long_exit_flags",
                        "precomputed_short_exit_flags",
                        "precomputed_dynamic_stop_loss_pcts",
                        "precomputed_dynamic_take_profit_pcts",
                    }
                }
                diagnostics_payload = consistent_payload.get("strategy_diagnostics")
                self._last_strategy_diagnostics = (
                    dict(diagnostics_payload)
                    if isinstance(diagnostics_payload, Mapping)
                    else None
                )
            else:
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
        else:
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
        candidate_log_label = (
            "Stage 2 Verified Candidate"
            if two_stage_enabled
            else "Best Optimizer Candidate"
        )
        if optimizer_low_edge_fallback_used:
            candidate_log_label = (
                "Stage 2 Low-Edge Candidate"
                if two_stage_enabled
                else "Low-Edge Optimizer Candidate"
            )
        self.log_message.emit(
            f"{candidate_log_label} for {self._symbol}: "
            f"{_format_profile_dict(best_profile)} "
            f"| profit_factor={self._format_profit_factor(float(best_summary['profit_factor']))} "
            f"| real_rrr={self._format_profit_factor(float(best_summary['real_rrr']))} "
            f"| pnl={float(best_summary['total_pnl_usd']):.2f} "
            f"| win_rate={float(best_summary['win_rate_pct']):.2f}% "
            f"| max_dd={float(best_summary.get('max_drawdown_pct', 0.0)):.2f}%"
        )
        self._log_backtest_metrics(best_result, prefix="Final verified optimization metrics")
        self.log_message.emit(
            f"Final Verified Result for {self._symbol}: "
            f"{_format_profile_dict(best_profile)} "
            f"| profit_factor={self._format_profit_factor(float(best_result.get('profit_factor', 0.0) or 0.0))} "
            f"| real_rrr={self._format_profit_factor(float(best_result.get('real_rrr', 0.0) or 0.0))} "
            f"| pnl={float(best_result.get('total_pnl_usd', 0.0) or 0.0):.2f} "
            f"| win_rate={float(best_result.get('win_rate_pct', 0.0) or 0.0):.2f}% "
            f"| max_dd={float(best_result.get('max_drawdown_pct', 0.0) or 0.0):.2f}%"
        )
        if strategy_name == "frama_cross":
            optimizer_min_confidence_pct = self._min_confidence_pct
            if self._use_setup_gate:
                optimizer_min_confidence_pct = _enforce_optimizer_min_confidence_floor(
                    optimizer_min_confidence_pct
                )
            optimizer_required_warmup, optimizer_required_candles = (
                self._resolve_required_runtime_candles(
                    strategy_name=strategy_name,
                    strategy_profile=best_profile,
                )
            )
            optimizer_window_start = TRACE_VALUE_UNAVAILABLE
            optimizer_window_end = TRACE_VALUE_UNAVAILABLE
            if len(optimization_candles_df) > 0 and "open_time" in optimization_candles_df.columns:
                with suppress(Exception):
                    optimizer_window_start = str(
                        pd.to_datetime(
                            optimization_candles_df["open_time"].iloc[0],
                            utc=False,
                            errors="coerce",
                        )
                    )
                    optimizer_window_end = str(
                        pd.to_datetime(
                            optimization_candles_df["open_time"].iloc[-1],
                            utc=False,
                            errors="coerce",
                        )
                    )
            regime_total = int(len(optimization_regime_mask)) if optimization_regime_mask is not None else 0
            regime_allowed = (
                int(sum(int(value) > 0 for value in optimization_regime_mask))
                if optimization_regime_mask is not None
                else 0
            )
            optimizer_input_snapshot = {
                "path": (
                    "compiled signal path + full strategy-consistent backtest engine"
                    if strategy_name in OPTIMIZER_JIT_STRATEGY_CONSISTENT_BACKTEST
                    else (
                        "compiled signal path + compiled core path"
                        if _jit_strategy_code(strategy_name) != 0
                        else "python signal path + full strategy-consistent backtest engine"
                    )
                ),
                "symbol": str(self._symbol),
                "interval": str(self._interval),
                "candles": int(len(optimization_candles_df)),
                "window_start": str(optimizer_window_start),
                "window_end": str(optimizer_window_end),
                "profile": str(_format_profile_dict(best_profile)),
                "use_setup_gate": bool(self._use_setup_gate),
                "min_confidence_pct": (
                    None
                    if optimizer_min_confidence_pct is None
                    else float(optimizer_min_confidence_pct)
                ),
                "required_warmup_candles": int(optimizer_required_warmup),
                "required_candles": int(optimizer_required_candles),
                "fee_pct": float(_resolve_backtest_fee_pct()),
                "slippage_pct_per_side": float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
                "stop_loss_pct": float(best_profile.get("stop_loss_pct", 0.0) or 0.0),
                "take_profit_pct": float(best_profile.get("take_profit_pct", 0.0) or 0.0),
                "trailing_activation_pct": float(best_profile.get("trailing_activation_pct", 0.0) or 0.0),
                "trailing_distance_pct": float(best_profile.get("trailing_distance_pct", 0.0) or 0.0),
                "breakeven_activation_pct": float(best_profile.get("breakeven_activation_pct", 0.0) or 0.0),
                "breakeven_buffer_pct": float(best_profile.get("breakeven_buffer_pct", 0.0) or 0.0),
                "regime_filter_active": bool(optimization_regime_mask is not None),
                "regime_allowed_candles": int(regime_allowed),
                "regime_total_candles": int(regime_total),
                "optimizer_early_stop_enabled": False,
            }
            final_input_snapshot = {
                "path": "final verification full backtest engine",
                "symbol": str(self._symbol),
                "interval": str(self._interval),
                "candles": int(len(optimization_candles_df)),
                "window_start": str(optimizer_window_start),
                "window_end": str(optimizer_window_end),
                "profile": str(_format_profile_dict(best_profile)),
                "use_setup_gate": bool(self._use_setup_gate),
                "min_confidence_pct": (
                    None
                    if self._min_confidence_pct is None
                    else float(self._min_confidence_pct)
                ),
                "required_warmup_candles": int(best_result.get("required_warmup_candles", 0) or 0),
                "required_candles": int(best_result.get("required_candles", 0) or 0),
                "fee_pct": float(best_result.get("effective_taker_fee_pct", 0.0) or 0.0),
                "slippage_pct_per_side": float(best_result.get("slippage_penalty_pct_per_side", 0.0) or 0.0),
                "stop_loss_pct": float(best_result.get("stop_loss_pct", 0.0) or 0.0),
                "take_profit_pct": float(best_result.get("take_profit_pct", 0.0) or 0.0),
                "trailing_activation_pct": float(best_result.get("trailing_activation_pct", 0.0) or 0.0),
                "trailing_distance_pct": float(best_result.get("trailing_distance_pct", 0.0) or 0.0),
                "breakeven_activation_pct": float(best_result.get("breakeven_activation_pct", 0.0) or 0.0),
                "breakeven_buffer_pct": float(best_result.get("breakeven_buffer_pct", 0.0) or 0.0),
                "regime_filter_active": bool(optimization_regime_mask is not None),
                "regime_allowed_candles": int(regime_allowed),
                "regime_total_candles": int(regime_total),
                "optimizer_early_stop_enabled": False,
            }
            self.log_message.emit(
                "FRAMA consistency inputs | optimizer: "
                f"path={optimizer_input_snapshot['path']} | candles={optimizer_input_snapshot['candles']} | "
                f"window={optimizer_input_snapshot['window_start']} -> {optimizer_input_snapshot['window_end']} | "
                f"profile={optimizer_input_snapshot['profile']} | gate={int(bool(optimizer_input_snapshot['use_setup_gate']))} | "
                f"min_conf={optimizer_input_snapshot['min_confidence_pct']} | req={optimizer_input_snapshot['required_warmup_candles']}/{optimizer_input_snapshot['required_candles']} | "
                f"fee={optimizer_input_snapshot['fee_pct']:.4f}% | slippage={optimizer_input_snapshot['slippage_pct_per_side']:.4f}%"
            )
            self.log_message.emit(
                "FRAMA consistency inputs | final: "
                f"path={final_input_snapshot['path']} | candles={final_input_snapshot['candles']} | "
                f"window={final_input_snapshot['window_start']} -> {final_input_snapshot['window_end']} | "
                f"profile={final_input_snapshot['profile']} | gate={int(bool(final_input_snapshot['use_setup_gate']))} | "
                f"min_conf={final_input_snapshot['min_confidence_pct']} | req={final_input_snapshot['required_warmup_candles']}/{final_input_snapshot['required_candles']} | "
                f"fee={final_input_snapshot['fee_pct']:.4f}% | slippage={final_input_snapshot['slippage_pct_per_side']:.4f}%"
            )
            input_diff_fields = (
                "symbol",
                "interval",
                "candles",
                "window_start",
                "window_end",
                "profile",
                "use_setup_gate",
                "required_warmup_candles",
                "required_candles",
                "fee_pct",
                "slippage_pct_per_side",
                "stop_loss_pct",
                "take_profit_pct",
                "trailing_activation_pct",
                "trailing_distance_pct",
                "breakeven_activation_pct",
                "breakeven_buffer_pct",
                "regime_filter_active",
                "regime_allowed_candles",
                "regime_total_candles",
                "optimizer_early_stop_enabled",
            )
            input_differences: list[str] = []
            for field_name in input_diff_fields:
                optimizer_value = optimizer_input_snapshot.get(field_name)
                final_value = final_input_snapshot.get(field_name)
                if isinstance(optimizer_value, float) or isinstance(final_value, float):
                    with suppress(Exception):
                        if math.isclose(float(optimizer_value), float(final_value), rel_tol=1e-9, abs_tol=1e-9):
                            continue
                if optimizer_value != final_value:
                    input_differences.append(
                        f"{field_name}: optimizer={optimizer_value} final={final_value}"
                    )
            if input_differences:
                self.log_message.emit(
                    "FRAMA input consistency warning: "
                    + "; ".join(input_differences[:6])
                )

        if strategy_name in {"ema_band_rejection", "frama_cross"}:
            consistency_ok, consistency_mismatches = _compare_optimizer_final_summary_metrics(
                best_summary,
                best_result,
            )
            best_result["optimizer_final_consistency_ok"] = bool(consistency_ok)
            best_result["optimizer_final_consistency_mismatches"] = list(consistency_mismatches)
            consistency_label = (
                "EMA Band Rejection"
                if strategy_name == "ema_band_rejection"
                else "FRAMA Cross"
            )
            if consistency_ok:
                if self._is_debug_mode():
                    self.log_message.emit(
                        f"{consistency_label} optimizer/final consistency check passed."
                    )
            else:
                self.log_message.emit(
                    f"{consistency_label} consistency mismatch between optimizer summary and final rerun: "
                    + "; ".join(consistency_mismatches[:4])
                )
                if optimizer_quality_status == "STRICT_PASS":
                    optimizer_quality_status = "WARN_CONSISTENCY"
        if optimizer_quality_status not in {"STRICT_PASS", "LOW_EDGE", "WARN_CONSISTENCY"}:
            optimizer_quality_status = "STRICT_PASS"
        self.log_message.emit(
            f"Optimizer verdict for {self._symbol}: {optimizer_quality_status}"
        )

        keep_top_count = min(
            len(ranked_optimization_summaries),
            max(1, int(_resolve_optimizer_validation_top_n())),
        )
        optimizer_effective_grid_bounds = _summarize_profile_grid_bounds(
            effective_grid_profiles_for_edge
        )
        if not optimizer_effective_grid_bounds:
            optimizer_effective_grid_bounds = _summarize_profile_grid_bounds(
                [best_profile] if best_profile else []
            )
        rejected_min_trades = int(
            stage1_gate_stats.get("fail_min_trades", 0)
            + stage2_gate_stats.get("fail_min_trades", 0)
        )
        rejected_breakeven_activation = int(
            stage1_gate_stats.get("fail_breakeven", 0)
            + stage2_gate_stats.get("fail_breakeven", 0)
        )
        rejected_avg_profit_per_trade_net = int(
            stage1_gate_stats.get("fail_avg_profit", 0)
            + stage2_gate_stats.get("fail_avg_profit", 0)
        )
        rejected_risk_guard = int(
            max(0, raw_grid_profile_count - constrained_grid_profile_count)
        )
        rejected_other = 0
        optimization_summaries_compact = [
            dict(summary)
            for summary in ranked_optimization_summaries[:keep_top_count]
        ]
        best_result.update(
            {
                "optimization_mode": True,
                "optimizer_mode": str(optimizer_mode_label),
                "optimizer_profile_mode": str(optimizer_profile_mode),
                "optimizer_two_stage_enabled": bool(two_stage_enabled),
                "optimizer_force_full_verification_for_winner": bool(force_full_verification_for_winner),
                "best_profile": best_profile,
                "best_optimizer_candidate_summary": dict(best_summary),
                "optimization_results": optimization_summaries_compact,
                "decision_matrix_summaries": optimization_summaries_compact,
                "evaluated_profiles": int(scan_profile_count),
                "sampled_profiles": int(stage1_evaluated_profiles),
                "theoretical_profiles": theoretical_profiles,
                "search_window_candles": len(optimization_candles_df),
                "validated_profiles": keep_top_count,
                "full_history_optimization": optimization_uses_full_history,
                "sampling_mode": (
                    "two_stage"
                    if two_stage_enabled
                    else ("sample" if is_sample_scan else "full")
                ),
                "sampling_coverage_pct": coverage_ratio * 100.0,
                "sampling_random_seed": int(OPTIMIZER_SAMPLE_RANDOM_SEED),
                "optimizer_worker_processes": int(initial_worker_count),
                "optimizer_max_sample_profiles": int(max_sample_profiles),
                "optimizer_force_full_scan": bool(force_full_scan),
                "optimizer_configured_search_window_candles": int(configured_search_window),
                "optimizer_stage1_profiles": int(stage1_evaluated_profiles),
                "optimizer_stage2_profiles": int(stage2_evaluated_profiles),
                "optimizer_stage1_worker_processes": int(stage1_worker_count),
                "optimizer_stage2_worker_processes": int(stage2_worker_count),
                "optimizer_stage1_search_window_candles": int(stage1_search_window_candles),
                "optimizer_stage2_search_window_candles": int(stage2_search_window_candles),
                "session_leaderboard": [],
                "session_day_breakdown": [],
                "optimizer_quality_status": str(optimizer_quality_status),
                "optimizer_strict_pass": bool(optimizer_quality_status == "STRICT_PASS"),
                "optimizer_low_edge": bool(optimizer_quality_status == "LOW_EDGE"),
                "optimizer_low_edge_fallback_used": bool(optimizer_low_edge_fallback_used),
                "optimizer_low_edge_reason": str(optimizer_low_edge_reason),
                "optimizer_grid_profile_tag": "standard_grid_v1",
                "optimizer_grid_total_before_guards": int(raw_grid_profile_count),
                "optimizer_grid_total_after_guards": int(constrained_grid_profile_count),
                "optimizer_effective_grid_profile_count": int(
                    len(effective_grid_profiles_for_edge)
                ),
                "optimizer_effective_grid_bounds": dict(optimizer_effective_grid_bounds),
                "optimizer_stage1_gate_stats": dict(stage1_gate_stats),
                "optimizer_stage2_gate_stats": dict(stage2_gate_stats),
                "rejected_min_trades": int(rejected_min_trades),
                "rejected_breakeven_activation": int(rejected_breakeven_activation),
                "rejected_avg_profit_per_trade_net": int(rejected_avg_profit_per_trade_net),
                "rejected_risk_guard": int(rejected_risk_guard),
                "rejected_other": int(rejected_other),
            }
        )
        # Persist winner immediately (batch-saving) so each strategy/coin result is durable
        try:
            if db is not None:
                self._persist_backtest_result(db, best_result, strategy_name=strategy_name)
        except Exception:
            # Don't fail if persistence fails; continue to clear caches.
            self.log_message.emit("Warning: failed to persist optimization winner to DB.")
        if self._isolated_db:
            try:
                checkpoint_run_id = self._persist_backtest_result_to_primary_db_with_retry(
                    best_result,
                    strategy_name=strategy_name,
                    stage_label="optimizer_checkpoint",
                )
                self._checkpoint_backtest_run_id = int(checkpoint_run_id)
                self.log_message.emit(
                    f"Optimizer checkpoint saved to primary DB for {self._symbol} {self._interval} ({strategy_name}) "
                    f"(run_id={self._checkpoint_backtest_run_id})."
                )
            except Exception as exc:
                self.log_message.emit(
                    "Warning: optimizer checkpoint persistence to primary DB failed: "
                    f"{exc}"
                )

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
        worker_cap_override: int | None = None,
    ) -> list[dict[str, float]]:
        total_profiles = len(profiles)
        if total_profiles == 0:
            return []

        strategy_signal_cache: dict[object, dict[str, object]] | None = None
        precomputed_signals: list[int] | None = None
        total_signals = 0
        approved_signals = 0
        blocked_signals = 0
        use_jit_strategy = _jit_strategy_code(strategy_name) != 0
        force_python_path = False
        runtime_worker_cap = worker_cap_override
        suggested_cap = int(OPTIMIZER_GUI_DEFAULT_MAX_WORKERS)
        runtime_worker_cap = (
            suggested_cap
            if runtime_worker_cap is None
            else min(int(runtime_worker_cap), suggested_cap)
        )
        worker_count = _memory_safe_worker_count(
            total_profiles,
            worker_cap_override=int(runtime_worker_cap),
        )
        if not self._is_quiet_mode():
            self.log_message.emit(
                f"{phase_label}: runtime worker governor active "
                f"(requested_cap={runtime_worker_cap}, workers={worker_count}, "
                f"force_python_path={'yes' if force_python_path else 'no'})."
            )

        if use_jit_strategy and strategy_name == "ema_band_rejection":
            try:
                reference_profile = (
                    dict(profiles[0])
                    if profiles
                    else {
                        "ema_fast": 5.0,
                        "ema_mid": 10.0,
                        "ema_slow": 20.0,
                        "slope_lookback": 5.0,
                        "min_ema_spread_pct": 0.05,
                        "min_slow_slope_pct": 0.0,
                        "pullback_requires_outer_band_touch": 0.0,
                        "use_rejection_quality_filter": 0.0,
                        "rejection_wick_min_ratio": 0.35,
                        "rejection_body_min_ratio": 0.20,
                        "use_rsi_filter": 0.0,
                        "rsi_length": 14.0,
                        "rsi_midline": 50.0,
                        "use_rsi_cross_filter": 0.0,
                        "rsi_midline_margin": 0.0,
                        "use_volume_filter": 0.0,
                        "volume_ma_length": 20.0,
                        "volume_multiplier": 1.0,
                        "use_atr_stop_buffer": 0.0,
                        "atr_length": 14.0,
                        "atr_stop_buffer_mult": 0.5,
                        "signal_cooldown_bars": 0.0,
                    }
                )
                open_arr, high_arr, low_arr, close_arr, volume_arr = _extract_ohlcv_numpy_arrays(candles_df)
                (
                    alignment_ok,
                    mismatch_count,
                    compare_count,
                    mismatch_pct,
                ) = _validate_ema_band_jit_alignment(
                    candles_df,
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    volume_arr,
                    reference_profile,
                )
                if alignment_ok:
                    if self._is_debug_mode():
                        self.log_message.emit(
                            "EMA Band Rejection JIT alignment check passed: "
                            f"{mismatch_count}/{compare_count} mismatches ({mismatch_pct:.2f}%)."
                        )
                    try:
                        probe_candle_count = max(1, min(len(candles_df), 12_000))
                        probe_candles_df = candles_df.iloc[-probe_candle_count:].copy(deep=False)
                        probe_regime_mask = (
                            None
                            if regime_mask is None
                            else list(regime_mask)[-probe_candle_count:]
                        )
                        probe_min_confidence_pct = self._min_confidence_pct
                        if self._use_setup_gate:
                            probe_min_confidence_pct = _enforce_optimizer_min_confidence_floor(
                                probe_min_confidence_pct
                            )
                        (
                            probe_open_arr,
                            probe_high_arr,
                            probe_low_arr,
                            probe_close_arr,
                            probe_volume_arr,
                        ) = _extract_ohlcv_numpy_arrays(probe_candles_df)
                        optimized_probe = _run_ema_band_rejection_consistent_profile_evaluation(
                            probe_candles_df,
                            probe_open_arr,
                            probe_high_arr,
                            probe_low_arr,
                            probe_close_arr,
                            probe_volume_arr,
                            strategy_profile=reference_profile,
                            use_setup_gate=self._use_setup_gate,
                            min_confidence_pct=probe_min_confidence_pct,
                            regime_mask=probe_regime_mask,
                            leverage_override=self._leverage_override,
                            symbol=str(self._symbol),
                            interval=str(self._interval),
                        )
                        python_probe = _run_python_profile_evaluation(
                            probe_candles_df,
                            strategy_name="ema_band_rejection",
                            strategy_profile=reference_profile,
                            use_setup_gate=self._use_setup_gate,
                            min_confidence_pct=probe_min_confidence_pct,
                            regime_mask=probe_regime_mask,
                            leverage_override=self._leverage_override,
                            symbol=str(self._symbol),
                            interval=str(self._interval),
                            precomputed_payload=None,
                            apply_optimizer_early_stop=False,
                        )
                        if optimized_probe is None and python_probe is None:
                            if self._is_debug_mode():
                                self.log_message.emit(
                                    "EMA Band Rejection end-to-end consistency probe skipped: "
                                    "no executable trades on probe sample."
                                )
                        elif optimized_probe is None or python_probe is None:
                            use_jit_strategy = False
                            force_python_path = True
                            self.log_message.emit(
                                "EMA Band Rejection end-to-end consistency probe failed: "
                                "one path returned no result. fallback to strategy-consistent python path."
                            )
                        else:
                            probe_ok, probe_mismatches = _compare_optimizer_final_summary_metrics(
                                optimized_probe[0],
                                python_probe[0],
                            )
                            if probe_ok:
                                if self._is_debug_mode():
                                    self.log_message.emit(
                                        "EMA Band Rejection end-to-end consistency probe passed: "
                                        "optimizer path matches full strategy-consistent engine."
                                    )
                            else:
                                use_jit_strategy = False
                                force_python_path = True
                                self.log_message.emit(
                                    "EMA Band Rejection end-to-end consistency probe failed: "
                                    + "; ".join(probe_mismatches[:3])
                                    + ". fallback to strategy-consistent python path."
                                )
                    except Exception as exc:
                        use_jit_strategy = False
                        force_python_path = True
                        self.log_message.emit(
                            "EMA Band Rejection end-to-end consistency probe error: "
                            f"{exc}. fallback to strategy-consistent python path."
                        )
                else:
                    use_jit_strategy = False
                    force_python_path = True
                    self.log_message.emit(
                        "EMA Band Rejection JIT alignment check failed: "
                        f"{mismatch_count}/{compare_count} mismatches ({mismatch_pct:.2f}%). "
                        "fallback to python path."
                    )
            except Exception as exc:
                use_jit_strategy = False
                force_python_path = True
                self.log_message.emit(
                    "EMA Band Rejection JIT alignment check error: "
                    f"{exc}. fallback to python path."
                )

        if use_jit_strategy:
            if self._is_debug_mode():
                if strategy_name in OPTIMIZER_JIT_STRATEGY_CONSISTENT_BACKTEST:
                    self.log_message.emit(
                        f"Numba JIT path active for {strategy_name}: "
                        "compiled signal path + full strategy-consistent backtest engine."
                    )
                else:
                    self.log_message.emit(
                        f"Numba JIT path active for {strategy_name}: compiled signal path + compiled core path."
                    )
        else:
            if self._is_debug_mode():
                self.log_message.emit(
                    f"Python vector path active for {strategy_name}: worker evaluation uses signal payload + backtest engine."
                )
            variant_profiles = _collect_strategy_variant_profiles(strategy_name, profiles)
            estimated_cache_bytes = _estimate_strategy_signal_cache_bytes(
                strategy_name=strategy_name,
                candle_count=len(candles_df),
                variant_count=len(variant_profiles),
            )
            estimated_replicated_cache_bytes = int(estimated_cache_bytes) * int(worker_count)
            should_build_shared_cache = True
            cache_skip_reason = ""
            if estimated_cache_bytes > int(OPTIMIZER_SIGNAL_CACHE_MAX_PARENT_BYTES):
                should_build_shared_cache = False
                cache_skip_reason = (
                    "parent cache estimate exceeds limit "
                    f"({estimated_cache_bytes / (1024 ** 2):.1f} MiB > "
                    f"{OPTIMIZER_SIGNAL_CACHE_MAX_PARENT_BYTES / (1024 ** 2):.1f} MiB)"
                )
            elif estimated_replicated_cache_bytes > int(OPTIMIZER_SIGNAL_CACHE_MAX_REPLICATED_BYTES):
                should_build_shared_cache = False
                cache_skip_reason = (
                    "replicated worker cache estimate exceeds limit "
                    f"({estimated_replicated_cache_bytes / (1024 ** 3):.2f} GiB > "
                    f"{OPTIMIZER_SIGNAL_CACHE_MAX_REPLICATED_BYTES / (1024 ** 3):.2f} GiB)"
                )
            if should_build_shared_cache:
                self.log_message.emit(
                    f"{phase_label}: preparing shared signal cache for {len(variant_profiles)} profile variants."
                )
                strategy_signal_cache = self._build_parallel_strategy_signal_cache(
                    candles_df,
                    strategy_name=strategy_name,
                    worker_count=worker_count,
                    variant_profiles=variant_profiles,
                    progress_start=cache_progress_start,
                    progress_span=cache_progress_span,
                    regime_mask=regime_mask,
                )
            else:
                strategy_signal_cache = None
                self.log_message.emit(
                    f"{phase_label}: shared signal cache disabled ({cache_skip_reason}). "
                    "Workers will compute signals on demand per profile to protect GUI stability."
                )

        self.progress_update.emit(
            optimization_progress_start,
            f"{phase_label} 0/{total_profiles}",
        )
        self.log_message.emit(
            f"{phase_label}: started ({total_profiles} profiles on {worker_count} workers)."
        )

        if use_jit_strategy:
            _warmup_numba_runtime(self.log_message.emit if self._is_debug_mode() else None)
        candles_payload = _candles_dataframe_to_worker_payload(candles_df)
        first_summary_logged = strategy_signal_cache is None
        completed_profiles = 0
        chunk_min, chunk_max = 8, 12
        target_chunk_size = int(
            math.ceil(
                total_profiles
                / float(max(1, worker_count * 4))
            )
        )
        profile_chunk_size = max(chunk_min, min(chunk_max, target_chunk_size))
        dispatch_profile_batch_size = max(1, worker_count * profile_chunk_size)
        self.log_message.emit(
            f"{phase_label}: optimizer chunking active "
            f"(chunk_size={profile_chunk_size}, dispatch_profiles={dispatch_profile_batch_size}, workers={worker_count})."
        )
        pool = _create_optimizer_pool(
            worker_count,
            candles_payload,
            regime_mask=regime_mask,
            strategy_signal_cache=strategy_signal_cache,
        )
        self._active_pool = pool
        interrupted = False
        phase_results: list[dict[str, float]] = []
        best_summary: dict[str, float] | None = None
        best_profile: OptimizationProfile | None = None
        phase_cache_dir: TemporaryDirectory[str] | None = None
        phase_cache_path: Path | None = None
        phase_cache_handle: object | None = None
        if total_profiles >= int(OPTIMIZER_INTERMEDIATE_CACHE_MIN_PROFILES):
            safe_symbol = "".join(
                character.lower() if character.isalnum() else "_"
                for character in str(self._symbol)
            ).strip("_") or "symbol"
            safe_strategy = "".join(
                character.lower() if character.isalnum() else "_"
                for character in str(strategy_name)
            ).strip("_") or "strategy"
            phase_cache_dir = TemporaryDirectory(prefix="optimizer_phase_cache_")
            phase_cache_path = Path(phase_cache_dir.name) / (
                f"{safe_symbol}_{safe_strategy}_summaries.jsonl"
            )
            phase_cache_handle = phase_cache_path.open("w", encoding="utf-8")
            self.log_message.emit(
                "Intermediate optimizer checkpoint cache active: "
                f"streaming summaries to {phase_cache_path}."
            )
        def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:
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
            return {
                "strategy_name": strategy_name,
                "strategy_profile": profile,
                "force_python_path": bool(force_python_path),
                "symbol": self._symbol,
                "interval": self._interval,
                "leverage_override": self._leverage_override,
                "use_setup_gate": self._use_setup_gate,
                "min_confidence_pct": self._min_confidence_pct,
                "precomputed_cache_key": (
                    cache_key
                    if cached_group is not None
                    else None
                ),
                "precomputed_signals": (
                    None
                    if cached_group is not None
                    else precomputed_signals
                ),
                "precomputed_long_exit_flags": (
                    None
                    if cached_group is not None
                    else None
                ),
                "precomputed_short_exit_flags": (
                    None
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
        try:
            for batch_start in range(0, total_profiles, dispatch_profile_batch_size):
                if self._should_stop():
                    interrupted = True
                    return []

                batch_profiles = list(
                    profiles[batch_start : batch_start + dispatch_profile_batch_size]
                )
                profile_chunks: list[list[OptimizationProfile]] = [
                    batch_profiles[index : index + profile_chunk_size]
                    for index in range(0, len(batch_profiles), profile_chunk_size)
                ]
                tasks: list[dict[str, object]] = []
                for profile_chunk in profile_chunks:
                    tasks.append(
                        {
                            "profile_tasks": [
                                _build_profile_task(profile)
                                for profile in profile_chunk
                            ],
                        }
                    )

                async_result = pool.map_async(_run_optimization_profile_chunk_worker, tasks)
                batch_wait_started_monotonic = time.monotonic()
                next_wait_heartbeat_monotonic = batch_wait_started_monotonic + 8.0
                while True:
                    if self._should_stop():
                        interrupted = True
                        return []
                    try:
                        batch_results = async_result.get(timeout=0.25)
                        break
                    except PoolTimeoutError:
                        now_monotonic = time.monotonic()
                        if now_monotonic >= next_wait_heartbeat_monotonic:
                            batch_elapsed_sec = max(
                                0.0,
                                now_monotonic - batch_wait_started_monotonic,
                            )
                            remaining_profiles = max(0, total_profiles - completed_profiles)
                            self.log_message.emit(
                                f"{phase_label}: running... completed={completed_profiles}/{total_profiles} "
                                f"(remaining={remaining_profiles}) | waiting on current worker batch "
                                f"({len(batch_profiles)} profiles in {len(tasks)} chunk-tasks, "
                                f"chunk_size={profile_chunk_size}, {worker_count} workers, {batch_elapsed_sec:.0f}s elapsed)."
                            )
                            next_wait_heartbeat_monotonic = now_monotonic + 8.0
                        continue

                submitted_profiles_in_batch = 0
                for chunk_result in batch_results:
                    if chunk_result is None:
                        continue
                    chunk_profile_results: list[dict[str, object]]
                    if isinstance(chunk_result, Mapping) and "chunk_results" in chunk_result:
                        raw_chunk_results = chunk_result.get("chunk_results", [])
                        chunk_profile_results = (
                            [dict(item) for item in raw_chunk_results if isinstance(item, Mapping)]
                            if isinstance(raw_chunk_results, Sequence)
                            else []
                        )
                        submitted_profiles_in_batch += int(
                            chunk_result.get(
                                "submitted_profiles",
                                len(chunk_profile_results),
                            )
                            or 0
                        )
                    elif isinstance(chunk_result, Mapping):
                        chunk_profile_results = [dict(chunk_result)]
                        submitted_profiles_in_batch += 1
                    else:
                        chunk_profile_results = []
                    for profile_result in chunk_profile_results:
                        summary = dict(profile_result["summary"])
                        if phase_cache_handle is not None:
                            phase_cache_handle.write(
                                json.dumps(summary, separators=(",", ":")) + "\n"
                            )
                        else:
                            phase_results.append(summary)
                        if (
                            not first_summary_logged
                            and strategy_signal_cache is not None
                            and not self._is_quiet_mode()
                        ):
                            self.log_message.emit(
                                "Backtest summary: "
                                f"Total Signals: {int(profile_result['total_signals'])} | "
                                f"Approved by Gate: {int(profile_result['approved_signals'])} | "
                                f"Blocked: {int(profile_result['blocked_signals'])}"
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

                if submitted_profiles_in_batch <= 0:
                    submitted_profiles_in_batch = len(batch_profiles)
                completed_profiles += submitted_profiles_in_batch
                completed_profiles = min(completed_profiles, total_profiles)
                if phase_cache_handle is not None:
                    phase_cache_handle.flush()
                best_profile_text = _format_profile_dict(best_profile) if best_profile is not None else "{}"
                # Throttle progress logging to configured modulo (or on completion)
                try:
                    modulo = settings.trading.optimizer_progress_print_modulo
                    should_log = completed_profiles >= total_profiles or (
                        modulo > 0 and (completed_profiles % modulo == 0)
                    )
                except Exception:
                    should_log = True
                if should_log and (self._is_debug_mode() or not self._is_quiet_mode() or completed_profiles >= total_profiles):
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
            if phase_cache_handle is not None:
                phase_cache_handle.flush()
                phase_cache_handle.close()
                phase_cache_handle = None
            if phase_cache_path is not None:
                with phase_cache_path.open("r", encoding="utf-8") as cache_reader:
                    phase_results = [
                        json.loads(line)
                        for line in cache_reader
                        if line.strip()
                    ]
                if self._is_debug_mode():
                    self.log_message.emit(
                        "Intermediate optimizer checkpoint cache restored: "
                        f"{len(phase_results)} summaries loaded from disk."
                    )
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
            with suppress(Exception):
                if phase_cache_handle is not None:
                    phase_cache_handle.close()
            with suppress(Exception):
                if phase_cache_dir is not None:
                    phase_cache_dir.cleanup()
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
        _ = worker_count
        variant_profiles = (
            list(variant_profiles)
            if variant_profiles is not None
            else _strategy_cache_profiles(strategy_name)
        )
        if not variant_profiles:
            return {}

        strategy_label = strategy_name.replace("_", " ").title()
        if not self._is_quiet_mode():
            self.log_message.emit(
                f"Precomputing {strategy_label} signal cache for {len(variant_profiles)} strategy variants "
                "on main thread (single-process vectorized pass, indicator-only cache keys)."
            )
        self.progress_update.emit(
            progress_start,
            f"Precomputing {strategy_label} signals... 0/{len(variant_profiles)}",
        )

        setup_gate = (
            SmartSetupGate(min_confidence_pct=self._min_confidence_pct)
            if self._use_setup_gate
            else None
        )
        strategy_signal_cache: dict[object, dict[str, object]] = {}
        completed_variants = 0
        next_log_percent = 5

        for strategy_profile in variant_profiles:
            if self._should_stop():
                return {}
            required_candles = _required_candle_count_for_profile(
                strategy_name,
                strategy_profile,
                use_setup_gate=self._use_setup_gate,
            )
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
                    use_setup_gate=self._use_setup_gate,
                    min_confidence_pct=self._min_confidence_pct,
                    strategy_profile=strategy_profile,
                    regime_mask=regime_mask,
                )
                vectorized_payload = {
                    "signals": signals,
                    "total_signals": total_signals,
                    "approved_signals": approved_signals,
                    "blocked_signals": blocked_signals,
                }
            cache_key = _strategy_cache_key(strategy_name, strategy_profile)
            strategy_signal_cache[cache_key] = _pack_strategy_signal_payload(vectorized_payload)
            completed_variants += 1
            self._emit_signal_cache_progress(
                completed_variants,
                len(variant_profiles),
                strategy_label=strategy_label,
                start_value=progress_start,
                progress_span=progress_span,
            )
            progress_percent = int((completed_variants / len(variant_profiles)) * 100)
            if (
                self._is_debug_mode()
                and (completed_variants == len(variant_profiles) or progress_percent >= next_log_percent)
            ):
                self.log_message.emit(
                    f"{self._SIGNAL_CACHE_PROGRESS_LOG_PREFIX} "
                    f"{strategy_label} signal cache progress: "
                    f"{completed_variants}/{len(variant_profiles)} ({progress_percent}%)"
                )
                while next_log_percent <= progress_percent:
                    next_log_percent += 5

        if self._is_debug_mode():
            self.log_message.emit(
                f"{strategy_label} signal cache prepared: {len(strategy_signal_cache)} strategy variants in RAM "
                "using compact packed payloads."
            )
        return strategy_signal_cache

    def _clear_signal_caches_and_gc(self) -> None:
        self._last_precomputed_exit_cache = None
        self._hmm_regime_mask = None
        self._hmm_regime_labels = None
        gc.collect()

    def _build_backtest_engine(
        self,
        db: Database,
        *,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_activation_pct: float | None = None,
        trailing_distance_pct: float | None = None,
        breakeven_activation_pct: float | None = None,
        breakeven_buffer_pct: float | None = None,
        tight_trailing_activation_pct: float | None = None,
        tight_trailing_distance_pct: float | None = None,
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
            breakeven_activation_pct=breakeven_activation_pct,
            breakeven_buffer_pct=breakeven_buffer_pct,
            tight_trailing_activation_pct=tight_trailing_activation_pct,
            tight_trailing_distance_pct=tight_trailing_distance_pct,
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
        if self._is_debug_mode():
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
                    breakeven_activation_pct=runtime_settings.breakeven_activation_pct,
                    breakeven_buffer_pct=runtime_settings.breakeven_buffer_pct,
                    tight_trailing_activation_pct=runtime_settings.tight_trailing_activation_pct,
                    tight_trailing_distance_pct=runtime_settings.tight_trailing_distance_pct,
                )
            )
        effective_fee_pct = engine.get_effective_taker_fee_pct()
        if self._is_debug_mode():
            self.log_message.emit(
                "Backtest fee stress active: "
                f"taker_fee={effective_fee_pct:.4f}% per side "
                f"(base={float(settings.trading.taker_fee_pct):.4f}%, "
                f"multiplier={BACKTEST_FEE_STRESS_MULTIPLIER:.2f}x)."
            )
            self.log_message.emit(
                "Backtest slippage penalty active: "
                f"slippage={BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE:.4f}% per side "
                f"({BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE:.4f}% round-trip)."
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
            "breakeven_activation_pct": float(runtime_settings.breakeven_activation_pct),
            "breakeven_buffer_pct": float(runtime_settings.breakeven_buffer_pct),
            "tight_trailing_activation_pct": float(runtime_settings.tight_trailing_activation_pct),
            "tight_trailing_distance_pct": float(runtime_settings.tight_trailing_distance_pct),
            "effective_taker_fee_pct": float(effective_fee_pct),
            "slippage_penalty_pct_per_side": float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
            "slippage_penalty_pct_per_trade": float(BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE),
            "estimated_round_trip_cost_pct": float(_resolve_backtest_round_trip_cost_pct()),
        }
        self._emit_trace_event(
            "runtime_settings",
            runtime_trace,
        )
        return runtime_trace

    def _log_backtest_metrics(self, result: dict, *, prefix: str = "Backtest metrics") -> None:
        if self._is_quiet_mode():
            return
        avg_win_to_cost_ratio = float(result.get("average_win_to_cost_ratio", 0.0) or 0.0)
        total_slippage_penalty = float(result.get("total_slippage_penalty_usd", 0.0) or 0.0)
        dynamic_stop_loss_overrides = int(result.get("dynamic_stop_loss_overrides_applied", 0) or 0)
        dynamic_take_profit_overrides = int(result.get("dynamic_take_profit_overrides_applied", 0) or 0)
        time_stop_exits = int(result.get("time_stop_exits", 0) or 0)
        self.log_message.emit(
            f"{prefix}: "
            f"Profit Factor = {self._format_profit_factor(float(result['profit_factor']))} | "
            f"Average Win = {float(result['average_win_usd']):.2f} USDT | "
            f"Average Loss = {float(result['average_loss_usd']):.2f} USDT | "
            f"Real RRR = {self._format_profit_factor(float(result['real_rrr']))} | "
            f"AvgWin/Cost = {self._format_profit_factor(avg_win_to_cost_ratio)} | "
            f"Slippage Penalty = {total_slippage_penalty:.2f} USDT"
        )
        if dynamic_stop_loss_overrides > 0 or dynamic_take_profit_overrides > 0:
            self.log_message.emit(
                "Dynamic SL/TP overrides applied: "
                f"SL={dynamic_stop_loss_overrides} | TP={dynamic_take_profit_overrides}"
            )
        if time_stop_exits > 0:
            self.log_message.emit(f"Time-Stop exits applied: {time_stop_exits}")

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
            if self._is_quiet_mode():
                return
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
            if self._is_quiet_mode():
                return
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
                if key in {
                    "precomputed_long_exit_flags",
                    "precomputed_short_exit_flags",
                    "precomputed_dynamic_stop_loss_pcts",
                    "precomputed_dynamic_take_profit_pcts",
                }
            }
            diagnostics_payload = vectorized_payload.get("strategy_diagnostics")
            if isinstance(diagnostics_payload, dict):
                self._last_strategy_diagnostics = dict(diagnostics_payload)
                if strategy_name == "ema_band_rejection":
                    trend_ok = bool(int(diagnostics_payload.get("latest_trend_ok", 0) or 0))
                    pullback_ok = bool(int(diagnostics_payload.get("latest_pullback_ok", 0) or 0))
                    rejection_ok = bool(int(diagnostics_payload.get("latest_rejection_ok", 0) or 0))
                    latest_signal = int(diagnostics_payload.get("latest_signal_direction", 0) or 0)
                    latest_signal_text = "Long" if latest_signal > 0 else ("Short" if latest_signal < 0 else "None")
                    yes_no = lambda value: "ja" if value else "nein"
                    self.log_message.emit(
                        "EMA Band Rejection status: "
                        f"Trend OK={yes_no(trend_ok)} | "
                        f"Pullback OK={yes_no(pullback_ok)} | "
                        f"Rejection OK={yes_no(rejection_ok)} | "
                        f"Signal={latest_signal_text} | "
                        f"params={{ema:{int(diagnostics_payload.get('ema_fast', 5) or 5)}/"
                        f"{int(diagnostics_payload.get('ema_mid', 10) or 10)}/"
                        f"{int(diagnostics_payload.get('ema_slow', 20) or 20)}, "
                        f"slope_lb:{int(diagnostics_payload.get('slope_lookback', 5) or 5)}, "
                        f"spread%:{float(diagnostics_payload.get('min_ema_spread_pct', 0.05) or 0.05):.3f}, "
                        f"rsi_filter:{int(diagnostics_payload.get('use_rsi_filter', 0) or 0)}, "
                        f"vol_filter:{int(diagnostics_payload.get('use_volume_filter', 0) or 0)}, "
                        f"atr_buffer:{int(diagnostics_payload.get('use_atr_stop_buffer', 0) or 0)}}}"
                    )
            return (
                list(vectorized_payload["signals"]),
                int(vectorized_payload["total_signals"]),
                int(vectorized_payload["approved_signals"]),
                int(vectorized_payload["blocked_signals"]),
            )
        effective_signal_profile = strategy_profile
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
                effective_signal_profile,
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
                normalized_direction = 1 if signal_direction > 0 else -1
                is_approved, _score, _reason = self._setup_gate.evaluate_signal_at_index(
                    candles_df,
                    index,
                    normalized_direction,
                    strategy_name,
                )
                if not is_approved:
                    blocked_signals += 1
                    signal_direction = 0
                else:
                    signal_direction = normalized_direction
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
    ) -> int:
        summary_payload = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "closed_trades",
                "optimization_results",
                "decision_matrix_summaries",
                "equity_curve_events",
            }
        }
        return int(
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
        ))

    def _persist_backtest_result_to_primary_db_with_retry(
        self,
        result: dict[str, object],
        *,
        strategy_name: str,
        stage_label: str,
    ) -> int:
        persist_error: Exception | None = None
        for attempt in range(1, self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS + 1):
            try:
                with Database(self._db_path) as result_db:
                    run_id = self._persist_backtest_result(
                        result_db,
                        result,
                        strategy_name=strategy_name,
                    )
                return int(run_id)
            except Exception as exc:
                persist_error = exc
                if (
                    attempt == 1
                    or attempt % self._PRIMARY_DB_PERSIST_LOG_EVERY_ATTEMPTS == 0
                    or attempt == self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS
                ):
                    self.log_message.emit(
                        "Primary DB persistence retry "
                        f"({stage_label}) {attempt}/{self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS}: "
                        f"{exc}"
                    )
                if attempt < self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS:
                    sleep(self._PRIMARY_DB_PERSIST_RETRY_SECONDS)
        if persist_error is not None:
            raise persist_error
        raise RuntimeError("Primary DB persistence failed without explicit error.")

    def _update_backtest_result_in_primary_db_with_retry(
        self,
        run_id: int,
        result: dict[str, object],
        *,
        strategy_name: str,
        stage_label: str,
    ) -> None:
        persist_error: Exception | None = None
        summary_payload = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "closed_trades",
                "optimization_results",
                "decision_matrix_summaries",
                "equity_curve_events",
            }
        }
        for attempt in range(1, self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS + 1):
            try:
                with Database(self._db_path) as result_db:
                    result_db.update_backtest_run(
                        int(run_id),
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
                return
            except Exception as exc:
                persist_error = exc
                if (
                    attempt == 1
                    or attempt % self._PRIMARY_DB_PERSIST_LOG_EVERY_ATTEMPTS == 0
                    or attempt == self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS
                ):
                    self.log_message.emit(
                        "Primary DB update retry "
                        f"({stage_label}) {attempt}/{self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS}: "
                        f"{exc}"
                    )
                if attempt < self._PRIMARY_DB_PERSIST_MAX_ATTEMPTS:
                    sleep(self._PRIMARY_DB_PERSIST_RETRY_SECONDS)
        if persist_error is not None:
            raise persist_error

    def _prepare_hmm_regime_mask(
        self,
        candles_df: pd.DataFrame,
        *,
        strategy_name: str,
    ) -> list[int] | None:
        self._hmm_regime_labels = None
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
            transmat_warning_events = [
                event
                for event in tuple(getattr(detection, "warning_events", tuple()) or tuple())
                if isinstance(event, Mapping)
                and str(event.get("warning_kind", "")).strip().lower()
                == "transmat_zero_sum_no_transition"
            ]
            if transmat_warning_events:
                warning_detail = _format_hmm_warning_detail(
                    detection,
                    warning_kind="transmat_zero_sum_no_transition",
                )
                self.log_message.emit(
                    "HMM regime filter structural warning. "
                    f"symbol={self._symbol} interval={self._interval} strategy={strategy_name} "
                    f"{warning_detail}."
                )
            if not bool(getattr(detection, "converged", True)):
                non_convergence_detail = _format_hmm_non_convergence_detail(detection)
                self.log_message.emit(
                    "HMM regime filter non-converged. "
                    f"symbol={self._symbol} interval={self._interval} strategy={strategy_name} "
                    f"(non_converged_windows={int(getattr(detection, 'non_converged_windows', 0) or 0)}). "
                    f"{non_convergence_detail}. Falling back to unfiltered signals."
                )
                return None
            regime_mask = detection.regime_mask.astype("int8").tolist()
            if len(detection.regime_labels) == len(candles_df):
                self._hmm_regime_labels = list(detection.regime_labels)
            allowed_candles = int(sum(regime_mask))
            total_candles = max(len(regime_mask), 1)
            if not self._is_quiet_mode():
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
                f"HMM regime detector failed ({exc}). Falling back to unfiltered signals."
            )
            return None

    def _reset_history_download_progress(self) -> None:
        self._history_progress_last_emit_monotonic = 0.0
        self._history_progress_last_percent = -1
        self._history_progress_last_stage = ""

    def _emit_history_download_progress(self, progress_value: int, text: str) -> None:
        normalized_progress = max(0, min(100, progress_value))
        scaled_value = min(50, 10 + int(normalized_progress * 0.4))
        self.progress_update.emit(scaled_value, text)
        compact_text = str(text).strip()
        if not compact_text:
            return
        display_symbol = self._symbol
        display_interval = self._interval
        progress_head = compact_text.split("...", 1)[0].strip()
        progress_tokens = progress_head.split()
        if len(progress_tokens) >= 2:
            candidate_symbol = progress_tokens[-2].strip().upper()
            candidate_interval = progress_tokens[-1].strip()
            if candidate_interval in settings.api.timeframes:
                display_symbol = candidate_symbol or display_symbol
                display_interval = candidate_interval
        stage_key = compact_text.split("...", 1)[0].strip()
        now_monotonic = time.monotonic()
        should_emit = (
            self._history_progress_last_percent < 0
            or normalized_progress >= 100
            or (normalized_progress - self._history_progress_last_percent) >= 2
            or stage_key != self._history_progress_last_stage
            or (now_monotonic - self._history_progress_last_emit_monotonic) >= 1.5
        )
        if not should_emit:
            return
        if self._is_quiet_mode() and normalized_progress < 100:
            self._history_progress_last_percent = normalized_progress
            self._history_progress_last_stage = stage_key
            self._history_progress_last_emit_monotonic = now_monotonic
            return
        self._history_progress_last_percent = normalized_progress
        self._history_progress_last_stage = stage_key
        self._history_progress_last_emit_monotonic = now_monotonic
        self.log_message.emit(
            f"{self._HISTORY_PROGRESS_LOG_PREFIX} "
            f"History sync {display_symbol} {display_interval}: "
            f"{normalized_progress}% | {compact_text}"
        )

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
