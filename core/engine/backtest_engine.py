from __future__ import annotations

import math
from typing import Any, Mapping

# Canonical import path for backtest runtime thread.
# Implementation remains in main_engine to preserve backward compatibility.
from main_engine import BacktestThread

ROBUST_PROFIT_FACTOR_CAP = 100.0
DEPLOY_MIN_WIN_RATE_PCT = 55.0
DEPLOY_MIN_TRADE_COUNT = 25
DEPLOY_MIN_ROBUST_PF = 1.2


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ratio_display_text(value: float) -> str:
    if math.isinf(value):
        return "MAX"
    if math.isnan(value):
        return "0.00"
    return f"{value:.2f}"


def _normalize_session_leaderboard(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, list):
        return []
    normalized_rows: list[dict[str, Any]] = []
    for row in raw_value:
        if not isinstance(row, Mapping):
            continue
        session_label = str(row.get("session", "") or "").strip().lower()
        if not session_label:
            continue
        normalized_rows.append(
            {
                "session": session_label,
                "avg_pf": _as_float(row.get("avg_pf"), 0.0),
                "avg_win_rate_pct": _as_float(row.get("avg_win_rate_pct"), 0.0),
                "avg_pnl_usd": _as_float(row.get("avg_pnl_usd"), 0.0),
                "runs": _as_int(row.get("runs"), 0),
                "top_runs": _as_int(row.get("top_runs"), 0),
            }
        )
    return normalized_rows


def generate_compact_summary(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        result = {}

    closed_trades = result.get("closed_trades")
    if isinstance(closed_trades, list):
        closed_trade_rows = list(closed_trades)
    else:
        closed_trade_rows = []

    net_pnl_usd = _as_float(result.get("total_pnl_usd"), 0.0)
    if not math.isfinite(net_pnl_usd) and closed_trade_rows:
        net_pnl_usd = sum(_as_float(trade.get("pnl"), 0.0) for trade in closed_trade_rows)
    elif abs(net_pnl_usd) <= 0.0 and closed_trade_rows:
        net_pnl_usd = sum(_as_float(trade.get("pnl"), 0.0) for trade in closed_trade_rows)

    trade_count = _as_int(result.get("total_trades"), 0)
    if trade_count <= 0 and closed_trade_rows:
        trade_count = len(closed_trade_rows)

    win_rate_pct = _as_float(result.get("win_rate_pct"), 0.0)
    average_win_usd = _as_float(result.get("average_win_usd"), 0.0)
    max_drawdown_pct = _as_float(result.get("max_drawdown_pct"), 0.0)

    raw_profit_factor = _as_float(result.get("profit_factor"), 0.0)
    if math.isinf(raw_profit_factor):
        robust_profit_factor = float(ROBUST_PROFIT_FACTOR_CAP)
        robust_profit_factor_display = "MAX"
    elif math.isnan(raw_profit_factor):
        robust_profit_factor = 0.0
        robust_profit_factor_display = "0.00"
    else:
        robust_profit_factor = max(0.0, min(raw_profit_factor, float(ROBUST_PROFIT_FACTOR_CAP)))
        robust_profit_factor_display = f"{robust_profit_factor:.2f}"

    average_trade_cost_usd = _as_float(result.get("average_trade_cost_usd"), 0.0)
    if average_trade_cost_usd <= 0.0:
        fee_part = _as_float(result.get("average_trade_fees_usd"), 0.0)
        slippage_part = _as_float(result.get("average_trade_slippage_usd"), 0.0)
        average_trade_cost_usd = max(0.0, fee_part + slippage_part)
    if average_trade_cost_usd <= 0.0 and trade_count > 0 and closed_trade_rows:
        total_cost_usd = sum(
            _as_float(trade.get("total_fees"), 0.0)
            + _as_float(trade.get("slippage_penalty_usd"), 0.0)
            for trade in closed_trade_rows
        )
        average_trade_cost_usd = max(0.0, total_cost_usd / float(max(trade_count, 1)))

    avg_win_fee_ratio = _as_float(result.get("average_win_to_cost_ratio"), 0.0)
    if avg_win_fee_ratio <= 0.0:
        if average_trade_cost_usd > 0.0:
            avg_win_fee_ratio = max(0.0, average_win_usd / average_trade_cost_usd)
        else:
            avg_win_fee_ratio = float("inf") if average_win_usd > 0.0 else 0.0

    deploy_eligible = (
        (win_rate_pct >= DEPLOY_MIN_WIN_RATE_PCT)
        and (trade_count >= DEPLOY_MIN_TRADE_COUNT)
        and (net_pnl_usd > 0.0)
        and (robust_profit_factor >= DEPLOY_MIN_ROBUST_PF)
    )
    optimization_mode = bool(result.get("optimization_mode"))
    evaluated_profiles = _as_int(result.get("evaluated_profiles"), 0)
    sampled_profiles = _as_int(result.get("sampled_profiles"), 0)
    validated_profiles = _as_int(result.get("validated_profiles"), 0)
    theoretical_profiles = _as_int(result.get("theoretical_profiles"), 0)
    sampling_coverage_pct = _as_float(result.get("sampling_coverage_pct"), 0.0)
    sampling_mode = str(result.get("sampling_mode", "") or "").strip().lower()
    search_window_candles = _as_int(result.get("search_window_candles"), 0)
    full_history_optimization = bool(result.get("full_history_optimization"))
    optimizer_worker_processes = _as_int(result.get("optimizer_worker_processes"), 0)
    optimizer_max_sample_profiles = _as_int(result.get("optimizer_max_sample_profiles"), 0)
    optimizer_force_full_scan = bool(result.get("optimizer_force_full_scan"))
    sampling_random_seed = _as_int(result.get("sampling_random_seed"), 0)
    session_leaderboard = _normalize_session_leaderboard(
        result.get("session_leaderboard")
        if result.get("session_leaderboard") is not None
        else result.get("session_day_breakdown")
    )
    session_top_session = ""
    session_top_avg_pf = 0.0
    session_top_avg_win_rate_pct = 0.0
    session_top_runs = 0
    if session_leaderboard:
        top_row = dict(session_leaderboard[0])
        session_top_session = str(top_row.get("session", "") or "").strip().lower()
        session_top_avg_pf = _as_float(top_row.get("avg_pf"), 0.0)
        session_top_avg_win_rate_pct = _as_float(top_row.get("avg_win_rate_pct"), 0.0)
        session_top_runs = _as_int(top_row.get("runs"), 0)

    return {
        "symbol": str(result.get("symbol", "") or "").strip().upper(),
        "strategy_name": str(result.get("strategy_name", "") or "").strip(),
        "interval": str(result.get("interval", "") or "").strip(),
        "net_pnl_usd": float(net_pnl_usd),
        "win_rate_pct": float(win_rate_pct),
        "robust_profit_factor": float(robust_profit_factor),
        "robust_profit_factor_display": robust_profit_factor_display,
        "trade_count": int(max(trade_count, 0)),
        "avg_win_usd": float(average_win_usd),
        "avg_win_fee_ratio": float(avg_win_fee_ratio),
        "avg_win_fee_ratio_display": _ratio_display_text(avg_win_fee_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "average_trade_cost_usd": float(average_trade_cost_usd),
        "slippage_penalty_pct_per_trade": _as_float(
            result.get("slippage_penalty_pct_per_trade"),
            0.1,
        ),
        "deploy_eligible": bool(deploy_eligible),
        "optimization_mode": optimization_mode,
        "evaluated_profiles": int(max(evaluated_profiles, 0)),
        "sampled_profiles": int(max(sampled_profiles, 0)),
        "validated_profiles": int(max(validated_profiles, 0)),
        "theoretical_profiles": int(max(theoretical_profiles, 0)),
        "sampling_coverage_pct": float(max(sampling_coverage_pct, 0.0)),
        "sampling_mode": sampling_mode or "n/a",
        "sampling_random_seed": int(max(sampling_random_seed, 0)),
        "search_window_candles": int(max(search_window_candles, 0)),
        "full_history_optimization": bool(full_history_optimization),
        "optimizer_worker_processes": int(max(optimizer_worker_processes, 0)),
        "optimizer_max_sample_profiles": int(max(optimizer_max_sample_profiles, 0)),
        "optimizer_force_full_scan": bool(optimizer_force_full_scan),
        "session_leaderboard": [dict(row) for row in session_leaderboard],
        "session_top_session": session_top_session,
        "session_top_avg_pf": float(session_top_avg_pf),
        "session_top_avg_win_rate_pct": float(session_top_avg_win_rate_pct),
        "session_top_runs": int(max(session_top_runs, 0)),
    }


__all__ = ("BacktestThread", "generate_compact_summary")
