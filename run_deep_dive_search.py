from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

from config import BACKTEST_BATCH_SYMBOLS, settings
from core.engine.backtest_engine import BacktestThread
from main_engine import MAX_OPTIMIZATION_GRID_PROFILES

DEEP_DIVE_SYMBOLS: tuple[str, ...] = tuple(
    symbol
    for symbol in (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "1000PEPEUSDT",
        "1000SHIBUSDT",
        "BNBUSDT",
        "AVAXUSDT",
        "NEARUSDT",
        "DOTUSDT",
    )
    if symbol in BACKTEST_BATCH_SYMBOLS
)
DEEP_DIVE_STRATEGIES: tuple[str, ...] = (
    "dual_thrust",
    "frama_cross",
    "ema_cross_volume",
)
DEEP_DIVE_INTERVALS: tuple[str, ...] = ("15m", "1h")
MEGA_WINNER_PROFIT_FACTOR = 2.0


def _float_range_tuple(start: float, stop: float, step: float) -> tuple[float, ...]:
    values: list[float] = []
    index = 0
    current = float(start)
    epsilon = abs(step) * 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        index += 1
        current = start + step * index
    return tuple(values)


def _entry_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(entry.get("profit_factor", 0.0) or 0.0),
        -float(entry.get("max_drawdown_pct", 0.0) or 0.0),
        float(entry.get("total_pnl_usd", 0.0) or 0.0),
        float(entry.get("win_rate_pct", 0.0) or 0.0),
    )


def _apply_deep_dive_runtime_overrides(*, sample_profiles: int) -> None:
    object.__setattr__(settings.trading, "default_leverage", 25)
    # Effective fee stress remains 0.0625% via engine multiplier 1.25x on this base fee.
    object.__setattr__(settings.trading, "taker_fee_pct", 0.05)
    object.__setattr__(settings.trading, "use_hmm_regime_filter", True)
    object.__setattr__(
        settings.trading,
        "hmm_allowed_regimes",
        ("Bull Trend", "Bear Trend", "High-Vol Range", "Low-Vol Range"),
    )
    # Force full-depth search inside engine profile cap.
    object.__setattr__(
        settings.trading,
        "optimization_max_sample_profiles",
        max(10_000, int(sample_profiles)),
    )
    object.__setattr__(
        settings.trading,
        "optimization_validation_top_n",
        max(5, int(settings.trading.optimization_validation_top_n)),
    )

    object.__setattr__(
        settings.strategy,
        "dual_thrust_opt_periods",
        (13, 21, 34, 55, 89, 144, 233, 377, 500, 610),
    )
    object.__setattr__(
        settings.strategy,
        "dual_thrust_opt_k_values",
        _float_range_tuple(0.1, 1.6, 0.05),
    )
    object.__setattr__(
        settings.strategy,
        "dual_thrust_stop_loss_pct_options",
        _float_range_tuple(1.0, 8.0, 0.5),
    )
    object.__setattr__(
        settings.strategy,
        "dual_thrust_take_profit_pct_options",
        _float_range_tuple(2.0, 30.0, 1.0),
    )
    object.__setattr__(
        settings.strategy,
        "dual_thrust_symbol_optimization_grids",
        {},
    )

    object.__setattr__(
        settings.strategy,
        "frama_fast_options",
        (4, 6, 8, 10, 12, 14, 16, 20, 24),
    )
    object.__setattr__(
        settings.strategy,
        "frama_slow_options",
        (30, 40, 50, 60, 80, 100, 120, 150, 200),
    )
    object.__setattr__(
        settings.strategy,
        "ema_fast_options",
        (5, 8, 9, 12, 13, 21, 34, 55),
    )
    object.__setattr__(
        settings.strategy,
        "ema_slow_options",
        (50, 89, 100, 144, 200, 233, 300, 500, 800),
    )
    object.__setattr__(
        settings.strategy,
        "volume_multiplier_options",
        (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
    )


def _run_single_optimization(
    *,
    symbol: str,
    strategy_name: str,
    interval: str,
    db_path: str,
) -> dict[str, Any]:
    result_holder: dict[str, Any] = {"result": None, "error": None}
    thread = BacktestThread(
        symbol=symbol,
        interval=interval,
        strategy_name=strategy_name,
        leverage=25,
        optimize_profile=True,
        isolated_db=True,
        db_path=db_path,
    )
    thread.backtest_finished.connect(
        lambda payload: result_holder.__setitem__("result", payload)
    )
    thread.backtest_error.connect(
        lambda err: result_holder.__setitem__("error", err)
    )

    started = time.time()
    thread.run()
    elapsed = time.time() - started

    payload = result_holder.get("result")
    run_output: dict[str, Any] = {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "interval": interval,
        "elapsed_sec": round(float(elapsed), 2),
        "error": result_holder.get("error"),
        "optimization_results_top5": [],
        "best_profile": None,
        "summary": {},
    }
    if not isinstance(payload, dict):
        return run_output

    optimization_results = payload.get("optimization_results", [])
    if isinstance(optimization_results, list):
        run_output["optimization_results_top5"] = [
            dict(row) for row in optimization_results[:5] if isinstance(row, dict)
        ]

    run_output["best_profile"] = payload.get("best_profile")
    run_output["summary"] = {
        "profit_factor": payload.get("profit_factor"),
        "total_pnl_usd": payload.get("total_pnl_usd"),
        "win_rate_pct": payload.get("win_rate_pct"),
        "total_trades": payload.get("total_trades"),
        "max_drawdown_pct": payload.get("max_drawdown_pct"),
        "effective_leverage": payload.get("effective_leverage"),
        "evaluated_profiles": payload.get("evaluated_profiles"),
        "sampled_profiles": payload.get("sampled_profiles"),
        "theoretical_profiles": payload.get("theoretical_profiles"),
        "sampling_mode": payload.get("sampling_mode"),
        "sampling_coverage_pct": payload.get("sampling_coverage_pct"),
    }
    return run_output


def _collect_top5_by_combo(
    run_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in run_rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        strategy_name = str(row.get("strategy_name", "")).strip()
        interval = str(row.get("interval", "")).strip()
        if not symbol or not strategy_name:
            continue

        strategy_bucket = grouped.setdefault(symbol, {}).setdefault(
            strategy_name,
            {
                "top5_combined": [],
                "top5_by_interval": {},
                "errors": [],
            },
        )
        if row.get("error"):
            strategy_bucket["errors"].append(
                {
                    "interval": interval,
                    "error": str(row["error"]),
                }
            )

        top_rows = row.get("optimization_results_top5", [])
        if not isinstance(top_rows, list):
            top_rows = []
        interval_rows = []
        for profile in top_rows:
            if not isinstance(profile, dict):
                continue
            enriched = dict(profile)
            enriched["interval"] = interval
            interval_rows.append(enriched)
            strategy_bucket["top5_combined"].append(enriched)
        strategy_bucket["top5_by_interval"][interval] = sorted(
            interval_rows,
            key=_entry_sort_key,
            reverse=True,
        )[:5]

    for symbol_bucket in grouped.values():
        for strategy_bucket in symbol_bucket.values():
            combined = strategy_bucket.get("top5_combined", [])
            strategy_bucket["top5_combined"] = sorted(
                [row for row in combined if isinstance(row, dict)],
                key=_entry_sort_key,
                reverse=True,
            )[:5]
    return grouped


def _collect_mega_winners(top5_by_combo: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    mega_rows: list[dict[str, Any]] = []
    for symbol, strategy_map in top5_by_combo.items():
        for strategy_name, bucket in strategy_map.items():
            combined_rows = bucket.get("top5_combined", [])
            if not isinstance(combined_rows, list):
                continue
            for profile in combined_rows:
                if not isinstance(profile, dict):
                    continue
                profit_factor = float(profile.get("profit_factor", 0.0) or 0.0)
                if profit_factor <= MEGA_WINNER_PROFIT_FACTOR:
                    continue
                mega_rows.append(
                    {
                        "symbol": symbol,
                        "strategy_name": strategy_name,
                        "interval": str(profile.get("interval", "")),
                        "profit_factor": profit_factor,
                        "max_drawdown_pct": float(profile.get("max_drawdown_pct", 0.0) or 0.0),
                        "total_pnl_usd": float(profile.get("total_pnl_usd", 0.0) or 0.0),
                        "win_rate_pct": float(profile.get("win_rate_pct", 0.0) or 0.0),
                        "total_trades": float(profile.get("total_trades", 0.0) or 0.0),
                        "profile": {
                            key: value
                            for key, value in profile.items()
                            if key
                            not in {
                                "profit_factor",
                                "max_drawdown_pct",
                                "total_pnl_usd",
                                "win_rate_pct",
                                "total_trades",
                                "interval",
                            }
                        },
                    }
                )
    return sorted(mega_rows, key=_entry_sort_key, reverse=True)


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the global deep-dive optimizer sweep and export final_deep_search_results.json."
    )
    parser.add_argument(
        "--out",
        default="final_deep_search_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--db-path",
        default="data/paper_trading.duckdb",
        help="Base DB path used for isolated optimizer workers.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional validation limit for runs (0 = all).",
    )
    parser.add_argument(
        "--sample-profiles",
        type=int,
        default=int(MAX_OPTIMIZATION_GRID_PROFILES),
        help="Per-strategy sampled profile budget (minimum is forced to 10,000).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only validate setup and write a plan payload without running optimizations.",
    )
    args = parser.parse_args()

    _apply_deep_dive_runtime_overrides(sample_profiles=int(args.sample_profiles))

    symbols = list(DEEP_DIVE_SYMBOLS)
    run_plan = [
        (symbol, strategy_name, interval)
        for symbol in symbols
        for strategy_name in DEEP_DIVE_STRATEGIES
        for interval in DEEP_DIVE_INTERVALS
    ]
    if args.max_runs > 0:
        run_plan = run_plan[: int(args.max_runs)]

    if args.plan_only:
        payload = {
            "meta": {
                "status": "planned",
                "planned_at_utc": datetime.now(UTC).isoformat(),
                "planned_runs": len(run_plan),
                "effective_fee_pct_per_side": float(settings.trading.taker_fee_pct) * 1.25,
                "configured_leverage": int(settings.trading.default_leverage),
                "sample_profiles": int(settings.trading.optimization_max_sample_profiles),
            },
            "plan": [
                {
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "interval": interval,
                }
                for symbol, strategy_name, interval in run_plan
            ],
        }
        _write_output(Path(args.out), payload)
        print(
            "[DeepDive] Plan validated and written: "
            f"{Path(args.out).resolve()} | runs={len(run_plan)}"
        )
        return

    started_at = datetime.now(UTC)
    run_rows: list[dict[str, Any]] = []

    print(
        "[DeepDive] Start: "
        f"{started_at.isoformat()} | runs={len(run_plan)} | "
        f"symbols={len(symbols)} | strategies={len(DEEP_DIVE_STRATEGIES)} | intervals={DEEP_DIVE_INTERVALS}"
    )

    for index, (symbol, strategy_name, interval) in enumerate(run_plan, start=1):
        print(
            f"[DeepDive] ({index}/{len(run_plan)}) "
            f"{symbol} | {strategy_name} | {interval} ..."
        )
        row = _run_single_optimization(
            symbol=symbol,
            strategy_name=strategy_name,
            interval=interval,
            db_path=str(args.db_path),
        )
        run_rows.append(row)
        if row.get("error"):
            print(f"[DeepDive] ERROR {symbol} {strategy_name} {interval}: {row['error']}")
        else:
            summary = row.get("summary", {})
            print(
                "[DeepDive] DONE "
                f"pf={float(summary.get('profit_factor', 0.0) or 0.0):.2f} "
                f"dd={float(summary.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
                f"trades={int(float(summary.get('total_trades', 0.0) or 0.0))} "
                f"elapsed={float(row.get('elapsed_sec', 0.0) or 0.0):.2f}s"
            )

        top5_by_combo = _collect_top5_by_combo(run_rows)
        mega_winners = _collect_mega_winners(top5_by_combo)
        checkpoint_payload = {
            "meta": {
                "status": "running",
                "started_at_utc": started_at.isoformat(),
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "planned_runs": len(run_plan),
                "completed_runs": len(run_rows),
                "effective_fee_pct_per_side": float(settings.trading.taker_fee_pct) * 1.25,
                "configured_leverage": int(settings.trading.default_leverage),
            },
            "runs": run_rows,
            "top5_by_coin_strategy": top5_by_combo,
            "mega_winners_pf_gt_2_0": mega_winners,
        }
        _write_output(Path(args.out), checkpoint_payload)

    finished_at = datetime.now(UTC)
    total_elapsed = (finished_at - started_at).total_seconds()
    final_top5 = _collect_top5_by_combo(run_rows)
    final_mega_winners = _collect_mega_winners(final_top5)

    final_payload = {
        "meta": {
            "status": "completed",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_sec": round(total_elapsed, 2),
            "planned_runs": len(run_plan),
            "completed_runs": len(run_rows),
            "effective_fee_pct_per_side": float(settings.trading.taker_fee_pct) * 1.25,
            "configured_leverage": int(settings.trading.default_leverage),
        },
        "runs": run_rows,
        "top5_by_coin_strategy": final_top5,
        "mega_winners_pf_gt_2_0": final_mega_winners,
    }
    _write_output(Path(args.out), final_payload)
    print(
        "[DeepDive] Complete: "
        f"{Path(args.out).resolve()} | mega_winners={len(final_mega_winners)}"
    )


if __name__ == "__main__":
    freeze_support()
    main()
