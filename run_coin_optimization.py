from __future__ import annotations

import argparse
import json
import time
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

from config import settings
from core.engine.backtest_engine import BacktestThread
from main_engine import resolve_optimizer_strategy_for_symbol


def run_coin_optimization(coin: str) -> dict[str, Any]:
    strategy = resolve_optimizer_strategy_for_symbol(
        coin,
        apply_backtest_optimizer_overrides=True,
    )
    result_holder: dict[str, Any] = {"result": None, "error": None}

    thread = BacktestThread(
        symbol=coin,
        interval=settings.live.default_interval,
        strategy_name=strategy,
        optimize_profile=True,
        isolated_db=True,
        db_path="data/paper_trading.duckdb",
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

    result_payload = result_holder.get("result")
    output: dict[str, Any] = {
        "coin": coin,
        "strategy": strategy,
        "elapsed_sec": round(elapsed, 2),
        "error": result_holder.get("error"),
    }

    if isinstance(result_payload, dict):
        output["result"] = {
            "profit_factor": result_payload.get("profit_factor"),
            "total_pnl_usd": result_payload.get("total_pnl_usd"),
            "win_rate_pct": result_payload.get("win_rate_pct"),
            "total_trades": result_payload.get("total_trades"),
            "max_drawdown_pct": result_payload.get("max_drawdown_pct"),
            "best_profile": result_payload.get("best_profile"),
        }
    else:
        output["result"] = None

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one optimized backtest safely from a real .py file (Windows spawn-safe)."
    )
    parser.add_argument(
        "--coin",
        default="ETHUSDT",
        help="Symbol to optimize (default: ETHUSDT).",
    )
    parser.add_argument(
        "--out",
        default="data/latest_single_coin_snapshot.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    snapshot = run_coin_optimization(args.coin.upper())
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    freeze_support()
    main()
