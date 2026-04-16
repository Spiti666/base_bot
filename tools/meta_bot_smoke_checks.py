from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.data.db import Database, PaperTradeCreate, PaperTradeUpdate
from gui import TradingTerminalWindow
from main_engine import BotEngineThread
from PyQt6.QtWidgets import QApplication


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _cleanup_db_files(db_path: Path) -> None:
    for suffix in ("", ".wal", ".shm"):
        candidate = db_path.with_suffix(db_path.suffix + suffix) if suffix else db_path
        if candidate.exists():
            candidate.unlink()


def _run_db_and_meta_smoke(db_path: Path) -> dict[str, Any]:
    now = datetime.now(tz=UTC).replace(tzinfo=None)
    entry_time = now - timedelta(hours=2)
    exit_time = now - timedelta(hours=1)
    trade_id: int

    with Database(db_path) as db:
        trade_id = db.insert_trade(
            PaperTradeCreate(
                symbol="BTCUSDT",
                side="LONG",
                entry_time=entry_time,
                entry_price=100.0,
                qty=2.0,
                leverage=5,
                strategy_name="frama_cross",
                timeframe="1m",
                regime_label_at_entry="range_balanced",
                regime_confidence=0.61,
                session_label="europe",
                signal_strength=0.64,
                confidence_score=0.67,
                atr_pct_at_entry=0.8,
                volume_ratio_at_entry=1.1,
                spread_estimate=0.02,
                move_already_extended_pct=0.1,
                entry_snapshot_json={
                    "symbol": "BTCUSDT",
                    "strategy_name": "frama_cross",
                    "timeframe": "1m",
                    "feature_snapshot_version": "smoke_v1",
                },
                profile_version="smoke_profile_v1",
                review_status="PENDING",
            )
        )

        before_reopen_trade = db.fetch_trade_by_id(trade_id)
        _assert(before_reopen_trade is not None, "Trade insert failed during migration smoke.")
        _assert(
            bool(before_reopen_trade.entry_snapshot_json),
            "Entry snapshot JSON was not persisted on trade open.",
        )
        db.update_trade(
            trade_id,
            PaperTradeUpdate(
                status="TP_HIT",
                exit_time=exit_time,
                exit_price=80.0,
                pnl=-1_000_000.0,
                total_fees=1.2,
            ),
        )

    # Re-open DB to verify migration compatibility and data persistence.
    with Database(db_path) as db:
        reopened_trade = db.fetch_trade_by_id(trade_id)
        _assert(reopened_trade is not None, "Trade vanished after DB reopen/migration.")
        _assert(
            reopened_trade.status == "TP_HIT",
            "Closed status was not persisted after migration reopen.",
        )

        required_tables = {
            "paper_trades",
            "trade_reviews",
            "strategy_health",
            "regime_observations",
            "adaptation_log",
        }
        existing_tables = {
            str(row[0]).strip()
            for row in db._connection.execute("SHOW TABLES").fetchall()  # noqa: SLF001
        }
        missing_tables = required_tables.difference(existing_tables)
        _assert(not missing_tables, f"Missing expected tables: {sorted(missing_tables)}")

        paper_cols = {
            str(row[1]).strip()
            for row in db._connection.execute("PRAGMA table_info('paper_trades')").fetchall()  # noqa: SLF001
        }
        for required_col in (
            "timeframe",
            "regime_label_at_entry",
            "entry_snapshot_json",
            "lifecycle_snapshot_json",
            "profile_version",
            "review_status",
        ):
            _assert(required_col in paper_cols, f"Missing required paper_trades column: {required_col}")

        bot = BotEngineThread(
            symbols=["BTCUSDT"],
            strategy_name="frama_cross",
            intervals=["1m"],
            symbol_intervals={"BTCUSDT": "1m"},
            db_path=db_path,
        )
        bot._db = db  # noqa: SLF001
        review_payload = bot.review_closed_trade(trade_id)
        _assert(review_payload is not None, "Trade close did not trigger a review payload.")

        review_row = db.fetch_trade_review_by_trade_id(trade_id)
        _assert(review_row is not None, "Trade review row missing after review_closed_trade().")

        health_snapshot = bot.recompute_strategy_health(
            symbol="BTCUSDT",
            strategy_name="frama_cross",
            timeframe="1m",
        )
        _assert(health_snapshot is not None, "Health recompute returned no snapshot.")

        meta_policy = bot.evaluate_meta_policy(
            symbol="BTCUSDT",
            strategy_name="frama_cross",
            interval="1m",
            current_context={
                "regime_payload": {
                    "regime_label": "trend_clean_down",
                    "regime_confidence": 0.77,
                }
            },
        )
        for key in (
            "allow_trade",
            "risk_multiplier",
            "state",
            "block_reason",
            "warning_reason",
            "meta_flags",
            "effective_policy_json",
        ):
            _assert(key in meta_policy, f"Meta policy key missing: {key}")
        _assert(
            not bool(meta_policy.get("allow_trade", True)),
            "Global guard smoke expected allow_trade=False due daily loss guard.",
        )

        daily_report = bot.build_daily_meta_report(report_date=now)
        weekly_report = bot.build_weekly_meta_report(end_time=now)
        for report_name, report_payload in (("daily", daily_report), ("weekly", weekly_report)):
            files = report_payload.get("files", {})
            _assert(isinstance(files, dict), f"{report_name} report files payload missing.")
            markdown_path = Path(str(files.get("markdown", "")))
            json_path = Path(str(files.get("json", "")))
            _assert(markdown_path.exists(), f"{report_name} markdown report file missing.")
            _assert(json_path.exists(), f"{report_name} json report file missing.")

        return {
            "trade_id": trade_id,
            "review_id": int(review_row.id),
            "health_state": str(health_snapshot.state),
            "meta_state": str(meta_policy.get("state", "")),
            "meta_block_reason": str(meta_policy.get("block_reason", "")),
        }


def _run_gui_refresh_smoke(db_path: Path, *, expect_populated: bool) -> dict[str, int]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    window = TradingTerminalWindow()
    try:
        window._db_path = db_path  # noqa: SLF001
        window._refresh_meta_views(force=True)  # noqa: SLF001
        health_rows = int(window.meta_health_table.rowCount())
        review_rows = int(window.meta_reviews_table.rowCount())
        adaptation_rows = int(window.meta_adaptation_table.rowCount())
        _assert(health_rows >= 1, "GUI smoke: health table did not render any row (incl. placeholder).")
        _assert(review_rows >= 1, "GUI smoke: reviews table did not render any row (incl. placeholder).")
        _assert(adaptation_rows >= 1, "GUI smoke: adaptation table did not render any row (incl. placeholder).")
        health_first_cell = window.meta_health_table.item(0, 0)
        reviews_first_cell = window.meta_reviews_table.item(0, 0)
        health_first_text = "" if health_first_cell is None else str(health_first_cell.text())
        reviews_first_text = "" if reviews_first_cell is None else str(reviews_first_cell.text())
        if expect_populated:
            _assert(
                "BTCUSDT" in "\n".join(
                    str(window.meta_health_table.item(row, 0).text())
                    for row in range(window.meta_health_table.rowCount())
                    if window.meta_health_table.item(row, 0) is not None
                ),
                "GUI smoke (filled): expected BTCUSDT health row.",
            )
            _assert(
                not reviews_first_text.startswith("No "),
                "GUI smoke (filled): review table still shows empty placeholder.",
            )
        else:
            _assert(
                health_first_text.startswith("No "),
                "GUI smoke (empty): health table placeholder missing.",
            )
            _assert(
                reviews_first_text.startswith("No "),
                "GUI smoke (empty): reviews table placeholder missing.",
            )
        return {
            "health_rows": health_rows,
            "review_rows": review_rows,
            "adaptation_rows": adaptation_rows,
        }
    finally:
        window.close()
        if created_app and app is not None:
            app.quit()


def main() -> int:
    db_path = Path("data/meta_smoke_checks.duckdb").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_db_files(db_path)
    with Database(db_path):
        pass
    gui_empty_result = _run_gui_refresh_smoke(db_path, expect_populated=False)
    core_result = _run_db_and_meta_smoke(db_path)
    gui_result = _run_gui_refresh_smoke(db_path, expect_populated=True)

    print("META SMOKE CHECKS: OK")
    print(f"- DB Path: {db_path}")
    print(f"- Trade ID: {core_result['trade_id']}")
    print(f"- Review ID: {core_result['review_id']}")
    print(f"- Health State: {core_result['health_state']}")
    print(f"- Meta State: {core_result['meta_state']}")
    print(f"- Meta Block Reason: {core_result['meta_block_reason']}")
    print(
        "- GUI Rows: "
        f"health={gui_result['health_rows']} "
        f"reviews={gui_result['review_rows']} "
        f"adaptation={gui_result['adaptation_rows']}"
    )
    print(
        "- GUI Empty Rows: "
        f"health={gui_empty_result['health_rows']} "
        f"reviews={gui_empty_result['review_rows']} "
        f"adaptation={gui_empty_result['adaptation_rows']}"
    )
    if os.environ.get("KEEP_META_SMOKE_DB", "").strip() != "1":
        _cleanup_db_files(db_path)
    else:
        print(f"- Kept DB file for inspection: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
