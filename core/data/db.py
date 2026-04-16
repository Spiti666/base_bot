from __future__ import annotations

import json
import math
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb


@dataclass(frozen=True, slots=True)
class CandleRecord:
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class PaperTradeCreate:
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    qty: float
    leverage: int
    status: str = "OPEN"
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None
    total_fees: float = 0.0
    high_water_mark: float | None = None
    strategy_name: str | None = None
    timeframe: str | None = None
    regime_label_at_entry: str | None = None
    regime_confidence: float | None = None
    session_label: str | None = None
    signal_strength: float | None = None
    confidence_score: float | None = None
    atr_pct_at_entry: float | None = None
    volume_ratio_at_entry: float | None = None
    spread_estimate: float | None = None
    move_already_extended_pct: float | None = None
    entry_snapshot_json: str | dict[str, Any] | None = None
    lifecycle_snapshot_json: str | dict[str, Any] | None = None
    profile_version: str | None = None
    review_status: str | None = None


@dataclass(frozen=True, slots=True)
class PaperTradeUpdate:
    symbol: str | None = None
    side: str | None = None
    entry_time: datetime | None = None
    entry_price: float | None = None
    qty: float | None = None
    leverage: int | None = None
    status: str | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None
    total_fees: float | None = None
    high_water_mark: float | None = None
    strategy_name: str | None = None
    timeframe: str | None = None
    regime_label_at_entry: str | None = None
    regime_confidence: float | None = None
    session_label: str | None = None
    signal_strength: float | None = None
    confidence_score: float | None = None
    atr_pct_at_entry: float | None = None
    volume_ratio_at_entry: float | None = None
    spread_estimate: float | None = None
    move_already_extended_pct: float | None = None
    entry_snapshot_json: str | dict[str, Any] | None = None
    lifecycle_snapshot_json: str | dict[str, Any] | None = None
    profile_version: str | None = None
    review_status: str | None = None


@dataclass(frozen=True, slots=True)
class PaperTrade:
    id: int
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    qty: float
    leverage: int
    status: str
    exit_time: datetime | None
    exit_price: float | None
    pnl: float | None
    total_fees: float
    high_water_mark: float
    strategy_name: str | None = None
    timeframe: str | None = None
    regime_label_at_entry: str | None = None
    regime_confidence: float | None = None
    session_label: str | None = None
    signal_strength: float | None = None
    confidence_score: float | None = None
    atr_pct_at_entry: float | None = None
    volume_ratio_at_entry: float | None = None
    spread_estimate: float | None = None
    move_already_extended_pct: float | None = None
    entry_snapshot_json: str | None = None
    lifecycle_snapshot_json: str | None = None
    profile_version: str | None = None
    review_status: str | None = None


@dataclass(frozen=True, slots=True)
class TradeReviewCreate:
    trade_id: int
    created_at: datetime
    updated_at: datetime
    symbol: str
    strategy_name: str | None
    timeframe: str | None
    trade_quality_score: float
    error_type_primary: str | None = None
    error_type_secondary: str | None = None
    better_no_trade_flag: bool = False
    better_exit_possible_flag: bool = False
    late_entry_flag: bool = False
    regime_mismatch_flag: bool = False
    overextended_entry_flag: bool = False
    sl_too_tight_flag: bool = False
    sl_too_wide_flag: bool = False
    avoidable_loss_flag: bool = False
    notes_auto_json: str | dict[str, Any] | None = None
    review_engine_version: str | None = None


@dataclass(frozen=True, slots=True)
class TradeReview:
    id: int
    trade_id: int
    created_at: datetime
    updated_at: datetime
    symbol: str
    strategy_name: str | None
    timeframe: str | None
    trade_quality_score: float
    error_type_primary: str | None
    error_type_secondary: str | None
    better_no_trade_flag: bool
    better_exit_possible_flag: bool
    late_entry_flag: bool
    regime_mismatch_flag: bool
    overextended_entry_flag: bool
    sl_too_tight_flag: bool
    sl_too_wide_flag: bool
    avoidable_loss_flag: bool
    notes_auto_json: str | None
    review_engine_version: str | None


@dataclass(frozen=True, slots=True)
class StrategyHealthSnapshot:
    symbol: str
    strategy_name: str
    timeframe: str
    computed_at: datetime
    trades_count: int
    pnl_sum: float
    winrate: float
    avg_pnl: float
    avg_fees: float
    late_entry_rate: float
    regime_mismatch_rate: float
    avoidable_loss_rate: float
    error_rate: float
    max_loss_streak: int
    health_score: float
    risk_multiplier: float
    state: str
    last_review_at: datetime | None
    window_size: int
    health_features_json: str | dict[str, Any] | None = None
    profile_version: str | None = None


@dataclass(frozen=True, slots=True)
class RegimeObservationCreate:
    observed_at: datetime
    symbol: str
    timeframe: str
    regime_label: str
    regime_confidence: float
    trend_bias: str
    volatility_state: str
    expansion_state: str
    liquidity_state: str
    session_label: str
    regime_features_json: str | dict[str, Any] | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    id: int
    observed_at: datetime
    symbol: str
    timeframe: str
    regime_label: str
    regime_confidence: float
    trend_bias: str
    volatility_state: str
    expansion_state: str
    liquidity_state: str
    session_label: str
    regime_features_json: str | None
    source: str | None


@dataclass(frozen=True, slots=True)
class AdaptationLogCreate:
    created_at: datetime
    symbol: str
    strategy_name: str | None
    timeframe: str | None
    event_type: str
    payload_json: str | dict[str, Any] | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class AdaptationLog:
    id: int
    created_at: datetime
    symbol: str
    strategy_name: str | None
    timeframe: str | None
    event_type: str
    payload_json: str | None
    source: str | None


class Database:
    _PAPER_TRADE_SELECT_COLUMNS: tuple[str, ...] = (
        "id",
        "symbol",
        "side",
        "entry_time",
        "entry_price",
        "qty",
        "leverage",
        "status",
        "exit_time",
        "exit_price",
        "pnl",
        "total_fees",
        "high_water_mark",
        "strategy_name",
        "timeframe",
        "regime_label_at_entry",
        "regime_confidence",
        "session_label",
        "signal_strength",
        "confidence_score",
        "atr_pct_at_entry",
        "volume_ratio_at_entry",
        "spread_estimate",
        "move_already_extended_pct",
        "entry_snapshot_json",
        "lifecycle_snapshot_json",
        "profile_version",
        "review_status",
    )
    _JSON_FIELDS: frozenset[str] = frozenset(
        {
            "entry_snapshot_json",
            "lifecycle_snapshot_json",
            "notes_auto_json",
            "health_features_json",
            "regime_features_json",
            "payload_json",
        }
    )

    def __init__(self, db_path: str | Path = "data/paper_trading.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS paper_trades_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS live_signals_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS backtest_runs_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS trade_reviews_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS strategy_health_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS regime_observations_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS adaptation_log_id_seq START 1
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                open_time TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id BIGINT PRIMARY KEY DEFAULT nextval('paper_trades_id_seq'),
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price DOUBLE NOT NULL,
                qty DOUBLE NOT NULL,
                leverage INTEGER NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'OPEN',
                exit_time TIMESTAMP,
                exit_price DOUBLE,
                pnl DOUBLE,
                total_fees DOUBLE NOT NULL DEFAULT 0,
                high_water_mark DOUBLE NOT NULL,
                strategy_name VARCHAR,
                timeframe VARCHAR,
                regime_label_at_entry VARCHAR,
                regime_confidence DOUBLE,
                session_label VARCHAR,
                signal_strength DOUBLE,
                confidence_score DOUBLE,
                atr_pct_at_entry DOUBLE,
                volume_ratio_at_entry DOUBLE,
                spread_estimate DOUBLE,
                move_already_extended_pct DOUBLE,
                entry_snapshot_json VARCHAR,
                lifecycle_snapshot_json VARCHAR,
                profile_version VARCHAR,
                review_status VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS total_fees DOUBLE DEFAULT 0
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS high_water_mark DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS strategy_name VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS timeframe VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS regime_label_at_entry VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS regime_confidence DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS session_label VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS signal_strength DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS confidence_score DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS atr_pct_at_entry DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS volume_ratio_at_entry DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS spread_estimate DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS move_already_extended_pct DOUBLE
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS entry_snapshot_json VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS lifecycle_snapshot_json VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS profile_version VARCHAR
            """
        )
        self._connection.execute(
            """
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS review_status VARCHAR
            """
        )
        self._connection.execute(
            """
            UPDATE paper_trades
            SET total_fees = 0
            WHERE total_fees IS NULL
            """
        )
        self._connection.execute(
            """
            UPDATE paper_trades
            SET high_water_mark = entry_price
            WHERE high_water_mark IS NULL
            """
        )
        self._connection.execute(
            """
            UPDATE paper_trades
            SET review_status = 'PENDING'
            WHERE review_status IS NULL
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_signals (
                id BIGINT PRIMARY KEY DEFAULT nextval('live_signals_id_seq'),
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                signal_time TIMESTAMP NOT NULL,
                signal_type VARCHAR NOT NULL,
                price DOUBLE NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id BIGINT PRIMARY KEY DEFAULT nextval('backtest_runs_id_seq'),
                created_at TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                optimization_mode BOOLEAN NOT NULL,
                leverage INTEGER NOT NULL,
                min_confidence_pct DOUBLE,
                total_pnl_usd DOUBLE NOT NULL,
                win_rate_pct DOUBLE NOT NULL,
                profit_factor DOUBLE NOT NULL,
                total_trades INTEGER NOT NULL,
                evaluated_profiles INTEGER,
                best_profile_json VARCHAR,
                summary_json VARCHAR NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_reviews (
                id BIGINT PRIMARY KEY DEFAULT nextval('trade_reviews_id_seq'),
                trade_id BIGINT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR,
                timeframe VARCHAR,
                trade_quality_score DOUBLE NOT NULL,
                error_type_primary VARCHAR,
                error_type_secondary VARCHAR,
                better_no_trade_flag BOOLEAN NOT NULL DEFAULT FALSE,
                better_exit_possible_flag BOOLEAN NOT NULL DEFAULT FALSE,
                late_entry_flag BOOLEAN NOT NULL DEFAULT FALSE,
                regime_mismatch_flag BOOLEAN NOT NULL DEFAULT FALSE,
                overextended_entry_flag BOOLEAN NOT NULL DEFAULT FALSE,
                sl_too_tight_flag BOOLEAN NOT NULL DEFAULT FALSE,
                sl_too_wide_flag BOOLEAN NOT NULL DEFAULT FALSE,
                avoidable_loss_flag BOOLEAN NOT NULL DEFAULT FALSE,
                notes_auto_json VARCHAR,
                review_engine_version VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_health (
                id BIGINT PRIMARY KEY DEFAULT nextval('strategy_health_id_seq'),
                computed_at TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                trades_count INTEGER NOT NULL,
                pnl_sum DOUBLE NOT NULL,
                winrate DOUBLE NOT NULL,
                avg_pnl DOUBLE NOT NULL,
                avg_fees DOUBLE NOT NULL,
                late_entry_rate DOUBLE NOT NULL,
                regime_mismatch_rate DOUBLE NOT NULL,
                avoidable_loss_rate DOUBLE NOT NULL,
                error_rate DOUBLE NOT NULL,
                max_loss_streak INTEGER NOT NULL,
                health_score DOUBLE NOT NULL,
                risk_multiplier DOUBLE NOT NULL,
                state VARCHAR NOT NULL,
                last_review_at TIMESTAMP,
                window_size INTEGER NOT NULL,
                health_features_json VARCHAR,
                profile_version VARCHAR,
                UNIQUE (symbol, strategy_name, timeframe)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_observations (
                id BIGINT PRIMARY KEY DEFAULT nextval('regime_observations_id_seq'),
                observed_at TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                regime_label VARCHAR NOT NULL,
                regime_confidence DOUBLE NOT NULL,
                trend_bias VARCHAR NOT NULL,
                volatility_state VARCHAR NOT NULL,
                expansion_state VARCHAR NOT NULL,
                liquidity_state VARCHAR NOT NULL,
                session_label VARCHAR NOT NULL,
                regime_features_json VARCHAR,
                source VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptation_log (
                id BIGINT PRIMARY KEY DEFAULT nextval('adaptation_log_id_seq'),
                created_at TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR,
                timeframe VARCHAR,
                event_type VARCHAR NOT NULL,
                payload_json VARCHAR,
                source VARCHAR
            )
            """
        )

    def upsert_candles(self, candles: Sequence[CandleRecord]) -> int:
        if not candles:
            return 0

        rows = [
            (
                candle.symbol,
                candle.interval,
                candle.open_time,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            for candle in candles
        ]
        self._connection.executemany(
            """
            INSERT INTO candles (
                symbol,
                interval,
                open_time,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            rows,
        )
        return len(rows)

    def insert_trade(self, trade: PaperTradeCreate) -> int:
        normalized_strategy_name = (
            str(trade.strategy_name).strip()
            if trade.strategy_name is not None and str(trade.strategy_name).strip()
            else None
        )
        normalized_timeframe = (
            str(trade.timeframe).strip()
            if trade.timeframe is not None and str(trade.timeframe).strip()
            else None
        )
        normalized_regime_label = (
            str(trade.regime_label_at_entry).strip()
            if trade.regime_label_at_entry is not None and str(trade.regime_label_at_entry).strip()
            else None
        )
        normalized_session_label = (
            str(trade.session_label).strip()
            if trade.session_label is not None and str(trade.session_label).strip()
            else None
        )
        normalized_profile_version = (
            str(trade.profile_version).strip()
            if trade.profile_version is not None and str(trade.profile_version).strip()
            else None
        )
        normalized_review_status = (
            str(trade.review_status).strip()
            if trade.review_status is not None and str(trade.review_status).strip()
            else "PENDING"
        )
        row = self._connection.execute(
            """
            INSERT INTO paper_trades (
                symbol,
                side,
                entry_time,
                entry_price,
                qty,
                leverage,
                status,
                exit_time,
                exit_price,
                pnl,
                total_fees,
                high_water_mark,
                strategy_name,
                timeframe,
                regime_label_at_entry,
                regime_confidence,
                session_label,
                signal_strength,
                confidence_score,
                atr_pct_at_entry,
                volume_ratio_at_entry,
                spread_estimate,
                move_already_extended_pct,
                entry_snapshot_json,
                lifecycle_snapshot_json,
                profile_version,
                review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                trade.symbol,
                trade.side,
                trade.entry_time,
                trade.entry_price,
                trade.qty,
                trade.leverage,
                trade.status,
                trade.exit_time,
                trade.exit_price,
                trade.pnl,
                trade.total_fees,
                trade.entry_price if trade.high_water_mark is None else trade.high_water_mark,
                normalized_strategy_name,
                normalized_timeframe,
                normalized_regime_label,
                self._to_optional_float(trade.regime_confidence),
                normalized_session_label,
                self._to_optional_float(trade.signal_strength),
                self._to_optional_float(trade.confidence_score),
                self._to_optional_float(trade.atr_pct_at_entry),
                self._to_optional_float(trade.volume_ratio_at_entry),
                self._to_optional_float(trade.spread_estimate),
                self._to_optional_float(trade.move_already_extended_pct),
                self._normalize_json_payload(trade.entry_snapshot_json),
                self._normalize_json_payload(trade.lifecycle_snapshot_json),
                normalized_profile_version,
                normalized_review_status,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to insert paper trade.")
        return int(row[0])

    def update_trade(self, trade_id: int, updates: PaperTradeUpdate) -> bool:
        update_data = {
            column: value
            for column, value in asdict(updates).items()
            if value is not None
        }
        if not update_data:
            return False
        for column in list(update_data.keys()):
            if column in self._JSON_FIELDS:
                update_data[column] = self._normalize_json_payload(update_data[column])

        assignments = ", ".join(f"{column} = ?" for column in update_data)
        parameters = [*update_data.values(), trade_id]
        row = self._connection.execute(
            f"""
            UPDATE paper_trades
            SET {assignments}
            WHERE id = ?
            RETURNING id
            """,
            parameters,
        ).fetchone()
        return row is not None

    def fetch_open_trades(self, symbol: str | None = None) -> list[PaperTrade]:
        sql = (
            "SELECT "
            + ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
            + " FROM paper_trades WHERE status = 'OPEN'"
        )
        parameters: list[str] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(symbol)

        sql += " ORDER BY entry_time ASC"
        rows = self._connection.execute(sql, parameters).fetchall()
        return [PaperTrade(*self._normalize_trade_row(row)) for row in rows]

    def archive_open_trades(
        self,
        *,
        status: str = "STALE_OPEN",
        symbol: str | None = None,
    ) -> int:
        if not status or status == "OPEN":
            raise ValueError("status must be a non-empty non-OPEN value.")

        sql = """
            UPDATE paper_trades
            SET status = ?
            WHERE status = 'OPEN'
        """
        parameters: list[str] = [status]
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(symbol)
        sql += " RETURNING id"

        rows = self._connection.execute(sql, parameters).fetchall()
        return len(rows)

    def get_candle_stats(self, symbol: str, interval: str) -> tuple[int, datetime | None, datetime | None]:
        count_row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE symbol = ? AND interval = ?
            """,
            [symbol, interval],
        ).fetchone()
        if count_row is None:
            return 0, None, None

        total_count = int(count_row[0])
        if total_count <= 0:
            return 0, None, None

        oldest_row = self._connection.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time ASC
            LIMIT 1
            """,
            [symbol, interval],
        ).fetchone()
        newest_row = self._connection.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC
            LIMIT 1
            """,
            [symbol, interval],
        ).fetchone()
        return (
            total_count,
            None if oldest_row is None else oldest_row[0],
            None if newest_row is None else newest_row[0],
        )

    def count_candles_in_range(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        if end_time <= start_time:
            return 0
        row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            """,
            [symbol, interval, start_time, end_time],
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])

    def get_candle_range_stats(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[int, datetime | None, datetime | None]:
        if end_time <= start_time:
            return 0, None, None
        count_row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            """,
            [symbol, interval, start_time, end_time],
        ).fetchone()
        if count_row is None:
            return 0, None, None

        total_count = int(count_row[0])
        if total_count <= 0:
            return 0, None, None

        oldest_row = self._connection.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            ORDER BY open_time ASC
            LIMIT 1
            """,
            [symbol, interval, start_time, end_time],
        ).fetchone()
        newest_row = self._connection.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            ORDER BY open_time DESC
            LIMIT 1
            """,
            [symbol, interval, start_time, end_time],
        ).fetchone()
        return (
            total_count,
            None if oldest_row is None else oldest_row[0],
            None if newest_row is None else newest_row[0],
        )

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        limit: int | None = None,
    ) -> list[CandleRecord]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")

        sql = """
            SELECT
                symbol,
                interval,
                open_time,
                open,
                high,
                low,
                close,
                volume
            FROM candles
            WHERE symbol = ? AND interval = ?
        """
        parameters: list[str | int] = [symbol, interval]
        if limit is None:
            sql += " ORDER BY open_time ASC"
        else:
            sql += " ORDER BY open_time DESC LIMIT ?"
            parameters.append(limit)

        rows = self._connection.execute(sql, parameters).fetchall()
        candles = [
            CandleRecord(
                symbol=row[0],
                interval=row[1],
                open_time=row[2],
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=float(row[7]),
            )
            for row in rows
        ]
        if limit is not None:
            candles.reverse()
        return candles

    def fetch_candles_since(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: datetime,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[CandleRecord]:
        if end_time is not None and end_time <= start_time:
            return []
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")

        sql = """
            SELECT
                symbol,
                interval,
                open_time,
                open,
                high,
                low,
                close,
                volume
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ?
        """
        parameters: list[str | int | datetime] = [symbol, interval, start_time]
        if end_time is not None:
            sql += " AND open_time < ?"
            parameters.append(end_time)
        sql += " ORDER BY open_time ASC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)

        rows = self._connection.execute(sql, parameters).fetchall()
        return [
            CandleRecord(
                symbol=row[0],
                interval=row[1],
                open_time=row[2],
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=float(row[7]),
            )
            for row in rows
        ]

    def fetch_recent_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[CandleRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive.")

        rows = self._connection.execute(
            """
            SELECT
                symbol,
                interval,
                open_time,
                open,
                high,
                low,
                close,
                volume
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC
            LIMIT ?
            """,
            [symbol, interval, limit],
        ).fetchall()
        candles = [
            CandleRecord(
                symbol=row[0],
                interval=row[1],
                open_time=row[2],
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=float(row[7]),
            )
            for row in rows
        ]
        candles.reverse()
        return candles

    def fetch_trade_by_id(self, trade_id: int) -> PaperTrade | None:
        select_clause = ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
        row = self._connection.execute(
            f"""
            SELECT {select_clause}
            FROM paper_trades
            WHERE id = ?
            """,
            [trade_id],
        ).fetchone()
        if row is None:
            return None
        return PaperTrade(*self._normalize_trade_row(row))

    def fetch_realized_pnl(self) -> float:
        row = self._connection.execute(
            """
            SELECT COALESCE(SUM(pnl), 0)
            FROM paper_trades
            WHERE status <> 'OPEN'
            """
        ).fetchone()
        if row is None:
            return 0.0
        return float(row[0])

    def fetch_realized_pnl_since(self, since: datetime) -> float:
        row = self._connection.execute(
            """
            SELECT COALESCE(SUM(pnl), 0)
            FROM paper_trades
            WHERE status <> 'OPEN' AND exit_time IS NOT NULL AND exit_time >= ?
            """,
            [since],
        ).fetchone()
        if row is None:
            return 0.0
        return float(row[0])

    def fetch_last_closed_trade(self, symbol: str) -> PaperTrade | None:
        select_clause = ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
        row = self._connection.execute(
            f"""
            SELECT {select_clause}
            FROM paper_trades
            WHERE symbol = ? AND status <> 'OPEN' AND exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 1
            """,
            [symbol],
        ).fetchone()
        if row is None:
            return None
        return PaperTrade(*self._normalize_trade_row(row))

    def fetch_recent_closed_trades(
        self,
        symbol: str,
        *,
        limit: int = 20,
    ) -> list[PaperTrade]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        select_clause = ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
        rows = self._connection.execute(
            f"""
            SELECT {select_clause}
            FROM paper_trades
            WHERE symbol = ? AND status <> 'OPEN' AND exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            [symbol, int(limit)],
        ).fetchall()
        return [PaperTrade(*self._normalize_trade_row(row)) for row in rows]

    def fetch_closed_trades(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> list[PaperTrade]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")
        sql = (
            "SELECT "
            + ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
            + " FROM paper_trades WHERE status <> 'OPEN' AND exit_time IS NOT NULL"
        )
        parameters: list[str | int] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(str(symbol).strip().upper())
        if strategy_name is not None:
            sql += " AND strategy_name = ?"
            parameters.append(str(strategy_name).strip())
        if timeframe is not None:
            sql += " AND timeframe = ?"
            parameters.append(str(timeframe).strip())
        sql += " ORDER BY exit_time DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        return [PaperTrade(*self._normalize_trade_row(row)) for row in rows]

    def upsert_trade_review(self, review: TradeReviewCreate) -> int:
        row = self._connection.execute(
            """
            INSERT INTO trade_reviews (
                trade_id,
                created_at,
                updated_at,
                symbol,
                strategy_name,
                timeframe,
                trade_quality_score,
                error_type_primary,
                error_type_secondary,
                better_no_trade_flag,
                better_exit_possible_flag,
                late_entry_flag,
                regime_mismatch_flag,
                overextended_entry_flag,
                sl_too_tight_flag,
                sl_too_wide_flag,
                avoidable_loss_flag,
                notes_auto_json,
                review_engine_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (trade_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                symbol = EXCLUDED.symbol,
                strategy_name = EXCLUDED.strategy_name,
                timeframe = EXCLUDED.timeframe,
                trade_quality_score = EXCLUDED.trade_quality_score,
                error_type_primary = EXCLUDED.error_type_primary,
                error_type_secondary = EXCLUDED.error_type_secondary,
                better_no_trade_flag = EXCLUDED.better_no_trade_flag,
                better_exit_possible_flag = EXCLUDED.better_exit_possible_flag,
                late_entry_flag = EXCLUDED.late_entry_flag,
                regime_mismatch_flag = EXCLUDED.regime_mismatch_flag,
                overextended_entry_flag = EXCLUDED.overextended_entry_flag,
                sl_too_tight_flag = EXCLUDED.sl_too_tight_flag,
                sl_too_wide_flag = EXCLUDED.sl_too_wide_flag,
                avoidable_loss_flag = EXCLUDED.avoidable_loss_flag,
                notes_auto_json = EXCLUDED.notes_auto_json,
                review_engine_version = EXCLUDED.review_engine_version
            RETURNING id
            """,
            (
                int(review.trade_id),
                review.created_at,
                review.updated_at,
                str(review.symbol).strip().upper(),
                self._normalize_optional_str(review.strategy_name),
                self._normalize_optional_str(review.timeframe),
                float(review.trade_quality_score),
                self._normalize_optional_str(review.error_type_primary),
                self._normalize_optional_str(review.error_type_secondary),
                bool(review.better_no_trade_flag),
                bool(review.better_exit_possible_flag),
                bool(review.late_entry_flag),
                bool(review.regime_mismatch_flag),
                bool(review.overextended_entry_flag),
                bool(review.sl_too_tight_flag),
                bool(review.sl_too_wide_flag),
                bool(review.avoidable_loss_flag),
                self._normalize_json_payload(review.notes_auto_json),
                self._normalize_optional_str(review.review_engine_version),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert trade review.")
        return int(row[0])

    def fetch_trade_review_by_trade_id(self, trade_id: int) -> TradeReview | None:
        row = self._connection.execute(
            """
            SELECT
                id,
                trade_id,
                created_at,
                updated_at,
                symbol,
                strategy_name,
                timeframe,
                trade_quality_score,
                error_type_primary,
                error_type_secondary,
                better_no_trade_flag,
                better_exit_possible_flag,
                late_entry_flag,
                regime_mismatch_flag,
                overextended_entry_flag,
                sl_too_tight_flag,
                sl_too_wide_flag,
                avoidable_loss_flag,
                notes_auto_json,
                review_engine_version
            FROM trade_reviews
            WHERE trade_id = ?
            """,
            [int(trade_id)],
        ).fetchone()
        if row is None:
            return None
        return self._normalize_trade_review_row(row)

    def fetch_trade_reviews_for_trade_ids(self, trade_ids: Sequence[int]) -> list[TradeReview]:
        normalized_trade_ids = [int(trade_id) for trade_id in trade_ids if int(trade_id) > 0]
        if not normalized_trade_ids:
            return []
        placeholders = ", ".join("?" for _ in normalized_trade_ids)
        rows = self._connection.execute(
            f"""
            SELECT
                id,
                trade_id,
                created_at,
                updated_at,
                symbol,
                strategy_name,
                timeframe,
                trade_quality_score,
                error_type_primary,
                error_type_secondary,
                better_no_trade_flag,
                better_exit_possible_flag,
                late_entry_flag,
                regime_mismatch_flag,
                overextended_entry_flag,
                sl_too_tight_flag,
                sl_too_wide_flag,
                avoidable_loss_flag,
                notes_auto_json,
                review_engine_version
            FROM trade_reviews
            WHERE trade_id IN ({placeholders})
            ORDER BY updated_at DESC
            """,
            normalized_trade_ids,
        ).fetchall()
        return [self._normalize_trade_review_row(row) for row in rows]

    def upsert_strategy_health(self, snapshot: StrategyHealthSnapshot) -> int:
        row = self._connection.execute(
            """
            INSERT INTO strategy_health (
                computed_at,
                symbol,
                strategy_name,
                timeframe,
                trades_count,
                pnl_sum,
                winrate,
                avg_pnl,
                avg_fees,
                late_entry_rate,
                regime_mismatch_rate,
                avoidable_loss_rate,
                error_rate,
                max_loss_streak,
                health_score,
                risk_multiplier,
                state,
                last_review_at,
                window_size,
                health_features_json,
                profile_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, strategy_name, timeframe) DO UPDATE SET
                computed_at = EXCLUDED.computed_at,
                trades_count = EXCLUDED.trades_count,
                pnl_sum = EXCLUDED.pnl_sum,
                winrate = EXCLUDED.winrate,
                avg_pnl = EXCLUDED.avg_pnl,
                avg_fees = EXCLUDED.avg_fees,
                late_entry_rate = EXCLUDED.late_entry_rate,
                regime_mismatch_rate = EXCLUDED.regime_mismatch_rate,
                avoidable_loss_rate = EXCLUDED.avoidable_loss_rate,
                error_rate = EXCLUDED.error_rate,
                max_loss_streak = EXCLUDED.max_loss_streak,
                health_score = EXCLUDED.health_score,
                risk_multiplier = EXCLUDED.risk_multiplier,
                state = EXCLUDED.state,
                last_review_at = EXCLUDED.last_review_at,
                window_size = EXCLUDED.window_size,
                health_features_json = EXCLUDED.health_features_json,
                profile_version = EXCLUDED.profile_version
            RETURNING id
            """,
            (
                snapshot.computed_at,
                str(snapshot.symbol).strip().upper(),
                str(snapshot.strategy_name).strip(),
                str(snapshot.timeframe).strip(),
                int(snapshot.trades_count),
                float(snapshot.pnl_sum),
                float(snapshot.winrate),
                float(snapshot.avg_pnl),
                float(snapshot.avg_fees),
                float(snapshot.late_entry_rate),
                float(snapshot.regime_mismatch_rate),
                float(snapshot.avoidable_loss_rate),
                float(snapshot.error_rate),
                int(snapshot.max_loss_streak),
                float(snapshot.health_score),
                float(snapshot.risk_multiplier),
                str(snapshot.state).strip(),
                snapshot.last_review_at,
                int(snapshot.window_size),
                self._normalize_json_payload(snapshot.health_features_json),
                self._normalize_optional_str(snapshot.profile_version),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert strategy health.")
        return int(row[0])

    def fetch_strategy_health(
        self,
        *,
        symbol: str,
        strategy_name: str,
        timeframe: str,
    ) -> StrategyHealthSnapshot | None:
        row = self._connection.execute(
            """
            SELECT
                symbol,
                strategy_name,
                timeframe,
                computed_at,
                trades_count,
                pnl_sum,
                winrate,
                avg_pnl,
                avg_fees,
                late_entry_rate,
                regime_mismatch_rate,
                avoidable_loss_rate,
                error_rate,
                max_loss_streak,
                health_score,
                risk_multiplier,
                state,
                last_review_at,
                window_size,
                health_features_json,
                profile_version
            FROM strategy_health
            WHERE symbol = ? AND strategy_name = ? AND timeframe = ?
            LIMIT 1
            """,
            [str(symbol).strip().upper(), str(strategy_name).strip(), str(timeframe).strip()],
        ).fetchone()
        if row is None:
            return None
        return self._normalize_strategy_health_row(row)

    def fetch_recent_strategy_health(
        self,
        *,
        limit: int = 200,
    ) -> list[StrategyHealthSnapshot]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        rows = self._connection.execute(
            """
            SELECT
                symbol,
                strategy_name,
                timeframe,
                computed_at,
                trades_count,
                pnl_sum,
                winrate,
                avg_pnl,
                avg_fees,
                late_entry_rate,
                regime_mismatch_rate,
                avoidable_loss_rate,
                error_rate,
                max_loss_streak,
                health_score,
                risk_multiplier,
                state,
                last_review_at,
                window_size,
                health_features_json,
                profile_version
            FROM strategy_health
            ORDER BY computed_at DESC
            LIMIT ?
            """,
            [int(limit)],
        ).fetchall()
        return [self._normalize_strategy_health_row(row) for row in rows]

    def fetch_strategy_health_rows(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        timeframe: str | None = None,
        state: str | None = None,
        limit: int = 1000,
    ) -> list[StrategyHealthSnapshot]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        sql = """
            SELECT
                symbol,
                strategy_name,
                timeframe,
                computed_at,
                trades_count,
                pnl_sum,
                winrate,
                avg_pnl,
                avg_fees,
                late_entry_rate,
                regime_mismatch_rate,
                avoidable_loss_rate,
                error_rate,
                max_loss_streak,
                health_score,
                risk_multiplier,
                state,
                last_review_at,
                window_size,
                health_features_json,
                profile_version
            FROM strategy_health
            WHERE 1=1
        """
        parameters: list[str | int] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(str(symbol).strip().upper())
        if strategy_name is not None:
            sql += " AND strategy_name = ?"
            parameters.append(str(strategy_name).strip())
        if timeframe is not None:
            sql += " AND timeframe = ?"
            parameters.append(str(timeframe).strip())
        if state is not None:
            sql += " AND state = ?"
            parameters.append(str(state).strip())
        sql += " ORDER BY computed_at DESC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        return [self._normalize_strategy_health_row(row) for row in rows]

    def fetch_meta_overview_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            """
            SELECT state, COUNT(*)
            FROM strategy_health
            GROUP BY state
            """
        ).fetchall()
        counts: dict[str, int] = {
            "healthy": 0,
            "degraded": 0,
            "watchlist": 0,
            "paper_only": 0,
            "paused": 0,
        }
        for row in rows:
            state_name = str(row[0]).strip()
            counts[state_name] = int(row[1])
        return counts

    def insert_regime_observation(self, observation: RegimeObservationCreate) -> int:
        row = self._connection.execute(
            """
            INSERT INTO regime_observations (
                observed_at,
                symbol,
                timeframe,
                regime_label,
                regime_confidence,
                trend_bias,
                volatility_state,
                expansion_state,
                liquidity_state,
                session_label,
                regime_features_json,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                observation.observed_at,
                str(observation.symbol).strip().upper(),
                str(observation.timeframe).strip(),
                str(observation.regime_label).strip(),
                float(observation.regime_confidence),
                str(observation.trend_bias).strip(),
                str(observation.volatility_state).strip(),
                str(observation.expansion_state).strip(),
                str(observation.liquidity_state).strip(),
                str(observation.session_label).strip(),
                self._normalize_json_payload(observation.regime_features_json),
                self._normalize_optional_str(observation.source),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to insert regime observation.")
        return int(row[0])

    def fetch_recent_regime_observations(
        self,
        *,
        symbol: str,
        timeframe: str,
        limit: int = 250,
    ) -> list[RegimeObservation]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        rows = self._connection.execute(
            """
            SELECT
                id,
                observed_at,
                symbol,
                timeframe,
                regime_label,
                regime_confidence,
                trend_bias,
                volatility_state,
                expansion_state,
                liquidity_state,
                session_label,
                regime_features_json,
                source
            FROM regime_observations
            WHERE symbol = ? AND timeframe = ?
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            [str(symbol).strip().upper(), str(timeframe).strip(), int(limit)],
        ).fetchall()
        return [self._normalize_regime_observation_row(row) for row in rows]

    def insert_adaptation_log(self, entry: AdaptationLogCreate) -> int:
        row = self._connection.execute(
            """
            INSERT INTO adaptation_log (
                created_at,
                symbol,
                strategy_name,
                timeframe,
                event_type,
                payload_json,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                entry.created_at,
                str(entry.symbol).strip().upper(),
                self._normalize_optional_str(entry.strategy_name),
                self._normalize_optional_str(entry.timeframe),
                str(entry.event_type).strip(),
                self._normalize_json_payload(entry.payload_json),
                self._normalize_optional_str(entry.source),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to insert adaptation log.")
        return int(row[0])

    def fetch_recent_adaptation_logs(
        self,
        *,
        symbol: str | None = None,
        limit: int = 200,
    ) -> list[AdaptationLog]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        sql = """
            SELECT
                id,
                created_at,
                symbol,
                strategy_name,
                timeframe,
                event_type,
                payload_json,
                source
            FROM adaptation_log
        """
        parameters: list[str | int] = []
        if symbol is not None:
            sql += " WHERE symbol = ?"
            parameters.append(str(symbol).strip().upper())
        sql += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        return [self._normalize_adaptation_log_row(row) for row in rows]

    def fetch_adaptation_logs(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        timeframe: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AdaptationLog]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        sql = """
            SELECT
                id,
                created_at,
                symbol,
                strategy_name,
                timeframe,
                event_type,
                payload_json,
                source
            FROM adaptation_log
            WHERE 1=1
        """
        parameters: list[str | int | datetime] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(str(symbol).strip().upper())
        if strategy_name is not None:
            sql += " AND strategy_name = ?"
            parameters.append(str(strategy_name).strip())
        if timeframe is not None:
            sql += " AND timeframe = ?"
            parameters.append(str(timeframe).strip())
        if event_type is not None:
            sql += " AND event_type = ?"
            parameters.append(str(event_type).strip())
        if since is not None:
            sql += " AND created_at >= ?"
            parameters.append(since)
        sql += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        return [self._normalize_adaptation_log_row(row) for row in rows]

    def fetch_closed_trades_since(
        self,
        *,
        start_time: datetime,
        end_time: datetime | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> list[PaperTrade]:
        if end_time is not None and end_time <= start_time:
            return []
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")
        sql = (
            "SELECT "
            + ", ".join(self._PAPER_TRADE_SELECT_COLUMNS)
            + " FROM paper_trades "
            + "WHERE status <> 'OPEN' AND exit_time IS NOT NULL AND exit_time >= ?"
        )
        parameters: list[str | int | datetime] = [start_time]
        if end_time is not None:
            sql += " AND exit_time < ?"
            parameters.append(end_time)
        if symbol is not None:
            sql += " AND symbol = ?"
            parameters.append(str(symbol).strip().upper())
        if strategy_name is not None:
            sql += " AND strategy_name = ?"
            parameters.append(str(strategy_name).strip())
        if timeframe is not None:
            sql += " AND timeframe = ?"
            parameters.append(str(timeframe).strip())
        sql += " ORDER BY exit_time DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        return [PaperTrade(*self._normalize_trade_row(row)) for row in rows]

    def fetch_recent_trade_reviews(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        timeframe: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        sql = """
            SELECT
                tr.id,
                tr.trade_id,
                tr.created_at,
                tr.updated_at,
                tr.symbol,
                tr.strategy_name,
                tr.timeframe,
                tr.trade_quality_score,
                tr.error_type_primary,
                tr.error_type_secondary,
                tr.better_no_trade_flag,
                tr.better_exit_possible_flag,
                tr.late_entry_flag,
                tr.regime_mismatch_flag,
                tr.overextended_entry_flag,
                tr.sl_too_tight_flag,
                tr.sl_too_wide_flag,
                tr.avoidable_loss_flag,
                tr.notes_auto_json,
                tr.review_engine_version,
                pt.regime_label_at_entry,
                pt.side,
                pt.entry_time,
                pt.exit_time,
                pt.pnl,
                pt.review_status
            FROM trade_reviews tr
            LEFT JOIN paper_trades pt
                ON pt.id = tr.trade_id
            WHERE 1=1
        """
        parameters: list[str | int] = []
        if symbol is not None:
            sql += " AND tr.symbol = ?"
            parameters.append(str(symbol).strip().upper())
        if strategy_name is not None:
            sql += " AND tr.strategy_name = ?"
            parameters.append(str(strategy_name).strip())
        if timeframe is not None:
            sql += " AND tr.timeframe = ?"
            parameters.append(str(timeframe).strip())
        sql += " ORDER BY tr.updated_at DESC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(sql, parameters).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "id": int(row[0]),
                    "trade_id": int(row[1]),
                    "created_at": row[2],
                    "updated_at": row[3],
                    "symbol": str(row[4]),
                    "strategy_name": self._normalize_optional_str(row[5]),
                    "timeframe": self._normalize_optional_str(row[6]),
                    "trade_quality_score": float(row[7]),
                    "error_type_primary": self._normalize_optional_str(row[8]),
                    "error_type_secondary": self._normalize_optional_str(row[9]),
                    "better_no_trade_flag": bool(row[10]),
                    "better_exit_possible_flag": bool(row[11]),
                    "late_entry_flag": bool(row[12]),
                    "regime_mismatch_flag": bool(row[13]),
                    "overextended_entry_flag": bool(row[14]),
                    "sl_too_tight_flag": bool(row[15]),
                    "sl_too_wide_flag": bool(row[16]),
                    "avoidable_loss_flag": bool(row[17]),
                    "notes_auto_json": self._normalize_optional_str(row[18]),
                    "review_engine_version": self._normalize_optional_str(row[19]),
                    "regime_label_at_entry": self._normalize_optional_str(row[20]),
                    "side": self._normalize_optional_str(row[21]),
                    "entry_time": row[22],
                    "exit_time": row[23],
                    "pnl": self._to_optional_float(row[24]),
                    "review_status": self._normalize_optional_str(row[25]),
                }
            )
        return payload

    def insert_backtest_run(
        self,
        *,
        symbol: str,
        strategy_name: str,
        interval: str,
        optimization_mode: bool,
        leverage: int,
        min_confidence_pct: float | None,
        total_pnl_usd: float,
        win_rate_pct: float,
        profit_factor: float,
        total_trades: int,
        evaluated_profiles: int | None = None,
        best_profile: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> int:
        row = self._connection.execute(
            """
            INSERT INTO backtest_runs (
                created_at,
                symbol,
                strategy_name,
                interval,
                optimization_mode,
                leverage,
                min_confidence_pct,
                total_pnl_usd,
                win_rate_pct,
                profit_factor,
                total_trades,
                evaluated_profiles,
                best_profile_json,
                summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                datetime.now(),
                symbol,
                strategy_name,
                interval,
                optimization_mode,
                leverage,
                min_confidence_pct,
                total_pnl_usd,
                win_rate_pct,
                profit_factor,
                total_trades,
                evaluated_profiles,
                json.dumps(best_profile or {}, default=self._json_default, ensure_ascii=True, sort_keys=True),
                json.dumps(summary or {}, default=self._json_default, ensure_ascii=True, sort_keys=True),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to insert backtest run.")
        return int(row[0])

    def update_backtest_run(
        self,
        run_id: int,
        *,
        symbol: str,
        strategy_name: str,
        interval: str,
        optimization_mode: bool,
        leverage: int,
        min_confidence_pct: float | None,
        total_pnl_usd: float,
        win_rate_pct: float,
        profit_factor: float,
        total_trades: int,
        evaluated_profiles: int | None = None,
        best_profile: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        row = self._connection.execute(
            """
            UPDATE backtest_runs
            SET
                symbol = ?,
                strategy_name = ?,
                interval = ?,
                optimization_mode = ?,
                leverage = ?,
                min_confidence_pct = ?,
                total_pnl_usd = ?,
                win_rate_pct = ?,
                profit_factor = ?,
                total_trades = ?,
                evaluated_profiles = ?,
                best_profile_json = ?,
                summary_json = ?
            WHERE id = ?
            RETURNING id
            """,
            (
                symbol,
                strategy_name,
                interval,
                optimization_mode,
                leverage,
                min_confidence_pct,
                total_pnl_usd,
                win_rate_pct,
                profit_factor,
                total_trades,
                evaluated_profiles,
                json.dumps(best_profile or {}, default=self._json_default, ensure_ascii=True, sort_keys=True),
                json.dumps(summary or {}, default=self._json_default, ensure_ascii=True, sort_keys=True),
                int(run_id),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to update backtest run id={run_id}.")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _normalize_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        with suppress(Exception):
            resolved_value = float(value)
            if math.isfinite(resolved_value):
                return resolved_value
        return None

    def _normalize_json_payload(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return json.dumps(value, default=self._json_default, ensure_ascii=True, sort_keys=True)

    @classmethod
    def _normalize_trade_row(cls, row: tuple[Any, ...]) -> tuple[Any, ...]:
        def _value(index: int, default: Any = None) -> Any:
            return row[index] if len(row) > index else default

        strategy_name = cls._normalize_optional_str(_value(13))
        timeframe = cls._normalize_optional_str(_value(14))
        regime_label_at_entry = cls._normalize_optional_str(_value(15))
        regime_confidence = cls._to_optional_float(_value(16))
        session_label = cls._normalize_optional_str(_value(17))
        signal_strength = cls._to_optional_float(_value(18))
        confidence_score = cls._to_optional_float(_value(19))
        atr_pct_at_entry = cls._to_optional_float(_value(20))
        volume_ratio_at_entry = cls._to_optional_float(_value(21))
        spread_estimate = cls._to_optional_float(_value(22))
        move_already_extended_pct = cls._to_optional_float(_value(23))
        entry_snapshot_json = cls._normalize_optional_str(_value(24))
        lifecycle_snapshot_json = cls._normalize_optional_str(_value(25))
        profile_version = cls._normalize_optional_str(_value(26))
        review_status = cls._normalize_optional_str(_value(27)) or "PENDING"
        entry_price = float(_value(4) if _value(4) is not None else 0.0)
        return (
            int(_value(0, 0)),
            _value(1),
            _value(2),
            _value(3),
            entry_price,
            float(_value(5) if _value(5) is not None else 0.0),
            int(_value(6, 0)),
            _value(7),
            _value(8),
            float(_value(9)) if _value(9) is not None else None,
            float(_value(10)) if _value(10) is not None else None,
            float(_value(11)) if _value(11) is not None else 0.0,
            float(_value(12)) if _value(12) is not None else entry_price,
            strategy_name,
            timeframe,
            regime_label_at_entry,
            regime_confidence,
            session_label,
            signal_strength,
            confidence_score,
            atr_pct_at_entry,
            volume_ratio_at_entry,
            spread_estimate,
            move_already_extended_pct,
            entry_snapshot_json,
            lifecycle_snapshot_json,
            profile_version,
            review_status,
        )

    @classmethod
    def _normalize_trade_review_row(cls, row: tuple[Any, ...]) -> TradeReview:
        return TradeReview(
            id=int(row[0]),
            trade_id=int(row[1]),
            created_at=row[2],
            updated_at=row[3],
            symbol=str(row[4]),
            strategy_name=cls._normalize_optional_str(row[5]),
            timeframe=cls._normalize_optional_str(row[6]),
            trade_quality_score=float(row[7]),
            error_type_primary=cls._normalize_optional_str(row[8]),
            error_type_secondary=cls._normalize_optional_str(row[9]),
            better_no_trade_flag=bool(row[10]),
            better_exit_possible_flag=bool(row[11]),
            late_entry_flag=bool(row[12]),
            regime_mismatch_flag=bool(row[13]),
            overextended_entry_flag=bool(row[14]),
            sl_too_tight_flag=bool(row[15]),
            sl_too_wide_flag=bool(row[16]),
            avoidable_loss_flag=bool(row[17]),
            notes_auto_json=cls._normalize_optional_str(row[18]),
            review_engine_version=cls._normalize_optional_str(row[19]),
        )

    @classmethod
    def _normalize_strategy_health_row(cls, row: tuple[Any, ...]) -> StrategyHealthSnapshot:
        return StrategyHealthSnapshot(
            symbol=str(row[0]),
            strategy_name=str(row[1]),
            timeframe=str(row[2]),
            computed_at=row[3],
            trades_count=int(row[4]),
            pnl_sum=float(row[5]),
            winrate=float(row[6]),
            avg_pnl=float(row[7]),
            avg_fees=float(row[8]),
            late_entry_rate=float(row[9]),
            regime_mismatch_rate=float(row[10]),
            avoidable_loss_rate=float(row[11]),
            error_rate=float(row[12]),
            max_loss_streak=int(row[13]),
            health_score=float(row[14]),
            risk_multiplier=float(row[15]),
            state=str(row[16]),
            last_review_at=row[17],
            window_size=int(row[18]),
            health_features_json=cls._normalize_optional_str(row[19]),
            profile_version=cls._normalize_optional_str(row[20]),
        )

    @classmethod
    def _normalize_regime_observation_row(cls, row: tuple[Any, ...]) -> RegimeObservation:
        return RegimeObservation(
            id=int(row[0]),
            observed_at=row[1],
            symbol=str(row[2]),
            timeframe=str(row[3]),
            regime_label=str(row[4]),
            regime_confidence=float(row[5]),
            trend_bias=str(row[6]),
            volatility_state=str(row[7]),
            expansion_state=str(row[8]),
            liquidity_state=str(row[9]),
            session_label=str(row[10]),
            regime_features_json=cls._normalize_optional_str(row[11]),
            source=cls._normalize_optional_str(row[12]),
        )

    @classmethod
    def _normalize_adaptation_log_row(cls, row: tuple[Any, ...]) -> AdaptationLog:
        return AdaptationLog(
            id=int(row[0]),
            created_at=row[1],
            symbol=str(row[2]),
            strategy_name=cls._normalize_optional_str(row[3]),
            timeframe=cls._normalize_optional_str(row[4]),
            event_type=str(row[5]),
            payload_json=cls._normalize_optional_str(row[6]),
            source=cls._normalize_optional_str(row[7]),
        )
