from __future__ import annotations

import json
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


class Database:
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
                high_water_mark DOUBLE NOT NULL
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
                high_water_mark
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        sql = """
            SELECT
                id,
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
                high_water_mark
            FROM paper_trades
            WHERE status = 'OPEN'
        """
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
        row = self._connection.execute(
            """
            SELECT
                id,
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
                high_water_mark
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
        row = self._connection.execute(
            """
            SELECT
                id,
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
                high_water_mark
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
    def _normalize_trade_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
        return (
            int(row[0]),
            row[1],
            row[2],
            row[3],
            float(row[4]),
            float(row[5]),
            int(row[6]),
            row[7],
            row[8],
            float(row[9]) if row[9] is not None else None,
            float(row[10]) if row[10] is not None else None,
            float(row[11]) if row[11] is not None else 0.0,
            float(row[12]) if row[12] is not None else float(row[4]),
        )
