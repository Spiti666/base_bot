from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import polars as pl
except Exception as exc:  # pragma: no cover - optional runtime dependency
    pl = None  # type: ignore[assignment]
    _POLARS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _POLARS_IMPORT_ERROR = None

from core.data.db import CandleRecord, Database


_BASE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _require_polars() -> None:
    if pl is None:
        raise RuntimeError(
            "PolarsDataLoader requires the 'polars' package. Install it via: pip install polars"
        ) from _POLARS_IMPORT_ERROR


class PolarsDataLoader:
    """Columnar data loader for backtest candle pipelines."""

    def __init__(self, frame: "pl.DataFrame") -> None:
        _require_polars()
        self._frame = self._normalize_frame(frame)

    @classmethod
    def from_candle_records(cls, candles: Sequence[CandleRecord]) -> "PolarsDataLoader":
        _require_polars()
        rows = [
            {
                "symbol": candle.symbol,
                "interval": candle.interval,
                "open_time": candle.open_time,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            }
            for candle in candles
        ]
        return cls(pl.DataFrame(rows, schema_overrides={"open_time": pl.Datetime(time_unit="us")}))

    @classmethod
    def from_row_dicts(cls, rows: Sequence[Mapping[str, Any]]) -> "PolarsDataLoader":
        _require_polars()
        normalized_rows = [
            {
                "symbol": str(row.get("symbol", "") or ""),
                "interval": str(row.get("interval", "") or ""),
                "open_time": row.get("open_time"),
                "open": float(row.get("open", 0.0) or 0.0),
                "high": float(row.get("high", 0.0) or 0.0),
                "low": float(row.get("low", 0.0) or 0.0),
                "close": float(row.get("close", 0.0) or 0.0),
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
            for row in rows
        ]
        return cls(
            pl.DataFrame(
                normalized_rows,
                schema_overrides={"open_time": pl.Datetime(time_unit="us")},
            )
        )

    @classmethod
    def from_db(
        cls,
        db: Database,
        *,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> "PolarsDataLoader":
        if start_time is None:
            effective_limit = max(1, int(limit if limit is not None else 100_000))
            candles = db.fetch_recent_candles(symbol, interval, limit=effective_limit)
        else:
            candles = db.fetch_candles_since(
                symbol,
                interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
        return cls.from_candle_records(candles)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "PolarsDataLoader":
        _require_polars()
        frame = pl.read_csv(
            str(csv_path),
            try_parse_dates=True,
            infer_schema_length=10_000,
        )
        return cls(frame)

    @classmethod
    def from_worker_payload(cls, payload: Mapping[str, Any]) -> "PolarsDataLoader":
        _require_polars()
        frame = pl.DataFrame(
            {
                "symbol": list(payload.get("symbol", [])),
                "interval": list(payload.get("interval", [])),
                "open_time": list(payload.get("open_time", [])),
                "open": list(payload.get("open", [])),
                "high": list(payload.get("high", [])),
                "low": list(payload.get("low", [])),
                "close": list(payload.get("close", [])),
                "volume": list(payload.get("volume", [])),
            },
            schema_overrides={"open_time": pl.Datetime(time_unit="us")},
        )
        return cls(frame)

    @classmethod
    def from_pandas_dataframe(cls, frame: Any) -> "PolarsDataLoader":
        _require_polars()
        return cls(pl.from_pandas(frame))

    @staticmethod
    def _normalize_frame(frame: "pl.DataFrame") -> "pl.DataFrame":
        _require_polars()
        normalized_frame = frame
        missing_columns = [column for column in _BASE_COLUMNS if column not in normalized_frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required candle columns: {', '.join(missing_columns)}")

        if normalized_frame.height == 0:
            return normalized_frame.select(list(_BASE_COLUMNS))

        normalized_frame = normalized_frame.with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("interval").cast(pl.Utf8),
            pl.col("open_time").cast(pl.Datetime(time_unit="us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        normalized_frame = normalized_frame.sort("open_time")
        return normalized_frame

    def with_setup_gate_ema(
        self,
        *,
        period: int = 200,
        source_column: str = "close",
        output_column: str = "setup_gate_ema_200",
    ) -> "PolarsDataLoader":
        _require_polars()
        if period <= 1:
            raise ValueError("period must be > 1 for EMA computation.")
        if source_column not in self._frame.columns:
            raise KeyError(f"Source column '{source_column}' is not available.")
        self._frame = self._frame.with_columns(
            pl.col(source_column).ewm_mean(span=period, adjust=False).alias(output_column)
        )
        return self

    def frame(self) -> "pl.DataFrame":
        return self._frame

    def select_base_columns(self) -> "pl.DataFrame":
        return self._frame.select(list(_BASE_COLUMNS))

    def to_worker_payload(self) -> dict[str, Any]:
        base_frame = self.select_base_columns()
        return {
            "symbol": base_frame.get_column("symbol").to_list(),
            "interval": base_frame.get_column("interval").to_list(),
            "open_time": base_frame.get_column("open_time").to_list(),
            "open": base_frame.get_column("open").to_numpy(),
            "high": base_frame.get_column("high").to_numpy(),
            "low": base_frame.get_column("low").to_numpy(),
            "close": base_frame.get_column("close").to_numpy(),
            "volume": base_frame.get_column("volume").to_numpy(),
        }

    def to_row_dicts(self) -> list[dict[str, Any]]:
        return self.select_base_columns().to_dicts()

    def get_numpy_arrays(self) -> dict[str, np.ndarray]:
        base_frame = self.select_base_columns()
        return {
            "open": np.asarray(base_frame.get_column("open").to_numpy(), dtype=np.float64),
            "high": np.asarray(base_frame.get_column("high").to_numpy(), dtype=np.float64),
            "low": np.asarray(base_frame.get_column("low").to_numpy(), dtype=np.float64),
            "close": np.asarray(base_frame.get_column("close").to_numpy(), dtype=np.float64),
            "volume": np.asarray(base_frame.get_column("volume").to_numpy(), dtype=np.float64),
        }

    def __len__(self) -> int:
        return int(self._frame.height)
