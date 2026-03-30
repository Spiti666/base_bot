from __future__ import annotations

from typing import Any

import pandas as pd

from config import settings


def build_ema_cross_volume_signal_frame(candles_dataframe: Any) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    required_columns = {"close", "volume"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"EMA Cross + Volume requires columns: {missing_list}")

    required_signal_columns = {
        "ema_fast",
        "ema_slow",
        "volume_sma",
        "volume_threshold",
        "recent_volume_spike",
        "ema_long_entry",
        "ema_short_entry",
    }
    if required_signal_columns.issubset(working_df.columns):
        return working_df

    fast_period = settings.strategy.ema_fast_period
    slow_period = settings.strategy.ema_slow_period
    volume_period = settings.strategy.volume_sma_period
    volume_multiplier = settings.strategy.volume_multiplier

    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    volume_series = pd.to_numeric(working_df["volume"], errors="coerce")

    working_df.loc[:, "ema_fast"] = close_series.ewm(span=fast_period, adjust=False).mean()
    working_df.loc[:, "ema_slow"] = close_series.ewm(span=slow_period, adjust=False).mean()
    working_df.loc[:, "volume_sma"] = volume_series.rolling(window=volume_period).mean()
    working_df.loc[:, "volume_threshold"] = working_df["volume_sma"] * float(volume_multiplier)
    volume_spike = volume_series.gt(working_df["volume_threshold"])
    working_df.loc[:, "recent_volume_spike"] = (
        volume_spike.fillna(False).rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
    )
    previous_fast = working_df["ema_fast"].shift(1)
    previous_slow = working_df["ema_slow"].shift(1)
    current_fast = working_df["ema_fast"]
    current_slow = working_df["ema_slow"]
    working_df.loc[:, "ema_long_entry"] = (
        previous_fast.le(previous_slow)
        & current_fast.gt(current_slow)
        & working_df["recent_volume_spike"]
    ).fillna(False)
    working_df.loc[:, "ema_short_entry"] = (
        previous_fast.ge(previous_slow)
        & current_fast.lt(current_slow)
        & working_df["recent_volume_spike"]
    ).fillna(False)
    return working_df


def run_python_ema_cross_volume(candles_dataframe: Any) -> int:
    working_df = build_ema_cross_volume_signal_frame(candles_dataframe)

    slow_period = settings.strategy.ema_slow_period
    volume_period = settings.strategy.volume_sma_period
    closes = _extract_series(working_df, "close")
    volumes = _extract_series(working_df, "volume")
    required_points = max(slow_period + 1, volume_period)
    if len(closes) < required_points or len(volumes) < required_points:
        return 0

    last_row = working_df.iloc[-1]
    if bool(last_row["ema_long_entry"]):
        return 1
    if bool(last_row["ema_short_entry"]):
        return -1
    return 0


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)


def _extract_series(candles_dataframe: Any, column_name: str) -> list[float]:
    try:
        series = candles_dataframe[column_name]
    except Exception as exc:
        raise ValueError(f"candles_dataframe must provide a '{column_name}' column.") from exc

    if hasattr(series, "tolist"):
        raw_values = series.tolist()
    else:
        raw_values = list(series)

    return [float(value) for value in raw_values]


def _calculate_ema_series(values: list[float], period: int) -> list[float]:
    multiplier = 2.0 / (period + 1.0)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _calculate_sma(values: list[float]) -> float:
    return sum(values) / len(values)
