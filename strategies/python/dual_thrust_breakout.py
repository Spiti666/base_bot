from __future__ import annotations

from typing import Any

import pandas as pd

from config import settings


def apply_dual_thrust(df: pd.DataFrame, n: int = 4, k1: float = 0.5, k2: float = 0.5) -> pd.DataFrame:
    working_df = df.copy(deep=True) if getattr(df, "_is_copy", None) is not None else df
    required_columns = {"open", "high", "low", "close"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dual Thrust requires columns: {missing_list}")

    if {"dual_range", "dual_buy_line", "dual_sell_line"}.issubset(working_df.columns):
        return working_df

    highest_high = working_df["high"].rolling(window=n).max().shift(1)
    highest_close = working_df["close"].rolling(window=n).max().shift(1)
    lowest_close = working_df["close"].rolling(window=n).min().shift(1)
    lowest_low = working_df["low"].rolling(window=n).min().shift(1)

    dual_range = pd.concat(
        [highest_high - lowest_close, highest_close - lowest_low],
        axis=1,
    ).max(axis=1)
    working_df.loc[:, "dual_range"] = dual_range
    working_df.loc[:, "dual_buy_line"] = working_df["open"] + (dual_range * k1)
    working_df.loc[:, "dual_sell_line"] = working_df["open"] - (dual_range * k2)
    return working_df


def build_dual_thrust_signal_frame(
    candles_dataframe: Any,
    period: int | None = None,
    k1: float | None = None,
    k2: float | None = None,
) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    resolved_period = settings.strategy.dual_thrust_period if period is None else int(period)
    resolved_k1 = settings.strategy.dual_thrust_k1 if k1 is None else float(k1)
    resolved_k2 = settings.strategy.dual_thrust_k2 if k2 is None else float(k2)
    required_signal_columns = {
        "dual_long_entry",
        "dual_short_entry",
        "dual_long_exit",
        "dual_short_exit",
    }
    if required_signal_columns.issubset(working_df.columns):
        return working_df

    working_df = apply_dual_thrust(
        working_df,
        n=resolved_period,
        k1=resolved_k1,
        k2=resolved_k2,
    )
    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    previous_close = close_series.shift(1)
    previous_buy_line = working_df["dual_buy_line"].shift(1).fillna(working_df["dual_buy_line"])
    previous_sell_line = working_df["dual_sell_line"].shift(1).fillna(working_df["dual_sell_line"])
    buy_line = working_df["dual_buy_line"]
    sell_line = working_df["dual_sell_line"]
    line_ready = buy_line.notna() & sell_line.notna()
    working_df.loc[:, "dual_long_entry"] = (
        line_ready
        & previous_close.le(previous_buy_line)
        & close_series.gt(buy_line)
    ).fillna(False)
    working_df.loc[:, "dual_short_entry"] = (
        line_ready
        & previous_close.ge(previous_sell_line)
        & close_series.lt(sell_line)
    ).fillna(False)
    working_df.loc[:, "dual_long_exit"] = (line_ready & close_series.lt(sell_line)).fillna(False)
    working_df.loc[:, "dual_short_exit"] = (line_ready & close_series.gt(buy_line)).fillna(False)
    return working_df


def run_python_dual_thrust(
    candles_dataframe: Any,
    period: int | None = None,
    k1: float | None = None,
    k2: float | None = None,
) -> int:
    working_df = _ensure_dataframe(candles_dataframe)
    resolved_period = settings.strategy.dual_thrust_period if period is None else int(period)
    resolved_k1 = settings.strategy.dual_thrust_k1 if k1 is None else float(k1)
    resolved_k2 = settings.strategy.dual_thrust_k2 if k2 is None else float(k2)
    if len(working_df) < resolved_period + 2:
        return 0

    working_df = build_dual_thrust_signal_frame(
        working_df,
        period=resolved_period,
        k1=resolved_k1,
        k2=resolved_k2,
    )
    current_row = working_df.iloc[-1]
    if bool(current_row["dual_long_entry"]):
        return 1
    if bool(current_row["dual_short_entry"]):
        return -1
    return 0


def run_python_dual_thrust_breakout(candles_dataframe: Any) -> int:
    return run_python_dual_thrust(candles_dataframe)


def should_exit_python_dual_thrust_breakout(candles_dataframe: Any, side: str) -> bool:
    working_df = _ensure_dataframe(candles_dataframe)
    period = settings.strategy.dual_thrust_period
    if len(working_df) < period + 1:
        return False

    working_df = build_dual_thrust_signal_frame(working_df)
    latest_row = working_df.iloc[-1]
    normalized_side = side.upper()
    if normalized_side == "LONG":
        return bool(latest_row["dual_long_exit"])
    if normalized_side == "SHORT":
        return bool(latest_row["dual_short_exit"])
    return False


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)
