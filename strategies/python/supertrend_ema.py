from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings


def build_supertrend_ema_signal_frame(candles_dataframe: Any) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    required_columns = {"close", "high", "low"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"SuperTrend + EMA requires columns: {missing_list}")

    required_signal_columns = {
        "supertrend_ema_ema_value",
        "supertrend_ema_line",
        "supertrend_ema_direction",
        "supertrend_ema_long_entry",
        "supertrend_ema_short_entry",
        "supertrend_ema_exit_signal",
        "supertrend_ema_long_exit",
        "supertrend_ema_short_exit",
        "supertrend_ema_dynamic_stop_loss_pct",
        "supertrend_ema_dynamic_take_profit_pct",
    }
    if required_signal_columns.issubset(working_df.columns):
        return working_df

    supertrend_length = int(settings.strategy.supertrend_ema_supertrend_length)
    supertrend_multiplier = float(settings.strategy.supertrend_ema_supertrend_multiplier)
    ema_length = int(settings.strategy.supertrend_ema_ema_length)
    _validate_params(
        supertrend_length=supertrend_length,
        supertrend_multiplier=supertrend_multiplier,
        ema_length=ema_length,
    )

    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    high_series = pd.to_numeric(working_df["high"], errors="coerce")
    low_series = pd.to_numeric(working_df["low"], errors="coerce")

    ema_series = close_series.ewm(span=ema_length, adjust=False).mean()
    supertrend_line, direction_series = _calculate_supertrend(
        high_series=high_series,
        low_series=low_series,
        close_series=close_series,
        length=supertrend_length,
        multiplier=supertrend_multiplier,
    )
    previous_direction = direction_series.shift(1)
    flip_up = previous_direction.lt(0) & direction_series.gt(0)
    flip_down = previous_direction.gt(0) & direction_series.lt(0)

    long_entry = (flip_up & close_series.gt(ema_series)).fillna(False)
    short_entry = (flip_down & close_series.lt(ema_series)).fillna(False)

    long_exit = flip_down.fillna(False)
    short_exit = flip_up.fillna(False)

    exit_signal = pd.Series(0, index=working_df.index, dtype="int8")
    exit_signal = exit_signal.mask(long_exit, 1)
    exit_signal = exit_signal.mask(short_exit, -1)

    long_stop_loss_pct = ((close_series - supertrend_line) / close_series * 100.0).where(long_entry)
    short_stop_loss_pct = ((supertrend_line - close_series) / close_series * 100.0).where(short_entry)
    dynamic_stop_loss_pct = pd.to_numeric(
        long_stop_loss_pct.combine_first(short_stop_loss_pct),
        errors="coerce",
    ).where(lambda values: values.gt(0.0))
    dynamic_take_profit_pct = dynamic_stop_loss_pct * 2.0

    working_df.loc[:, "supertrend_ema_ema_value"] = ema_series
    working_df.loc[:, "supertrend_ema_line"] = supertrend_line
    working_df.loc[:, "supertrend_ema_direction"] = direction_series
    working_df.loc[:, "supertrend_ema_long_entry"] = long_entry
    working_df.loc[:, "supertrend_ema_short_entry"] = short_entry
    working_df.loc[:, "supertrend_ema_exit_signal"] = exit_signal
    working_df.loc[:, "supertrend_ema_long_exit"] = long_exit
    working_df.loc[:, "supertrend_ema_short_exit"] = short_exit
    working_df.loc[:, "supertrend_ema_dynamic_stop_loss_pct"] = dynamic_stop_loss_pct
    working_df.loc[:, "supertrend_ema_dynamic_take_profit_pct"] = dynamic_take_profit_pct
    return working_df


def run_python_supertrend_ema(candles_dataframe: Any) -> int:
    working_df = build_supertrend_ema_signal_frame(candles_dataframe)
    required_points = max(
        int(settings.strategy.supertrend_ema_supertrend_length) + 3,
        int(settings.strategy.supertrend_ema_ema_length) + 2,
        6,
    )
    if len(working_df) < required_points:
        return 0

    last_row = working_df.iloc[-1]
    if bool(last_row["supertrend_ema_long_entry"]):
        return 1
    if bool(last_row["supertrend_ema_short_entry"]):
        return -1
    return 0


def should_exit_python_supertrend_ema(candles_dataframe: Any, side: str) -> bool:
    working_df = build_supertrend_ema_signal_frame(candles_dataframe)
    if working_df.empty:
        return False
    latest_row = working_df.iloc[-1]
    normalized_side = str(side).strip().upper()
    if normalized_side == "LONG":
        return int(latest_row["supertrend_ema_exit_signal"]) == 1
    if normalized_side == "SHORT":
        return int(latest_row["supertrend_ema_exit_signal"]) == -1
    return False


def _validate_params(
    *,
    supertrend_length: int,
    supertrend_multiplier: float,
    ema_length: int,
) -> None:
    if supertrend_length <= 1:
        raise ValueError("supertrend_ema_supertrend_length must be greater than 1.")
    if supertrend_multiplier <= 0.0:
        raise ValueError("supertrend_ema_supertrend_multiplier must be positive.")
    if ema_length <= 1:
        raise ValueError("supertrend_ema_ema_length must be greater than 1.")


def _calculate_true_range(
    *,
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
) -> pd.Series:
    previous_close = close_series.shift(1)
    return pd.concat(
        [
            (high_series - low_series).abs(),
            (high_series - previous_close).abs(),
            (low_series - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _calculate_supertrend(
    *,
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    length: int,
    multiplier: float,
) -> tuple[pd.Series, pd.Series]:
    length = max(int(length), 1)
    true_range = _calculate_true_range(
        high_series=high_series,
        low_series=low_series,
        close_series=close_series,
    )
    atr = true_range.ewm(alpha=1.0 / float(length), adjust=False, min_periods=length).mean()
    hl2 = (high_series + low_series) * 0.5
    basic_upper = hl2 + (float(multiplier) * atr)
    basic_lower = hl2 - (float(multiplier) * atr)

    row_count = len(close_series)
    final_upper = np.full(row_count, np.nan, dtype="float64")
    final_lower = np.full(row_count, np.nan, dtype="float64")
    supertrend = np.full(row_count, np.nan, dtype="float64")
    direction = np.zeros(row_count, dtype="int8")
    close_values = close_series.to_numpy(dtype="float64", copy=False)
    upper_values = basic_upper.to_numpy(dtype="float64", copy=False)
    lower_values = basic_lower.to_numpy(dtype="float64", copy=False)

    for index in range(row_count):
        current_close = close_values[index]
        current_upper = upper_values[index]
        current_lower = lower_values[index]

        if index == 0:
            final_upper[index] = current_upper
            final_lower[index] = current_lower
            if np.isnan(current_upper) and np.isnan(current_lower):
                direction[index] = 0
                supertrend[index] = np.nan
            elif np.isnan(current_upper):
                direction[index] = 1
                supertrend[index] = current_lower
            else:
                direction[index] = -1
                supertrend[index] = current_upper
            continue

        previous_upper = final_upper[index - 1]
        previous_lower = final_lower[index - 1]
        previous_close_value = close_values[index - 1]

        if np.isnan(current_upper):
            final_upper[index] = previous_upper
        elif (
            np.isnan(previous_upper)
            or current_upper < previous_upper
            or (not np.isnan(previous_close_value) and previous_close_value > previous_upper)
        ):
            final_upper[index] = current_upper
        else:
            final_upper[index] = previous_upper

        if np.isnan(current_lower):
            final_lower[index] = previous_lower
        elif (
            np.isnan(previous_lower)
            or current_lower > previous_lower
            or (not np.isnan(previous_close_value) and previous_close_value < previous_lower)
        ):
            final_lower[index] = current_lower
        else:
            final_lower[index] = previous_lower

        previous_direction = int(direction[index - 1])
        if np.isnan(final_upper[index]) and np.isnan(final_lower[index]):
            direction[index] = previous_direction
            supertrend[index] = supertrend[index - 1]
            continue
        if np.isnan(current_close):
            direction[index] = previous_direction
            supertrend[index] = supertrend[index - 1]
            continue

        if previous_direction <= 0:
            if np.isnan(final_upper[index]):
                direction[index] = 1
                supertrend[index] = final_lower[index]
            elif current_close <= final_upper[index]:
                direction[index] = -1
                supertrend[index] = final_upper[index]
            else:
                direction[index] = 1
                supertrend[index] = final_lower[index]
        else:
            if np.isnan(final_lower[index]):
                direction[index] = -1
                supertrend[index] = final_upper[index]
            elif current_close >= final_lower[index]:
                direction[index] = 1
                supertrend[index] = final_lower[index]
            else:
                direction[index] = -1
                supertrend[index] = final_upper[index]

    return (
        pd.Series(supertrend, index=close_series.index, dtype="float64"),
        pd.Series(direction, index=close_series.index, dtype="int8"),
    )


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)
