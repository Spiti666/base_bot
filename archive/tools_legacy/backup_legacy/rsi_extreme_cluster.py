from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import settings


_EPSILON = 1e-9
_DIAGNOSTICS_ATTR_KEY = "rsi_extreme_cluster_diagnostics"


@dataclass(frozen=True, slots=True)
class _ExtremeClusterParams:
    rsi_period: int
    rsi_sma_period: int
    cluster_count: int
    band_points: float
    band_pct: float
    min_cluster_size: int
    reset_on_trade: bool


def extract_rsi_extreme_cluster_diagnostics(candles_dataframe: Any) -> dict[str, object]:
    working_df = build_rsi_extreme_cluster_signal_frame(candles_dataframe)
    diagnostics = working_df.attrs.get(_DIAGNOSTICS_ATTR_KEY)
    if isinstance(diagnostics, dict):
        return dict(diagnostics)

    params = _resolve_params()
    long_entry_count = (
        int(working_df["rsi_extreme_cluster_long_entry"].fillna(False).astype(bool).sum())
        if "rsi_extreme_cluster_long_entry" in working_df.columns
        else 0
    )
    short_entry_count = (
        int(working_df["rsi_extreme_cluster_short_entry"].fillna(False).astype(bool).sum())
        if "rsi_extreme_cluster_short_entry" in working_df.columns
        else 0
    )
    return {
        "rsi_period": int(params.rsi_period),
        "rsi_sma_period": int(params.rsi_sma_period),
        "cluster_count": int(params.cluster_count),
        "band_points": float(params.band_points),
        "band_pct": float(params.band_pct),
        "min_cluster_size": int(params.min_cluster_size),
        "reset_on_trade": bool(params.reset_on_trade),
        "long_entry_count": int(long_entry_count),
        "short_entry_count": int(short_entry_count),
        "valid_cluster_events": "unavailable",
        "rejected_out_of_band_values": "unavailable",
        "reset_count": "unavailable",
    }


def build_rsi_extreme_cluster_signal_frame(candles_dataframe: Any) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    required_columns = {"close"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"RSI Extreme Cluster requires columns: {missing_list}")

    required_signal_columns = {
        "rsi_extreme_cluster_value",
        "rsi_extreme_cluster_short_avg",
        "rsi_extreme_cluster_long_avg",
        "rsi_extreme_cluster_short_entry",
        "rsi_extreme_cluster_long_entry",
    }
    if required_signal_columns.issubset(working_df.columns):
        return working_df

    params = _resolve_params()
    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    rsi_series = _calculate_rsi_series(close_series, period=params.rsi_period)
    cluster_series = rsi_series.rolling(
        window=params.rsi_sma_period,
        min_periods=params.rsi_sma_period,
    ).mean()

    value_array = cluster_series.to_numpy(dtype=np.float64, copy=False)
    row_count = value_array.shape[0]
    short_entry_flags = np.zeros(row_count, dtype=bool)
    long_entry_flags = np.zeros(row_count, dtype=bool)
    short_avg_values = np.full(row_count, np.nan, dtype=np.float64)
    long_avg_values = np.full(row_count, np.nan, dtype=np.float64)
    high_anchor_values = np.full(row_count, np.nan, dtype=np.float64)
    low_anchor_values = np.full(row_count, np.nan, dtype=np.float64)

    high_anchor: float | None = None
    low_anchor: float | None = None
    short_cluster_values: list[float] = []
    long_cluster_values: list[float] = []
    long_entry_count = 0
    short_entry_count = 0
    valid_cluster_events = 0
    rejected_out_of_band_values = 0
    reset_count = 0

    for index, current_value_raw in enumerate(value_array):
        if not np.isfinite(current_value_raw):
            continue
        current_value = float(current_value_raw)
        previous_value = float(value_array[index - 1]) if index > 0 else np.nan

        previous_high_anchor = high_anchor
        previous_low_anchor = low_anchor
        if high_anchor is None or current_value > high_anchor:
            high_anchor = current_value
            if (
                previous_high_anchor is not None
                and current_value > previous_high_anchor
                and low_anchor is not None
                and current_value - low_anchor >= _counter_extreme_distance(low_anchor, params)
            ):
                low_anchor = None
                long_cluster_values.clear()
                reset_count += 1
        if low_anchor is None or current_value < low_anchor:
            low_anchor = current_value
            if (
                previous_low_anchor is not None
                and current_value < previous_low_anchor
                and high_anchor is not None
                and high_anchor - current_value >= _counter_extreme_distance(high_anchor, params)
            ):
                high_anchor = None
                short_cluster_values.clear()
                reset_count += 1

        if high_anchor is not None:
            if _is_short_value_in_band(current_value, high_anchor, params):
                short_cluster_values.append(current_value)
                valid_cluster_events += 1
            else:
                rejected_out_of_band_values += 1
        if low_anchor is not None:
            if _is_long_value_in_band(current_value, low_anchor, params):
                long_cluster_values.append(current_value)
                valid_cluster_events += 1
            else:
                rejected_out_of_band_values += 1

        short_cluster_values = _trim_short_cluster_values(
            short_cluster_values,
            high_anchor=high_anchor,
            params=params,
        )
        long_cluster_values = _trim_long_cluster_values(
            long_cluster_values,
            low_anchor=low_anchor,
            params=params,
        )

        if high_anchor is not None:
            high_anchor_values[index] = high_anchor
        if low_anchor is not None:
            low_anchor_values[index] = low_anchor

        short_avg = _calculate_short_weighted_average(
            short_cluster_values,
            high_anchor=high_anchor,
            params=params,
        )
        long_avg = _calculate_long_weighted_average(
            long_cluster_values,
            low_anchor=low_anchor,
            params=params,
        )
        if short_avg is not None:
            short_avg_values[index] = short_avg
        if long_avg is not None:
            long_avg_values[index] = long_avg

        short_signal = (
            index > 0
            and np.isfinite(previous_value)
            and short_avg is not None
            and len(short_cluster_values) >= params.min_cluster_size
            and previous_value > short_avg
            and current_value <= short_avg
        )
        long_signal = (
            index > 0
            and np.isfinite(previous_value)
            and long_avg is not None
            and len(long_cluster_values) >= params.min_cluster_size
            and previous_value < long_avg
            and current_value >= long_avg
        )

        if short_signal:
            short_entry_flags[index] = True
            short_entry_count += 1
            reset_count += 1
            if params.reset_on_trade:
                high_anchor = None
                low_anchor = None
            short_cluster_values.clear()
            long_cluster_values.clear()
            continue
        if long_signal:
            long_entry_flags[index] = True
            long_entry_count += 1
            reset_count += 1
            if params.reset_on_trade:
                high_anchor = None
                low_anchor = None
            short_cluster_values.clear()
            long_cluster_values.clear()
            continue

    working_df.loc[:, "rsi_extreme_cluster_rsi"] = rsi_series
    working_df.loc[:, "rsi_extreme_cluster_value"] = pd.Series(
        value_array,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_short_avg"] = pd.Series(
        short_avg_values,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_long_avg"] = pd.Series(
        long_avg_values,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_high_anchor"] = pd.Series(
        high_anchor_values,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_low_anchor"] = pd.Series(
        low_anchor_values,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_short_entry"] = pd.Series(
        short_entry_flags,
        index=working_df.index,
    )
    working_df.loc[:, "rsi_extreme_cluster_long_entry"] = pd.Series(
        long_entry_flags,
        index=working_df.index,
    )
    working_df.attrs[_DIAGNOSTICS_ATTR_KEY] = {
        "rsi_period": int(params.rsi_period),
        "rsi_sma_period": int(params.rsi_sma_period),
        "cluster_count": int(params.cluster_count),
        "band_points": float(params.band_points),
        "band_pct": float(params.band_pct),
        "min_cluster_size": int(params.min_cluster_size),
        "reset_on_trade": bool(params.reset_on_trade),
        "long_entry_count": int(long_entry_count),
        "short_entry_count": int(short_entry_count),
        "valid_cluster_events": int(valid_cluster_events),
        "rejected_out_of_band_values": int(rejected_out_of_band_values),
        "reset_count": int(reset_count),
    }
    return working_df


def run_python_rsi_extreme_cluster(candles_dataframe: Any) -> int:
    working_df = build_rsi_extreme_cluster_signal_frame(candles_dataframe)
    params = _resolve_params()
    required_points = max(
        params.rsi_period + params.rsi_sma_period,
        params.min_cluster_size + 2,
    )
    if len(working_df) < required_points:
        return 0

    last_row = working_df.iloc[-1]
    if bool(last_row["rsi_extreme_cluster_long_entry"]):
        return 1
    if bool(last_row["rsi_extreme_cluster_short_entry"]):
        return -1
    return 0


def _resolve_params() -> _ExtremeClusterParams:
    params = _ExtremeClusterParams(
        rsi_period=int(settings.strategy.rsi_period),
        rsi_sma_period=int(settings.strategy.rsi_sma_period),
        cluster_count=int(settings.strategy.cluster_count),
        band_points=float(settings.strategy.band_points),
        band_pct=float(settings.strategy.band_pct),
        min_cluster_size=int(settings.strategy.min_cluster_size),
        reset_on_trade=bool(settings.strategy.reset_on_trade),
    )
    _validate_params(params)
    return params


def _validate_params(params: _ExtremeClusterParams) -> None:
    if params.rsi_period <= 0:
        raise ValueError("rsi_period must be > 0 for rsi_extreme_cluster.")
    if params.rsi_sma_period <= 0:
        raise ValueError("rsi_sma_period must be > 0 for rsi_extreme_cluster.")
    if params.cluster_count <= 0:
        raise ValueError("cluster_count must be > 0 for rsi_extreme_cluster.")
    if params.band_points <= 0.0:
        raise ValueError("band_points must be > 0 for rsi_extreme_cluster.")
    if params.band_pct < 0.0:
        raise ValueError("band_pct must be >= 0 for rsi_extreme_cluster.")
    if params.min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be > 0 for rsi_extreme_cluster.")
    if params.min_cluster_size > params.cluster_count:
        raise ValueError(
            "min_cluster_size must be <= cluster_count for rsi_extreme_cluster."
        )


def _counter_extreme_distance(anchor: float, params: _ExtremeClusterParams) -> float:
    anchor_distance = abs(anchor) * (params.band_pct / 100.0)
    return max(params.band_points, anchor_distance)


def _is_short_value_in_band(
    value: float,
    high_anchor: float,
    params: _ExtremeClusterParams,
) -> bool:
    if value > high_anchor + _EPSILON:
        return False
    points_distance = high_anchor - value
    if points_distance > params.band_points + _EPSILON:
        return False
    pct_distance = (points_distance / max(abs(high_anchor), _EPSILON)) * 100.0
    return pct_distance <= params.band_pct + _EPSILON


def _is_long_value_in_band(
    value: float,
    low_anchor: float,
    params: _ExtremeClusterParams,
) -> bool:
    if value < low_anchor - _EPSILON:
        return False
    points_distance = value - low_anchor
    if points_distance > params.band_points + _EPSILON:
        return False
    pct_distance = (points_distance / max(abs(low_anchor), _EPSILON)) * 100.0
    return pct_distance <= params.band_pct + _EPSILON


def _trim_short_cluster_values(
    values: list[float],
    *,
    high_anchor: float | None,
    params: _ExtremeClusterParams,
) -> list[float]:
    if high_anchor is None:
        return []
    filtered_values = [
        float(value)
        for value in values
        if _is_short_value_in_band(float(value), high_anchor, params)
    ]
    filtered_values.sort(reverse=True)
    return filtered_values[: params.cluster_count]


def _trim_long_cluster_values(
    values: list[float],
    *,
    low_anchor: float | None,
    params: _ExtremeClusterParams,
) -> list[float]:
    if low_anchor is None:
        return []
    filtered_values = [
        float(value)
        for value in values
        if _is_long_value_in_band(float(value), low_anchor, params)
    ]
    filtered_values.sort()
    return filtered_values[: params.cluster_count]


def _calculate_short_weighted_average(
    values: list[float],
    *,
    high_anchor: float | None,
    params: _ExtremeClusterParams,
) -> float | None:
    if high_anchor is None or not values:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for value in values:
        distance_points = max(0.0, high_anchor - float(value))
        weight = max(0.0, 1.0 - (distance_points / params.band_points))
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _calculate_long_weighted_average(
    values: list[float],
    *,
    low_anchor: float | None,
    params: _ExtremeClusterParams,
) -> float | None:
    if low_anchor is None or not values:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for value in values:
        distance_points = max(0.0, float(value) - low_anchor)
        weight = max(0.0, 1.0 - (distance_points / params.band_points))
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _calculate_rsi_series(close_series: pd.Series, *, period: int) -> pd.Series:
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()

    gain_values = average_gain.to_numpy(dtype=np.float64, copy=False)
    loss_values = average_loss.to_numpy(dtype=np.float64, copy=False)
    rsi_values = np.full(close_series.shape[0], np.nan, dtype=np.float64)

    for index in range(close_series.shape[0]):
        gain_value = gain_values[index]
        loss_value = loss_values[index]
        if not np.isfinite(gain_value) or not np.isfinite(loss_value):
            continue
        if loss_value <= _EPSILON and gain_value <= _EPSILON:
            rsi_values[index] = 50.0
            continue
        if loss_value <= _EPSILON:
            rsi_values[index] = 100.0
            continue
        rs = gain_value / loss_value
        rsi_values[index] = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi_values, index=close_series.index)


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)
