from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when numba is unavailable
    _NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore[override]
        def _decorator(function):
            return function

        return _decorator


FRAMA_EPSILON = 1e-12
FRAMA_MIN_ALPHA = 0.01
FRAMA_MAX_ALPHA = 1.0
FRAMA_CROSS_MIN_GAP_RATIO = 0.0005
FRAMA_VOLUME_CONFIRM_RATIO = 0.8
FRAMA_VOLUME_SMA_WINDOW = 20


@njit(nopython=True, cache=True)
def _frama_numba(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    period: int,
) -> np.ndarray:
    eps = FRAMA_EPSILON
    result = np.empty(close_values.shape[0], dtype=np.float64)
    length = close_values.shape[0]
    if length == 0:
        return result

    first_close = close_values[0]
    if np.isnan(first_close) or first_close <= 0.0:
        first_close = 1.0
    result[0] = first_close

    if period < 2:
        for index in range(1, length):
            close_price = close_values[index]
            if np.isnan(close_price) or close_price <= 0.0:
                close_price = result[index - 1]
            if np.isnan(close_price) or close_price <= 0.0:
                close_price = first_close
            result[index] = close_price
        return result

    log_two = np.log(2.0 + eps)
    half_period = period // 2
    if half_period < 1:
        half_period = 1

    for index in range(1, length):
        close_price = close_values[index]
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = result[index - 1]
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = first_close

        previous_frama = result[index - 1]
        if np.isnan(previous_frama) or previous_frama <= 0.0:
            previous_frama = close_price

        if index < period - 1:
            result[index] = close_price
            continue

        window_start = index - period + 1
        middle = window_start + half_period
        if middle > index:
            middle = index
        if middle <= window_start:
            middle = window_start + 1

        first_high = high_values[window_start]
        first_low = low_values[window_start]
        if np.isnan(first_high) or first_high <= 0.0:
            first_high = close_values[window_start]
        if np.isnan(first_low) or first_low <= 0.0:
            first_low = close_values[window_start]
        if np.isnan(first_high) or first_high <= 0.0:
            first_high = close_price
        if np.isnan(first_low) or first_low <= 0.0:
            first_low = close_price

        for cursor in range(window_start + 1, middle):
            local_high = high_values[cursor]
            local_low = low_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_values[cursor]
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_price
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_price
            if local_high > first_high:
                first_high = local_high
            if local_low < first_low:
                first_low = local_low

        first_count = middle - window_start
        if first_count < 1:
            first_count = 1
        n1 = (first_high - first_low) / (first_count + eps)

        second_high = high_values[middle]
        second_low = low_values[middle]
        if np.isnan(second_high) or second_high <= 0.0:
            second_high = close_values[middle]
        if np.isnan(second_low) or second_low <= 0.0:
            second_low = close_values[middle]
        if np.isnan(second_high) or second_high <= 0.0:
            second_high = close_price
        if np.isnan(second_low) or second_low <= 0.0:
            second_low = close_price

        for cursor in range(middle + 1, index + 1):
            local_high = high_values[cursor]
            local_low = low_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_values[cursor]
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_price
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_price
            if local_high > second_high:
                second_high = local_high
            if local_low < second_low:
                second_low = local_low

        second_count = index - middle + 1
        if second_count < 1:
            second_count = 1
        n2 = (second_high - second_low) / (second_count + eps)

        full_high = high_values[window_start]
        full_low = low_values[window_start]
        if np.isnan(full_high) or full_high <= 0.0:
            full_high = close_values[window_start]
        if np.isnan(full_low) or full_low <= 0.0:
            full_low = close_values[window_start]
        if np.isnan(full_high) or full_high <= 0.0:
            full_high = close_price
        if np.isnan(full_low) or full_low <= 0.0:
            full_low = close_price

        for cursor in range(window_start + 1, index + 1):
            local_high = high_values[cursor]
            local_low = low_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_values[cursor]
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_values[cursor]
            if np.isnan(local_high) or local_high <= 0.0:
                local_high = close_price
            if np.isnan(local_low) or local_low <= 0.0:
                local_low = close_price
            if local_high > full_high:
                full_high = local_high
            if local_low < full_low:
                full_low = local_low

        full_count = index - window_start + 1
        if full_count < 1:
            full_count = 1
        n3 = (full_high - full_low) / (full_count + eps)

        if np.isnan(n1) or n1 <= 0.0:
            n1 = eps
        if np.isnan(n2) or n2 <= 0.0:
            n2 = eps
        if np.isnan(n3) or n3 <= 0.0:
            n3 = eps

        fractal_dimension = (np.log(n1 + n2 + eps) - np.log(n3 + eps)) / (log_two + eps)
        if np.isnan(fractal_dimension):
            fractal_dimension = 1.0
        if fractal_dimension < 1.0:
            fractal_dimension = 1.0
        elif fractal_dimension > 2.0:
            fractal_dimension = 2.0

        alpha = np.exp(-4.6 * (fractal_dimension - 1.0))
        if np.isnan(alpha) or alpha <= 0.0:
            alpha = FRAMA_MIN_ALPHA
        elif alpha < FRAMA_MIN_ALPHA:
            alpha = FRAMA_MIN_ALPHA
        elif alpha > FRAMA_MAX_ALPHA:
            alpha = FRAMA_MAX_ALPHA

        frama_value = alpha * close_price + (1.0 - alpha) * previous_frama
        if np.isnan(frama_value) or frama_value <= 0.0:
            frama_value = close_price
        result[index] = frama_value

    for index in range(length):
        if np.isnan(result[index]) or result[index] <= 0.0:
            fallback_price = close_values[index]
            if np.isnan(fallback_price) or fallback_price <= 0.0:
                fallback_price = result[index - 1] if index > 0 else first_close
            if np.isnan(fallback_price) or fallback_price <= 0.0:
                fallback_price = first_close
            result[index] = fallback_price
    return result


def _frama_numpy(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    period: int,
) -> np.ndarray:
    eps = FRAMA_EPSILON
    result = np.empty(close_values.shape[0], dtype=np.float64)
    length = close_values.shape[0]
    if length == 0:
        return result

    first_close = float(close_values[0])
    if np.isnan(first_close) or first_close <= 0.0:
        first_close = 1.0
    result[0] = first_close

    if period < 2:
        for index in range(1, length):
            close_price = float(close_values[index])
            if np.isnan(close_price) or close_price <= 0.0:
                close_price = float(result[index - 1])
            if np.isnan(close_price) or close_price <= 0.0:
                close_price = first_close
            result[index] = close_price
        return result

    log_two = float(np.log(2.0 + eps))
    half_period = max(period // 2, 1)

    for index in range(1, length):
        close_price = float(close_values[index])
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = float(result[index - 1])
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = first_close

        previous_frama = float(result[index - 1])
        if np.isnan(previous_frama) or previous_frama <= 0.0:
            previous_frama = close_price

        if index < period - 1:
            result[index] = close_price
            continue

        window_start = index - period + 1
        middle = window_start + half_period
        if middle > index:
            middle = index
        if middle <= window_start:
            middle = window_start + 1

        first_slice = slice(window_start, middle)
        second_slice = slice(middle, index + 1)
        full_slice = slice(window_start, index + 1)

        first_count = max(middle - window_start, 1)
        second_count = max(index - middle + 1, 1)
        full_count = max(index - window_start + 1, 1)

        n1 = float(np.max(high_values[first_slice]) - np.min(low_values[first_slice])) / float(
            first_count + eps
        )
        n2 = float(np.max(high_values[second_slice]) - np.min(low_values[second_slice])) / float(
            second_count + eps
        )
        n3 = float(np.max(high_values[full_slice]) - np.min(low_values[full_slice])) / float(
            full_count + eps
        )

        if np.isnan(n1) or n1 <= 0.0:
            n1 = eps
        if np.isnan(n2) or n2 <= 0.0:
            n2 = eps
        if np.isnan(n3) or n3 <= 0.0:
            n3 = eps

        fractal_dimension = float(
            (np.log(n1 + n2 + eps) - np.log(n3 + eps)) / (log_two + eps)
        )
        if np.isnan(fractal_dimension):
            fractal_dimension = 1.0
        fractal_dimension = float(np.clip(fractal_dimension, 1.0, 2.0))

        alpha = float(np.exp(-4.6 * (fractal_dimension - 1.0)))
        if np.isnan(alpha) or alpha <= 0.0:
            alpha = FRAMA_MIN_ALPHA
        else:
            alpha = float(np.clip(alpha, FRAMA_MIN_ALPHA, FRAMA_MAX_ALPHA))

        frama_value = alpha * close_price + (1.0 - alpha) * previous_frama
        if np.isnan(frama_value) or frama_value <= 0.0:
            frama_value = close_price
        result[index] = frama_value

    for index in range(length):
        if np.isnan(result[index]) or result[index] <= 0.0:
            fallback_price = float(close_values[index])
            if np.isnan(fallback_price) or fallback_price <= 0.0:
                fallback_price = float(result[index - 1]) if index > 0 else first_close
            if np.isnan(fallback_price) or fallback_price <= 0.0:
                fallback_price = first_close
            result[index] = fallback_price
    return result


@njit(nopython=True, cache=True)
def _build_crossover_masks_numba(
    fast_values: np.ndarray,
    slow_values: np.ndarray,
    close_values: np.ndarray,
    volume_values: np.ndarray,
    warmup_bars: int,
    volume_confirm_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = FRAMA_EPSILON
    length = close_values.shape[0]
    volume_sma = np.empty(length, dtype=np.float64)
    long_entries = np.zeros(length, dtype=np.bool_)
    short_entries = np.zeros(length, dtype=np.bool_)
    running_sum = 0.0

    for index in range(length):
        close_price = close_values[index]
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = close_values[index - 1] if index > 0 else 1.0
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = 1.0

        current_volume = volume_values[index]
        if np.isnan(current_volume) or current_volume <= 0.0:
            current_volume = close_price

        running_sum += current_volume
        if index >= FRAMA_VOLUME_SMA_WINDOW:
            previous_volume = volume_values[index - FRAMA_VOLUME_SMA_WINDOW]
            if np.isnan(previous_volume) or previous_volume <= 0.0:
                previous_volume = close_values[index - FRAMA_VOLUME_SMA_WINDOW]
            if np.isnan(previous_volume) or previous_volume <= 0.0:
                previous_volume = current_volume
            running_sum -= previous_volume

        count = FRAMA_VOLUME_SMA_WINDOW if index + 1 >= FRAMA_VOLUME_SMA_WINDOW else index + 1
        sma_value = running_sum / (count + eps)
        if np.isnan(sma_value) or sma_value <= 0.0:
            sma_value = close_price
        volume_sma[index] = sma_value

        if index == 0 or index < warmup_bars:
            continue

        fast_now = fast_values[index]
        slow_now = slow_values[index]
        fast_prev = fast_values[index - 1]
        slow_prev = slow_values[index - 1]
        if (
            np.isnan(fast_now)
            or np.isnan(slow_now)
            or np.isnan(fast_prev)
            or np.isnan(slow_prev)
        ):
            continue

        gap_abs = abs(fast_now - slow_now)
        min_gap = close_price * FRAMA_CROSS_MIN_GAP_RATIO
        if np.isnan(min_gap) or min_gap <= 0.0:
            min_gap = close_price * (FRAMA_CROSS_MIN_GAP_RATIO + eps)
        if gap_abs + eps < min_gap:
            continue

        if current_volume + eps < (sma_value * volume_confirm_ratio):
            continue

        if fast_now > slow_now and fast_prev <= slow_prev:
            long_entries[index] = True
        elif fast_now < slow_now and fast_prev >= slow_prev:
            short_entries[index] = True

    return volume_sma, long_entries, short_entries


def _calculate_frama_series(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    period: int,
) -> np.ndarray:
    normalized_period = max(int(period), 2)
    if _NUMBA_AVAILABLE:
        return _frama_numba(close_values, high_values, low_values, normalized_period)
    return _frama_numpy(close_values, high_values, low_values, normalized_period)


def _sanitize_ohlcv_arrays(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    volume_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sanitized_close = np.asarray(close_values, dtype=np.float64).copy()
    sanitized_high = np.asarray(high_values, dtype=np.float64).copy()
    sanitized_low = np.asarray(low_values, dtype=np.float64).copy()
    sanitized_volume = np.asarray(volume_values, dtype=np.float64).copy()
    length = sanitized_close.shape[0]
    if length == 0:
        return sanitized_close, sanitized_high, sanitized_low, sanitized_volume

    for index in range(length):
        close_price = sanitized_close[index]
        if np.isnan(close_price) or close_price <= 0.0:
            if index > 0:
                close_price = sanitized_close[index - 1]
            else:
                close_price = 1.0
        if np.isnan(close_price) or close_price <= 0.0:
            close_price = 1.0
        sanitized_close[index] = close_price

    for index in range(length):
        close_price = sanitized_close[index]

        high_price = sanitized_high[index]
        if np.isnan(high_price) or high_price <= 0.0:
            high_price = close_price
        low_price = sanitized_low[index]
        if np.isnan(low_price) or low_price <= 0.0:
            low_price = close_price
        if high_price < low_price:
            high_price, low_price = low_price, high_price
        sanitized_high[index] = high_price
        sanitized_low[index] = low_price

        volume_value = sanitized_volume[index]
        if np.isnan(volume_value) or volume_value <= 0.0:
            volume_value = close_price
        if np.isnan(volume_value) or volume_value <= 0.0:
            volume_value = 1.0
        sanitized_volume[index] = volume_value

    return sanitized_close, sanitized_high, sanitized_low, sanitized_volume


def build_frama_cross_signal_frame(candles_dataframe: Any) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    required_columns = {"close", "high", "low", "volume"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"FRAMA Cross requires columns: {missing_list}")

    required_signal_columns = {
        "frama_fast",
        "frama_slow",
        "frama_volume_sma_20",
        "frama_long_entry",
        "frama_short_entry",
    }
    if required_signal_columns.issubset(working_df.columns):
        if not working_df[list(required_signal_columns)].isna().to_numpy().any():
            return working_df

    fast_period = int(settings.strategy.frama_fast_period)
    slow_period = int(settings.strategy.frama_slow_period)
    if fast_period >= slow_period:
        raise ValueError("FRAMA Cross requires frama_fast_period to be smaller than frama_slow_period.")

    close_values = pd.to_numeric(working_df["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    high_values = pd.to_numeric(working_df["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    low_values = pd.to_numeric(working_df["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    volume_values = pd.to_numeric(working_df["volume"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    close_values, high_values, low_values, volume_values = _sanitize_ohlcv_arrays(
        close_values,
        high_values,
        low_values,
        volume_values,
    )

    fast_frama = _calculate_frama_series(close_values, high_values, low_values, fast_period)
    slow_frama = _calculate_frama_series(close_values, high_values, low_values, slow_period)

    warmup_end = min(len(close_values), max(slow_period, 1))
    if warmup_end > 0:
        fast_frama[:warmup_end] = close_values[:warmup_end]
        slow_frama[:warmup_end] = close_values[:warmup_end]

    volume_confirm_ratio = float(settings.strategy.volume_multiplier)
    if not np.isfinite(volume_confirm_ratio) or volume_confirm_ratio <= 0.0:
        volume_confirm_ratio = FRAMA_VOLUME_CONFIRM_RATIO

    volume_sma_20, long_entries, short_entries = _build_crossover_masks_numba(
        fast_frama,
        slow_frama,
        close_values,
        volume_values,
        warmup_end,
        volume_confirm_ratio,
    )

    working_df.loc[:, "frama_fast"] = pd.Series(fast_frama, index=working_df.index)
    working_df.loc[:, "frama_slow"] = pd.Series(slow_frama, index=working_df.index)
    working_df.loc[:, "frama_volume_sma_20"] = pd.Series(volume_sma_20, index=working_df.index)
    working_df.loc[:, "frama_long_entry"] = pd.Series(long_entries, index=working_df.index)
    working_df.loc[:, "frama_short_entry"] = pd.Series(short_entries, index=working_df.index)
    return working_df


def run_python_frama_cross(candles_dataframe: Any) -> int:
    working_df = build_frama_cross_signal_frame(candles_dataframe)
    required_points = max(int(settings.strategy.frama_slow_period) + 1, 3)
    if len(working_df) < required_points:
        return 0

    last_row = working_df.iloc[-1]
    if bool(last_row["frama_long_entry"]):
        return 1
    if bool(last_row["frama_short_entry"]):
        return -1
    return 0


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)
