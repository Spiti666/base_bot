from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    result = np.empty(length, dtype=np.float64)
    if length == 0:
        return result
    if period <= 1:
        for index in range(length):
            value = values[index]
            if np.isnan(value):
                value = 0.0
            result[index] = value
        return result
    alpha = 2.0 / (float(period) + 1.0)
    first_value = values[0]
    if np.isnan(first_value):
        first_value = 0.0
    result[0] = first_value
    for index in range(1, length):
        value = values[index]
        if np.isnan(value):
            value = result[index - 1]
        result[index] = (value - result[index - 1]) * alpha + result[index - 1]
    return result


@njit(cache=True, fastmath=True)
def _rolling_sma(values: np.ndarray, window: int) -> np.ndarray:
    length = values.shape[0]
    result = np.empty(length, dtype=np.float64)
    if length == 0:
        return result
    if window <= 1:
        for index in range(length):
            value = values[index]
            result[index] = 0.0 if np.isnan(value) else value
        return result
    running_sum = 0.0
    for index in range(length):
        value = values[index]
        if np.isnan(value):
            value = 0.0
        running_sum += value
        if index >= window:
            old_value = values[index - window]
            if np.isnan(old_value):
                old_value = 0.0
            running_sum -= old_value
        count = window if index + 1 >= window else index + 1
        result[index] = running_sum / float(count)
    return result


@njit(cache=True, fastmath=True)
def _compute_rsi_wilder(close_arr: np.ndarray, period: int) -> np.ndarray:
    length = close_arr.shape[0]
    rsi = np.empty(length, dtype=np.float64)
    for index in range(length):
        rsi[index] = 50.0
    if length == 0:
        return rsi
    if period < 2:
        period = 2
    if length <= period:
        return rsi

    gains = np.zeros(length, dtype=np.float64)
    losses = np.zeros(length, dtype=np.float64)
    for index in range(1, length):
        delta = close_arr[index] - close_arr[index - 1]
        if delta > 0.0:
            gains[index] = delta
        elif delta < 0.0:
            losses[index] = -delta

    avg_gain = 0.0
    avg_loss = 0.0
    for index in range(1, period + 1):
        avg_gain += gains[index]
        avg_loss += losses[index]
    avg_gain /= float(period)
    avg_loss /= float(period)

    if avg_loss <= 0.0:
        rsi[period] = 50.0 if avg_gain <= 0.0 else 100.0
    else:
        relative_strength = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + relative_strength))

    for index in range(period + 1, length):
        avg_gain = ((avg_gain * float(period - 1)) + gains[index]) / float(period)
        avg_loss = ((avg_loss * float(period - 1)) + losses[index]) / float(period)
        if avg_loss <= 0.0:
            rsi[index] = 50.0 if avg_gain <= 0.0 else 100.0
        else:
            relative_strength = avg_gain / avg_loss
            rsi[index] = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi


@njit(cache=True, fastmath=True)
def _sanitize_ohlcv(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = close_arr.shape[0]
    open_out = np.empty(length, dtype=np.float64)
    high_out = np.empty(length, dtype=np.float64)
    low_out = np.empty(length, dtype=np.float64)
    close_out = np.empty(length, dtype=np.float64)
    volume_out = np.empty(length, dtype=np.float64)
    if length == 0:
        return open_out, high_out, low_out, close_out, volume_out

    for index in range(length):
        close_value = close_arr[index]
        if np.isnan(close_value) or close_value <= 0.0:
            close_value = close_out[index - 1] if index > 0 else 1.0
        if close_value <= 0.0:
            close_value = 1.0
        close_out[index] = close_value

        open_value = open_arr[index]
        if np.isnan(open_value) or open_value <= 0.0:
            open_value = close_value
        open_out[index] = open_value

        high_value = high_arr[index]
        if np.isnan(high_value) or high_value <= 0.0:
            high_value = close_value
        low_value = low_arr[index]
        if np.isnan(low_value) or low_value <= 0.0:
            low_value = close_value
        if high_value < low_value:
            tmp = high_value
            high_value = low_value
            low_value = tmp
        high_out[index] = high_value
        low_out[index] = low_value

        volume_value = volume_arr[index]
        if np.isnan(volume_value) or volume_value <= 0.0:
            volume_value = close_value
        volume_out[index] = volume_value
    return open_out, high_out, low_out, close_out, volume_out


@njit(cache=True, fastmath=True)
def compute_ema_cross_volume_signals(
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    ema_fast_period: int,
    ema_slow_period: int,
    volume_period: int,
    volume_multiplier: float,
) -> np.ndarray:
    length = close_arr.shape[0]
    signals = np.zeros(length, dtype=np.int8)
    if length == 0:
        return signals

    fast = ema_series(close_arr, ema_fast_period)
    slow = ema_series(close_arr, ema_slow_period)
    volume_sma = _rolling_sma(volume_arr, volume_period)
    threshold = volume_sma * volume_multiplier

    recent_spike = np.zeros(length, dtype=np.bool_)
    for index in range(length):
        spike_now = volume_arr[index] > threshold[index]
        if index == 0:
            recent_spike[index] = spike_now
            continue
        if index == 1:
            recent_spike[index] = spike_now or (volume_arr[index - 1] > threshold[index - 1])
            continue
        recent_spike[index] = (
            spike_now
            or (volume_arr[index - 1] > threshold[index - 1])
            or (volume_arr[index - 2] > threshold[index - 2])
        )

    for index in range(1, length):
        if not recent_spike[index]:
            continue
        if fast[index - 1] <= slow[index - 1] and fast[index] > slow[index]:
            signals[index] = np.int8(1)
        elif fast[index - 1] >= slow[index - 1] and fast[index] < slow[index]:
            signals[index] = np.int8(-1)
    return signals


@njit(cache=True, fastmath=True)
def frama_series(
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    period: int,
) -> np.ndarray:
    eps = 1e-12
    min_alpha = 0.01
    max_alpha = 1.0
    length = close_arr.shape[0]
    result = np.empty(length, dtype=np.float64)
    if length == 0:
        return result
    first_close = close_arr[0]
    if np.isnan(first_close) or first_close <= 0.0:
        first_close = 1.0
    result[0] = first_close
    if period < 2:
        for index in range(1, length):
            value = close_arr[index]
            if np.isnan(value) or value <= 0.0:
                value = result[index - 1]
            result[index] = value
        return result

    half_period = period // 2
    if half_period < 1:
        half_period = 1
    log_two = np.log(2.0 + eps)

    for index in range(1, length):
        close_value = close_arr[index]
        if np.isnan(close_value) or close_value <= 0.0:
            close_value = result[index - 1]
        if close_value <= 0.0:
            close_value = first_close
        previous = result[index - 1]
        if np.isnan(previous) or previous <= 0.0:
            previous = close_value
        if index < period - 1:
            result[index] = close_value
            continue

        window_start = index - period + 1
        middle = window_start + half_period
        if middle > index:
            middle = index
        if middle <= window_start:
            middle = window_start + 1

        first_high = high_arr[window_start]
        first_low = low_arr[window_start]
        if np.isnan(first_high) or first_high <= 0.0:
            first_high = close_value
        if np.isnan(first_low) or first_low <= 0.0:
            first_low = close_value
        for cursor in range(window_start + 1, middle):
            high_value = high_arr[cursor]
            low_value = low_arr[cursor]
            if np.isnan(high_value) or high_value <= 0.0:
                high_value = close_value
            if np.isnan(low_value) or low_value <= 0.0:
                low_value = close_value
            if high_value > first_high:
                first_high = high_value
            if low_value < first_low:
                first_low = low_value
        first_count = middle - window_start
        if first_count < 1:
            first_count = 1
        n1 = (first_high - first_low) / (float(first_count) + eps)

        second_high = high_arr[middle]
        second_low = low_arr[middle]
        if np.isnan(second_high) or second_high <= 0.0:
            second_high = close_value
        if np.isnan(second_low) or second_low <= 0.0:
            second_low = close_value
        for cursor in range(middle + 1, index + 1):
            high_value = high_arr[cursor]
            low_value = low_arr[cursor]
            if np.isnan(high_value) or high_value <= 0.0:
                high_value = close_value
            if np.isnan(low_value) or low_value <= 0.0:
                low_value = close_value
            if high_value > second_high:
                second_high = high_value
            if low_value < second_low:
                second_low = low_value
        second_count = index - middle + 1
        if second_count < 1:
            second_count = 1
        n2 = (second_high - second_low) / (float(second_count) + eps)

        full_high = high_arr[window_start]
        full_low = low_arr[window_start]
        if np.isnan(full_high) or full_high <= 0.0:
            full_high = close_value
        if np.isnan(full_low) or full_low <= 0.0:
            full_low = close_value
        for cursor in range(window_start + 1, index + 1):
            high_value = high_arr[cursor]
            low_value = low_arr[cursor]
            if np.isnan(high_value) or high_value <= 0.0:
                high_value = close_value
            if np.isnan(low_value) or low_value <= 0.0:
                low_value = close_value
            if high_value > full_high:
                full_high = high_value
            if low_value < full_low:
                full_low = low_value
        full_count = index - window_start + 1
        if full_count < 1:
            full_count = 1
        n3 = (full_high - full_low) / (float(full_count) + eps)

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
            alpha = min_alpha
        elif alpha < min_alpha:
            alpha = min_alpha
        elif alpha > max_alpha:
            alpha = max_alpha

        frama_value = alpha * close_value + (1.0 - alpha) * previous
        if np.isnan(frama_value) or frama_value <= 0.0:
            frama_value = close_value
        result[index] = frama_value

    return result


@njit(cache=True, fastmath=True)
def compute_frama_cross_signals(
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    volume_arr: np.ndarray,
    frama_fast_period: int,
    frama_slow_period: int,
    volume_multiplier: float,
) -> np.ndarray:
    length = close_arr.shape[0]
    signals = np.zeros(length, dtype=np.int8)
    if length == 0:
        return signals

    fast = frama_series(close_arr, high_arr, low_arr, frama_fast_period)
    slow = frama_series(close_arr, high_arr, low_arr, frama_slow_period)
    volume_sma_20 = _rolling_sma(volume_arr, 20)
    warmup = frama_slow_period
    if warmup < 1:
        warmup = 1

    for index in range(1, length):
        if index < warmup:
            continue
        close_value = close_arr[index]
        gap_abs = abs(fast[index] - slow[index])
        min_gap = close_value * 0.0005
        if gap_abs < min_gap:
            continue
        if volume_arr[index] < (volume_sma_20[index] * volume_multiplier):
            continue
        if fast[index - 1] <= slow[index - 1] and fast[index] > slow[index]:
            signals[index] = np.int8(1)
        elif fast[index - 1] >= slow[index - 1] and fast[index] < slow[index]:
            signals[index] = np.int8(-1)
    return signals


@njit(cache=True, fastmath=True)
def dual_thrust_lines(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    period: int,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = close_arr.shape[0]
    buy_line = np.empty(length, dtype=np.float64)
    sell_line = np.empty(length, dtype=np.float64)
    for index in range(length):
        buy_line[index] = np.nan
        sell_line[index] = np.nan
    if length == 0:
        return buy_line, sell_line
    if period < 1:
        period = 1

    for index in range(length):
        if index < period:
            continue
        window_start = index - period
        hh = high_arr[window_start]
        hc = close_arr[window_start]
        lc = close_arr[window_start]
        ll = low_arr[window_start]
        for cursor in range(window_start + 1, index):
            high_value = high_arr[cursor]
            close_value = close_arr[cursor]
            low_value = low_arr[cursor]
            if high_value > hh:
                hh = high_value
            if close_value > hc:
                hc = close_value
            if close_value < lc:
                lc = close_value
            if low_value < ll:
                ll = low_value
        rng = hh - lc
        alt = hc - ll
        if alt > rng:
            rng = alt
        buy_line[index] = open_arr[index] + (rng * k1)
        sell_line[index] = open_arr[index] - (rng * k2)
    return buy_line, sell_line


@njit(cache=True, fastmath=True)
def compute_dual_thrust_signals(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    period: int,
    k1: float,
    k2: float,
) -> np.ndarray:
    length = close_arr.shape[0]
    signals = np.zeros(length, dtype=np.int8)
    if length == 0:
        return signals
    buy_line, sell_line = dual_thrust_lines(open_arr, high_arr, low_arr, close_arr, period, k1, k2)
    for index in range(1, length):
        buy_now = buy_line[index]
        sell_now = sell_line[index]
        buy_prev = buy_line[index - 1]
        sell_prev = sell_line[index - 1]
        if np.isnan(buy_now) or np.isnan(sell_now) or np.isnan(buy_prev) or np.isnan(sell_prev):
            continue
        if close_arr[index - 1] <= buy_prev and close_arr[index] > buy_now:
            signals[index] = np.int8(1)
        elif close_arr[index - 1] >= sell_prev and close_arr[index] < sell_now:
            signals[index] = np.int8(-1)
    return signals


@njit(cache=True, fastmath=True)
def compute_ema_band_rejection_signals(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    ema_fast_period: int,
    ema_mid_period: int,
    ema_slow_period: int,
    slope_lookback: int,
    min_ema_spread_pct: float,
    min_slow_slope_pct: float,
    pullback_requires_outer_band_touch: int,
    use_rejection_quality_filter: int,
    rejection_wick_min_ratio: float,
    rejection_body_min_ratio: float,
    use_rsi_filter: int,
    rsi_length: int,
    rsi_midline: float,
    use_rsi_cross_filter: int,
    rsi_midline_margin: float,
    use_volume_filter: int,
    volume_ma_length: int,
    volume_multiplier: float,
    signal_cooldown_bars: int,
) -> np.ndarray:
    open_s, high_s, low_s, close_s, volume_s = _sanitize_ohlcv(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
    )
    length = close_s.shape[0]
    signals = np.zeros(length, dtype=np.int8)
    if length == 0:
        return signals

    if ema_fast_period < 1:
        ema_fast_period = 1
    if ema_mid_period < 1:
        ema_mid_period = 1
    if ema_slow_period < 2:
        ema_slow_period = 2
    if slope_lookback < 1:
        slope_lookback = 1
    if rsi_length < 2:
        rsi_length = 2
    if volume_ma_length < 1:
        volume_ma_length = 1
    if min_ema_spread_pct < 0.0:
        min_ema_spread_pct = 0.0
    if min_slow_slope_pct < 0.0:
        min_slow_slope_pct = 0.0
    if rejection_wick_min_ratio < 0.0:
        rejection_wick_min_ratio = 0.0
    if rejection_body_min_ratio < 0.0:
        rejection_body_min_ratio = 0.0
    if rsi_midline_margin < 0.0:
        rsi_midline_margin = 0.0
    if volume_multiplier < 1.0:
        volume_multiplier = 1.0
    if signal_cooldown_bars < 0:
        signal_cooldown_bars = 0

    ema_fast = ema_series(close_s, ema_fast_period)
    ema_mid = ema_series(close_s, ema_mid_period)
    ema_slow = ema_series(close_s, ema_slow_period)

    volume_ma = _rolling_sma(volume_s, volume_ma_length)
    rsi_values = _compute_rsi_wilder(close_s, rsi_length)

    setup_long = np.zeros(length, dtype=np.bool_)
    setup_short = np.zeros(length, dtype=np.bool_)
    for index in range(1, length):
        close_value = close_s[index]
        prev_close_value = close_s[index - 1]
        if close_value <= 0.0 or prev_close_value <= 0.0:
            continue

        ema_fast_now = ema_fast[index]
        ema_mid_now = ema_mid[index]
        ema_slow_now = ema_slow[index]
        ema_fast_prev = ema_fast[index - 1]
        ema_slow_prev = ema_slow[index - 1]

        zone_low_now = ema_fast_now if ema_fast_now < ema_slow_now else ema_slow_now
        zone_high_now = ema_slow_now if ema_fast_now < ema_slow_now else ema_fast_now
        zone_low_prev = ema_fast_prev if ema_fast_prev < ema_slow_prev else ema_slow_prev
        zone_high_prev = ema_slow_prev if ema_fast_prev < ema_slow_prev else ema_fast_prev

        in_band_zone = (high_s[index] >= zone_low_now) and (low_s[index] <= zone_high_now)
        came_from_above = prev_close_value > zone_high_prev
        came_from_below = prev_close_value < zone_low_prev
        pullback_long = in_band_zone and came_from_above
        pullback_short = in_band_zone and came_from_below
        if pullback_requires_outer_band_touch > 0:
            pullback_long = pullback_long and (low_s[index] <= ema_slow_now)
            pullback_short = pullback_short and (high_s[index] >= ema_slow_now)

        candle_bullish = close_value > open_s[index]
        candle_bearish = close_value < open_s[index]
        rejection_long = candle_bullish and (close_value > ema_mid_now) and (close_value >= ema_slow_now)
        rejection_short = candle_bearish and (close_value < ema_mid_now) and (close_value <= ema_slow_now)
        if use_rejection_quality_filter > 0:
            candle_range = high_s[index] - low_s[index]
            if candle_range <= 0.0:
                rejection_long = False
                rejection_short = False
            else:
                body = abs(close_value - open_s[index])
                body_ratio = body / candle_range
                upper_wick = high_s[index] - max(open_s[index], close_value)
                lower_wick = min(open_s[index], close_value) - low_s[index]
                long_wick_ratio = lower_wick / candle_range
                short_wick_ratio = upper_wick / candle_range
                rejection_long = (
                    rejection_long
                    and (long_wick_ratio >= rejection_wick_min_ratio)
                    and (body_ratio >= rejection_body_min_ratio)
                )
                rejection_short = (
                    rejection_short
                    and (short_wick_ratio >= rejection_wick_min_ratio)
                    and (body_ratio >= rejection_body_min_ratio)
                )

        slow_slope_pct = 0.0
        if index >= slope_lookback:
            previous_slow = ema_slow[index - slope_lookback]
            if abs(previous_slow) > 0.0:
                slow_slope_pct = ((ema_slow_now - previous_slow) / abs(previous_slow)) * 100.0
        spread_pct = (abs(ema_fast_now - ema_slow_now) / close_value) * 100.0
        spread_ok = spread_pct >= min_ema_spread_pct

        trend_long = (
            (ema_fast_now > ema_mid_now)
            and (ema_mid_now > ema_slow_now)
            and (slow_slope_pct >= min_slow_slope_pct)
            and spread_ok
        )
        trend_short = (
            (ema_fast_now < ema_mid_now)
            and (ema_mid_now < ema_slow_now)
            and (slow_slope_pct <= (-min_slow_slope_pct))
            and spread_ok
        )

        long_rsi_ok = True
        short_rsi_ok = True
        if use_rsi_filter > 0:
            rsi_value = rsi_values[index]
            upper_midline = rsi_midline + rsi_midline_margin
            lower_midline = rsi_midline - rsi_midline_margin
            long_rsi_ok = rsi_value > upper_midline
            short_rsi_ok = rsi_value < lower_midline
            if use_rsi_cross_filter > 0 and index > 0:
                previous_rsi = rsi_values[index - 1]
                long_rsi_ok = long_rsi_ok and (previous_rsi <= lower_midline)
                short_rsi_ok = short_rsi_ok and (previous_rsi >= upper_midline)

        volume_ok = True
        if use_volume_filter > 0:
            volume_ok = volume_s[index] > (volume_ma[index] * volume_multiplier)

        setup_long[index] = trend_long and pullback_long and rejection_long and long_rsi_ok and volume_ok
        setup_short[index] = trend_short and pullback_short and rejection_short and short_rsi_ok and volume_ok

    next_allowed_index = 0
    for index in range(1, length):
        long_now = setup_long[index - 1]
        short_now = setup_short[index - 1]
        candidate_signal = np.int8(0)
        if long_now and not short_now:
            candidate_signal = np.int8(1)
        elif short_now and not long_now:
            candidate_signal = np.int8(-1)
        if candidate_signal == 0:
            continue
        if index < next_allowed_index:
            continue
        signals[index] = candidate_signal
        next_allowed_index = index + signal_cooldown_bars + 1
    return signals


@njit(cache=True, fastmath=True)
def generate_strategy_signals(
    strategy_code: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    param_a: float,
    param_b: float,
    param_c: float,
    volume_period: int,
) -> np.ndarray:
    """
    strategy_code:
    1 -> ema_cross_volume
    2 -> frama_cross
    3 -> dual_thrust
    ema_band_rejection uses dedicated `compute_ema_band_rejection_signals(...)`.
    """
    open_s, high_s, low_s, close_s, volume_s = _sanitize_ohlcv(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
    )
    if strategy_code == 1:
        return compute_ema_cross_volume_signals(
            close_s,
            volume_s,
            int(param_a),
            int(param_b),
            int(volume_period),
            float(param_c),
        )
    if strategy_code == 2:
        return compute_frama_cross_signals(
            close_s,
            high_s,
            low_s,
            volume_s,
            int(param_a),
            int(param_b),
            float(param_c),
        )
    if strategy_code == 3:
        return compute_dual_thrust_signals(
            open_s,
            high_s,
            low_s,
            close_s,
            int(param_a),
            float(param_b),
            float(param_c),
        )
    return np.zeros(close_s.shape[0], dtype=np.int8)
