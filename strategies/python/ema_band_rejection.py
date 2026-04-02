from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EMA_FAST = 5
DEFAULT_EMA_MID = 10
DEFAULT_EMA_SLOW = 20
DEFAULT_SLOPE_LOOKBACK = 5
DEFAULT_MIN_EMA_SPREAD_PCT = 0.05
DEFAULT_MIN_SLOW_SLOPE_PCT = 0.0
DEFAULT_PULLBACK_REQUIRES_OUTER_BAND_TOUCH = False
DEFAULT_USE_REJECTION_QUALITY_FILTER = False
DEFAULT_REJECTION_WICK_MIN_RATIO = 0.35
DEFAULT_REJECTION_BODY_MIN_RATIO = 0.20
DEFAULT_USE_RSI_FILTER = False
DEFAULT_RSI_LENGTH = 14
DEFAULT_RSI_MIDLINE = 50.0
DEFAULT_USE_RSI_CROSS_FILTER = False
DEFAULT_RSI_MIDLINE_MARGIN = 0.0
DEFAULT_USE_VOLUME_FILTER = False
DEFAULT_VOLUME_MA_LENGTH = 20
DEFAULT_VOLUME_MULTIPLIER = 1.0
DEFAULT_USE_ATR_STOP_BUFFER = False
DEFAULT_ATR_LENGTH = 14
DEFAULT_ATR_STOP_BUFFER_MULT = 0.5
DEFAULT_SIGNAL_COOLDOWN_BARS = 0


def _ensure_dataframe(candles_dataframe: Any) -> pd.DataFrame:
    if isinstance(candles_dataframe, pd.DataFrame):
        return candles_dataframe
    return pd.DataFrame(candles_dataframe)


def _calculate_rsi(close_series: pd.Series, length: int) -> pd.Series:
    resolved_length = max(2, int(length))
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(resolved_length)
    avg_gains = gains.ewm(alpha=alpha, adjust=False, min_periods=resolved_length).mean()
    avg_losses = losses.ewm(alpha=alpha, adjust=False, min_periods=resolved_length).mean()
    relative_strength = avg_gains / avg_losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.fillna(50.0)


def _calculate_atr(
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    length: int,
) -> pd.Series:
    resolved_length = max(2, int(length))
    previous_close = close_series.shift(1)
    true_range = pd.concat(
        [
            (high_series - low_series).abs(),
            (high_series - previous_close).abs(),
            (low_series - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=resolved_length, min_periods=1).mean()


def build_ema_band_rejection_signal_frame(
    candles_dataframe: Any,
    *,
    ema_fast: int = DEFAULT_EMA_FAST,
    ema_mid: int = DEFAULT_EMA_MID,
    ema_slow: int = DEFAULT_EMA_SLOW,
    slope_lookback: int = DEFAULT_SLOPE_LOOKBACK,
    min_ema_spread_pct: float = DEFAULT_MIN_EMA_SPREAD_PCT,
    min_slow_slope_pct: float = DEFAULT_MIN_SLOW_SLOPE_PCT,
    pullback_requires_outer_band_touch: bool = DEFAULT_PULLBACK_REQUIRES_OUTER_BAND_TOUCH,
    use_rejection_quality_filter: bool = DEFAULT_USE_REJECTION_QUALITY_FILTER,
    rejection_wick_min_ratio: float = DEFAULT_REJECTION_WICK_MIN_RATIO,
    rejection_body_min_ratio: float = DEFAULT_REJECTION_BODY_MIN_RATIO,
    use_rsi_filter: bool = DEFAULT_USE_RSI_FILTER,
    rsi_length: int = DEFAULT_RSI_LENGTH,
    rsi_midline: float = DEFAULT_RSI_MIDLINE,
    use_rsi_cross_filter: bool = DEFAULT_USE_RSI_CROSS_FILTER,
    rsi_midline_margin: float = DEFAULT_RSI_MIDLINE_MARGIN,
    use_volume_filter: bool = DEFAULT_USE_VOLUME_FILTER,
    volume_ma_length: int = DEFAULT_VOLUME_MA_LENGTH,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
    use_atr_stop_buffer: bool = DEFAULT_USE_ATR_STOP_BUFFER,
    atr_length: int = DEFAULT_ATR_LENGTH,
    atr_stop_buffer_mult: float = DEFAULT_ATR_STOP_BUFFER_MULT,
    signal_cooldown_bars: int = DEFAULT_SIGNAL_COOLDOWN_BARS,
) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"EMA Band Rejection requires columns: {missing_list}")

    resolved_ema_fast = max(1, int(ema_fast))
    resolved_ema_mid = max(1, int(ema_mid))
    resolved_ema_slow = max(2, int(ema_slow))
    resolved_slope_lookback = max(1, int(slope_lookback))
    resolved_min_spread = max(0.0, float(min_ema_spread_pct))
    resolved_min_slow_slope_pct = max(0.0, float(min_slow_slope_pct))
    resolved_pullback_requires_outer_band_touch = bool(pullback_requires_outer_band_touch)
    resolved_use_rejection_quality_filter = bool(use_rejection_quality_filter)
    resolved_rejection_wick_min_ratio = max(0.0, float(rejection_wick_min_ratio))
    resolved_rejection_body_min_ratio = max(0.0, float(rejection_body_min_ratio))
    resolved_rsi_length = max(2, int(rsi_length))
    resolved_rsi_midline = float(rsi_midline)
    resolved_use_rsi_cross_filter = bool(use_rsi_cross_filter)
    resolved_rsi_midline_margin = max(0.0, float(rsi_midline_margin))
    resolved_volume_ma_length = max(1, int(volume_ma_length))
    resolved_volume_multiplier = max(1.0, float(volume_multiplier))
    resolved_atr_length = max(2, int(atr_length))
    resolved_atr_mult = max(0.0, float(atr_stop_buffer_mult))
    resolved_signal_cooldown_bars = max(0, int(signal_cooldown_bars))
    use_rsi = bool(use_rsi_filter)
    use_volume = bool(use_volume_filter)
    use_atr = bool(use_atr_stop_buffer)

    open_series = pd.to_numeric(working_df["open"], errors="coerce")
    high_series = pd.to_numeric(working_df["high"], errors="coerce")
    low_series = pd.to_numeric(working_df["low"], errors="coerce")
    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    volume_series = pd.to_numeric(working_df["volume"], errors="coerce")

    ema_fast_series = close_series.ewm(span=resolved_ema_fast, adjust=False).mean()
    ema_mid_series = close_series.ewm(span=resolved_ema_mid, adjust=False).mean()
    ema_slow_series = close_series.ewm(span=resolved_ema_slow, adjust=False).mean()
    ema_slow_reference = ema_slow_series.shift(resolved_slope_lookback)
    ema_slow_slope_pct = (
        (ema_slow_series - ema_slow_reference)
        / ema_slow_reference.abs().replace(0.0, np.nan)
    ) * 100.0
    ema_spread_pct = (
        (ema_fast_series - ema_slow_series).abs()
        / close_series.replace(0.0, np.nan).abs()
    ) * 100.0
    spread_ok = ema_spread_pct >= resolved_min_spread

    zone_low = np.minimum(ema_fast_series, ema_slow_series)
    zone_high = np.maximum(ema_fast_series, ema_slow_series)
    in_band_zone = high_series.ge(zone_low) & low_series.le(zone_high)
    came_from_above = close_series.shift(1).gt(zone_high.shift(1))
    came_from_below = close_series.shift(1).lt(zone_low.shift(1))
    pullback_long = in_band_zone & came_from_above
    pullback_short = in_band_zone & came_from_below
    if resolved_pullback_requires_outer_band_touch:
        pullback_long = pullback_long & low_series.le(ema_slow_series)
        pullback_short = pullback_short & high_series.ge(ema_slow_series)

    candle_bullish = close_series.gt(open_series)
    candle_bearish = close_series.lt(open_series)
    rejection_long = candle_bullish & close_series.gt(ema_mid_series) & close_series.ge(ema_slow_series)
    rejection_short = candle_bearish & close_series.lt(ema_mid_series) & close_series.le(ema_slow_series)
    if resolved_use_rejection_quality_filter:
        candle_range = (high_series - low_series).abs()
        safe_range = candle_range.replace(0.0, np.nan)
        body_size = (close_series - open_series).abs()
        upper_wick = high_series - np.maximum(open_series, close_series)
        lower_wick = np.minimum(open_series, close_series) - low_series
        body_ratio = body_size / safe_range
        upper_wick_ratio = upper_wick / safe_range
        lower_wick_ratio = lower_wick / safe_range
        rejection_long = (
            rejection_long
            & lower_wick_ratio.ge(resolved_rejection_wick_min_ratio)
            & body_ratio.ge(resolved_rejection_body_min_ratio)
        )
        rejection_short = (
            rejection_short
            & upper_wick_ratio.ge(resolved_rejection_wick_min_ratio)
            & body_ratio.ge(resolved_rejection_body_min_ratio)
        )

    trend_long = (
        ema_fast_series.gt(ema_mid_series)
        & ema_mid_series.gt(ema_slow_series)
        & ema_slow_slope_pct.ge(resolved_min_slow_slope_pct)
        & spread_ok.fillna(False)
    )
    trend_short = (
        ema_fast_series.lt(ema_mid_series)
        & ema_mid_series.lt(ema_slow_series)
        & ema_slow_slope_pct.le(-resolved_min_slow_slope_pct)
        & spread_ok.fillna(False)
    )

    if use_rsi:
        rsi_series = _calculate_rsi(close_series, resolved_rsi_length)
        upper_midline = resolved_rsi_midline + resolved_rsi_midline_margin
        lower_midline = resolved_rsi_midline - resolved_rsi_midline_margin
        long_rsi_ok = rsi_series.gt(upper_midline)
        short_rsi_ok = rsi_series.lt(lower_midline)
        if resolved_use_rsi_cross_filter:
            long_rsi_ok = long_rsi_ok & rsi_series.shift(1).le(lower_midline)
            short_rsi_ok = short_rsi_ok & rsi_series.shift(1).ge(upper_midline)
    else:
        rsi_series = pd.Series(50.0, index=working_df.index, dtype="float64")
        long_rsi_ok = pd.Series(True, index=working_df.index, dtype="bool")
        short_rsi_ok = pd.Series(True, index=working_df.index, dtype="bool")

    if use_volume:
        volume_ma = volume_series.rolling(window=resolved_volume_ma_length, min_periods=1).mean()
        volume_ok = volume_series.gt(volume_ma * resolved_volume_multiplier)
    else:
        volume_ma = volume_series.rolling(window=resolved_volume_ma_length, min_periods=1).mean()
        volume_ok = pd.Series(True, index=working_df.index, dtype="bool")

    setup_long = trend_long & pullback_long & rejection_long & long_rsi_ok & volume_ok
    setup_short = trend_short & pullback_short & rejection_short & short_rsi_ok & volume_ok

    long_entry = setup_long.shift(1, fill_value=False).astype(bool)
    short_entry = setup_short.shift(1, fill_value=False).astype(bool)
    conflicting_entries = long_entry & short_entry
    long_entry = long_entry & ~conflicting_entries
    short_entry = short_entry & ~conflicting_entries
    cooldown_blocked_signals = 0
    if resolved_signal_cooldown_bars > 0:
        long_values = long_entry.to_numpy(dtype=bool, copy=False)
        short_values = short_entry.to_numpy(dtype=bool, copy=False)
        filtered_long = np.zeros(len(long_values), dtype=bool)
        filtered_short = np.zeros(len(short_values), dtype=bool)
        next_allowed_index = 0
        for idx in range(len(long_values)):
            long_flag = bool(long_values[idx])
            short_flag = bool(short_values[idx])
            if not long_flag and not short_flag:
                continue
            if idx < next_allowed_index:
                cooldown_blocked_signals += 1
                continue
            if long_flag and not short_flag:
                filtered_long[idx] = True
                next_allowed_index = idx + resolved_signal_cooldown_bars + 1
            elif short_flag and not long_flag:
                filtered_short[idx] = True
                next_allowed_index = idx + resolved_signal_cooldown_bars + 1
        long_entry = pd.Series(filtered_long, index=working_df.index, dtype="bool")
        short_entry = pd.Series(filtered_short, index=working_df.index, dtype="bool")

    atr_series = _calculate_atr(high_series, low_series, close_series, resolved_atr_length)
    atr_buffer = (
        atr_series.shift(1) * resolved_atr_mult
        if use_atr
        else pd.Series(0.0, index=working_df.index, dtype="float64")
    )
    rejection_high = high_series.shift(1)
    rejection_low = low_series.shift(1)
    long_stop_price = rejection_low - atr_buffer
    short_stop_price = rejection_high + atr_buffer

    # Signals are shifted by one candle, so entry aligns to this candle.
    # Backtest execution enters on candle close; use close as sizing reference
    # so dynamic stop-loss percent remains unit-consistent with execution.
    entry_reference_price = close_series.replace(0.0, np.nan)
    long_stop_pct = ((entry_reference_price - long_stop_price) / entry_reference_price) * 100.0
    short_stop_pct = ((short_stop_price - entry_reference_price) / entry_reference_price) * 100.0
    long_stop_pct = long_stop_pct.where(long_stop_pct.gt(0.0))
    short_stop_pct = short_stop_pct.where(short_stop_pct.gt(0.0))

    dynamic_stop_loss_pct = pd.Series(np.nan, index=working_df.index, dtype="float64")
    dynamic_stop_loss_pct = dynamic_stop_loss_pct.mask(long_entry, long_stop_pct)
    dynamic_stop_loss_pct = dynamic_stop_loss_pct.mask(short_entry, short_stop_pct)
    dynamic_stop_loss_pct = dynamic_stop_loss_pct.clip(lower=0.05, upper=95.0)

    signal_direction = pd.Series(0, index=working_df.index, dtype="int8")
    signal_direction = signal_direction.mask(long_entry, 1)
    signal_direction = signal_direction.mask(short_entry, -1)

    working_df.loc[:, "ema_band_ema_fast"] = ema_fast_series
    working_df.loc[:, "ema_band_ema_mid"] = ema_mid_series
    working_df.loc[:, "ema_band_ema_slow"] = ema_slow_series
    working_df.loc[:, "ema_band_ema_slow_slope_pct"] = ema_slow_slope_pct
    working_df.loc[:, "ema_band_ema_spread_pct"] = ema_spread_pct
    working_df.loc[:, "ema_band_zone_low"] = zone_low
    working_df.loc[:, "ema_band_zone_high"] = zone_high
    working_df.loc[:, "ema_band_rsi"] = rsi_series
    working_df.loc[:, "ema_band_volume_ma"] = volume_ma
    working_df.loc[:, "ema_band_atr"] = atr_series
    working_df.loc[:, "ema_band_trend_long"] = trend_long.fillna(False)
    working_df.loc[:, "ema_band_trend_short"] = trend_short.fillna(False)
    working_df.loc[:, "ema_band_pullback_long"] = pullback_long.fillna(False)
    working_df.loc[:, "ema_band_pullback_short"] = pullback_short.fillna(False)
    working_df.loc[:, "ema_band_rejection_long"] = rejection_long.fillna(False)
    working_df.loc[:, "ema_band_rejection_short"] = rejection_short.fillna(False)
    working_df.loc[:, "ema_band_setup_long"] = setup_long.fillna(False)
    working_df.loc[:, "ema_band_setup_short"] = setup_short.fillna(False)
    working_df.loc[:, "ema_band_long_entry"] = long_entry.fillna(False)
    working_df.loc[:, "ema_band_short_entry"] = short_entry.fillna(False)
    working_df.loc[:, "ema_band_long_exit"] = short_entry.fillna(False)
    working_df.loc[:, "ema_band_short_exit"] = long_entry.fillna(False)
    working_df.loc[:, "ema_band_dynamic_stop_loss_pct"] = dynamic_stop_loss_pct
    working_df.loc[:, "ema_band_signal_direction"] = signal_direction
    working_df.loc[:, "ema_band_signal_cooldown_bars"] = float(resolved_signal_cooldown_bars)
    working_df.loc[:, "ema_band_cooldown_blocked_signals"] = float(cooldown_blocked_signals)
    return working_df


def run_python_ema_band_rejection(candles_dataframe: Any) -> int:
    working_df = build_ema_band_rejection_signal_frame(candles_dataframe)
    if len(working_df) < 3:
        return 0
    latest = working_df.iloc[-1]
    try:
        return int(latest["ema_band_signal_direction"])
    except Exception:
        return 0
