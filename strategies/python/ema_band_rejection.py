from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings

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
DEFAULT_TREND_PERSISTENCE_BARS = 1
DEFAULT_MAX_PULLBACK_BARS = 0
DEFAULT_ENTRY_OFFSET_PCT = 0.0
DEFAULT_USE_LATE_ENTRY_GUARD = False
DEFAULT_LATE_ENTRY_MAX_MOVE_1_BAR_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_MOVE_2_BARS_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_MOVE_3_BARS_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_DISTANCE_REF_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_DISTANCE_FAST_REF_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_DISTANCE_MID_REF_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_ATR_MULT = 0.0
DEFAULT_USE_PULLBACK_REENTRY = False
DEFAULT_PULLBACK_REENTRY_MIN_TOUCH = 0.0
DEFAULT_PULLBACK_REENTRY_RECONFIRM_REQUIRED = False


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


def _safe_pct_distance(
    price_series: pd.Series,
    reference_series: pd.Series,
) -> pd.Series:
    return (
        (price_series - reference_series)
        / reference_series.abs().replace(0.0, np.nan)
    ) * 100.0


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
    trend_persistence_bars: int = DEFAULT_TREND_PERSISTENCE_BARS,
    max_pullback_bars: int = DEFAULT_MAX_PULLBACK_BARS,
    entry_offset_pct: float = DEFAULT_ENTRY_OFFSET_PCT,
    use_late_entry_guard: bool = DEFAULT_USE_LATE_ENTRY_GUARD,
    late_entry_max_move_1_bar_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_1_BAR_PCT,
    late_entry_max_move_2_bars_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_2_BARS_PCT,
    late_entry_max_move_3_bars_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_3_BARS_PCT,
    late_entry_max_distance_ref_pct: float = DEFAULT_LATE_ENTRY_MAX_DISTANCE_REF_PCT,
    late_entry_max_distance_fast_ref_pct: float = DEFAULT_LATE_ENTRY_MAX_DISTANCE_FAST_REF_PCT,
    late_entry_max_distance_mid_ref_pct: float = DEFAULT_LATE_ENTRY_MAX_DISTANCE_MID_REF_PCT,
    late_entry_max_atr_mult: float = DEFAULT_LATE_ENTRY_MAX_ATR_MULT,
    use_pullback_reentry: bool = DEFAULT_USE_PULLBACK_REENTRY,
    pullback_reentry_min_touch: float = DEFAULT_PULLBACK_REENTRY_MIN_TOUCH,
    pullback_reentry_reconfirm_required: bool = DEFAULT_PULLBACK_REENTRY_RECONFIRM_REQUIRED,
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
    resolved_trend_persistence_bars = max(1, int(trend_persistence_bars))
    resolved_max_pullback_bars = max(0, int(max_pullback_bars))
    resolved_entry_offset_pct = max(0.0, float(entry_offset_pct))
    resolved_use_late_entry_guard = bool(use_late_entry_guard)
    resolved_late_entry_max_move_1_bar_pct = max(0.0, float(late_entry_max_move_1_bar_pct))
    resolved_late_entry_max_move_2_bars_pct = max(0.0, float(late_entry_max_move_2_bars_pct))
    resolved_late_entry_max_move_3_bars_pct = max(0.0, float(late_entry_max_move_3_bars_pct))
    resolved_late_entry_max_distance_ref_pct = max(0.0, float(late_entry_max_distance_ref_pct))
    resolved_late_entry_max_distance_fast_ref_pct = max(
        0.0,
        float(late_entry_max_distance_fast_ref_pct),
    )
    resolved_late_entry_max_distance_mid_ref_pct = max(
        0.0,
        float(late_entry_max_distance_mid_ref_pct),
    )
    resolved_late_entry_max_atr_mult = max(0.0, float(late_entry_max_atr_mult))
    resolved_use_pullback_reentry = bool(use_pullback_reentry)
    resolved_pullback_reentry_min_touch = max(0.0, float(pullback_reentry_min_touch))
    resolved_pullback_reentry_reconfirm_required = bool(pullback_reentry_reconfirm_required)
    use_rsi = bool(use_rsi_filter)
    use_volume = bool(use_volume_filter)
    use_atr = bool(use_atr_stop_buffer)

    open_series = pd.to_numeric(working_df["open"], errors="coerce")
    high_series = pd.to_numeric(working_df["high"], errors="coerce")
    low_series = pd.to_numeric(working_df["low"], errors="coerce")
    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    volume_series = pd.to_numeric(working_df["volume"], errors="coerce")
    # Compute ATR once early because both late-entry diagnostics and stop buffers depend on it.
    # Keep a safe fallback so signal generation does not fail hard on unexpected ATR edge cases.
    atr_series = pd.Series(np.nan, index=working_df.index, dtype="float64")
    try:
        atr_series = _calculate_atr(high_series, low_series, close_series, resolved_atr_length)
    except Exception:
        pass

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
    if resolved_trend_persistence_bars > 1:
        trend_long_confirmed = (
            trend_long.fillna(False)
            .rolling(
                window=resolved_trend_persistence_bars,
                min_periods=resolved_trend_persistence_bars,
            )
            .sum()
            .ge(float(resolved_trend_persistence_bars))
            .fillna(False)
        )
        trend_short_confirmed = (
            trend_short.fillna(False)
            .rolling(
                window=resolved_trend_persistence_bars,
                min_periods=resolved_trend_persistence_bars,
            )
            .sum()
            .ge(float(resolved_trend_persistence_bars))
            .fillna(False)
        )
    else:
        trend_long_confirmed = trend_long.fillna(False)
        trend_short_confirmed = trend_short.fillna(False)
    if resolved_max_pullback_bars > 0:
        pullback_window = resolved_max_pullback_bars + 1
        pullback_long_recent = (
            pullback_long.fillna(False)
            .rolling(window=pullback_window, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )
        pullback_short_recent = (
            pullback_short.fillna(False)
            .rolling(window=pullback_window, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )
    else:
        pullback_long_recent = pullback_long.fillna(False)
        pullback_short_recent = pullback_short.fillna(False)

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

    setup_long = (
        trend_long_confirmed
        & pullback_long_recent
        & rejection_long
        & long_rsi_ok
        & volume_ok
    )
    setup_short = (
        trend_short_confirmed
        & pullback_short_recent
        & rejection_short
        & short_rsi_ok
        & volume_ok
    )

    long_entry = setup_long.shift(1, fill_value=False).astype(bool)
    short_entry = setup_short.shift(1, fill_value=False).astype(bool)
    conflicting_entries = long_entry & short_entry
    long_entry = long_entry & ~conflicting_entries
    short_entry = short_entry & ~conflicting_entries
    if resolved_entry_offset_pct > 0.0:
        long_entry_offset_threshold = high_series.shift(1) * (
            1.0 + (resolved_entry_offset_pct / 100.0)
        )
        short_entry_offset_threshold = low_series.shift(1) * (
            1.0 - (resolved_entry_offset_pct / 100.0)
        )
        long_entry = long_entry & close_series.ge(long_entry_offset_threshold).fillna(False)
        short_entry = short_entry & close_series.le(short_entry_offset_threshold).fillna(False)
        conflicting_entries = long_entry & short_entry
        long_entry = long_entry & ~conflicting_entries
        short_entry = short_entry & ~conflicting_entries

    move_last_1_bar_pct = (
        (close_series - close_series.shift(1))
        / close_series.shift(1).abs().replace(0.0, np.nan)
    ) * 100.0
    move_last_2_bars_pct = (
        (close_series - close_series.shift(2))
        / close_series.shift(2).abs().replace(0.0, np.nan)
    ) * 100.0
    move_last_3_bars_pct = (
        (close_series - close_series.shift(3))
        / close_series.shift(3).abs().replace(0.0, np.nan)
    ) * 100.0
    distance_fast_ref_pct = _safe_pct_distance(close_series, ema_fast_series)
    distance_mid_ref_pct = _safe_pct_distance(close_series, ema_mid_series)
    distance_ref_long_pct = np.maximum(distance_fast_ref_pct, distance_mid_ref_pct)
    distance_ref_short_pct = np.maximum(-distance_fast_ref_pct, -distance_mid_ref_pct)
    atr_safe = atr_series.replace(0.0, np.nan)
    long_extension_price = np.maximum(close_series - ema_fast_series, close_series - ema_mid_series)
    short_extension_price = np.maximum(ema_fast_series - close_series, ema_mid_series - close_series)
    atr_extension_long_mult = long_extension_price / atr_safe
    atr_extension_short_mult = short_extension_price / atr_safe
    distance_to_reference_pct = pd.Series(np.nan, index=working_df.index, dtype="float64")
    distance_to_reference_pct = distance_to_reference_pct.mask(
        long_entry,
        pd.to_numeric(distance_ref_long_pct, errors="coerce"),
    )
    distance_to_reference_pct = distance_to_reference_pct.mask(
        short_entry,
        pd.to_numeric(distance_ref_short_pct, errors="coerce"),
    )
    atr_extension_mult = pd.Series(np.nan, index=working_df.index, dtype="float64")
    atr_extension_mult = atr_extension_mult.mask(
        long_entry,
        pd.to_numeric(atr_extension_long_mult, errors="coerce"),
    )
    atr_extension_mult = atr_extension_mult.mask(
        short_entry,
        pd.to_numeric(atr_extension_short_mult, errors="coerce"),
    )

    late_entry_guard_blocked = pd.Series(False, index=working_df.index, dtype="bool")
    late_entry_guard_block_reason = pd.Series("", index=working_df.index, dtype="object")
    if resolved_use_late_entry_guard or resolved_use_pullback_reentry:
        long_entry_values = long_entry.to_numpy(dtype=bool, copy=False)
        short_entry_values = short_entry.to_numpy(dtype=bool, copy=False)
        move_1_values = pd.to_numeric(move_last_1_bar_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        move_2_values = pd.to_numeric(move_last_2_bars_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        move_3_values = pd.to_numeric(move_last_3_bars_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        distance_fast_values = pd.to_numeric(distance_fast_ref_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        distance_mid_values = pd.to_numeric(distance_mid_ref_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        distance_long_values = pd.to_numeric(distance_ref_long_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        distance_short_values = pd.to_numeric(distance_ref_short_pct, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        atr_long_values = pd.to_numeric(atr_extension_long_mult, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        atr_short_values = pd.to_numeric(atr_extension_short_mult, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        low_values = pd.to_numeric(low_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        high_values = pd.to_numeric(high_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        ema_mid_values = pd.to_numeric(ema_mid_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        bullish_values = candle_bullish.fillna(False).to_numpy(dtype=bool, copy=False)
        bearish_values = candle_bearish.fillna(False).to_numpy(dtype=bool, copy=False)
        close_values = pd.to_numeric(close_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)

        blocked_values = np.zeros(len(working_df), dtype=bool)
        reason_values = np.array([""] * len(working_df), dtype=object)
        touch_tolerance = resolved_pullback_reentry_min_touch / 100.0
        for index in range(len(working_df)):
            is_long = bool(long_entry_values[index])
            is_short = bool(short_entry_values[index])
            if not is_long and not is_short:
                continue
            direction = 1 if is_long else -1
            reason = ""
            directional_move_1 = float(move_1_values[index]) if np.isfinite(move_1_values[index]) else 0.0
            directional_move_2 = float(move_2_values[index]) if np.isfinite(move_2_values[index]) else 0.0
            directional_move_3 = float(move_3_values[index]) if np.isfinite(move_3_values[index]) else 0.0
            directional_fast_distance = (
                float(distance_fast_values[index]) if np.isfinite(distance_fast_values[index]) else 0.0
            )
            directional_mid_distance = (
                float(distance_mid_values[index]) if np.isfinite(distance_mid_values[index]) else 0.0
            )
            directional_distance_ref = (
                float(distance_long_values[index]) if is_long else float(distance_short_values[index])
            )
            directional_atr_extension = (
                float(atr_long_values[index]) if is_long else float(atr_short_values[index])
            )
            if direction < 0:
                directional_move_1 = -directional_move_1
                directional_move_2 = -directional_move_2
                directional_move_3 = -directional_move_3
                directional_fast_distance = -directional_fast_distance
                directional_mid_distance = -directional_mid_distance
            if not np.isfinite(directional_distance_ref):
                directional_distance_ref = 0.0
            if not np.isfinite(directional_atr_extension):
                directional_atr_extension = 0.0
            if resolved_use_late_entry_guard:
                if (
                    resolved_late_entry_max_move_3_bars_pct > 0.0
                    and directional_move_3 > resolved_late_entry_max_move_3_bars_pct
                ):
                    reason = "overextended_last_3_bars"
                elif (
                    resolved_late_entry_max_move_2_bars_pct > 0.0
                    and directional_move_2 > resolved_late_entry_max_move_2_bars_pct
                ):
                    reason = "overextended_last_2_bars"
                elif (
                    resolved_late_entry_max_move_1_bar_pct > 0.0
                    and directional_move_1 > resolved_late_entry_max_move_1_bar_pct
                ):
                    reason = "overextended_last_1_bar"
                elif (
                    resolved_late_entry_max_distance_fast_ref_pct > 0.0
                    and directional_fast_distance > resolved_late_entry_max_distance_fast_ref_pct
                ):
                    reason = "overextended_from_reference"
                elif (
                    resolved_late_entry_max_distance_mid_ref_pct > 0.0
                    and directional_mid_distance > resolved_late_entry_max_distance_mid_ref_pct
                ):
                    reason = "overextended_from_reference"
                elif (
                    resolved_late_entry_max_distance_ref_pct > 0.0
                    and directional_distance_ref > resolved_late_entry_max_distance_ref_pct
                ):
                    reason = "overextended_from_reference"
                elif (
                    resolved_late_entry_max_atr_mult > 0.0
                    and directional_atr_extension > resolved_late_entry_max_atr_mult
                ):
                    reason = "late_entry_guard_long" if is_long else "late_entry_guard_short"
            if not reason and resolved_use_pullback_reentry:
                previous_index = max(0, index - 1)
                previous_mid = ema_mid_values[previous_index]
                long_touch_ready = False
                short_touch_ready = False
                if np.isfinite(previous_mid) and previous_mid > 0.0:
                    long_touch_ready = bool(
                        np.isfinite(low_values[previous_index])
                        and low_values[previous_index] <= previous_mid * (1.0 + touch_tolerance)
                    )
                    short_touch_ready = bool(
                        np.isfinite(high_values[previous_index])
                        and high_values[previous_index] >= previous_mid * (1.0 - touch_tolerance)
                    )
                touch_ready = long_touch_ready if is_long else short_touch_ready
                if not touch_ready:
                    reason = "pullback_reentry_not_ready"
                elif resolved_pullback_reentry_reconfirm_required:
                    if is_long:
                        reconfirm_ready = bool(
                            bullish_values[index]
                            and np.isfinite(close_values[index])
                            and np.isfinite(ema_mid_values[index])
                            and close_values[index] >= ema_mid_values[index]
                        )
                    else:
                        reconfirm_ready = bool(
                            bearish_values[index]
                            and np.isfinite(close_values[index])
                            and np.isfinite(ema_mid_values[index])
                            and close_values[index] <= ema_mid_values[index]
                        )
                    if not reconfirm_ready:
                        reason = "pullback_reentry_not_ready"
            if reason:
                blocked_values[index] = True
                reason_values[index] = reason

        late_entry_guard_blocked = pd.Series(blocked_values, index=working_df.index, dtype="bool")
        late_entry_guard_block_reason = pd.Series(reason_values, index=working_df.index, dtype="object")
        long_entry = long_entry & ~late_entry_guard_blocked
        short_entry = short_entry & ~late_entry_guard_blocked

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
    working_df.loc[:, "ema_band_move_last_1_bar_pct"] = move_last_1_bar_pct
    working_df.loc[:, "ema_band_move_last_2_bars_pct"] = move_last_2_bars_pct
    working_df.loc[:, "ema_band_move_last_3_bars_pct"] = move_last_3_bars_pct
    working_df.loc[:, "ema_band_distance_fast_ref_pct"] = distance_fast_ref_pct
    working_df.loc[:, "ema_band_distance_mid_ref_pct"] = distance_mid_ref_pct
    working_df.loc[:, "ema_band_distance_to_reference_pct"] = distance_to_reference_pct
    working_df.loc[:, "ema_band_atr_extension_mult"] = atr_extension_mult
    working_df.loc[:, "ema_band_late_entry_guard_enabled"] = float(resolved_use_late_entry_guard)
    working_df.loc[:, "ema_band_late_entry_guard_blocked"] = late_entry_guard_blocked.fillna(False)
    working_df.loc[:, "ema_band_late_entry_guard_block_reason"] = late_entry_guard_block_reason
    working_df.loc[:, "ema_band_late_entry_guard_blocked_signals"] = float(
        late_entry_guard_blocked.fillna(False).sum()
    )
    working_df.loc[:, "ema_band_pullback_reentry_enabled"] = float(resolved_use_pullback_reentry)
    return working_df


def run_python_ema_band_rejection(candles_dataframe: Any) -> int:
    working_df = build_ema_band_rejection_signal_frame(
        candles_dataframe,
        use_late_entry_guard=bool(getattr(settings.strategy, "use_late_entry_guard", False)),
        late_entry_max_move_1_bar_pct=float(
            getattr(settings.strategy, "late_entry_max_move_1_bar_pct", 0.0)
        ),
        late_entry_max_move_2_bars_pct=float(
            getattr(settings.strategy, "late_entry_max_move_2_bars_pct", 0.0)
        ),
        late_entry_max_move_3_bars_pct=float(
            getattr(settings.strategy, "late_entry_max_move_3_bars_pct", 0.0)
        ),
        late_entry_max_distance_ref_pct=float(
            getattr(settings.strategy, "late_entry_max_distance_ref_pct", 0.0)
        ),
        late_entry_max_distance_fast_ref_pct=float(
            getattr(settings.strategy, "late_entry_max_distance_fast_ref_pct", 0.0)
        ),
        late_entry_max_distance_mid_ref_pct=float(
            getattr(settings.strategy, "late_entry_max_distance_mid_ref_pct", 0.0)
        ),
        late_entry_max_atr_mult=float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
        use_pullback_reentry=bool(getattr(settings.strategy, "use_pullback_reentry", False)),
        pullback_reentry_min_touch=float(
            getattr(settings.strategy, "pullback_reentry_min_touch", 0.0)
        ),
        pullback_reentry_reconfirm_required=bool(
            getattr(settings.strategy, "pullback_reentry_reconfirm_required", False)
        ),
    )
    if len(working_df) < 3:
        return 0
    latest = working_df.iloc[-1]
    try:
        return int(latest["ema_band_signal_direction"])
    except Exception:
        return 0
