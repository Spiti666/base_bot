from __future__ import annotations

from typing import Any

import pandas as pd

from config import settings

DEFAULT_USE_LATE_ENTRY_GUARD = False
DEFAULT_LATE_ENTRY_MAX_MOVE_1_BAR_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_MOVE_2_BARS_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_MOVE_3_BARS_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_DISTANCE_REF_PCT = 0.0
DEFAULT_LATE_ENTRY_MAX_ATR_MULT = 0.0
DEFAULT_MAX_BREAKOUT_CANDLE_BODY_PCT = 0.0
DEFAULT_MAX_BREAKOUT_CANDLE_RANGE_ATR_MULT = 0.0


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
    breakout_buffer_pct: float = 0.0,
    min_range_pct: float = 0.0,
    cooldown_bars_after_exit: int = 0,
    use_late_entry_guard: bool = DEFAULT_USE_LATE_ENTRY_GUARD,
    late_entry_max_move_1_bar_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_1_BAR_PCT,
    late_entry_max_move_2_bars_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_2_BARS_PCT,
    late_entry_max_move_3_bars_pct: float = DEFAULT_LATE_ENTRY_MAX_MOVE_3_BARS_PCT,
    late_entry_max_distance_ref_pct: float = DEFAULT_LATE_ENTRY_MAX_DISTANCE_REF_PCT,
    late_entry_max_atr_mult: float = DEFAULT_LATE_ENTRY_MAX_ATR_MULT,
    max_breakout_candle_body_pct: float = DEFAULT_MAX_BREAKOUT_CANDLE_BODY_PCT,
    max_breakout_candle_range_atr_mult: float = DEFAULT_MAX_BREAKOUT_CANDLE_RANGE_ATR_MULT,
) -> pd.DataFrame:
    working_df = _ensure_dataframe(candles_dataframe)
    resolved_period = settings.strategy.dual_thrust_period if period is None else int(period)
    resolved_k1 = settings.strategy.dual_thrust_k1 if k1 is None else float(k1)
    resolved_k2 = settings.strategy.dual_thrust_k2 if k2 is None else float(k2)
    resolved_breakout_buffer_pct = max(0.0, float(breakout_buffer_pct))
    resolved_min_range_pct = max(0.0, float(min_range_pct))
    resolved_cooldown_bars_after_exit = max(0, int(cooldown_bars_after_exit))
    resolved_use_late_entry_guard = bool(use_late_entry_guard)
    resolved_late_entry_max_move_1_bar_pct = max(0.0, float(late_entry_max_move_1_bar_pct))
    resolved_late_entry_max_move_2_bars_pct = max(0.0, float(late_entry_max_move_2_bars_pct))
    resolved_late_entry_max_move_3_bars_pct = max(0.0, float(late_entry_max_move_3_bars_pct))
    resolved_late_entry_max_distance_ref_pct = max(0.0, float(late_entry_max_distance_ref_pct))
    resolved_late_entry_max_atr_mult = max(0.0, float(late_entry_max_atr_mult))
    resolved_max_breakout_candle_body_pct = max(0.0, float(max_breakout_candle_body_pct))
    resolved_max_breakout_candle_range_atr_mult = max(0.0, float(max_breakout_candle_range_atr_mult))
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
    buffer_ratio = resolved_breakout_buffer_pct / 100.0
    if buffer_ratio > 0.0:
        buy_line = working_df["dual_buy_line"] * (1.0 + buffer_ratio)
        sell_line = working_df["dual_sell_line"] * (1.0 - buffer_ratio)
    else:
        buy_line = working_df["dual_buy_line"]
        sell_line = working_df["dual_sell_line"]
    close_series = pd.to_numeric(working_df["close"], errors="coerce")
    previous_close = close_series.shift(1)
    previous_buy_line = buy_line.shift(1).fillna(buy_line)
    previous_sell_line = sell_line.shift(1).fillna(sell_line)
    line_ready = buy_line.notna() & sell_line.notna()
    if resolved_min_range_pct > 0.0:
        open_reference = (
            pd.to_numeric(working_df["open"], errors="coerce")
            .abs()
            .replace(0.0, float("nan"))
        )
        dual_range_pct = (
            pd.to_numeric(working_df["dual_range"], errors="coerce")
            / open_reference
        ) * 100.0
        line_ready = line_ready & dual_range_pct.ge(resolved_min_range_pct).fillna(False)
    long_entry_series = (
        line_ready
        & previous_close.le(previous_buy_line)
        & close_series.gt(buy_line)
    ).fillna(False)
    short_entry_series = (
        line_ready
        & previous_close.ge(previous_sell_line)
        & close_series.lt(sell_line)
    ).fillna(False)
    long_exit_series = (line_ready & close_series.lt(sell_line)).fillna(False)
    short_exit_series = (line_ready & close_series.gt(buy_line)).fillna(False)

    move_last_1_bar_pct = (
        (close_series - close_series.shift(1))
        / close_series.shift(1).abs().replace(0.0, float("nan"))
    ) * 100.0
    move_last_2_bars_pct = (
        (close_series - close_series.shift(2))
        / close_series.shift(2).abs().replace(0.0, float("nan"))
    ) * 100.0
    move_last_3_bars_pct = (
        (close_series - close_series.shift(3))
        / close_series.shift(3).abs().replace(0.0, float("nan"))
    ) * 100.0
    distance_to_buy_line_pct = (
        (close_series - buy_line)
        / buy_line.abs().replace(0.0, float("nan"))
    ) * 100.0
    distance_to_sell_line_pct = (
        (sell_line - close_series)
        / sell_line.abs().replace(0.0, float("nan"))
    ) * 100.0
    previous_close_series = close_series.shift(1)
    true_range = pd.concat(
        [
            (pd.to_numeric(working_df["high"], errors="coerce") - pd.to_numeric(working_df["low"], errors="coerce")).abs(),
            (pd.to_numeric(working_df["high"], errors="coerce") - previous_close_series).abs(),
            (pd.to_numeric(working_df["low"], errors="coerce") - previous_close_series).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = true_range.rolling(window=14, min_periods=1).mean()
    atr_extension_long_mult = (
        (close_series - buy_line).clip(lower=0.0)
        / atr_series.replace(0.0, float("nan"))
    )
    atr_extension_short_mult = (
        (sell_line - close_series).clip(lower=0.0)
        / atr_series.replace(0.0, float("nan"))
    )
    breakout_candle_body_pct = (
        (pd.to_numeric(working_df["close"], errors="coerce") - pd.to_numeric(working_df["open"], errors="coerce"))
        .abs()
        / pd.to_numeric(working_df["open"], errors="coerce").abs().replace(0.0, float("nan"))
    ) * 100.0
    breakout_candle_range_atr_mult = (
        (pd.to_numeric(working_df["high"], errors="coerce") - pd.to_numeric(working_df["low"], errors="coerce")).abs()
        / atr_series.replace(0.0, float("nan"))
    )

    late_entry_guard_blocked = pd.Series(False, index=working_df.index, dtype="bool")
    late_entry_guard_block_reason = pd.Series("", index=working_df.index, dtype="object")
    if resolved_use_late_entry_guard:
        long_values = long_entry_series.to_numpy(dtype=bool, copy=False)
        short_values = short_entry_series.to_numpy(dtype=bool, copy=False)
        move_1_values = pd.to_numeric(move_last_1_bar_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        move_2_values = pd.to_numeric(move_last_2_bars_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        move_3_values = pd.to_numeric(move_last_3_bars_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        distance_buy_values = pd.to_numeric(distance_to_buy_line_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        distance_sell_values = pd.to_numeric(distance_to_sell_line_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        atr_long_values = pd.to_numeric(atr_extension_long_mult, errors="coerce").to_numpy(dtype=float, copy=False)
        atr_short_values = pd.to_numeric(atr_extension_short_mult, errors="coerce").to_numpy(dtype=float, copy=False)
        body_values = pd.to_numeric(breakout_candle_body_pct, errors="coerce").to_numpy(dtype=float, copy=False)
        range_atr_values = pd.to_numeric(breakout_candle_range_atr_mult, errors="coerce").to_numpy(dtype=float, copy=False)
        blocked_values = [False] * len(working_df)
        reason_values = [""] * len(working_df)
        for index in range(len(working_df)):
            is_long = bool(long_values[index])
            is_short = bool(short_values[index])
            if not is_long and not is_short:
                continue
            reason = ""
            directional_move_1 = float(move_1_values[index]) if pd.notna(move_1_values[index]) else 0.0
            directional_move_2 = float(move_2_values[index]) if pd.notna(move_2_values[index]) else 0.0
            directional_move_3 = float(move_3_values[index]) if pd.notna(move_3_values[index]) else 0.0
            directional_distance = float(distance_buy_values[index]) if is_long else float(distance_sell_values[index])
            directional_atr_extension = float(atr_long_values[index]) if is_long else float(atr_short_values[index])
            body_pct = float(body_values[index]) if pd.notna(body_values[index]) else 0.0
            range_atr_mult = float(range_atr_values[index]) if pd.notna(range_atr_values[index]) else 0.0
            if is_short:
                directional_move_1 = -directional_move_1
                directional_move_2 = -directional_move_2
                directional_move_3 = -directional_move_3
            if (
                resolved_max_breakout_candle_body_pct > 0.0
                and body_pct > resolved_max_breakout_candle_body_pct
            ):
                reason = "breakout_too_extended"
            elif (
                resolved_max_breakout_candle_range_atr_mult > 0.0
                and range_atr_mult > resolved_max_breakout_candle_range_atr_mult
            ):
                reason = "breakout_too_extended"
            elif (
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
                resolved_late_entry_max_distance_ref_pct > 0.0
                and pd.notna(directional_distance)
                and directional_distance > resolved_late_entry_max_distance_ref_pct
            ):
                reason = "overextended_from_reference"
            elif (
                resolved_late_entry_max_atr_mult > 0.0
                and pd.notna(directional_atr_extension)
                and directional_atr_extension > resolved_late_entry_max_atr_mult
            ):
                reason = "late_entry_guard_long" if is_long else "late_entry_guard_short"
            if reason:
                blocked_values[index] = True
                reason_values[index] = reason
        late_entry_guard_blocked = pd.Series(blocked_values, index=working_df.index, dtype="bool")
        late_entry_guard_block_reason = pd.Series(reason_values, index=working_df.index, dtype="object")
        long_entry_series = long_entry_series & ~late_entry_guard_blocked
        short_entry_series = short_entry_series & ~late_entry_guard_blocked

    working_df.loc[:, "dual_long_entry"] = long_entry_series
    working_df.loc[:, "dual_short_entry"] = short_entry_series
    working_df.loc[:, "dual_long_exit"] = long_exit_series
    working_df.loc[:, "dual_short_exit"] = short_exit_series
    if resolved_cooldown_bars_after_exit > 0:
        long_entry_values = (
            working_df["dual_long_entry"].fillna(False).to_numpy(dtype=bool, copy=False)
        )
        short_entry_values = (
            working_df["dual_short_entry"].fillna(False).to_numpy(dtype=bool, copy=False)
        )
        long_exit_values = (
            working_df["dual_long_exit"].fillna(False).to_numpy(dtype=bool, copy=False)
        )
        short_exit_values = (
            working_df["dual_short_exit"].fillna(False).to_numpy(dtype=bool, copy=False)
        )
        filtered_long_entries = [False] * len(long_entry_values)
        filtered_short_entries = [False] * len(short_entry_values)
        next_allowed_entry_index = 0
        for index in range(len(long_entry_values)):
            if index >= next_allowed_entry_index:
                if bool(long_entry_values[index]) and not bool(short_entry_values[index]):
                    filtered_long_entries[index] = True
                elif bool(short_entry_values[index]) and not bool(long_entry_values[index]):
                    filtered_short_entries[index] = True
            if bool(long_exit_values[index]) or bool(short_exit_values[index]):
                next_allowed_entry_index = max(
                    next_allowed_entry_index,
                    index + resolved_cooldown_bars_after_exit + 1,
                )
        working_df.loc[:, "dual_long_entry"] = pd.Series(
            filtered_long_entries,
            index=working_df.index,
            dtype="bool",
        )
        working_df.loc[:, "dual_short_entry"] = pd.Series(
            filtered_short_entries,
            index=working_df.index,
            dtype="bool",
        )
    working_df.loc[:, "dual_move_last_1_bar_pct"] = move_last_1_bar_pct
    working_df.loc[:, "dual_move_last_2_bars_pct"] = move_last_2_bars_pct
    working_df.loc[:, "dual_move_last_3_bars_pct"] = move_last_3_bars_pct
    working_df.loc[:, "dual_distance_to_reference_pct"] = pd.Series(float("nan"), index=working_df.index, dtype="float64")
    working_df.loc[:, "dual_distance_to_reference_pct"] = working_df["dual_distance_to_reference_pct"].mask(
        working_df["dual_long_entry"].fillna(False),
        pd.to_numeric(distance_to_buy_line_pct, errors="coerce"),
    )
    working_df.loc[:, "dual_distance_to_reference_pct"] = working_df["dual_distance_to_reference_pct"].mask(
        working_df["dual_short_entry"].fillna(False),
        pd.to_numeric(distance_to_sell_line_pct, errors="coerce"),
    )
    working_df.loc[:, "dual_atr_extension_mult"] = pd.Series(float("nan"), index=working_df.index, dtype="float64")
    working_df.loc[:, "dual_atr_extension_mult"] = working_df["dual_atr_extension_mult"].mask(
        working_df["dual_long_entry"].fillna(False),
        pd.to_numeric(atr_extension_long_mult, errors="coerce"),
    )
    working_df.loc[:, "dual_atr_extension_mult"] = working_df["dual_atr_extension_mult"].mask(
        working_df["dual_short_entry"].fillna(False),
        pd.to_numeric(atr_extension_short_mult, errors="coerce"),
    )
    working_df.loc[:, "dual_breakout_candle_body_pct"] = breakout_candle_body_pct
    working_df.loc[:, "dual_breakout_candle_range_atr_mult"] = breakout_candle_range_atr_mult
    working_df.loc[:, "dual_late_entry_guard_enabled"] = float(resolved_use_late_entry_guard)
    working_df.loc[:, "dual_late_entry_guard_blocked"] = late_entry_guard_blocked.fillna(False)
    working_df.loc[:, "dual_late_entry_guard_block_reason"] = late_entry_guard_block_reason
    working_df.loc[:, "dual_late_entry_guard_blocked_signals"] = float(
        late_entry_guard_blocked.fillna(False).sum()
    )
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
        late_entry_max_atr_mult=float(getattr(settings.strategy, "late_entry_max_atr_mult", 0.0)),
        max_breakout_candle_body_pct=float(
            getattr(settings.strategy, "max_breakout_candle_body_pct", 0.0)
        ),
        max_breakout_candle_range_atr_mult=float(
            getattr(settings.strategy, "max_breakout_candle_range_atr_mult", 0.0)
        ),
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
