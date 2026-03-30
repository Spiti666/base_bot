from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from config import settings


class SmartSetupGate:
    EMA_PERIOD = 50
    VOLUME_SMA_PERIOD = 20
    ATR_PERIOD = 14
    MACRO_EMA_PERIOD = 200
    EXTREME_VOLUME_MULTIPLIER = 2.0
    VOLATILITY_RANGE_ATR_MULTIPLIER = 3.0
    EMA_SLOPE_LOOKBACK = 5
    CRASH_SLOPE_THRESHOLD_PCT = -0.5
    MEAN_REVERSION_STRATEGIES: set[str] = set()
    _PREPARED_CACHE_MAX_ENTRIES = 8

    def __init__(
        self,
        *,
        min_confidence_pct: float | None = None,
    ) -> None:
        self._min_confidence_pct = (
            settings.trading.min_confidence_pct
            if min_confidence_pct is None
            else float(min_confidence_pct)
        )
        if not 0.0 <= self._min_confidence_pct <= 100.0:
            raise ValueError("min_confidence_pct must be between 0 and 100.")
        self._prepared_cache_lock = Lock()
        self._prepared_cache: OrderedDict[
            tuple[int, int, object, float | None],
            dict[str, np.ndarray],
        ] = OrderedDict()

    @classmethod
    def required_candle_count(cls) -> int:
        return max(
            cls.EMA_PERIOD,
            cls.VOLUME_SMA_PERIOD,
            cls.ATR_PERIOD + 1,
            cls.MACRO_EMA_PERIOD,
        )

    def evaluate_signal(
        self,
        candles_df: Any,
        signal_direction: int,
        strategy_name: str,
    ) -> tuple[bool, float, str]:
        last_index = len(candles_df) - 1
        return self.evaluate_signal_at_index(
            candles_df,
            last_index,
            signal_direction,
            strategy_name,
        )

    def evaluate_signal_at_index(
        self,
        candles_df: Any,
        index: int,
        signal_direction: int,
        strategy_name: str,
    ) -> tuple[bool, float, str]:
        if signal_direction not in (-1, 1):
            return False, 0.0, "Neutral signal."

        prepared = self._prepare_runtime_arrays(candles_df)
        closes = prepared["close"]
        opens = prepared["open"]
        volumes = prepared["volume"]
        highs = prepared["high"]
        lows = prepared["low"]
        row_count = int(closes.size)
        if row_count <= 0:
            return False, 0.0, "Not enough candles for Setup Gate (need 200)."
        if index < 0:
            index = row_count + int(index)
        if index < 0 or index >= row_count:
            return False, 0.0, "Signal index out of bounds."

        required_candles = self.required_candle_count()
        if index + 1 < required_candles:
            return (
                False,
                0.0,
                f"Not enough candles for Setup Gate (need {required_candles}).",
            )

        current_price = float(closes[index])
        current_volume = float(volumes[index])
        ema_50 = float(prepared["ema_50"][index])
        macro_ema = float(prepared["macro_ema"][index])
        volume_sma = float(prepared["volume_sma"][index])
        atr_14 = float(prepared["atr_14"][index])
        current_range = float(prepared["current_range"][index])
        dominant_wick_ratio = float(prepared["dominant_wick_ratio"][index])
        ema_slope_pct = float(prepared["ema_slope_pct"][index])

        if signal_direction > 0 and current_price < macro_ema:
            return (
                False,
                0.0,
                f"Macro context blocked LONG: price {current_price:.4f} < EMA200 {macro_ema:.4f}",
            )
        if signal_direction < 0 and current_price > macro_ema:
            return (
                False,
                0.0,
                f"Macro context blocked SHORT: price {current_price:.4f} > EMA200 {macro_ema:.4f}",
            )

        score = 50.0
        reasons: list[str] = []
        if strategy_name in self.MEAN_REVERSION_STRATEGIES:
            if signal_direction > 0:
                if current_price < ema_50:
                    score += 25.0
                    reasons.append(
                        f"Mean reversion aligned: price below EMA50 ({current_price:.4f} < {ema_50:.4f})"
                    )
                else:
                    score -= 25.0
                    reasons.append(
                        f"Mean reversion weak: price above EMA50 ({current_price:.4f} >= {ema_50:.4f})"
                    )
            else:
                if current_price > ema_50:
                    score += 25.0
                    reasons.append(
                        f"Mean reversion aligned: price above EMA50 ({current_price:.4f} > {ema_50:.4f})"
                    )
                else:
                    score -= 25.0
                    reasons.append(
                        f"Mean reversion weak: price below EMA50 ({current_price:.4f} <= {ema_50:.4f})"
                    )
        else:
            if signal_direction > 0:
                if current_price > ema_50:
                    score += 25.0
                    reasons.append(
                        f"Trend aligned: price above EMA50 ({current_price:.4f} > {ema_50:.4f})"
                    )
                else:
                    score -= 25.0
                    reasons.append(
                        f"Trend weak: price below EMA50 ({current_price:.4f} <= {ema_50:.4f})"
                    )
            else:
                if current_price < ema_50:
                    score += 25.0
                    reasons.append(
                        f"Trend aligned: price below EMA50 ({current_price:.4f} < {ema_50:.4f})"
                    )
                else:
                    score -= 25.0
                    reasons.append(
                        f"Trend weak: price above EMA50 ({current_price:.4f} >= {ema_50:.4f})"
                    )

        if current_volume > volume_sma:
            score += 25.0
            reasons.append(
                f"Volume confirmed: {current_volume:.4f} > SMA20 {volume_sma:.4f}"
            )
        else:
            reasons.append(
                f"Volume unconfirmed: {current_volume:.4f} <= SMA20 {volume_sma:.4f}"
            )

        if (
            strategy_name in self.MEAN_REVERSION_STRATEGIES
            and signal_direction > 0
            and np.isfinite(volume_sma)
            and volume_sma > 0.0
            and current_volume > volume_sma * self.EXTREME_VOLUME_MULTIPLIER
        ):
            score -= 10.0
            reasons.append(
                f"Possible panic selling: volume {current_volume:.4f} > {self.EXTREME_VOLUME_MULTIPLIER:.1f}x SMA20"
            )

        if strategy_name in self.MEAN_REVERSION_STRATEGIES and signal_direction > 0:
            if np.isfinite(ema_slope_pct) and ema_slope_pct <= self.CRASH_SLOPE_THRESHOLD_PCT:
                score -= 20.0
                reasons.append(
                    f"Trend strength filter: EMA50 slope {ema_slope_pct:.2f}% over {self.EMA_SLOPE_LOOKBACK} candles"
                )

        volatility_multiplier = self._volatility_multiplier_for_strategy(strategy_name)
        if np.isfinite(atr_14) and atr_14 > 0.0 and current_range > atr_14 * volatility_multiplier:
            score -= 40.0
            reasons.append(
                f"Volatility filter: candle range {current_range:.4f} > "
                f"{volatility_multiplier:.1f}x ATR14 {atr_14:.4f}"
            )

        if dominant_wick_ratio > 0.60:
            score -= 30.0
            reasons.append(
                f"Candle quality filter: dominant wick {dominant_wick_ratio * 100.0:.1f}% of candle range"
            )

        capped_score = max(0.0, min(100.0, score))
        is_approved = capped_score >= self._min_confidence_pct
        return is_approved, capped_score, "; ".join(reasons)

    def _prepare_runtime_arrays(self, candles_df: Any) -> dict[str, np.ndarray]:
        frame_cache_key = self._frame_cache_key(candles_df)
        with self._prepared_cache_lock:
            cached = self._prepared_cache.get(frame_cache_key)
            if cached is not None:
                self._prepared_cache.move_to_end(frame_cache_key)
                return cached

        close = self._extract_numpy(candles_df, "close")
        open_ = self._extract_numpy(candles_df, "open")
        high = self._extract_numpy(candles_df, "high")
        low = self._extract_numpy(candles_df, "low")
        volume = self._extract_numpy(candles_df, "volume")

        ema_50 = (
            pd.Series(close)
            .ewm(span=self.EMA_PERIOD, adjust=False)
            .mean()
            .to_numpy(dtype=np.float64, copy=False)
        )
        macro_ema = (
            pd.Series(close)
            .ewm(span=self.MACRO_EMA_PERIOD, adjust=False)
            .mean()
            .to_numpy(dtype=np.float64, copy=False)
        )
        volume_sma = (
            pd.Series(volume)
            .rolling(window=self.VOLUME_SMA_PERIOD, min_periods=self.VOLUME_SMA_PERIOD)
            .mean()
            .to_numpy(dtype=np.float64, copy=False)
        )

        prev_close = np.empty_like(close)
        prev_close[0] = np.nan
        prev_close[1:] = close[:-1]
        tr = np.maximum.reduce(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        )
        atr_14 = (
            pd.Series(tr)
            .rolling(window=self.ATR_PERIOD, min_periods=self.ATR_PERIOD)
            .mean()
            .to_numpy(dtype=np.float64, copy=False)
        )

        current_range = high - low
        upper_wick = np.maximum(high - np.maximum(open_, close), 0.0)
        lower_wick = np.maximum(np.minimum(open_, close) - low, 0.0)
        dominant_wick_ratio = np.divide(
            np.maximum(upper_wick, lower_wick),
            current_range,
            out=np.zeros_like(current_range, dtype=np.float64),
            where=current_range > 0.0,
        )

        ema_slope_pct = np.full(close.shape[0], np.nan, dtype=np.float64)
        lookback = int(self.EMA_SLOPE_LOOKBACK)
        if lookback > 0 and close.shape[0] > lookback:
            start_values = ema_50[:-lookback]
            end_values = ema_50[lookback:]
            ema_slope_pct[lookback:] = np.divide(
                end_values - start_values,
                start_values,
                out=np.zeros_like(end_values, dtype=np.float64),
                where=start_values != 0.0,
            ) * 100.0

        prepared = {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "ema_50": ema_50,
            "macro_ema": macro_ema,
            "volume_sma": volume_sma,
            "atr_14": atr_14,
            "current_range": current_range,
            "dominant_wick_ratio": dominant_wick_ratio,
            "ema_slope_pct": ema_slope_pct,
        }
        with self._prepared_cache_lock:
            self._prepared_cache[frame_cache_key] = prepared
            while len(self._prepared_cache) > self._PREPARED_CACHE_MAX_ENTRIES:
                self._prepared_cache.popitem(last=False)
        return prepared

    @staticmethod
    def _frame_cache_key(candles_df: Any) -> tuple[int, int, object, float | None]:
        row_count = int(len(candles_df))
        if row_count <= 0:
            return (id(candles_df), 0, None, None)
        tail_marker: object
        try:
            if hasattr(candles_df, "columns") and "open_time" in candles_df.columns:
                tail_marker = candles_df["open_time"].iat[-1]
            elif hasattr(candles_df, "index"):
                tail_marker = candles_df.index[-1]
            else:
                tail_marker = row_count - 1
        except Exception:
            tail_marker = row_count - 1
        try:
            close_tail = float(candles_df["close"].iat[-1])  # type: ignore[index]
        except Exception:
            close_tail = None
        return (id(candles_df), row_count, tail_marker, close_tail)

    @staticmethod
    def _extract_numpy(candles_df: Any, column_name: str) -> np.ndarray:
        try:
            series = candles_df[column_name]
        except Exception as exc:
            raise ValueError(f"candles_df must provide a '{column_name}' column.") from exc
        if hasattr(series, "to_numpy"):
            values = series.to_numpy(dtype=np.float64, copy=False)
            return np.asarray(values, dtype=np.float64)
        return np.asarray([float(value) for value in series], dtype=np.float64)

    @classmethod
    def _volatility_multiplier_for_strategy(cls, strategy_name: str) -> float:
        return float(cls.VOLATILITY_RANGE_ATR_MULTIPLIER)

    @staticmethod
    def _extract_series(candles_df: Any, column_name: str) -> list[float]:
        try:
            series = candles_df[column_name]
        except Exception as exc:
            raise ValueError(f"candles_df must provide a '{column_name}' column.") from exc

        if hasattr(series, "tolist"):
            raw_values = series.tolist()
        else:
            raw_values = list(series)

        return [float(value) for value in raw_values]

    @staticmethod
    def _calculate_ema_series(values: list[float], period: int) -> list[float]:
        multiplier = 2.0 / (period + 1.0)
        ema_values = [values[0]]
        for value in values[1:]:
            ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    @staticmethod
    def _calculate_sma(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _calculate_ema_slope_pct(ema_values: list[float], *, lookback: int) -> float:
        if lookback <= 0 or len(ema_values) <= lookback:
            raise ValueError("Not enough EMA values to calculate slope.")
        start_value = ema_values[-(lookback + 1)]
        end_value = ema_values[-1]
        if start_value == 0.0:
            return 0.0
        return ((end_value - start_value) / start_value) * 100.0

    @staticmethod
    def _calculate_atr(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        *,
        period: int,
    ) -> float:
        true_ranges: list[float] = []
        for index in range(1, len(closes)):
            current_high = highs[index]
            current_low = lows[index]
            previous_close = closes[index - 1]
            true_ranges.append(
                max(
                    current_high - current_low,
                    abs(current_high - previous_close),
                    abs(current_low - previous_close),
                )
            )
        if len(true_ranges) < period:
            raise ValueError("Not enough candles to calculate ATR.")
        return sum(true_ranges[-period:]) / period

    @staticmethod
    def _calculate_dominant_wick_ratio(
        *,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> float:
        total_range = high_price - low_price
        if total_range <= 0.0:
            return 0.0

        upper_wick = max(high_price - max(open_price, close_price), 0.0)
        lower_wick = max(min(open_price, close_price) - low_price, 0.0)
        return max(upper_wick, lower_wick) / total_range
