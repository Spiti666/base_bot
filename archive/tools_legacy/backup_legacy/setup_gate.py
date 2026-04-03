from __future__ import annotations

from typing import Any

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

    def __init__(self, *, min_confidence_pct: float | None = None) -> None:
        self._min_confidence_pct = (
            settings.trading.min_confidence_pct
            if min_confidence_pct is None
            else float(min_confidence_pct)
        )
        if not 0.0 <= self._min_confidence_pct <= 100.0:
            raise ValueError("min_confidence_pct must be between 0 and 100.")

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
        if signal_direction not in (-1, 1):
            return False, 0.0, "Neutral signal."

        closes = self._extract_series(candles_df, "close")
        opens = self._extract_series(candles_df, "open")
        volumes = self._extract_series(candles_df, "volume")
        highs = self._extract_series(candles_df, "high")
        lows = self._extract_series(candles_df, "low")
        required_candles = self.required_candle_count()
        if (
            len(closes) < required_candles
            or len(opens) < required_candles
            or len(volumes) < required_candles
            or len(highs) < required_candles
            or len(lows) < required_candles
        ):
            return (
                False,
                0.0,
                f"Not enough candles for Setup Gate (need {required_candles}).",
            )

        current_price = closes[-1]
        current_volume = volumes[-1]
        volume_sma = self._calculate_sma(volumes[-self.VOLUME_SMA_PERIOD :])
        ema_series = self._calculate_ema_series(closes, self.EMA_PERIOD)
        ema_50 = ema_series[-1]
        atr_14 = self._calculate_atr(highs, lows, closes, period=self.ATR_PERIOD)
        macro_ema = self._calculate_ema_series(closes, self.MACRO_EMA_PERIOD)[-1]
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
        current_range = highs[-1] - lows[-1]
        dominant_wick_ratio = self._calculate_dominant_wick_ratio(
            open_price=opens[-1],
            high_price=highs[-1],
            low_price=lows[-1],
            close_price=closes[-1],
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
            and volume_sma > 0.0
            and current_volume > volume_sma * self.EXTREME_VOLUME_MULTIPLIER
        ):
            score -= 10.0
            reasons.append(
                f"Possible panic selling: volume {current_volume:.4f} > {self.EXTREME_VOLUME_MULTIPLIER:.1f}x SMA20"
            )

        if strategy_name in self.MEAN_REVERSION_STRATEGIES and signal_direction > 0:
            ema_slope_pct = self._calculate_ema_slope_pct(
                ema_series,
                lookback=self.EMA_SLOPE_LOOKBACK,
            )
            if ema_slope_pct <= self.CRASH_SLOPE_THRESHOLD_PCT:
                score -= 20.0
                reasons.append(
                    f"Trend strength filter: EMA50 slope {ema_slope_pct:.2f}% over {self.EMA_SLOPE_LOOKBACK} candles"
                )

        volatility_multiplier = self._volatility_multiplier_for_strategy(strategy_name)
        if atr_14 > 0.0 and current_range > atr_14 * volatility_multiplier:
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
