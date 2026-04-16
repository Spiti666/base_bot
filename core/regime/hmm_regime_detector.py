from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

_HMM_NON_CONVERGENCE_PREFIX = "Model is not converging."
_HMM_TRANSMAT_ZERO_SUM_PREFIX = (
    "Some rows of transmat_ have zero sum because no transition from the state was ever observed."
)


class _SuppressHMMNonConvergenceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            message = str(record.getMessage())
        except Exception:
            return True
        return _HMM_NON_CONVERGENCE_PREFIX not in message


@contextmanager
def _suppress_hmmlearn_non_convergence_logs():
    logger = logging.getLogger("hmmlearn.base")
    filter_instance = _SuppressHMMNonConvergenceFilter()
    logger.addFilter(filter_instance)
    try:
        yield
    finally:
        logger.removeFilter(filter_instance)


@dataclass(frozen=True, slots=True)
class HMMRegimeDetectionResult:
    regime_mask: np.ndarray
    regime_labels: list[str]
    allowed_ratio: float
    window_count: int
    converged: bool
    non_converged_windows: int
    non_convergence_events: tuple[dict[str, object], ...]
    warning_events: tuple[dict[str, object], ...]


class HMMRegimeDetector:
    BULL_TREND = "Bull Trend"
    BEAR_TREND = "Bear Trend"
    HIGH_VOL_RANGE = "High-Vol Range"
    LOW_VOL_RANGE = "Low-Vol Range"
    UNKNOWN = "Unknown"
    WARMUP = "Warmup"

    _EPSILON = 1e-12

    def __init__(
        self,
        *,
        n_components: int = 4,
        train_window_candles: int = 8000,
        apply_window_candles: int = 2000,
        warmup_candles: int = 2000,
        stability_candles: int = 5,
        allowed_regimes: Sequence[str] | None = None,
        random_state: int = 42,
        n_iter: int = 200,
    ) -> None:
        self._n_components = max(int(n_components), 2)
        self._train_window_candles = max(int(train_window_candles), self._n_components * 10)
        self._apply_window_candles = max(int(apply_window_candles), 1)
        self._warmup_candles = max(int(warmup_candles), 0)
        self._stability_candles = max(int(stability_candles), 1)
        self._allowed_regimes = set(
            allowed_regimes or (self.BULL_TREND, self.LOW_VOL_RANGE)
        )
        self._random_state = int(random_state)
        self._n_iter = max(int(n_iter), 50)

    def detect(self, candles_dataframe: pd.DataFrame) -> HMMRegimeDetectionResult:
        features = self._build_feature_matrix(candles_dataframe)
        length = features.shape[0]
        if length == 0:
            return HMMRegimeDetectionResult(
                regime_mask=np.zeros(0, dtype=np.int8),
                regime_labels=[],
                allowed_ratio=0.0,
                window_count=0,
                converged=True,
                non_converged_windows=0,
                non_convergence_events=tuple(),
                warning_events=tuple(),
            )

        warmup = min(self._warmup_candles, length)
        regime_labels = [self.WARMUP] * length
        regime_mask = np.ones(length, dtype=np.int8)
        window_count = 0
        non_converged_windows = 0
        non_convergence_events: list[dict[str, object]] = []
        warning_events: list[dict[str, object]] = []

        for apply_start in range(warmup, length, self._apply_window_candles):
            train_end = apply_start
            train_start = max(0, train_end - self._train_window_candles)
            if train_end - train_start < self._n_components * 10:
                apply_end = min(apply_start + self._apply_window_candles, length)
                for index in range(apply_start, apply_end):
                    regime_labels[index] = self.WARMUP
                    regime_mask[index] = 1
                continue

            train_features_raw = features[train_start:train_end]
            train_features, scaling = self._normalize_features(train_features_raw)
            model = GaussianHMM(
                n_components=self._n_components,
                covariance_type="diag",
                n_iter=self._n_iter,
                random_state=self._random_state + window_count,
            )
            apply_end = min(apply_start + self._apply_window_candles, length)
            try:
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always")
                    with _suppress_hmmlearn_non_convergence_logs():
                        model.fit(train_features)
                for warning_record in fit_warnings:
                    warning_text = str(getattr(warning_record, "message", "") or "").strip()
                    if not warning_text:
                        continue
                    if _HMM_TRANSMAT_ZERO_SUM_PREFIX not in warning_text:
                        continue
                    warning_events.append(
                        {
                            "warning_kind": "transmat_zero_sum_no_transition",
                            "window_index": int(window_count),
                            "train_start": int(train_start),
                            "train_end": int(train_end),
                            "apply_start": int(apply_start),
                            "apply_end": int(apply_end),
                            "warning_message": warning_text,
                        }
                    )
                monitor = getattr(model, "monitor_", None)
                if monitor is not None and not bool(getattr(monitor, "converged", True)):
                    non_converged_windows += 1
                    previous_score: float | None = None
                    current_score: float | None = None
                    delta_score: float | None = None
                    history = list(getattr(monitor, "history", []) or [])
                    if len(history) >= 2:
                        previous_score = float(history[-2])
                        current_score = float(history[-1])
                        delta_score = float(current_score - previous_score)
                    non_convergence_events.append(
                        {
                            "window_index": int(window_count),
                            "train_start": int(train_start),
                            "train_end": int(train_end),
                            "apply_start": int(apply_start),
                            "apply_end": int(apply_end),
                            "previous_score": previous_score,
                            "current_score": current_score,
                            "delta_score": delta_score,
                        }
                    )
                    for index in range(apply_start, apply_end):
                        regime_labels[index] = self.UNKNOWN
                        regime_mask[index] = 1
                    window_count += 1
                    continue
                train_states = model.predict(train_features)
                state_mapping = self._map_states_from_training(
                    train_states=train_states,
                    train_features=train_features_raw,
                )
                apply_features = self._apply_scaling(features[apply_start:apply_end], scaling)
                apply_states = model.predict(apply_features)
                for relative_index, state in enumerate(apply_states):
                    absolute_index = apply_start + relative_index
                    regime_name = state_mapping.get(int(state), self.UNKNOWN)
                    regime_labels[absolute_index] = regime_name
                    regime_mask[absolute_index] = 1 if regime_name in self._allowed_regimes else 0
            except Exception:
                for index in range(apply_start, apply_end):
                    regime_labels[index] = self.UNKNOWN
                    regime_mask[index] = 1
            window_count += 1

        stable_labels = self._stabilize_regimes(regime_labels, warmup=warmup)
        for index in range(length):
            if index < warmup:
                regime_mask[index] = 1
                continue
            regime_mask[index] = 1 if stable_labels[index] in self._allowed_regimes else 0

        allowed_ratio = float(regime_mask.sum()) / float(max(length, 1))
        return HMMRegimeDetectionResult(
            regime_mask=regime_mask,
            regime_labels=stable_labels,
            allowed_ratio=allowed_ratio,
            window_count=window_count,
            converged=(non_converged_windows == 0),
            non_converged_windows=non_converged_windows,
            non_convergence_events=tuple(non_convergence_events),
            warning_events=tuple(warning_events),
        )

    def _build_feature_matrix(self, candles_dataframe: pd.DataFrame) -> np.ndarray:
        required_columns = {"close", "volume"}
        missing_columns = required_columns.difference(candles_dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"HMM regime detector requires columns: {missing}")

        close = pd.to_numeric(candles_dataframe["close"], errors="coerce")
        volume = pd.to_numeric(candles_dataframe["volume"], errors="coerce")
        close = close.ffill().bfill().fillna(0.0)
        volume = volume.ffill().bfill().fillna(0.0)

        shifted_close = close.shift(1)
        safe_ratio = (close + self._EPSILON) / (shifted_close + self._EPSILON)
        log_returns = np.log(safe_ratio).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        volatility_20 = (
            log_returns.rolling(window=20, min_periods=1).std().fillna(0.0)
        )
        volume_mean_20 = volume.rolling(window=20, min_periods=1).mean().fillna(0.0)
        volume_std_20 = volume.rolling(window=20, min_periods=1).std().fillna(0.0)
        volume_zscore = ((volume - volume_mean_20) / (volume_std_20 + self._EPSILON)).fillna(0.0)

        features = np.column_stack(
            (
                log_returns.to_numpy(dtype=np.float64),
                volatility_20.to_numpy(dtype=np.float64),
                volume_zscore.to_numpy(dtype=np.float64),
            )
        )
        features[~np.isfinite(features)] = 0.0
        return features

    def _normalize_features(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < self._EPSILON, 1.0, std)
        normalized = (features - mean) / std
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized, (mean, std)

    def _apply_scaling(
        self,
        features: np.ndarray,
        scaling: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        mean, std = scaling
        normalized = (features - mean) / std
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized

    def _map_states_from_training(
        self,
        *,
        train_states: np.ndarray,
        train_features: np.ndarray,
    ) -> dict[int, str]:
        state_ids = list(range(self._n_components))
        return_means = np.full(self._n_components, np.nan, dtype=np.float64)
        vol_means = np.full(self._n_components, np.nan, dtype=np.float64)

        for state_id in state_ids:
            mask = train_states == state_id
            if not np.any(mask):
                continue
            state_features = train_features[mask]
            return_means[state_id] = np.mean(state_features[:, 0])
            vol_means[state_id] = np.mean(state_features[:, 1])

        mapping = {state_id: self.UNKNOWN for state_id in state_ids}
        used_states: set[int] = set()

        bull_state = self._pick_state(return_means, descending=True, exclude=used_states)
        if bull_state is not None:
            mapping[bull_state] = self.BULL_TREND
            used_states.add(bull_state)

        bear_state = self._pick_state(return_means, descending=False, exclude=used_states)
        if bear_state is not None:
            mapping[bear_state] = self.BEAR_TREND
            used_states.add(bear_state)

        high_vol_state = self._pick_state(vol_means, descending=True, exclude=used_states)
        if high_vol_state is not None:
            mapping[high_vol_state] = self.HIGH_VOL_RANGE
            used_states.add(high_vol_state)

        low_vol_state = self._pick_state(vol_means, descending=False, exclude=used_states)
        if low_vol_state is not None:
            mapping[low_vol_state] = self.LOW_VOL_RANGE
            used_states.add(low_vol_state)

        return mapping

    @staticmethod
    def _pick_state(
        metric_values: np.ndarray,
        *,
        descending: bool,
        exclude: set[int],
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        for state_id, metric in enumerate(metric_values.tolist()):
            if state_id in exclude:
                continue
            if not np.isfinite(metric):
                continue
            candidates.append((float(metric), state_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=descending)
        return int(candidates[0][1])

    def _stabilize_regimes(self, labels: list[str], *, warmup: int) -> list[str]:
        length = len(labels)
        if length == 0:
            return labels
        if warmup >= length:
            return labels

        stable_labels = list(labels)
        active_label = stable_labels[warmup - 1] if warmup > 0 else stable_labels[0]
        candidate_label = active_label
        candidate_count = 0

        for index in range(warmup, length):
            current_label = labels[index]
            if current_label == active_label:
                candidate_label = active_label
                candidate_count = 0
            else:
                if current_label == candidate_label:
                    candidate_count += 1
                else:
                    candidate_label = current_label
                    candidate_count = 1
                if candidate_count >= self._stability_candles:
                    active_label = candidate_label
                    candidate_count = 0
            stable_labels[index] = active_label
        return stable_labels
