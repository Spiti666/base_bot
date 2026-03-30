from __future__ import annotations

AVAILABLE_STRATEGIES: tuple[str, ...] = (
    "ema_cross_volume",
    "frama_cross",
    "dual_thrust",
    "rsi_extreme_cluster",
)

DEFAULT_STRATEGY_NAME = "ema_cross_volume"

DEFAULT_COIN_STRATEGIES: dict[str, str] = {
    "BTCUSDT": "dual_thrust_breakout",
    "ETHUSDT": "dual_thrust",
    "XRPUSDT": "dual_thrust_breakout",
    "SOLUSDT": "frama_cross",
    "ADAUSDT": "frama_cross",
    "DOGEUSDT": "dual_thrust_breakout",
    "1000SHIBUSDT": "dual_thrust",
    "BNBUSDT": "dual_thrust_breakout",
    "AVAXUSDT": "dual_thrust_breakout",
    "NEARUSDT": "dual_thrust",
    "SUIUSDT": "dual_thrust_breakout",
    "1000PEPEUSDT": "dual_thrust_breakout",
    "DOTUSDT": "dual_thrust_breakout",
}

DEFAULT_COIN_STRATEGY_PARAMS: dict[str, dict[str, float]] = {
    "BTCUSDT": {"dual_thrust_k1": 0.5, "dual_thrust_k2": 0.7, "dual_thrust_period": 144.0},
    "ETHUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.6, "dual_thrust_period": 34.0},
    "SOLUSDT": {"frama_fast_period": 6.0, "frama_slow_period": 40.0},
    "XRPUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.4, "dual_thrust_period": 144.0},
    "ADAUSDT": {"frama_fast_period": 9.0, "frama_slow_period": 40.0},
    "DOGEUSDT": {"dual_thrust_k1": 0.7, "dual_thrust_k2": 0.5, "dual_thrust_period": 144.0},
    "1000SHIBUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.4, "dual_thrust_period": 89.0},
    "BNBUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.7, "dual_thrust_period": 89.0},
    "AVAXUSDT": {"dual_thrust_k1": 0.4, "dual_thrust_k2": 0.6, "dual_thrust_period": 144.0},
    "NEARUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.6, "dual_thrust_period": 89.0},
    "SUIUSDT": {"dual_thrust_k1": 0.7, "dual_thrust_k2": 0.4, "dual_thrust_period": 89.0},
    "1000PEPEUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.6, "dual_thrust_period": 89.0},
    "DOTUSDT": {"dual_thrust_k1": 0.6, "dual_thrust_k2": 0.6, "dual_thrust_period": 55.0},
}

DEFAULT_SCAN_STRATEGIES: tuple[str, ...] = (
    "dual_thrust",
    "frama_cross",
    "ema_cross_volume",
    "rsi_extreme_cluster",
)
