from __future__ import annotations

BACKTEST_HISTORY_START_UTC = "2023-01-01T00:00:00+00:00"
OPTIMIZER_HISTORY_START_UTC = "2024-01-01T00:00:00+00:00"

TRADING_BASE_DEFAULTS: dict[str, object] = {
    "interval": "15m",
    "start_capital": 2500.0,
    "risk_per_trade_pct": 2.0,
    "taker_fee_pct": 0.0500,
    "use_setup_gate": True,
    "use_hmm_regime_filter": True,
    "hmm_allowed_regimes": (
        "Bull Trend",
        "Bear Trend",
        "High-Vol Range",
        "Low-Vol Range",
    ),
    "min_confidence_pct": 60.0,
    "take_profit_pct": 2.5,
    "stop_loss_pct": 1.5,
    "use_trailing_stop": True,
    "trailing_activation_pct": 1.5,
    "trailing_distance_pct": 0.8,
    "min_leverage": 1,
    "max_leverage": 100,
    "default_leverage": 25,
    "max_open_positions": 3,
}

OPTIMIZATION_DEFAULTS: dict[str, object] = {
    "optimization_take_profit_pct_options": (5.0, 10.0, 15.0, 25.0),
    "optimization_stop_loss_pct_options": (3.0, 5.0, 7.0),
    "optimization_trailing_activation_pct_options": (3.0, 6.0, 10.0),
    "optimization_trailing_distance_pct_options": (0.2, 0.5, 1.0, 2.0),
    "optimization_max_sample_profiles": 10_000,
    "optimization_search_window_candles": 100000,
    "optimization_validation_top_n": 8,
}

UNIVERSAL_RISK_GRID_DEFAULTS: dict[str, tuple[float, ...]] = {
    "universal_stop_loss_pct_options": (3.0, 5.0, 7.0),
    "universal_take_profit_pct_options": (5.0, 10.0, 15.0, 25.0),
    "universal_trailing_activation_pct_options": (3.0, 6.0, 10.0),
    "universal_trailing_distance_pct_options": (0.2, 0.5, 1.0, 2.0),
}

DEFAULT_COIN_PROFILE_VALUES: dict[str, dict[str, object]] = {
    "BTCUSDT": {
        "interval": "15m",
        "dual_thrust_k1": 0.5,
        "dual_thrust_k2": 0.7,
        "dual_thrust_period": 144.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "ETHUSDT": {
        "interval": "15m",
        "stop_loss_pct": 4.5,
        "take_profit_pct": 7.0,
        "trailing_activation_pct": 3.5,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
    "SOLUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.5,
        "take_profit_pct": 10.0,
        "trailing_activation_pct": 4.5,
        "trailing_distance_pct": 0.15,
        "default_leverage": 25,
    },
    "XRPUSDT": {
        "interval": "15m",
        "stop_loss_pct": 4.0,
        "take_profit_pct": 6.0,
        "trailing_activation_pct": 3.0,
        "trailing_distance_pct": 0.25,
        "default_leverage": 25,
    },
    "ADAUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.0,
        "take_profit_pct": 9.0,
        "trailing_activation_pct": 4.0,
        "trailing_distance_pct": 0.15,
        "default_leverage": 25,
    },
    "DOGEUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.0,
        "take_profit_pct": 8.0,
        "trailing_activation_pct": 5.0,
        "trailing_distance_pct": 1.0,
        "default_leverage": 25,
    },
    "BNBUSDT": {
        "interval": "15m",
        "stop_loss_pct": 4.5,
        "take_profit_pct": 8.0,
        "trailing_activation_pct": 4.0,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
    "AVAXUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.5,
        "take_profit_pct": 9.0,
        "trailing_activation_pct": 4.0,
        "trailing_distance_pct": 0.15,
        "default_leverage": 25,
    },
    "NEARUSDT": {
        "interval": "15m",
        "stop_loss_pct": 10.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 8.0,
        "trailing_distance_pct": 4.0,
        "default_leverage": 25,
    },
    "1000SHIBUSDT": {
        "interval": "15m",
        "stop_loss_pct": 6.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "1000PEPEUSDT": {
        "interval": "15m",
        "stop_loss_pct": 6.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "DOTUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.0,
        "take_profit_pct": 9.0,
        "trailing_activation_pct": 4.0,
        "trailing_distance_pct": 0.15,
        "default_leverage": 25,
    },
}
