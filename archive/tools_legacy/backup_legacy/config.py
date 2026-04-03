from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from config_sections.backtest_optimizer_defaults import (
    BACKTEST_HISTORY_START_UTC,
    DEFAULT_COIN_PROFILE_VALUES,
    OPTIMIZATION_DEFAULTS,
    OPTIMIZER_HISTORY_START_UTC,
    TRADING_BASE_DEFAULTS,
    UNIVERSAL_RISK_GRID_DEFAULTS,
)
from config_sections.live_defaults import LIVE_AVAILABLE_SYMBOLS, LIVE_DEFAULT_INTERVAL
from config_sections.production_registry import (
    PRODUCTION_PROFILE_INTERVAL,
    PRODUCTION_PROFILE_REGISTRY,
    PRODUCTION_STRATEGY_ALIASES,
)
from config_sections.strategy_defaults import (
    AVAILABLE_STRATEGIES,
    DEFAULT_COIN_STRATEGIES,
    DEFAULT_COIN_STRATEGY_PARAMS,
    DEFAULT_SCAN_STRATEGIES,
    DEFAULT_STRATEGY_NAME,
)

# Dedicated backtest universe (aligned with active live symbols).
BACKTEST_BATCH_SYMBOLS: tuple[str, ...] = tuple(dict.fromkeys(LIVE_AVAILABLE_SYMBOLS))
MAX_BACKTEST_CANDLES = 100_000


DEFAULT_COIN_PROFILE_VALUES = {
    "BTCUSDT": {
        "interval": "15m",
        "stop_loss_pct": 8.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 1.0,
        "default_leverage": 25,
    },
    "ETHUSDT": {
        "interval": "15m",
        "stop_loss_pct": 8.0,
        "take_profit_pct": 25.0,
        "trailing_activation_pct": 8.0,
        "trailing_distance_pct": 5.0,
        "default_leverage": 25,
    },
    "SOLUSDT": {
        "interval": "15m",
        "stop_loss_pct": 7.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 0.5,
        "default_leverage": 25,
    },
    "XRPUSDT": {
        "interval": "15m",
        "stop_loss_pct": 3.0,
        "take_profit_pct": 5.0,
        "trailing_activation_pct": 3.0,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
    "ADAUSDT": {
        "interval": "15m",
        "stop_loss_pct": 2.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 0.15,
        "default_leverage": 25,
    },
    "DOGEUSDT": {
        "interval": "15m",
        "stop_loss_pct": 7.0,
        "take_profit_pct": 10.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "BNBUSDT": {
        "interval": "15m",
        "stop_loss_pct": 5.0,
        "take_profit_pct": 5.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "AVAXUSDT": {
        "interval": "15m",
        "stop_loss_pct": 7.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 2.0,
        "default_leverage": 25,
    },
    "NEARUSDT": {
        "interval": "15m",
        "stop_loss_pct": 7.0,
        "take_profit_pct": 25.0,
        "trailing_activation_pct": 10.0,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
    "1000PEPEUSDT": {
        "interval": "15m",
        "stop_loss_pct": 3.0,
        "take_profit_pct": 5.0,
        "trailing_activation_pct": 3.0,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
    "1000SHIBUSDT": {
        "interval": "15m",
        "stop_loss_pct": 6.0,
        "take_profit_pct": 15.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 0.5,
        "default_leverage": 25,
    },
    "DOTUSDT": {
        "interval": "15m",
        "stop_loss_pct": 3.0,
        "take_profit_pct": 5.0,
        "trailing_activation_pct": 6.0,
        "trailing_distance_pct": 0.2,
        "default_leverage": 25,
    },
}

DEFAULT_COIN_STRATEGIES = {
    "BTCUSDT": "dual_thrust",
    "ETHUSDT": "dual_thrust",
    "XRPUSDT": "frama_cross",
    "SOLUSDT": "ema_cross_volume",
    "ADAUSDT": "dual_thrust",
    "DOGEUSDT": "ema_cross_volume",
    "BNBUSDT": "ema_cross_volume",
    "AVAXUSDT": "ema_cross_volume",
    "NEARUSDT": "ema_cross_volume",
    "1000PEPEUSDT": "frama_cross",
    "1000SHIBUSDT": "dual_thrust",
    "DOTUSDT": "frama_cross",
}

DEFAULT_COIN_STRATEGY_PARAMS = {
    "BTCUSDT": {"dual_thrust_k1": 0.5, "dual_thrust_k2": 0.6, "dual_thrust_period": 144.0},
    "ETHUSDT": {"dual_thrust_k1": 0.2, "dual_thrust_k2": 1.6, "dual_thrust_period": 377.0},
    "SOLUSDT": {"ema_fast_period": 21.0, "ema_slow_period": 800.0, "volume_multiplier": 3.0},
    "XRPUSDT": {"frama_fast_period": 6.0, "frama_slow_period": 150.0, "volume_multiplier": 4.0},
    "ADAUSDT": {"dual_thrust_k1": 1.2, "dual_thrust_k2": 0.5, "dual_thrust_period": 55.0},
    "DOGEUSDT": {"ema_fast_period": 55.0, "ema_slow_period": 300.0, "volume_multiplier": 3.0},
    "BNBUSDT": {"ema_fast_period": 21.0, "ema_slow_period": 250.0, "volume_multiplier": 4.0},
    "AVAXUSDT": {"ema_fast_period": 25.0, "ema_slow_period": 250.0, "volume_multiplier": 3.0},
    "NEARUSDT": {"ema_fast_period": 50.0, "ema_slow_period": 144.0, "volume_multiplier": 3.0},
    "1000PEPEUSDT": {"frama_fast_period": 12.0, "frama_slow_period": 200.0, "volume_multiplier": 2.5},
    "1000SHIBUSDT": {"dual_thrust_k1": 1.0, "dual_thrust_k2": 0.6, "dual_thrust_period": 55.0},
    "DOTUSDT": {"frama_fast_period": 6.0, "frama_slow_period": 40.0, "volume_multiplier": 4.0},
}

@dataclass(frozen=True, slots=True)
class CoinProfileSettings:
    interval: str | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_activation_pct: float | None = None
    trailing_distance_pct: float | None = None
    default_leverage: int | None = None


def _build_default_coin_strategies() -> dict[str, str]:
    return dict(DEFAULT_COIN_STRATEGIES)


def _build_default_coin_strategy_params() -> dict[str, dict[str, float]]:
    return {
        symbol: dict(params)
        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items()
    }


def _build_default_coin_profiles() -> dict[str, CoinProfileSettings]:
    return {
        symbol: CoinProfileSettings(
            interval=(
                None if profile.get("interval") is None else str(profile["interval"])
            ),
            stop_loss_pct=(
                None if profile.get("stop_loss_pct") is None else float(profile["stop_loss_pct"])
            ),
            take_profit_pct=(
                None if profile.get("take_profit_pct") is None else float(profile["take_profit_pct"])
            ),
            trailing_activation_pct=(
                None
                if profile.get("trailing_activation_pct") is None
                else float(profile["trailing_activation_pct"])
            ),
            trailing_distance_pct=(
                None
                if profile.get("trailing_distance_pct") is None
                else float(profile["trailing_distance_pct"])
            ),
            default_leverage=(
                None if profile.get("default_leverage") is None else int(profile["default_leverage"])
            ),
        )
        for symbol, profile in DEFAULT_COIN_PROFILE_VALUES.items()
    }


def _float_range_tuple(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("step must be positive.")
    values: list[float] = []
    index = 0
    current = float(start)
    epsilon = step * 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        index += 1
        current = start + step * index
    return tuple(values)


@dataclass(frozen=True, slots=True)
class RequestLimits:
    public_requests_per_second: int = 10
    private_requests_per_second: int = 10
    candles_per_request: int = 200
    request_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.public_requests_per_second <= 0:
            raise ValueError("public_requests_per_second must be positive.")
        if self.private_requests_per_second <= 0:
            raise ValueError("private_requests_per_second must be positive.")
        if self.candles_per_request <= 0:
            raise ValueError("candles_per_request must be positive.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class APISettings:
    base_url: str = "https://fapi.bitunix.com"
    timeframes: tuple[str, ...] = (
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    )
    request_limits: RequestLimits = field(default_factory=RequestLimits)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty.")
        if not self.timeframes:
            raise ValueError("At least one timeframe must be configured.")


@dataclass(frozen=True, slots=True)
class LiveSettings:
    default_interval: str = LIVE_DEFAULT_INTERVAL
    available_symbols: tuple[str, ...] = LIVE_AVAILABLE_SYMBOLS

    def __post_init__(self) -> None:
        if not self.default_interval:
            raise ValueError("default_interval must not be empty.")
        if not self.available_symbols:
            raise ValueError("available_symbols must not be empty.")


@dataclass(frozen=True, slots=True)
class StrategySettings:
    dual_thrust_period: int = 4
    dual_thrust_k1: float = 0.5
    dual_thrust_k2: float = 0.5
    dual_thrust_opt_periods: tuple[int, ...] = (21, 55, 144, 377, 610)
    dual_thrust_opt_k_values: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.4,
        1.6,
    )
    dual_thrust_stop_loss_pct_options: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    dual_thrust_take_profit_pct_options: tuple[float, ...] = (5.0, 10.0, 15.0, 25.0)
    dual_thrust_trailing_activation_pct_options: tuple[float, ...] = (6.0, 8.0, 10.0, 15.0)
    dual_thrust_trailing_distance_pct_options: tuple[float, ...] = (
        0.15,
        0.3,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    )
    dual_thrust_symbol_optimization_grids: dict[str, dict[str, tuple[float, ...]]] = field(
        default_factory=dict
    )
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_fast_options: tuple[int, ...] = (5, 8, 10, 12, 15, 21, 25, 30, 34, 40, 50, 55)
    ema_slow_options: tuple[int, ...] = (50, 80, 100, 144, 200, 250, 300, 400, 500, 600, 800, 1000)
    frama_fast_period: int = 16
    frama_slow_period: int = 55
    frama_fast_options: tuple[int, ...] = (6, 12, 20)
    frama_slow_options: tuple[int, ...] = (40, 80, 150, 200)
    volume_sma_period: int = 20
    volume_multiplier: float = 1.0
    volume_multiplier_options: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    available_strategies: tuple[str, ...] = AVAILABLE_STRATEGIES
    default_strategy_name: str = DEFAULT_STRATEGY_NAME
    coin_strategies: dict[str, str] = field(default_factory=_build_default_coin_strategies)
    coin_strategy_params: dict[str, dict[str, float]] = field(
        default_factory=_build_default_coin_strategy_params
    )

    def __post_init__(self) -> None:
        production_coin_strategies = dict(self.coin_strategies)
        production_coin_strategy_params = {
            symbol: dict(params)
            for symbol, params in self.coin_strategy_params.items()
        }
        for symbol, profile in PRODUCTION_PROFILE_REGISTRY.items():
            strategy_name = str(profile["strategy_name"])
            strategy_name = PRODUCTION_STRATEGY_ALIASES.get(strategy_name, strategy_name)
            raw_strategy_params = profile.get("strategy_params", {})
            if not isinstance(raw_strategy_params, dict):
                raw_strategy_params = {}
            strategy_params = {
                str(param_name): float(param_value)
                for param_name, param_value in raw_strategy_params.items()
            }
            production_coin_strategies[symbol] = strategy_name
            production_coin_strategy_params[symbol] = strategy_params
        # Apply local optimizer overrides as final defaults.
        for symbol, strategy_name in DEFAULT_COIN_STRATEGIES.items():
            normalized_strategy_name = PRODUCTION_STRATEGY_ALIASES.get(
                str(strategy_name),
                str(strategy_name),
            )
            production_coin_strategies[symbol] = normalized_strategy_name
        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items():
            production_coin_strategy_params[symbol] = {
                str(param_name): float(param_value)
                for param_name, param_value in params.items()
            }
        object.__setattr__(self, "coin_strategies", production_coin_strategies)
        object.__setattr__(self, "coin_strategy_params", production_coin_strategy_params)

        if self.dual_thrust_period <= 0:
            raise ValueError("dual_thrust_period must be positive.")
        if self.dual_thrust_k1 <= 0.0:
            raise ValueError("dual_thrust_k1 must be positive.")
        if self.dual_thrust_k2 <= 0.0:
            raise ValueError("dual_thrust_k2 must be positive.")
        if not self.dual_thrust_opt_periods:
            raise ValueError("dual_thrust_opt_periods must not be empty.")
        if any(value <= 0 for value in self.dual_thrust_opt_periods):
            raise ValueError("dual_thrust_opt_periods must contain only positive values.")
        if not self.dual_thrust_opt_k_values:
            raise ValueError("dual_thrust_opt_k_values must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_opt_k_values):
            raise ValueError("dual_thrust_opt_k_values must contain only positive values.")
        if not self.dual_thrust_stop_loss_pct_options:
            raise ValueError("dual_thrust_stop_loss_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_stop_loss_pct_options):
            raise ValueError("dual_thrust_stop_loss_pct_options must contain only positive values.")
        if not self.dual_thrust_take_profit_pct_options:
            raise ValueError("dual_thrust_take_profit_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_take_profit_pct_options):
            raise ValueError("dual_thrust_take_profit_pct_options must contain only positive values.")
        if not self.dual_thrust_trailing_activation_pct_options:
            raise ValueError("dual_thrust_trailing_activation_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_trailing_activation_pct_options):
            raise ValueError(
                "dual_thrust_trailing_activation_pct_options must contain only positive values."
            )
        if not self.dual_thrust_trailing_distance_pct_options:
            raise ValueError("dual_thrust_trailing_distance_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_trailing_distance_pct_options):
            raise ValueError(
                "dual_thrust_trailing_distance_pct_options must contain only positive values."
            )
        if not any(
            float(trailing_distance_pct) < float(trailing_activation_pct)
            for trailing_activation_pct in self.dual_thrust_trailing_activation_pct_options
            for trailing_distance_pct in self.dual_thrust_trailing_distance_pct_options
        ):
            raise ValueError(
                "dual_thrust risk options must provide at least one trailing_distance_pct < trailing_activation_pct combination."
            )
        for symbol, symbol_grid in self.dual_thrust_symbol_optimization_grids.items():
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol:
                raise ValueError(
                    "dual_thrust_symbol_optimization_grids keys must be non-empty symbols."
                )
            stop_loss_options = tuple(
                float(value)
                for value in symbol_grid.get("stop_loss_pct_options", ())
            )
            take_profit_options = tuple(
                float(value)
                for value in symbol_grid.get("take_profit_pct_options", ())
            )
            if not stop_loss_options:
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}] must define stop_loss_pct_options."
                )
            if not take_profit_options:
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}] must define take_profit_pct_options."
                )
            if any(value <= 0.0 for value in stop_loss_options):
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}].stop_loss_pct_options must contain only positive values."
                )
            if any(value <= 0.0 for value in take_profit_options):
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}].take_profit_pct_options must contain only positive values."
                )
        if self.ema_fast_period <= 0:
            raise ValueError("ema_fast_period must be positive.")
        if self.ema_slow_period <= 0:
            raise ValueError("ema_slow_period must be positive.")
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be less than ema_slow_period.")
        if not self.ema_fast_options:
            raise ValueError("ema_fast_options must not be empty.")
        if any(value <= 0 for value in self.ema_fast_options):
            raise ValueError("ema_fast_options must contain only positive values.")
        if not self.ema_slow_options:
            raise ValueError("ema_slow_options must not be empty.")
        if any(value <= 0 for value in self.ema_slow_options):
            raise ValueError("ema_slow_options must contain only positive values.")
        if self.frama_fast_period <= 1:
            raise ValueError("frama_fast_period must be greater than 1.")
        if self.frama_slow_period <= 1:
            raise ValueError("frama_slow_period must be greater than 1.")
        if self.frama_fast_period >= self.frama_slow_period:
            raise ValueError("frama_fast_period must be less than frama_slow_period.")
        if not self.frama_fast_options:
            raise ValueError("frama_fast_options must not be empty.")
        if any(value <= 1 for value in self.frama_fast_options):
            raise ValueError("frama_fast_options must contain only values greater than 1.")
        if not self.frama_slow_options:
            raise ValueError("frama_slow_options must not be empty.")
        if any(value <= 1 for value in self.frama_slow_options):
            raise ValueError("frama_slow_options must contain only values greater than 1.")
        if self.volume_sma_period <= 0:
            raise ValueError("volume_sma_period must be positive.")
        if self.volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive.")
        if not self.volume_multiplier_options:
            raise ValueError("volume_multiplier_options must not be empty.")
        if any(value <= 0.0 for value in self.volume_multiplier_options):
            raise ValueError("volume_multiplier_options must contain only positive values.")
        if not self.available_strategies:
            raise ValueError("available_strategies must not be empty.")
        if self.default_strategy_name not in self.available_strategies:
            raise ValueError("default_strategy_name must be one of available_strategies.")
        for symbol, strategy_name in self.coin_strategies.items():
            if not symbol:
                raise ValueError("coin_strategies keys must not be empty.")
            if strategy_name not in self.available_strategies:
                raise ValueError(
                    f"coin_strategies[{symbol}] must be one of available_strategies."
                )

@dataclass(frozen=True, slots=True)
class TradingSettings:
    interval: str = str(TRADING_BASE_DEFAULTS["interval"])
    backtest_history_start_utc: str = BACKTEST_HISTORY_START_UTC
    optimizer_history_start_utc: str = OPTIMIZER_HISTORY_START_UTC
    start_capital: float = float(TRADING_BASE_DEFAULTS["start_capital"])
    risk_per_trade_pct: float = float(TRADING_BASE_DEFAULTS["risk_per_trade_pct"])
    taker_fee_pct: float = float(TRADING_BASE_DEFAULTS["taker_fee_pct"])
    use_setup_gate: bool = bool(TRADING_BASE_DEFAULTS["use_setup_gate"])
    use_hmm_regime_filter: bool = True
    hmm_allowed_regimes: tuple[str, ...] = (
        "Bull Trend",
        "Low-Vol Range",
    )
    min_confidence_pct: float = float(TRADING_BASE_DEFAULTS["min_confidence_pct"])
    take_profit_pct: float = float(TRADING_BASE_DEFAULTS["take_profit_pct"])
    stop_loss_pct: float = float(TRADING_BASE_DEFAULTS["stop_loss_pct"])
    use_trailing_stop: bool = bool(TRADING_BASE_DEFAULTS["use_trailing_stop"])
    trailing_activation_pct: float = float(TRADING_BASE_DEFAULTS["trailing_activation_pct"])
    trailing_distance_pct: float = float(TRADING_BASE_DEFAULTS["trailing_distance_pct"])
    min_leverage: int = int(TRADING_BASE_DEFAULTS["min_leverage"])
    max_leverage: int = int(TRADING_BASE_DEFAULTS["max_leverage"])
    default_leverage: int = int(TRADING_BASE_DEFAULTS["default_leverage"])
    max_open_positions: int = int(TRADING_BASE_DEFAULTS["max_open_positions"])
    # Aggressive optimization defaults while preserving slot count for stable profile volume.
    optimization_take_profit_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_take_profit_pct_options"]
    )
    optimization_stop_loss_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_stop_loss_pct_options"]
    )
    optimization_trailing_activation_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_trailing_activation_pct_options"]
    )
    optimization_trailing_distance_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_trailing_distance_pct_options"]
    )
    optimization_max_sample_profiles: int = int(
        OPTIMIZATION_DEFAULTS["optimization_max_sample_profiles"]
    )
    optimization_search_window_candles: int = int(
        OPTIMIZATION_DEFAULTS["optimization_search_window_candles"]
    )
    optimization_validation_top_n: int = int(
        OPTIMIZATION_DEFAULTS["optimization_validation_top_n"]
    )
    # Universal risk grid used for large automated scans
    # This is applied across strategies when running the universal scan
    universal_stop_loss_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_stop_loss_pct_options"]
    )
    universal_take_profit_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_take_profit_pct_options"]
    )
    universal_trailing_activation_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_trailing_activation_pct_options"]
    )
    universal_trailing_distance_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_trailing_distance_pct_options"]
    )
    # Terminal I/O throttling for optimizer progress prints (every N profiles)
    optimizer_progress_print_modulo: int = 2500
    coin_profiles: dict[str, CoinProfileSettings] = field(default_factory=_build_default_coin_profiles)

    def __post_init__(self) -> None:
        production_coin_profiles = dict(self.coin_profiles)
        for symbol, profile in PRODUCTION_PROFILE_REGISTRY.items():
            production_coin_profiles[symbol] = CoinProfileSettings(
                interval=PRODUCTION_PROFILE_INTERVAL,
                take_profit_pct=float(profile["take_profit_pct"]),
                stop_loss_pct=float(profile["stop_loss_pct"]),
                trailing_activation_pct=float(profile["trailing_activation_pct"]),
                trailing_distance_pct=float(profile["trailing_distance_pct"]),
                default_leverage=int(profile.get("default_leverage", 20) or 20),
            )
        # Apply local optimizer overrides as final defaults.
        for symbol, profile in DEFAULT_COIN_PROFILE_VALUES.items():
            production_coin_profiles[symbol] = CoinProfileSettings(
                interval=(
                    None if profile.get("interval") is None else str(profile["interval"])
                ),
                stop_loss_pct=(
                    None if profile.get("stop_loss_pct") is None else float(profile["stop_loss_pct"])
                ),
                take_profit_pct=(
                    None if profile.get("take_profit_pct") is None else float(profile["take_profit_pct"])
                ),
                trailing_activation_pct=(
                    None
                    if profile.get("trailing_activation_pct") is None
                    else float(profile["trailing_activation_pct"])
                ),
                trailing_distance_pct=(
                    None
                    if profile.get("trailing_distance_pct") is None
                    else float(profile["trailing_distance_pct"])
                ),
                default_leverage=(
                    None
                    if profile.get("default_leverage") is None
                    else int(profile["default_leverage"])
                ),
            )
        object.__setattr__(self, "coin_profiles", production_coin_profiles)
        if PRODUCTION_PROFILE_REGISTRY and not self.use_hmm_regime_filter:
            object.__setattr__(self, "use_hmm_regime_filter", True)

        if not self.interval:
            raise ValueError("interval must not be empty.")
        if not self.backtest_history_start_utc.strip():
            raise ValueError("backtest_history_start_utc must not be empty.")
        if not self.optimizer_history_start_utc.strip():
            raise ValueError("optimizer_history_start_utc must not be empty.")
        try:
            parsed_backtest_start = datetime.fromisoformat(
                self.backtest_history_start_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "backtest_history_start_utc must be a valid ISO-8601 datetime."
            ) from exc
        if parsed_backtest_start.tzinfo is None:
            parsed_backtest_start = parsed_backtest_start.replace(tzinfo=UTC)
        if parsed_backtest_start.astimezone(UTC) > datetime.now(tz=UTC):
            raise ValueError("backtest_history_start_utc must not be in the future.")
        try:
            parsed_optimizer_start = datetime.fromisoformat(
                self.optimizer_history_start_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "optimizer_history_start_utc must be a valid ISO-8601 datetime."
            ) from exc
        if parsed_optimizer_start.tzinfo is None:
            parsed_optimizer_start = parsed_optimizer_start.replace(tzinfo=UTC)
        if parsed_optimizer_start.astimezone(UTC) > datetime.now(tz=UTC):
            raise ValueError("optimizer_history_start_utc must not be in the future.")
        if self.start_capital <= 0:
            raise ValueError("start_capital must be positive.")
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be positive.")
        if self.taker_fee_pct < 0:
            raise ValueError("taker_fee_pct must be greater than or equal to 0.")
        if self.use_hmm_regime_filter and not self.hmm_allowed_regimes:
            raise ValueError(
                "hmm_allowed_regimes must not be empty when use_hmm_regime_filter is enabled."
            )
        if any(not str(regime).strip() for regime in self.hmm_allowed_regimes):
            raise ValueError("hmm_allowed_regimes must contain non-empty regime names.")
        if not 0.0 <= self.min_confidence_pct <= 100.0:
            raise ValueError("min_confidence_pct must be between 0 and 100.")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive.")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive.")
        if self.trailing_activation_pct < 0:
            raise ValueError("trailing_activation_pct must be greater than or equal to 0.")
        if self.trailing_distance_pct < 0:
            raise ValueError("trailing_distance_pct must be greater than or equal to 0.")
        if self.min_leverage <= 0:
            raise ValueError("min_leverage must be positive.")
        if self.max_leverage < self.min_leverage:
            raise ValueError("max_leverage must be greater than or equal to min_leverage.")
        if not self.min_leverage <= self.default_leverage <= self.max_leverage:
            raise ValueError("default_leverage must be within the leverage range.")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive.")
        if not self.optimization_take_profit_pct_options:
            raise ValueError("optimization_take_profit_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.optimization_take_profit_pct_options):
            raise ValueError("optimization_take_profit_pct_options must contain only positive values.")
        if not self.optimization_stop_loss_pct_options:
            raise ValueError("optimization_stop_loss_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.optimization_stop_loss_pct_options):
            raise ValueError("optimization_stop_loss_pct_options must contain only positive values.")
        if not self.optimization_trailing_activation_pct_options:
            raise ValueError("optimization_trailing_activation_pct_options must not be empty.")
        if any(value < 0.0 for value in self.optimization_trailing_activation_pct_options):
            raise ValueError(
                "optimization_trailing_activation_pct_options must contain only values >= 0."
            )
        if not self.optimization_trailing_distance_pct_options:
            raise ValueError("optimization_trailing_distance_pct_options must not be empty.")
        if any(value < 0.0 for value in self.optimization_trailing_distance_pct_options):
            raise ValueError(
                "optimization_trailing_distance_pct_options must contain only values >= 0."
            )
        if self.optimization_max_sample_profiles <= 0:
            raise ValueError("optimization_max_sample_profiles must be positive.")
        if self.optimization_search_window_candles <= 0:
            raise ValueError("optimization_search_window_candles must be positive.")
        if self.optimization_validation_top_n <= 0:
            raise ValueError("optimization_validation_top_n must be positive.")
        if self.optimizer_progress_print_modulo <= 0:
            raise ValueError("optimizer_progress_print_modulo must be positive.")
        for symbol, profile in self.coin_profiles.items():
            if not symbol:
                raise ValueError("coin_profiles keys must not be empty.")
            if profile.stop_loss_pct is not None and profile.stop_loss_pct <= 0:
                raise ValueError(f"coin_profiles[{symbol}].stop_loss_pct must be positive.")
            if profile.take_profit_pct is not None and profile.take_profit_pct <= 0:
                raise ValueError(f"coin_profiles[{symbol}].take_profit_pct must be positive.")
            if (
                profile.trailing_activation_pct is not None
                and profile.trailing_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].trailing_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.trailing_distance_pct is not None
                and profile.trailing_distance_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].trailing_distance_pct must be greater than or equal to 0."
                )
            if profile.default_leverage is not None and not (
                self.min_leverage <= profile.default_leverage <= self.max_leverage
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].default_leverage must be within the leverage range."
                )


@dataclass(frozen=True, slots=True)
class Settings:
    api: APISettings = field(default_factory=APISettings)
    live: LiveSettings = field(default_factory=LiveSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    trading: TradingSettings = field(default_factory=TradingSettings)

    def __post_init__(self) -> None:
        allowed_timeframes = set(self.api.timeframes)
        if self.live.default_interval not in allowed_timeframes:
            raise ValueError(
                f"live.default_interval must be one of api.timeframes: {self.live.default_interval}"
            )
        if self.trading.interval not in allowed_timeframes:
            raise ValueError(
                f"trading.interval must be one of api.timeframes: {self.trading.interval}"
            )
        for symbol, profile in self.trading.coin_profiles.items():
            if profile.interval is not None and profile.interval not in allowed_timeframes:
                raise ValueError(
                    f"coin_profiles[{symbol}].interval must be one of api.timeframes."
                )


settings = Settings()
USE_SETUP_GATE = settings.trading.use_setup_gate
COIN_STRATEGIES = settings.strategy.coin_strategies
COIN_STRATEGY_PARAMS = settings.strategy.coin_strategy_params

# Default scan strategies for automated multi-strategy runs
SCAN_STRATEGIES = tuple(DEFAULT_SCAN_STRATEGIES)
