from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Mapping

from config_defaults import ACTIVE_COINS
from config_sections.backtest_optimizer_defaults import (
    BACKTEST_HISTORY_START_UTC,
    DEFAULT_COIN_PROFILE_VALUES,
    OPTIMIZATION_DEFAULTS,
    OPTIMIZER_HISTORY_START_UTC,
    STRATEGY_GRID_DEFAULTS,
    TRADING_BASE_DEFAULTS,
    UNIVERSAL_RISK_GRID_DEFAULTS,
)
from config_sections.live_defaults import LIVE_DEFAULT_INTERVAL
from config_sections.strategy_defaults import (
    AVAILABLE_STRATEGIES,
    DEFAULT_COIN_STRATEGIES,
    DEFAULT_COIN_STRATEGY_PARAMS,
    DEFAULT_STRATEGY_NAME,
)
from production_registry import (
    PRODUCTION_PROFILE_INTERVAL,
    PRODUCTION_PROFILE_REGISTRY,
    PRODUCTION_STRATEGY_ALIASES,
)


BITUNIX_MAX_KLINE_LIMIT = 200


@dataclass(frozen=True, slots=True)
class CoinProfileSettings:
    interval: str | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    breakeven_activation_pct: float | None = None
    breakeven_buffer_pct: float | None = None
    trailing_activation_pct: float | None = None
    trailing_distance_pct: float | None = None
    tight_trailing_activation_pct: float | None = None
    tight_trailing_distance_pct: float | None = None
    default_leverage: int | None = None


def build_coin_profile_settings(raw_profile: Mapping[str, object]) -> CoinProfileSettings:
    return CoinProfileSettings(
        interval=(
            None if raw_profile.get("interval") is None else str(raw_profile["interval"])
        ),
        stop_loss_pct=(
            None if raw_profile.get("stop_loss_pct") is None else float(raw_profile["stop_loss_pct"])
        ),
        take_profit_pct=(
            None if raw_profile.get("take_profit_pct") is None else float(raw_profile["take_profit_pct"])
        ),
        breakeven_activation_pct=(
            None
            if raw_profile.get("breakeven_activation_pct") is None
            else float(raw_profile["breakeven_activation_pct"])
        ),
        breakeven_buffer_pct=(
            None
            if raw_profile.get("breakeven_buffer_pct") is None
            else float(raw_profile["breakeven_buffer_pct"])
        ),
        trailing_activation_pct=(
            None
            if raw_profile.get("trailing_activation_pct") is None
            else float(raw_profile["trailing_activation_pct"])
        ),
        trailing_distance_pct=(
            None
            if raw_profile.get("trailing_distance_pct") is None
            else float(raw_profile["trailing_distance_pct"])
        ),
        tight_trailing_activation_pct=(
            None
            if raw_profile.get("tight_trailing_activation_pct") is None
            else float(raw_profile["tight_trailing_activation_pct"])
        ),
        tight_trailing_distance_pct=(
            None
            if raw_profile.get("tight_trailing_distance_pct") is None
            else float(raw_profile["tight_trailing_distance_pct"])
        ),
        default_leverage=(
            None if raw_profile.get("default_leverage") is None else int(raw_profile["default_leverage"])
        ),
    )


def build_coin_profile_settings_map(
    raw_profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, CoinProfileSettings]:
    return {
        str(symbol).strip().upper(): build_coin_profile_settings(raw_profile)
        for symbol, raw_profile in raw_profiles.items()
        if str(symbol).strip()
    }


def _build_default_coin_strategies() -> dict[str, str]:
    return dict(DEFAULT_COIN_STRATEGIES)


def _build_default_coin_strategy_params() -> dict[str, dict[str, float]]:
    return {
        symbol: dict(params)
        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items()
    }


def _build_default_coin_profiles() -> dict[str, CoinProfileSettings]:
    return build_coin_profile_settings_map(DEFAULT_COIN_PROFILE_VALUES)


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
    candles_per_request: int = BITUNIX_MAX_KLINE_LIMIT
    request_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.public_requests_per_second <= 0:
            raise ValueError("public_requests_per_second must be positive.")
        if self.private_requests_per_second <= 0:
            raise ValueError("private_requests_per_second must be positive.")
        if self.candles_per_request <= 0:
            raise ValueError("candles_per_request must be positive.")
        if self.candles_per_request > BITUNIX_MAX_KLINE_LIMIT:
            raise ValueError(
                f"candles_per_request must be <= {BITUNIX_MAX_KLINE_LIMIT} for Bitunix klines."
            )
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
    available_symbols: tuple[str, ...] = tuple(ACTIVE_COINS)

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
    dual_thrust_opt_periods: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["dual_thrust_opt_periods"]
    )
    dual_thrust_opt_k_values: tuple[float, ...] = tuple(
        float(value) for value in STRATEGY_GRID_DEFAULTS["dual_thrust_opt_k_values"]
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
    ema_fast_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["ema_fast_options"]
    )
    ema_slow_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["ema_slow_options"]
    )
    frama_fast_period: int = 16
    frama_slow_period: int = 55
    frama_fast_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["frama_fast_options"]
    )
    frama_slow_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["frama_slow_options"]
    )
    volume_sma_period: int = 20
    volume_multiplier: float = 1.0
    volume_multiplier_options: tuple[float, ...] = (1.0, 1.25, 1.5)
    use_late_entry_guard: bool = False
    late_entry_max_move_1_bar_pct: float = 0.0
    late_entry_max_move_2_bars_pct: float = 0.0
    late_entry_max_move_3_bars_pct: float = 0.0
    late_entry_max_distance_ref_pct: float = 0.0
    late_entry_max_distance_fast_ref_pct: float = 0.0
    late_entry_max_distance_mid_ref_pct: float = 0.0
    late_entry_max_atr_mult: float = 0.0
    use_pullback_reentry: bool = False
    pullback_reentry_min_touch: float = 0.0
    pullback_reentry_reconfirm_required: bool = False
    max_breakout_candle_body_pct: float = 0.0
    max_breakout_candle_range_atr_mult: float = 0.0
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
            strategy_name = str(profile.get("strategy_name", profile.get("strategy", "")))
            strategy_name = PRODUCTION_STRATEGY_ALIASES.get(strategy_name, strategy_name)
            raw_strategy_params = profile.get("strategy_params", profile.get("params", {}))
            if not isinstance(raw_strategy_params, Mapping):
                raw_strategy_params = {}
            strategy_params = {
                str(param_name): float(param_value)
                for param_name, param_value in raw_strategy_params.items()
            }
            production_coin_strategies[symbol] = strategy_name
            production_coin_strategy_params[symbol] = strategy_params

        for symbol, strategy_name in DEFAULT_COIN_STRATEGIES.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY or symbol in production_coin_strategies:
                continue
            normalized_strategy_name = PRODUCTION_STRATEGY_ALIASES.get(
                str(strategy_name),
                str(strategy_name),
            )
            production_coin_strategies[symbol] = normalized_strategy_name

        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY or symbol in production_coin_strategy_params:
                continue
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
        if self.late_entry_max_move_1_bar_pct < 0.0:
            raise ValueError("late_entry_max_move_1_bar_pct must be greater than or equal to 0.")
        if self.late_entry_max_move_2_bars_pct < 0.0:
            raise ValueError("late_entry_max_move_2_bars_pct must be greater than or equal to 0.")
        if self.late_entry_max_move_3_bars_pct < 0.0:
            raise ValueError("late_entry_max_move_3_bars_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_fast_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_fast_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_mid_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_mid_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_atr_mult < 0.0:
            raise ValueError("late_entry_max_atr_mult must be greater than or equal to 0.")
        if self.pullback_reentry_min_touch < 0.0:
            raise ValueError("pullback_reentry_min_touch must be greater than or equal to 0.")
        if self.max_breakout_candle_body_pct < 0.0:
            raise ValueError("max_breakout_candle_body_pct must be greater than or equal to 0.")
        if self.max_breakout_candle_range_atr_mult < 0.0:
            raise ValueError("max_breakout_candle_range_atr_mult must be greater than or equal to 0.")
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
    risk_per_trade_pct: float = 1.0
    taker_fee_pct: float = float(TRADING_BASE_DEFAULTS["taker_fee_pct"])
    use_setup_gate: bool = bool(TRADING_BASE_DEFAULTS["use_setup_gate"])
    use_hmm_regime_filter: bool = bool(TRADING_BASE_DEFAULTS["use_hmm_regime_filter"])
    hmm_allowed_regimes: tuple[str, ...] = tuple(TRADING_BASE_DEFAULTS["hmm_allowed_regimes"])
    min_confidence_pct: float = float(TRADING_BASE_DEFAULTS["min_confidence_pct"])
    take_profit_pct: float = float(TRADING_BASE_DEFAULTS["take_profit_pct"])
    stop_loss_pct: float = float(TRADING_BASE_DEFAULTS["stop_loss_pct"])
    chandelier_period: int = int(TRADING_BASE_DEFAULTS["chandelier_period"])
    chandelier_multiplier: float = float(TRADING_BASE_DEFAULTS["chandelier_multiplier"])
    use_trailing_stop: bool = bool(TRADING_BASE_DEFAULTS["use_trailing_stop"])
    trailing_activation_pct: float = float(TRADING_BASE_DEFAULTS["trailing_activation_pct"])
    trailing_distance_pct: float = float(TRADING_BASE_DEFAULTS["trailing_distance_pct"])
    min_leverage: int = int(TRADING_BASE_DEFAULTS["min_leverage"])
    max_leverage: int = int(TRADING_BASE_DEFAULTS["max_leverage"])
    default_leverage: int = int(TRADING_BASE_DEFAULTS["default_leverage"])
    max_open_positions: int = 5
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
    random_search_samples: int = int(
        OPTIMIZATION_DEFAULTS["random_search_samples"]
    )
    optimization_search_window_candles: int = int(
        OPTIMIZATION_DEFAULTS["optimization_search_window_candles"]
    )
    optimization_validation_top_n: int = int(
        OPTIMIZATION_DEFAULTS["optimization_validation_top_n"]
    )
    optimizer_profile_mode: str = "all_coins_pass1"
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
    universal_chandelier_period_options: tuple[int, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_chandelier_period_options"]
    )
    universal_chandelier_multiplier_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_chandelier_multiplier_options"]
    )
    optimizer_progress_print_modulo: int = 2500
    coin_profiles: dict[str, CoinProfileSettings] = field(default_factory=_build_default_coin_profiles)

    def __post_init__(self) -> None:
        production_coin_profiles = dict(self.coin_profiles)
        for symbol, profile in PRODUCTION_PROFILE_REGISTRY.items():
            resolved_interval = str(profile.get("interval", PRODUCTION_PROFILE_INTERVAL) or PRODUCTION_PROFILE_INTERVAL)
            production_coin_profiles[symbol] = CoinProfileSettings(
                interval=resolved_interval,
                take_profit_pct=float(profile["take_profit_pct"]),
                stop_loss_pct=float(profile["stop_loss_pct"]),
                breakeven_activation_pct=(
                    None
                    if profile.get("breakeven_activation_pct") is None
                    else float(profile["breakeven_activation_pct"])
                ),
                breakeven_buffer_pct=(
                    None
                    if profile.get("breakeven_buffer_pct") is None
                    else float(profile["breakeven_buffer_pct"])
                ),
                trailing_activation_pct=float(profile["trailing_activation_pct"]),
                trailing_distance_pct=float(profile["trailing_distance_pct"]),
                tight_trailing_activation_pct=(
                    None
                    if profile.get("tight_trailing_activation_pct") is None
                    else float(profile["tight_trailing_activation_pct"])
                ),
                tight_trailing_distance_pct=(
                    None
                    if profile.get("tight_trailing_distance_pct") is None
                    else float(profile["tight_trailing_distance_pct"])
                ),
                default_leverage=int(profile.get("default_leverage", 20) or 20),
            )

        for symbol, profile in DEFAULT_COIN_PROFILE_VALUES.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY or symbol in production_coin_profiles:
                continue
            production_coin_profiles[symbol] = build_coin_profile_settings(profile)

        object.__setattr__(self, "coin_profiles", production_coin_profiles)

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
        if self.chandelier_period <= 1:
            raise ValueError("chandelier_period must be greater than 1.")
        if self.chandelier_multiplier <= 0.0:
            raise ValueError("chandelier_multiplier must be positive.")
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
        if self.random_search_samples <= 0:
            raise ValueError("random_search_samples must be positive.")
        if self.optimization_search_window_candles <= 0:
            raise ValueError("optimization_search_window_candles must be positive.")
        if self.optimization_validation_top_n <= 0:
            raise ValueError("optimization_validation_top_n must be positive.")
        if self.optimizer_profile_mode not in {"all_coins_pass1", "strict_verification"}:
            raise ValueError(
                "optimizer_profile_mode must be one of {'all_coins_pass1', 'strict_verification'}."
            )
        if not self.universal_chandelier_period_options:
            raise ValueError("universal_chandelier_period_options must not be empty.")
        if any(value <= 1 for value in self.universal_chandelier_period_options):
            raise ValueError(
                "universal_chandelier_period_options must contain values greater than 1."
            )
        if not self.universal_chandelier_multiplier_options:
            raise ValueError("universal_chandelier_multiplier_options must not be empty.")
        if any(value <= 0.0 for value in self.universal_chandelier_multiplier_options):
            raise ValueError(
                "universal_chandelier_multiplier_options must contain only positive values."
            )
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
            if (
                profile.breakeven_activation_pct is not None
                and profile.breakeven_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].breakeven_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.breakeven_buffer_pct is not None
                and profile.breakeven_buffer_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].breakeven_buffer_pct must be greater than or equal to 0."
                )
            if (
                profile.tight_trailing_activation_pct is not None
                and profile.tight_trailing_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].tight_trailing_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.tight_trailing_distance_pct is not None
                and profile.tight_trailing_distance_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].tight_trailing_distance_pct must be greater than or equal to 0."
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


__all__ = (
    "BITUNIX_MAX_KLINE_LIMIT",
    "CoinProfileSettings",
    "RequestLimits",
    "APISettings",
    "LiveSettings",
    "StrategySettings",
    "TradingSettings",
    "Settings",
    "build_coin_profile_settings",
    "build_coin_profile_settings_map",
    "_float_range_tuple",
)

