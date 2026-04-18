from __future__ import annotations

from typing import Mapping

from config_defaults import (
    EMA_BAND_REJECTION_1H_EXCLUDED_COINS,
    MAX_BACKTEST_CANDLES,
    normalize_symbol_list,
)
from config_loader import ResolvedConfig, load_resolved_config
from config_schema import (
    APISettings,
    CoinProfileSettings,
    LiveSettings,
    RequestLimits,
    Settings,
    StrategySettings,
    TradingSettings,
)
from config_sections.strategy_defaults import DEFAULT_SCAN_STRATEGIES
from production_registry import (
    PRODUCTION_PROFILE_INTERVAL,
    PRODUCTION_PROFILE_REGISTRY,
    PRODUCTION_STRATEGY_ALIASES,
)
from runtime_profiles import load_runtime_profile_state, save_runtime_profile_state


def _copy_nested_float_dict(
    payload: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        str(symbol): {
            str(param_name): float(param_value)
            for param_name, param_value in dict(params).items()
        }
        for symbol, params in payload.items()
    }


def _copy_nested_object_dict(
    payload: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        str(symbol): dict(profile)
        for symbol, profile in payload.items()
    }


def _replace_list_contents(target: list[str], source: list[str]) -> None:
    target[:] = list(source)


def _replace_dict_contents(target: dict, source: Mapping) -> None:
    target.clear()
    target.update(source)


def _build_strategy_params_for_deploy(
    *,
    strategy_name: str,
    best_profile: Mapping[str, object],
    base_strategy_params: Mapping[str, object],
    runtime_settings: Settings,
) -> dict[str, float]:
    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    if strategy_name == "dual_thrust":
        return {
            "dual_thrust_period": _to_float(
                best_profile.get(
                    "dual_thrust_period",
                    base_strategy_params.get(
                        "dual_thrust_period",
                        runtime_settings.strategy.dual_thrust_period,
                    ),
                ),
                runtime_settings.strategy.dual_thrust_period,
            ),
            "dual_thrust_k1": _to_float(
                best_profile.get(
                    "dual_thrust_k1",
                    base_strategy_params.get(
                        "dual_thrust_k1",
                        runtime_settings.strategy.dual_thrust_k1,
                    ),
                ),
                runtime_settings.strategy.dual_thrust_k1,
            ),
            "dual_thrust_k2": _to_float(
                best_profile.get(
                    "dual_thrust_k2",
                    base_strategy_params.get(
                        "dual_thrust_k2",
                        runtime_settings.strategy.dual_thrust_k2,
                    ),
                ),
                runtime_settings.strategy.dual_thrust_k2,
            ),
        }
    if strategy_name == "frama_cross":
        return {
            "frama_fast_period": _to_float(
                best_profile.get(
                    "frama_fast_period",
                    base_strategy_params.get(
                        "frama_fast_period",
                        runtime_settings.strategy.frama_fast_period,
                    ),
                ),
                runtime_settings.strategy.frama_fast_period,
            ),
            "frama_slow_period": _to_float(
                best_profile.get(
                    "frama_slow_period",
                    base_strategy_params.get(
                        "frama_slow_period",
                        runtime_settings.strategy.frama_slow_period,
                    ),
                ),
                runtime_settings.strategy.frama_slow_period,
            ),
            "volume_multiplier": _to_float(
                best_profile.get(
                    "volume_multiplier",
                    base_strategy_params.get(
                        "volume_multiplier",
                        runtime_settings.strategy.volume_multiplier,
                    ),
                ),
                runtime_settings.strategy.volume_multiplier,
            ),
        }
    if strategy_name == "ema_cross_volume":
        return {
            "ema_fast_period": _to_float(
                best_profile.get(
                    "ema_fast_period",
                    base_strategy_params.get(
                        "ema_fast_period",
                        runtime_settings.strategy.ema_fast_period,
                    ),
                ),
                runtime_settings.strategy.ema_fast_period,
            ),
            "ema_slow_period": _to_float(
                best_profile.get(
                    "ema_slow_period",
                    base_strategy_params.get(
                        "ema_slow_period",
                        runtime_settings.strategy.ema_slow_period,
                    ),
                ),
                runtime_settings.strategy.ema_slow_period,
            ),
            "volume_multiplier": _to_float(
                best_profile.get(
                    "volume_multiplier",
                    base_strategy_params.get(
                        "volume_multiplier",
                        runtime_settings.strategy.volume_multiplier,
                    ),
                ),
                runtime_settings.strategy.volume_multiplier,
            ),
        }
    return {
        str(key): float(value)
        for key, value in base_strategy_params.items()
        if str(key)
    }


def _resolve_strategy_name_for_deploy(
    *,
    symbol: str,
    best_profile: Mapping[str, object],
    runtime_settings: Settings,
) -> str:
    explicit_strategy = str(
        best_profile.get(
            "strategy_name",
            best_profile.get("strategy", ""),
        )
    ).strip()
    base_strategy_name = str(
        explicit_strategy
        or DEFAULT_COIN_STRATEGIES.get(symbol)
        or runtime_settings.strategy.default_strategy_name
    ).strip()
    strategy_name = PRODUCTION_STRATEGY_ALIASES.get(base_strategy_name, base_strategy_name)
    if any(key in best_profile for key in ("dual_thrust_period", "dual_thrust_k1", "dual_thrust_k2")):
        strategy_name = "dual_thrust"
    elif any(key in best_profile for key in ("frama_fast_period", "frama_slow_period")):
        strategy_name = "frama_cross"
    elif any(key in best_profile for key in ("ema_fast_period", "ema_slow_period")):
        strategy_name = "ema_cross_volume"
    if strategy_name not in runtime_settings.strategy.available_strategies:
        strategy_name = str(runtime_settings.strategy.default_strategy_name)
    return strategy_name


def _apply_resolved_config_inplace(resolved: ResolvedConfig) -> None:
    _replace_list_contents(ACTIVE_COINS, resolved.active_coins)
    _replace_list_contents(BACKTEST_ONLY_COINS, resolved.backtest_only_coins)

    global BACKTEST_BATCH_SYMBOLS
    BACKTEST_BATCH_SYMBOLS = tuple(resolved.backtest_batch_symbols)

    _replace_dict_contents(DEFAULT_COIN_PROFILE_VALUES, resolved.default_coin_profile_values)
    _replace_dict_contents(DEFAULT_COIN_STRATEGIES, resolved.default_coin_strategies)
    _replace_dict_contents(DEFAULT_COIN_STRATEGY_PARAMS, resolved.default_coin_strategy_params)

    object.__setattr__(settings, "api", resolved.settings.api)
    object.__setattr__(settings, "live", resolved.settings.live)
    object.__setattr__(settings, "strategy", resolved.settings.strategy)
    object.__setattr__(settings, "trading", resolved.settings.trading)

    global USE_SETUP_GATE
    USE_SETUP_GATE = settings.trading.use_setup_gate

    _replace_dict_contents(COIN_STRATEGIES, settings.strategy.coin_strategies)
    _replace_dict_contents(COIN_STRATEGY_PARAMS, settings.strategy.coin_strategy_params)


def _reload_runtime_settings() -> None:
    _apply_resolved_config_inplace(load_resolved_config())


_INITIAL_RESOLVED = load_resolved_config()

ACTIVE_COINS: list[str] = list(_INITIAL_RESOLVED.active_coins)
BACKTEST_ONLY_COINS: list[str] = list(_INITIAL_RESOLVED.backtest_only_coins)
BACKTEST_BATCH_SYMBOLS: tuple[str, ...] = tuple(_INITIAL_RESOLVED.backtest_batch_symbols)

DEFAULT_COIN_PROFILE_VALUES: dict[str, dict[str, object]] = _copy_nested_object_dict(
    _INITIAL_RESOLVED.default_coin_profile_values
)
DEFAULT_COIN_STRATEGIES: dict[str, str] = dict(_INITIAL_RESOLVED.default_coin_strategies)
DEFAULT_COIN_STRATEGY_PARAMS: dict[str, dict[str, float]] = _copy_nested_float_dict(
    _INITIAL_RESOLVED.default_coin_strategy_params
)

settings = _INITIAL_RESOLVED.settings

USE_SETUP_GATE = settings.trading.use_setup_gate
COIN_STRATEGIES = settings.strategy.coin_strategies
COIN_STRATEGY_PARAMS = settings.strategy.coin_strategy_params

# Default scan strategies for automated multi-strategy runs
SCAN_STRATEGIES: tuple[str, ...] = tuple(DEFAULT_SCAN_STRATEGIES)

# Optional extra backtest-only strategies. Keep empty unless explicitly extended.
BACKTEST_ONLY_AVAILABLE_STRATEGIES: tuple[str, ...] = ()

# Optional optimizer strategy overrides per coin (research only).
BACKTEST_OPTIMIZER_COIN_STRATEGIES: dict[str, str] = {}

# Optional frozen winner presets for EMA Band Rejection 1h backtest mode.
EMA_BAND_REJECTION_1H_WINNERS: dict[str, dict[str, object]] = {}


def migrate_to_live(
    symbol: str,
    best_profile: dict[str, float] | None,
    *,
    min_confidence: float,
) -> None:
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must not be empty.")
    if not isinstance(best_profile, dict) or not best_profile:
        raise ValueError("best_profile must be a non-empty dict.")

    runtime_state = load_runtime_profile_state()
    base_profile = dict(runtime_state.default_coin_profile_values.get("BTCUSDT", {}))
    base_profile.update(runtime_state.default_coin_profile_values.get(normalized_symbol, {}))
    base_strategy_params = dict(
        runtime_state.default_coin_strategy_params.get(normalized_symbol, {})
    )

    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    take_profit_pct = _to_float(
        best_profile.get(
            "take_profit_pct",
            base_profile.get("take_profit_pct", settings.trading.take_profit_pct),
        ),
        settings.trading.take_profit_pct,
    )
    stop_loss_pct = _to_float(
        best_profile.get(
            "stop_loss_pct",
            base_profile.get("stop_loss_pct", settings.trading.stop_loss_pct),
        ),
        settings.trading.stop_loss_pct,
    )
    trailing_activation_pct = _to_float(
        best_profile.get(
            "trailing_activation_pct",
            base_profile.get("trailing_activation_pct", settings.trading.trailing_activation_pct),
        ),
        settings.trading.trailing_activation_pct,
    )
    trailing_distance_pct = _to_float(
        best_profile.get(
            "trailing_distance_pct",
            base_profile.get("trailing_distance_pct", settings.trading.trailing_distance_pct),
        ),
        settings.trading.trailing_distance_pct,
    )

    default_breakeven_activation = float(
        base_profile.get("breakeven_activation_pct")
        if base_profile.get("breakeven_activation_pct") is not None
        else 3.5
    )
    default_breakeven_buffer = float(
        base_profile.get("breakeven_buffer_pct")
        if base_profile.get("breakeven_buffer_pct") is not None
        else 0.2
    )
    default_tight_trailing_activation = float(
        base_profile.get("tight_trailing_activation_pct")
        if base_profile.get("tight_trailing_activation_pct") is not None
        else 8.0
    )
    default_tight_trailing_distance = float(
        base_profile.get("tight_trailing_distance_pct")
        if base_profile.get("tight_trailing_distance_pct") is not None
        else 0.3
    )

    breakeven_activation_pct = _to_float(
        best_profile.get("breakeven_activation_pct", default_breakeven_activation),
        default_breakeven_activation,
    )
    breakeven_buffer_pct = _to_float(
        best_profile.get("breakeven_buffer_pct", default_breakeven_buffer),
        default_breakeven_buffer,
    )
    tight_trailing_activation_pct = _to_float(
        best_profile.get("tight_trailing_activation_pct", default_tight_trailing_activation),
        default_tight_trailing_activation,
    )
    tight_trailing_distance_pct = _to_float(
        best_profile.get("tight_trailing_distance_pct", default_tight_trailing_distance),
        default_tight_trailing_distance,
    )

    strategy_name = _resolve_strategy_name_for_deploy(
        symbol=normalized_symbol,
        best_profile=best_profile,
        runtime_settings=settings,
    )
    strategy_params = _build_strategy_params_for_deploy(
        strategy_name=strategy_name,
        best_profile=best_profile,
        base_strategy_params=base_strategy_params,
        runtime_settings=settings,
    )

    min_confidence_value = max(0.0, min(float(min_confidence), 100.0))
    deployed_profile = {
        "interval": str(base_profile.get("interval") or settings.trading.interval),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "breakeven_activation_pct": breakeven_activation_pct,
        "breakeven_buffer_pct": breakeven_buffer_pct,
        "trailing_activation_pct": trailing_activation_pct,
        "trailing_distance_pct": trailing_distance_pct,
        "tight_trailing_activation_pct": tight_trailing_activation_pct,
        "tight_trailing_distance_pct": tight_trailing_distance_pct,
        "default_leverage": int(
            base_profile.get("default_leverage", settings.trading.default_leverage)
            or settings.trading.default_leverage
        ),
        "min_confidence": min_confidence_value,
    }

    runtime_state.default_coin_profile_values[normalized_symbol] = dict(deployed_profile)
    runtime_state.default_coin_strategies[normalized_symbol] = strategy_name
    runtime_state.default_coin_strategy_params[normalized_symbol] = {
        str(param_name): float(param_value)
        for param_name, param_value in strategy_params.items()
    }

    next_active_coins = normalize_symbol_list([*runtime_state.active_coins, normalized_symbol])
    next_active_set = set(next_active_coins)
    runtime_state.active_coins = next_active_coins
    runtime_state.backtest_only_coins = [
        candidate_symbol
        for candidate_symbol in normalize_symbol_list(runtime_state.backtest_only_coins)
        if candidate_symbol not in next_active_set
    ]

    save_runtime_profile_state(runtime_state)
    _reload_runtime_settings()


__all__ = (
    "ACTIVE_COINS",
    "BACKTEST_ONLY_COINS",
    "BACKTEST_BATCH_SYMBOLS",
    "MAX_BACKTEST_CANDLES",
    "DEFAULT_COIN_PROFILE_VALUES",
    "DEFAULT_COIN_STRATEGIES",
    "DEFAULT_COIN_STRATEGY_PARAMS",
    "PRODUCTION_PROFILE_INTERVAL",
    "PRODUCTION_PROFILE_REGISTRY",
    "PRODUCTION_STRATEGY_ALIASES",
    "EMA_BAND_REJECTION_1H_EXCLUDED_COINS",
    "RequestLimits",
    "APISettings",
    "LiveSettings",
    "StrategySettings",
    "TradingSettings",
    "CoinProfileSettings",
    "Settings",
    "settings",
    "USE_SETUP_GATE",
    "COIN_STRATEGIES",
    "COIN_STRATEGY_PARAMS",
    "SCAN_STRATEGIES",
    "BACKTEST_ONLY_AVAILABLE_STRATEGIES",
    "BACKTEST_OPTIMIZER_COIN_STRATEGIES",
    "EMA_BAND_REJECTION_1H_WINNERS",
    "migrate_to_live",
)
