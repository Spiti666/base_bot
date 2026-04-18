from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from config_defaults import (
    build_backtest_batch_symbols,
    normalize_symbol_list,
)
from config_schema import (
    LiveSettings,
    Settings,
    StrategySettings,
    TradingSettings,
    build_coin_profile_settings_map,
)
from runtime_profiles import RuntimeProfileState, load_runtime_profile_state


@dataclass(slots=True)
class ResolvedConfig:
    active_coins: list[str]
    backtest_only_coins: list[str]
    backtest_batch_symbols: tuple[str, ...]
    default_coin_profile_values: dict[str, dict[str, object]]
    default_coin_strategies: dict[str, str]
    default_coin_strategy_params: dict[str, dict[str, float]]
    settings: Settings


def _normalize_runtime_state(state: RuntimeProfileState) -> RuntimeProfileState:
    normalized_active_coins = normalize_symbol_list(state.active_coins)
    active_set = set(normalized_active_coins)
    normalized_backtest_only_coins = [
        symbol
        for symbol in normalize_symbol_list(state.backtest_only_coins)
        if symbol not in active_set
    ]
    return RuntimeProfileState(
        active_coins=normalized_active_coins,
        backtest_only_coins=normalized_backtest_only_coins,
        default_coin_profile_values={
            str(symbol).strip().upper(): dict(profile)
            for symbol, profile in state.default_coin_profile_values.items()
            if str(symbol).strip() and isinstance(profile, Mapping)
        },
        default_coin_strategies={
            str(symbol).strip().upper(): str(strategy_name)
            for symbol, strategy_name in state.default_coin_strategies.items()
            if str(symbol).strip()
        },
        default_coin_strategy_params={
            str(symbol).strip().upper(): {
                str(param_name): float(param_value)
                for param_name, param_value in dict(params).items()
            }
            for symbol, params in state.default_coin_strategy_params.items()
            if str(symbol).strip() and isinstance(params, Mapping)
        },
    )


def load_resolved_config() -> ResolvedConfig:
    runtime_state = _normalize_runtime_state(load_runtime_profile_state())
    backtest_batch_symbols = build_backtest_batch_symbols(
        runtime_state.active_coins,
        runtime_state.backtest_only_coins,
    )
    resolved_default_coin_profile_values = {
        symbol: dict(profile)
        for symbol, profile in runtime_state.default_coin_profile_values.items()
    }
    resolved_default_coin_strategies = {
        symbol: str(strategy_name)
        for symbol, strategy_name in runtime_state.default_coin_strategies.items()
    }
    resolved_default_coin_strategy_params = {
        symbol: {
            str(param_name): float(param_value)
            for param_name, param_value in params.items()
        }
        for symbol, params in runtime_state.default_coin_strategy_params.items()
    }

    live_settings = LiveSettings(available_symbols=tuple(runtime_state.active_coins))
    strategy_settings = StrategySettings(
        coin_strategies=resolved_default_coin_strategies,
        coin_strategy_params=resolved_default_coin_strategy_params,
    )
    trading_settings = TradingSettings(
        coin_profiles=build_coin_profile_settings_map(resolved_default_coin_profile_values)
    )
    settings = Settings(
        live=live_settings,
        strategy=strategy_settings,
        trading=trading_settings,
    )
    return ResolvedConfig(
        active_coins=list(runtime_state.active_coins),
        backtest_only_coins=list(runtime_state.backtest_only_coins),
        backtest_batch_symbols=backtest_batch_symbols,
        default_coin_profile_values=resolved_default_coin_profile_values,
        default_coin_strategies=resolved_default_coin_strategies,
        default_coin_strategy_params=resolved_default_coin_strategy_params,
        settings=settings,
    )

