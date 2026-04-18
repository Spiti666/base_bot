from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from config_defaults import (
    ACTIVE_COINS,
    BACKTEST_ONLY_COINS,
    RUNTIME_PROFILES_PATH,
    normalize_symbol_list,
)
from config_sections.backtest_optimizer_defaults import DEFAULT_COIN_PROFILE_VALUES
from config_sections.strategy_defaults import (
    DEFAULT_COIN_STRATEGIES,
    DEFAULT_COIN_STRATEGY_PARAMS,
)


@dataclass(slots=True)
class RuntimeProfileState:
    active_coins: list[str]
    backtest_only_coins: list[str]
    default_coin_profile_values: dict[str, dict[str, object]]
    default_coin_strategies: dict[str, str]
    default_coin_strategy_params: dict[str, dict[str, float]]

    def to_json_payload(self) -> dict[str, object]:
        return {
            "active_coins": list(self.active_coins),
            "backtest_only_coins": list(self.backtest_only_coins),
            "default_coin_profile_values": deepcopy(self.default_coin_profile_values),
            "default_coin_strategies": dict(self.default_coin_strategies),
            "default_coin_strategy_params": {
                symbol: dict(params)
                for symbol, params in self.default_coin_strategy_params.items()
            },
        }


def _default_state() -> RuntimeProfileState:
    return RuntimeProfileState(
        active_coins=normalize_symbol_list(ACTIVE_COINS),
        backtest_only_coins=normalize_symbol_list(BACKTEST_ONLY_COINS),
        default_coin_profile_values=deepcopy(DEFAULT_COIN_PROFILE_VALUES),
        default_coin_strategies={
            str(symbol).strip().upper(): str(strategy_name)
            for symbol, strategy_name in DEFAULT_COIN_STRATEGIES.items()
            if str(symbol).strip()
        },
        default_coin_strategy_params={
            str(symbol).strip().upper(): {
                str(param_name): float(param_value)
                for param_name, param_value in dict(params).items()
            }
            for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items()
            if str(symbol).strip()
        },
    )


def _safe_json_load(path: Path) -> dict[str, object] | None:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    return raw_payload


def _sanitize_profile_values(
    raw_profiles: object,
    fallback: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = deepcopy(fallback)
    if not isinstance(raw_profiles, Mapping):
        return merged
    for symbol, raw_profile in raw_profiles.items():
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            continue
        if not isinstance(raw_profile, Mapping):
            continue
        sanitized_profile: dict[str, object] = {}
        for key, value in raw_profile.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            sanitized_profile[normalized_key] = value
        merged[normalized_symbol] = sanitized_profile
    return merged


def _sanitize_coin_strategies(
    raw_strategies: object,
    fallback: dict[str, str],
) -> dict[str, str]:
    merged: dict[str, str] = dict(fallback)
    if not isinstance(raw_strategies, Mapping):
        return merged
    for symbol, strategy_name in raw_strategies.items():
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            continue
        normalized_strategy_name = str(strategy_name).strip()
        if not normalized_strategy_name:
            continue
        merged[normalized_symbol] = normalized_strategy_name
    return merged


def _sanitize_strategy_params(
    raw_params: object,
    fallback: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {
        symbol: dict(params)
        for symbol, params in fallback.items()
    }
    if not isinstance(raw_params, Mapping):
        return merged
    for symbol, raw_symbol_params in raw_params.items():
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol or not isinstance(raw_symbol_params, Mapping):
            continue
        symbol_params: dict[str, float] = {}
        for param_name, param_value in raw_symbol_params.items():
            normalized_param_name = str(param_name).strip()
            if not normalized_param_name:
                continue
            try:
                symbol_params[normalized_param_name] = float(param_value)
            except (TypeError, ValueError):
                continue
        merged[normalized_symbol] = symbol_params
    return merged


def _sanitize_symbol_lists(
    active_coins: object,
    backtest_only_coins: object,
    *,
    fallback_active: list[str],
    fallback_backtest_only: list[str],
) -> tuple[list[str], list[str]]:
    normalized_active = (
        normalize_symbol_list(active_coins)
        if isinstance(active_coins, list)
        else list(fallback_active)
    )
    normalized_backtest_only = (
        normalize_symbol_list(backtest_only_coins)
        if isinstance(backtest_only_coins, list)
        else list(fallback_backtest_only)
    )
    active_set = set(normalized_active)
    normalized_backtest_only = [
        symbol
        for symbol in normalized_backtest_only
        if symbol not in active_set
    ]
    return normalized_active, normalized_backtest_only


def load_runtime_profile_state(path: Path | None = None) -> RuntimeProfileState:
    resolved_path = RUNTIME_PROFILES_PATH if path is None else Path(path)
    defaults = _default_state()
    payload = _safe_json_load(resolved_path)
    if payload is None:
        return defaults

    active_coins, backtest_only_coins = _sanitize_symbol_lists(
        payload.get("active_coins"),
        payload.get("backtest_only_coins"),
        fallback_active=defaults.active_coins,
        fallback_backtest_only=defaults.backtest_only_coins,
    )
    return RuntimeProfileState(
        active_coins=active_coins,
        backtest_only_coins=backtest_only_coins,
        default_coin_profile_values=_sanitize_profile_values(
            payload.get("default_coin_profile_values"),
            defaults.default_coin_profile_values,
        ),
        default_coin_strategies=_sanitize_coin_strategies(
            payload.get("default_coin_strategies"),
            defaults.default_coin_strategies,
        ),
        default_coin_strategy_params=_sanitize_strategy_params(
            payload.get("default_coin_strategy_params"),
            defaults.default_coin_strategy_params,
        ),
    )


def save_runtime_profile_state(
    state: RuntimeProfileState,
    path: Path | None = None,
) -> None:
    resolved_path = RUNTIME_PROFILES_PATH if path is None else Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_json_payload()
    temp_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(resolved_path)
