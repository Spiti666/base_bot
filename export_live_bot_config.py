from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_COIN_STRATEGIES,
    settings,
)
from config_sections.production_registry import PRODUCTION_STRATEGY_ALIASES

RISK_FIELDS: tuple[str, ...] = (
    "interval",
    "default_leverage",
    "stop_loss_pct",
    "take_profit_pct",
    "breakeven_activation_pct",
    "breakeven_buffer_pct",
    "trailing_activation_pct",
    "trailing_distance_pct",
    "tight_trailing_activation_pct",
    "tight_trailing_distance_pct",
)

STRATEGY_PARAM_ALIASES: dict[str, str] = {
    # Legacy/short names mapped to current canonical runtime keys.
    "supertrend_length": "supertrend_ema_supertrend_length",
    "supertrend_multiplier": "supertrend_ema_supertrend_multiplier",
    "ema_length": "supertrend_ema_ema_length",
}

STRATEGY_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "ema_cross_volume": ("ema_fast_period", "ema_slow_period", "volume_multiplier"),
    "frama_cross": ("frama_fast_period", "frama_slow_period", "volume_multiplier"),
    "dual_thrust": ("dual_thrust_period", "dual_thrust_k1", "dual_thrust_k2"),
    "supertrend_ema": (
        "supertrend_ema_supertrend_length",
        "supertrend_ema_supertrend_multiplier",
        "supertrend_ema_ema_length",
    ),
}


def _normalize_strategy_name(strategy_name: str | None) -> str | None:
    if strategy_name is None:
        return None
    normalized = str(strategy_name).strip()
    if not normalized:
        return None
    return PRODUCTION_STRATEGY_ALIASES.get(normalized, normalized)


def _resolve_runtime_risk_profile(symbol: str) -> dict[str, Any]:
    profile = settings.trading.coin_profiles.get(symbol)
    interval = settings.trading.interval
    leverage = settings.trading.default_leverage
    stop_loss_pct = settings.trading.stop_loss_pct
    take_profit_pct = settings.trading.take_profit_pct
    breakeven_activation_pct: float | None = None
    breakeven_buffer_pct: float | None = None
    trailing_activation_pct = settings.trading.trailing_activation_pct
    trailing_distance_pct = settings.trading.trailing_distance_pct
    tight_trailing_activation_pct: float | None = None
    tight_trailing_distance_pct: float | None = None

    if profile is not None:
        if profile.interval is not None:
            interval = profile.interval
        if profile.default_leverage is not None:
            leverage = profile.default_leverage
        if profile.stop_loss_pct is not None:
            stop_loss_pct = profile.stop_loss_pct
        if profile.take_profit_pct is not None:
            take_profit_pct = profile.take_profit_pct
        if profile.breakeven_activation_pct is not None:
            breakeven_activation_pct = profile.breakeven_activation_pct
        if profile.breakeven_buffer_pct is not None:
            breakeven_buffer_pct = profile.breakeven_buffer_pct
        if profile.trailing_activation_pct is not None:
            trailing_activation_pct = profile.trailing_activation_pct
        if profile.trailing_distance_pct is not None:
            trailing_distance_pct = profile.trailing_distance_pct
        if profile.tight_trailing_activation_pct is not None:
            tight_trailing_activation_pct = profile.tight_trailing_activation_pct
        if profile.tight_trailing_distance_pct is not None:
            tight_trailing_distance_pct = profile.tight_trailing_distance_pct

    return {
        "interval": interval,
        "default_leverage": leverage,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "breakeven_activation_pct": breakeven_activation_pct,
        "breakeven_buffer_pct": breakeven_buffer_pct,
        "trailing_activation_pct": trailing_activation_pct,
        "trailing_distance_pct": trailing_distance_pct,
        "tight_trailing_activation_pct": tight_trailing_activation_pct,
        "tight_trailing_distance_pct": tight_trailing_distance_pct,
    }


def _build_effective_strategy_params(symbol: str) -> dict[str, float]:
    params: dict[str, float] = {
        "dual_thrust_period": float(settings.strategy.dual_thrust_period),
        "dual_thrust_k1": float(settings.strategy.dual_thrust_k1),
        "dual_thrust_k2": float(settings.strategy.dual_thrust_k2),
        "ema_fast_period": float(settings.strategy.ema_fast_period),
        "ema_slow_period": float(settings.strategy.ema_slow_period),
        "volume_sma_period": float(settings.strategy.volume_sma_period),
        "volume_multiplier": float(settings.strategy.volume_multiplier),
        "frama_fast_period": float(settings.strategy.frama_fast_period),
        "frama_slow_period": float(settings.strategy.frama_slow_period),
        "supertrend_ema_supertrend_length": float(settings.strategy.supertrend_ema_supertrend_length),
        "supertrend_ema_supertrend_multiplier": float(
            settings.strategy.supertrend_ema_supertrend_multiplier
        ),
        "supertrend_ema_ema_length": float(settings.strategy.supertrend_ema_ema_length),
        "chandelier_period": float(settings.trading.chandelier_period),
        "chandelier_multiplier": float(settings.trading.chandelier_multiplier),
    }

    coin_overrides = settings.strategy.coin_strategy_params.get(symbol, {}) or {}
    for raw_key, raw_value in coin_overrides.items():
        normalized_key = STRATEGY_PARAM_ALIASES.get(str(raw_key), str(raw_key))
        try:
            params[normalized_key] = float(raw_value)
        except (TypeError, ValueError):
            # Keep raw value if it cannot be converted cleanly.
            params[normalized_key] = raw_value  # type: ignore[assignment]
    return params


def _validate_live_config(
    *,
    strategy_name: str | None,
    runtime_risk: dict[str, Any],
    strategy_params: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    interval = str(runtime_risk.get("interval") or "").strip()
    if not interval:
        issues.append("missing interval")
    elif interval not in settings.api.timeframes:
        issues.append(f"unsupported interval '{interval}'")

    leverage = runtime_risk.get("default_leverage")
    if leverage is None:
        issues.append("missing default_leverage")
    else:
        try:
            leverage_value = int(leverage)
            if not settings.trading.min_leverage <= leverage_value <= settings.trading.max_leverage:
                issues.append(
                    f"default_leverage out of range ({leverage_value}, allowed {settings.trading.min_leverage}-{settings.trading.max_leverage})"
                )
        except (TypeError, ValueError):
            issues.append(f"invalid default_leverage '{leverage}'")

    def _require_positive(field_name: str) -> None:
        value = runtime_risk.get(field_name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            issues.append(f"invalid {field_name} '{value}'")
            return
        if numeric_value <= 0.0:
            issues.append(f"{field_name} must be > 0 (got {numeric_value})")

    def _require_non_negative(field_name: str) -> None:
        value = runtime_risk.get(field_name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            issues.append(f"invalid {field_name} '{value}'")
            return
        if numeric_value < 0.0:
            issues.append(f"{field_name} must be >= 0 (got {numeric_value})")

    _require_positive("stop_loss_pct")
    _require_positive("take_profit_pct")
    _require_non_negative("trailing_activation_pct")
    _require_non_negative("trailing_distance_pct")

    if strategy_name is None:
        return issues

    required_params = STRATEGY_REQUIRED_PARAMS.get(strategy_name)
    if required_params is None:
        issues.append(f"unknown strategy '{strategy_name}' for validator")
        return issues

    missing_params = [
        param_name
        for param_name in required_params
        if param_name not in strategy_params or strategy_params.get(param_name) is None
    ]
    if missing_params:
        issues.append("missing strategy params: " + ", ".join(missing_params))
        return issues

    def _param(name: str) -> float:
        return float(strategy_params[name])

    try:
        if strategy_name == "ema_cross_volume":
            if _param("ema_fast_period") >= _param("ema_slow_period"):
                issues.append("ema_fast_period must be < ema_slow_period")
            if _param("volume_multiplier") <= 0:
                issues.append("volume_multiplier must be > 0")
        elif strategy_name == "frama_cross":
            if _param("frama_fast_period") >= _param("frama_slow_period"):
                issues.append("frama_fast_period must be < frama_slow_period")
            if _param("volume_multiplier") <= 0:
                issues.append("volume_multiplier must be > 0")
        elif strategy_name == "dual_thrust":
            if _param("dual_thrust_period") <= 0:
                issues.append("dual_thrust_period must be > 0")
            if _param("dual_thrust_k1") <= 0 or _param("dual_thrust_k2") <= 0:
                issues.append("dual_thrust_k1/dual_thrust_k2 must be > 0")
        elif strategy_name == "supertrend_ema":
            if _param("supertrend_ema_supertrend_length") <= 1:
                issues.append("supertrend_ema_supertrend_length must be > 1")
            if _param("supertrend_ema_supertrend_multiplier") <= 0:
                issues.append("supertrend_ema_supertrend_multiplier must be > 0")
            if _param("supertrend_ema_ema_length") <= 1:
                issues.append("supertrend_ema_ema_length must be > 1")
    except (TypeError, ValueError) as exc:
        issues.append(f"invalid strategy param value ({exc})")

    return issues


def _format_pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.4f}%"
    except (TypeError, ValueError):
        return "-"


def _format_runtime_line(runtime_risk: dict[str, Any]) -> str:
    return (
        f"Interval: {runtime_risk['interval']} | "
        f"Leverage: {int(runtime_risk['default_leverage'])}x | "
        f"SL: {_format_pct(runtime_risk['stop_loss_pct'])} | "
        f"TP: {_format_pct(runtime_risk['take_profit_pct'])} | "
        f"BE: {_format_pct(runtime_risk['breakeven_activation_pct'])} | "
        f"TrailOn: {_format_pct(runtime_risk['trailing_activation_pct'])} | "
        f"TrailDist: {_format_pct(runtime_risk['trailing_distance_pct'])} | "
        f"TightOn: {_format_pct(runtime_risk['tight_trailing_activation_pct'])} | "
        f"TightDist: {_format_pct(runtime_risk['tight_trailing_distance_pct'])}"
    )


def export_live_bot_config(output_path: str | Path | None = None) -> str:
    """Export live runtime strategy/risk config for all active symbols."""
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    active_symbols = list(settings.live.available_symbols)

    lines: list[str] = []
    lines.append("LIVE BOT CONFIG EXPORT")
    lines.append(f"Generated: {now_utc}")
    lines.append(f"Active Symbols: {len(active_symbols)}")
    lines.append("")

    ok_count = 0
    issue_count = 0

    for symbol in active_symbols:
        runtime_strategy = _normalize_strategy_name(settings.strategy.coin_strategies.get(symbol))
        default_strategy = _normalize_strategy_name(DEFAULT_COIN_STRATEGIES.get(symbol))
        runtime_risk = _resolve_runtime_risk_profile(symbol)
        strategy_params = _build_effective_strategy_params(symbol)
        is_inactive = runtime_strategy is None
        issues = _validate_live_config(
            strategy_name=runtime_strategy,
            runtime_risk=runtime_risk,
            strategy_params=strategy_params,
        )
        integrity_status = "OK" if not issues else "ISSUES"
        if issues:
            issue_count += 1
        else:
            ok_count += 1

        if is_inactive:
            lines.append(f"[{symbol}] Strategy: INACTIVE/WATCHLIST")
        else:
            lines.append(f"[{symbol}] Strategy: {runtime_strategy}")
        lines.append("  " + _format_runtime_line(runtime_risk))

        if runtime_strategy is not None:
            required_params = STRATEGY_REQUIRED_PARAMS.get(runtime_strategy, ())
            formatted_params = ", ".join(
                f"{param}={strategy_params.get(param)}"
                for param in required_params
            )
            if not formatted_params:
                formatted_params = "-"
            lines.append(f"  Strategy Params: {formatted_params}")
        else:
            lines.append("  Strategy Params: -")

        strategy_info = default_strategy if default_strategy is not None else "-"
        lines.append(f"  Default Mapping: strategy={strategy_info}")
        lines.append(f"  Integrity: {integrity_status}")
        if issues:
            for issue in issues:
                lines.append(f"    - {issue}")
        lines.append("")

    lines.append("SUMMARY")
    lines.append(f"  Symbols checked: {len(active_symbols)}")
    lines.append(f"  Integrity OK: {ok_count}")
    lines.append(f"  Integrity issues: {issue_count}")

    report_text = "\n".join(lines).rstrip() + "\n"

    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(report_text, encoding="utf-8")

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export live runtime bot config (strategy + risk + strategy params) "
            "for all active symbols."
        )
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for a text report.",
    )
    args = parser.parse_args()

    report = export_live_bot_config(args.out)
    print(report)
    if args.out:
        print(f"Saved report to: {Path(args.out)}")


if __name__ == "__main__":
    main()
