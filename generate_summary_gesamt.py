from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config_sections.production_registry import PRODUCTION_STRATEGY_ALIASES

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "summary_gesamt.txt"

BACKTEST_FILE_PATTERNS: tuple[str, ...] = (
    "backtest_compact_summary_*.txt",
    "archive/backtest_compact_summaries/backtest_compact_summary_*.txt",
)

STRATEGY_LABEL_TO_NAME: dict[str, str] = {
    "ema cross + volume": "ema_cross_volume",
    "ema band rejection": "ema_band_rejection",
    "frama cross": "frama_cross",
    "dual thrust": "dual_thrust",
}

STRATEGY_NAME_TO_LABEL: dict[str, str] = {
    "ema_cross_volume": "EMA Cross + Volume",
    "ema_band_rejection": "EMA Band Rejection",
    "frama_cross": "FRAMA Cross",
    "dual_thrust": "Dual Thrust",
}

RISK_PARAM_ORDER: tuple[str, ...] = (
    "stop_loss_pct",
    "take_profit_pct",
    "trailing_activation_pct",
    "trailing_distance_pct",
    "tight_trailing_activation_pct",
    "tight_trailing_distance_pct",
    "breakeven_activation_pct",
    "breakeven_buffer_pct",
)

PROFILE_META_KEYS: set[str] = {
    "longest_consecutive_losses",
    "max_drawdown_pct",
    "avg_profit_per_trade_net_pct",
}

LIVE_PROFILE_META_KEYS: set[str] = {
    "strategy_name",
    "interval",
    "strategy_params",
    "default_leverage",
    "audit_meta",
}


@dataclass(frozen=True)
class LiveProfile:
    symbol: str
    strategy_name: str
    strategy_label: str
    interval: str
    leverage: int
    strategy_params: dict[str, float]
    risk: dict[str, float]

    @property
    def combined_profile(self) -> dict[str, float]:
        merged = dict(self.strategy_params)
        merged.update(self.risk)
        return merged

    @property
    def summary_live_profile(self) -> dict[str, dict[str, float]]:
        return {
            "strategy_params": dict(self.strategy_params),
            "risk": dict(self.risk),
        }


@dataclass
class BacktestRow:
    symbol: str
    strategy_name: str
    strategy_label: str
    interval: str
    source: str
    source_dt: datetime
    columns: dict[str, str]
    best_profile: dict[str, float]


@dataclass
class MatchResult:
    status: str
    source: str
    row: BacktestRow | None


def _parse_source_dt(path: Path) -> datetime:
    match = re.search(r"backtest_compact_summary_(\d{8})_(\d{6})\.txt$", path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _normalize_strategy_name(strategy_name: str | None) -> str:
    raw = str(strategy_name or "").strip()
    if not raw:
        return ""
    return PRODUCTION_STRATEGY_ALIASES.get(raw, raw)


def _strategy_name_from_label(strategy_label: str) -> str:
    return STRATEGY_LABEL_TO_NAME.get(str(strategy_label).strip().lower(), "")


def _strategy_label_from_name(strategy_name: str) -> str:
    return STRATEGY_NAME_TO_LABEL.get(strategy_name, strategy_name.replace("_", " ").title())


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    numeric = _parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _parse_profile_dict(payload: str) -> dict[str, float]:
    payload = payload.strip()
    parsed: Any
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in parsed.items():
        numeric = _parse_float(value)
        if numeric is None:
            continue
        normalized[str(key)] = numeric
    return normalized


def _collect_backtest_files() -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in BACKTEST_FILE_PATTERNS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                files[path.resolve()] = None
    return sorted(files.keys())


def _parse_backtest_rows(file_paths: list[Path]) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for path in file_paths:
        source = path.name
        source_dt = _parse_source_dt(path)
        header_columns: list[str] = []
        last_row: BacktestRow | None = None
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("symbol | strategy | interval |"):
                header_columns = [part.strip() for part in line.split("|")]
                last_row = None
                continue

            if "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 4 and re.fullmatch(r"[A-Z0-9]+USDT", parts[0]):
                    strategy_name = _strategy_name_from_label(parts[1])
                    if not strategy_name:
                        last_row = None
                        continue
                    columns: dict[str, str] = {}
                    if header_columns:
                        for index, column_name in enumerate(header_columns):
                            if index < len(parts):
                                columns[column_name] = parts[index]
                    row = BacktestRow(
                        symbol=parts[0],
                        strategy_name=strategy_name,
                        strategy_label=parts[1],
                        interval=parts[2],
                        source=source,
                        source_dt=source_dt,
                        columns=columns,
                        best_profile={},
                    )
                    rows.append(row)
                    last_row = row
                    continue

            if stripped.startswith("best_profile="):
                if last_row is None:
                    continue
                profile_payload = stripped.split("=", 1)[1].strip()
                last_row.best_profile = _parse_profile_dict(profile_payload)
    return rows


def _ordered_strategy_params(raw_params: Any) -> dict[str, float]:
    if not isinstance(raw_params, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in raw_params.items():
        numeric = _parse_float(value)
        if numeric is None:
            continue
        normalized[str(key)] = numeric
    return normalized


def _ordered_risk_params(raw_profile: dict[str, Any]) -> dict[str, float]:
    risk: dict[str, float] = {}
    for key in RISK_PARAM_ORDER:
        if key not in raw_profile:
            continue
        numeric = _parse_float(raw_profile.get(key))
        if numeric is None:
            continue
        risk[key] = numeric

    extra_risk_keys = sorted(
        key
        for key, value in raw_profile.items()
        if key not in LIVE_PROFILE_META_KEYS
        and key not in risk
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    for key in extra_risk_keys:
        numeric = _parse_float(raw_profile.get(key))
        if numeric is None:
            continue
        risk[key] = numeric
    return risk


def _build_live_profiles() -> list[LiveProfile]:
    profiles: list[LiveProfile] = []
    for symbol in sorted(config.PRODUCTION_PROFILE_REGISTRY.keys()):
        raw_profile = dict(config.PRODUCTION_PROFILE_REGISTRY.get(symbol, {}))
        strategy_name = _normalize_strategy_name(str(raw_profile.get("strategy_name") or ""))
        if not strategy_name:
            continue
        interval = str(raw_profile.get("interval") or "").strip()
        leverage = _parse_int(raw_profile.get("default_leverage")) or int(config.settings.trading.default_leverage)
        strategy_params = _ordered_strategy_params(raw_profile.get("strategy_params"))
        risk = _ordered_risk_params(raw_profile)
        profiles.append(
            LiveProfile(
                symbol=symbol,
                strategy_name=strategy_name,
                strategy_label=_strategy_label_from_name(strategy_name),
                interval=interval,
                leverage=leverage,
                strategy_params=strategy_params,
                risk=risk,
            )
        )
    return profiles


def _profile_core(best_profile: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in best_profile.items()
        if key not in PROFILE_META_KEYS
    }


def _is_exact_profile_match(live_profile: dict[str, float], best_profile: dict[str, float]) -> bool:
    live_keys = set(live_profile.keys())
    best_keys = set(best_profile.keys())
    if live_keys != best_keys:
        return False
    for key in live_keys:
        if abs(live_profile[key] - best_profile[key]) > 1e-9:
            return False
    return True


def _profile_distance(live_profile: dict[str, float], best_profile: dict[str, float]) -> float:
    all_keys = set(live_profile.keys()) | set(best_profile.keys())
    if not all_keys:
        return 0.0
    distance = 0.0
    for key in all_keys:
        if key not in live_profile or key not in best_profile:
            distance += 1.0
            continue
        left = live_profile[key]
        right = best_profile[key]
        scale = max(abs(left), abs(right), 1.0)
        distance += abs(left - right) / scale
    return distance


def _select_match(live: LiveProfile, rows: list[BacktestRow]) -> MatchResult:
    filtered = [
        row
        for row in rows
        if row.symbol == live.symbol
        and row.strategy_name == live.strategy_name
        and row.interval == live.interval
        and row.best_profile
    ]
    if not filtered:
        return MatchResult(status="MISSING_PROFILE", source="-", row=None)

    live_combined = live.combined_profile
    exact_rows: list[BacktestRow] = []
    scored_rows: list[tuple[float, float, float, BacktestRow]] = []

    for row in filtered:
        core = _profile_core(row.best_profile)
        if _is_exact_profile_match(live_combined, core):
            exact_rows.append(row)
        distance = _profile_distance(live_combined, core)
        pnl = _parse_float(row.columns.get("pnl_usd"))
        scored_rows.append((distance, -row.source_dt.timestamp(), -(pnl or 0.0), row))

    if exact_rows:
        exact_rows.sort(
            key=lambda row: (
                -row.source_dt.timestamp(),
                -(_parse_float(row.columns.get("pnl_usd")) or 0.0),
            )
        )
        winner = exact_rows[0]
        return MatchResult(status="EXACT_PROFILE", source=winner.source, row=winner)

    scored_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    winner = scored_rows[0][3]
    return MatchResult(status="NEAREST_PROFILE", source=winner.source, row=winner)


def _format_metrics(row: BacktestRow | None) -> str:
    if row is None:
        return "-"
    pnl = _parse_float(row.columns.get("pnl_usd"))
    pf = _parse_float(row.columns.get("robust_pf"))
    if pf is None:
        pf = _parse_float(row.columns.get("profit_factor"))
    win_rate = _parse_float(row.columns.get("win_rate_pct"))
    trades = _parse_int(row.columns.get("trades"))
    max_dd = _parse_float(row.columns.get("max_dd_pct"))
    real_rrr = _parse_float(row.columns.get("real_rrr"))

    def _fmt_num(value: float | None, *, suffix: str = "", precision: int = 2) -> str:
        if value is None:
            return "-"
        return f"{value:.{precision}f}{suffix}"

    trades_text = "-" if trades is None else str(trades)
    return (
        f"PnL={_fmt_num(pnl)} | "
        f"PF={_fmt_num(pf)} | "
        f"WR={_fmt_num(win_rate, suffix='%')} | "
        f"Trades={trades_text} | "
        f"MaxDD={_fmt_num(max_dd, suffix='%')} | "
        f"RRR={_fmt_num(real_rrr)}"
    )


def generate_summary_gesamt(output_path: Path = DEFAULT_OUTPUT_PATH) -> str:
    backtest_files = _collect_backtest_files()
    rows = _parse_backtest_rows(backtest_files)
    live_profiles = _build_live_profiles()

    lines: list[str] = []
    lines.append("Live Strategy vs Backtest Performance (summary_gesamt.txt)")
    lines.append(f"Generated (UTC): {datetime.now(UTC).strftime('%Y-%m-%d')}")
    lines.append("Source of live truth: config.PRODUCTION_PROFILE_REGISTRY")
    lines.append(f"Backtest rows scanned: {len(rows)}")
    lines.append(f"Live profiles scanned: {len(live_profiles)}")
    if not backtest_files:
        lines.append("Hinweis: keine Backtest-Compact-Dateien gefunden (backtest_compact_summary_*.txt).")
        lines.append("Die Live-Profile werden ohne Backtest-Match ausgewiesen.")
    lines.append("")

    exact_count = 0
    matched_count = 0
    missing_symbols: list[str] = []

    for index, live in enumerate(live_profiles, start=1):
        match = _select_match(live, rows)
        if match.status == "EXACT_PROFILE":
            exact_count += 1
            matched_count += 1
        elif match.status == "NEAREST_PROFILE":
            matched_count += 1
        else:
            missing_symbols.append(live.symbol)

        lines.append(
            f"{index:02d}. {live.symbol} | {live.strategy_label} | {live.interval} | live_leverage={live.leverage}"
        )
        lines.append(f"  Backtest match: {match.status} | source={match.source}")
        lines.append(f"  Metrics: {_format_metrics(match.row)}")
        if match.row is None:
            lines.append("  best_profile={}")
        else:
            lines.append(f"  best_profile={match.row.best_profile}")
        lines.append(f"  live_profile={live.summary_live_profile}")
        lines.append("")

    total = len(live_profiles)
    lines.append("Summary")
    lines.append(f"Matched coins: {matched_count}/{total}")
    lines.append(f"Exact param matches: {exact_count}/{total}")
    if missing_symbols:
        lines.append(f"Missing matches: {', '.join(missing_symbols)}")
    else:
        lines.append("Missing matches: none")

    report_text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary_gesamt.txt from config.PRODUCTION_PROFILE_REGISTRY and backtest compact summaries."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output file path (default: summary_gesamt.txt in project root).",
    )
    args = parser.parse_args()
    report = generate_summary_gesamt(Path(args.out))
    print(report)
    print(f"Saved summary to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
