from __future__ import annotations

import config


TARGET_COINS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",  # PEPE
    "1000SHIBUSDT",  # SHIB
    "SOLUSDT",
)

TARGET_INTERVALS: dict[str, str] = {
    symbol: ("15m" if symbol == "SOLUSDT" else "5m")
    for symbol in TARGET_COINS
}


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _build_row(
    symbol: str,
    *,
    interval: str,
    strategy: str,
    sl: float | None,
    tp: float | None,
    be: float | None,
    leverage: int | None,
    warn: bool,
    notes: list[str],
) -> list[str]:
    return [
        "[!!!]" if warn else "",
        f"{symbol} | {interval}",
        strategy,
        _fmt(sl),
        _fmt(tp),
        _fmt(be),
        _fmt(leverage),
        ", ".join(notes) if notes else "",
    ]


def main() -> None:
    profiles = config.settings.trading.coin_profiles
    strategies = config.settings.strategy.coin_strategies

    rows: list[list[str]] = []
    target_set = set(TARGET_COINS)
    present_symbols = set(profiles) | set(strategies)

    for symbol in TARGET_COINS:
        profile = profiles.get(symbol)
        strategy = str(strategies.get(symbol, "-"))
        expected_interval = TARGET_INTERVALS[symbol]
        warn = False
        notes: list[str] = []

        if profile is None:
            warn = True
            notes.append("fehlend_in_config")
            rows.append(
                _build_row(
                    symbol,
                    interval="-",
                    strategy=strategy,
                    sl=None,
                    tp=None,
                    be=None,
                    leverage=None,
                    warn=warn,
                    notes=notes,
                )
            )
            continue

        interval = str(profile.interval) if profile.interval is not None else "-"
        if interval != expected_interval:
            warn = True
            notes.append(f"intervall_soll={expected_interval}")

        rows.append(
            _build_row(
                symbol,
                interval=interval,
                strategy=strategy,
                sl=profile.stop_loss_pct,
                tp=profile.take_profit_pct,
                be=profile.breakeven_activation_pct,
                leverage=profile.default_leverage,
                warn=warn,
                notes=notes,
            )
        )

    extra_symbols = sorted(present_symbols - target_set)
    for symbol in extra_symbols:
        profile = profiles.get(symbol)
        strategy = str(strategies.get(symbol, "-"))
        interval = "-"
        sl = None
        tp = None
        be = None
        leverage = None
        if profile is not None:
            interval = str(profile.interval) if profile.interval is not None else "-"
            sl = profile.stop_loss_pct
            tp = profile.take_profit_pct
            be = profile.breakeven_activation_pct
            leverage = profile.default_leverage
        rows.append(
            _build_row(
                symbol,
                interval=interval,
                strategy=strategy,
                sl=sl,
                tp=tp,
                be=be,
                leverage=leverage,
                warn=True,
                notes=["nicht_im_12er_universum"],
            )
        )

    headers = ["Warn", "Symbol | Intervall", "Strategie", "SL", "TP", "BE", "Hebel", "Hinweis"]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    sep = "-++-".join("-" * width for width in widths)
    header_line = " || ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    print(header_line)
    print(sep)
    for row in rows:
        print(" || ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    warn_count = sum(1 for row in rows if row[0] == "[!!!]")
    print()
    print(f"Coins geprüft: {len(TARGET_COINS)} | Zusätzliche Einträge: {len(extra_symbols)}")
    print(f"Warnungen: {warn_count}")


if __name__ == "__main__":
    main()
