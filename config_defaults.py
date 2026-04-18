from __future__ import annotations

from pathlib import Path
from typing import Iterable


ACTIVE_COINS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "BNBUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ARBUSDT",
    "SUIUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "LINKUSDT",
    "TRXUSDT",
    "FILUSDT",
    "AAVEUSDT",
]

BACKTEST_ONLY_COINS: list[str] = [
    "TONUSDT",
    "WLDUSDT",
    "APTUSDT",
    "FETUSDT",
    "TAOUSDT",
    "ENAUSDT",
    "QNTUSDT",
    "ETCUSDT",
    "OPUSDT",
    "SEIUSDT",
    "INJUSDT",
    "ATOMUSDT",
    "POLUSDT",
    "TIAUSDT",
    "HBARUSDT",
    "UNIUSDT",
    "XLMUSDT",
    "ALGOUSDT",
    "ICPUSDT",
    "RENDERUSDT",
    "SANDUSDT",
    "GALAUSDT",
    "JUPUSDT",
    "PENDLEUSDT",
    "RUNEUSDT",
    "CRVUSDT",
    "ONDOUSDT",
    "KASUSDT",
    "IMXUSDT",
    "PYTHUSDT",
    "1000BONKUSDT",
    "JASMYUSDT",
    "LDOUSDT",
    "STXUSDT",
    "DYDXUSDT",
    "ARUSDT",
    "SUSHIUSDT",
    "EGLDUSDT",
    "COMPUSDT",
    "THETAUSDT",
]

MAX_BACKTEST_CANDLES = 100_000

EMA_BAND_REJECTION_1H_EXCLUDED_COINS: tuple[str, ...] = (
    "BNBUSDT",
    "LINKUSDT",
    "AAVEUSDT",
    "TRXUSDT",
    "LTCUSDT",
)

RUNTIME_PROFILES_PATH = Path("data/runtime_profiles.json")


def normalize_symbol_list(symbols: Iterable[object]) -> list[str]:
    return list(
        dict.fromkeys(
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        )
    )


def build_backtest_batch_symbols(
    active_coins: Iterable[object],
    backtest_only_coins: Iterable[object],
) -> tuple[str, ...]:
    normalized_active = normalize_symbol_list(active_coins)
    normalized_backtest_only = [
        symbol
        for symbol in normalize_symbol_list(backtest_only_coins)
        if symbol not in set(normalized_active)
    ]
    return tuple(dict.fromkeys((*normalized_active, *normalized_backtest_only)))

