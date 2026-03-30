from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from core.data.db import PaperTrade

PAPER_TRADES_PATH = Path("paper_trades.json")


def save_paper_trades(
    trades: Sequence[PaperTrade | dict[str, Any]],
    path: str | Path = PAPER_TRADES_PATH,
) -> None:
    file_path = Path(path)
    serializable_trades = [_serialize_trade(trade) for trade in trades]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_trades, handle, indent=2, sort_keys=True)
    temp_path.replace(file_path)


def load_paper_trades(path: str | Path = PAPER_TRADES_PATH) -> list[PaperTrade]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    trades: list[PaperTrade] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            trades.append(_deserialize_trade(item))
        except (KeyError, TypeError, ValueError):
            continue
    return trades


def _serialize_trade(trade: PaperTrade | dict[str, Any]) -> dict[str, Any]:
    data = asdict(trade) if is_dataclass(trade) else dict(trade)
    for key in ("entry_time", "exit_time"):
        value = data.get(key)
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


def _deserialize_trade(payload: dict[str, Any]) -> PaperTrade:
    return PaperTrade(
        id=int(payload.get("id", 0)),
        symbol=str(payload["symbol"]),
        side=str(payload["side"]),
        entry_time=_deserialize_datetime(payload["entry_time"]),
        entry_price=float(payload["entry_price"]),
        qty=float(payload["qty"]),
        leverage=int(payload["leverage"]),
        status=str(payload.get("status", "OPEN")),
        exit_time=_deserialize_optional_datetime(payload.get("exit_time")),
        exit_price=_deserialize_optional_float(payload.get("exit_price")),
        pnl=_deserialize_optional_float(payload.get("pnl")),
        total_fees=float(payload.get("total_fees", 0.0) or 0.0),
        high_water_mark=float(payload.get("high_water_mark", payload["entry_price"])),
    )


def _deserialize_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("Expected ISO datetime string.")


def _deserialize_optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    return _deserialize_datetime(value)


def _deserialize_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
