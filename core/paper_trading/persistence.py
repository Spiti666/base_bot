from __future__ import annotations

import json
import logging
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
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger("paper_engine.persistence").exception(
            "Failed to load paper trades from %s: %s",
            file_path,
            exc,
        )
        return []

    if not isinstance(payload, list):
        logging.getLogger("paper_engine.persistence").warning(
            "Invalid paper_trades payload type in %s (expected list, got %s).",
            file_path,
            type(payload).__name__,
        )
        return []

    trades: list[PaperTrade] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            logging.getLogger("paper_engine.persistence").warning(
                "Skipping invalid paper trade entry at index %d in %s (expected dict).",
                index,
                file_path,
            )
            continue
        try:
            trades.append(_deserialize_trade(item))
        except (KeyError, TypeError, ValueError) as exc:
            logging.getLogger("paper_engine.persistence").warning(
                "Skipping malformed paper trade entry at index %d in %s: %s",
                index,
                file_path,
                exc,
            )
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
    strategy_name_raw = payload.get("strategy_name")
    strategy_name = None
    if strategy_name_raw is not None:
        normalized_strategy_name = str(strategy_name_raw).strip()
        strategy_name = normalized_strategy_name or None
    timeframe_raw = payload.get("timeframe")
    timeframe = None
    if timeframe_raw is not None:
        normalized_timeframe = str(timeframe_raw).strip()
        timeframe = normalized_timeframe or None
    regime_label_raw = payload.get("regime_label_at_entry")
    regime_label_at_entry = None
    if regime_label_raw is not None:
        normalized_regime_label = str(regime_label_raw).strip()
        regime_label_at_entry = normalized_regime_label or None
    session_label_raw = payload.get("session_label")
    session_label = None
    if session_label_raw is not None:
        normalized_session_label = str(session_label_raw).strip()
        session_label = normalized_session_label or None
    profile_version_raw = payload.get("profile_version")
    profile_version = None
    if profile_version_raw is not None:
        normalized_profile_version = str(profile_version_raw).strip()
        profile_version = normalized_profile_version or None
    review_status_raw = payload.get("review_status")
    review_status = None
    if review_status_raw is not None:
        normalized_review_status = str(review_status_raw).strip()
        review_status = normalized_review_status or None
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
        strategy_name=strategy_name,
        timeframe=timeframe,
        regime_label_at_entry=regime_label_at_entry,
        regime_confidence=_deserialize_optional_float(payload.get("regime_confidence")),
        session_label=session_label,
        signal_strength=_deserialize_optional_float(payload.get("signal_strength")),
        confidence_score=_deserialize_optional_float(payload.get("confidence_score")),
        atr_pct_at_entry=_deserialize_optional_float(payload.get("atr_pct_at_entry")),
        volume_ratio_at_entry=_deserialize_optional_float(payload.get("volume_ratio_at_entry")),
        spread_estimate=_deserialize_optional_float(payload.get("spread_estimate")),
        move_already_extended_pct=_deserialize_optional_float(payload.get("move_already_extended_pct")),
        entry_snapshot_json=_deserialize_optional_text(payload.get("entry_snapshot_json")),
        lifecycle_snapshot_json=_deserialize_optional_text(payload.get("lifecycle_snapshot_json")),
        profile_version=profile_version,
        review_status=review_status,
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


def _deserialize_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
