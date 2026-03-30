from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import websockets

from core.data.db import CandleRecord


LogCallback = Callable[[str], Any]
CandleClosedCallback = Callable[[str, CandleRecord], Any]
CandleUpdateCallback = Callable[[str, CandleRecord], Any]
ReconnectCallback = Callable[[], Any]


class MultiTimeframeWebSocketManager:
    PUBLIC_WS_URL = "wss://fapi.bitunix.com/public/"
    _INTERVAL_TO_CHANNEL = {
        "1m": "market_kline_1min",
        "5m": "market_kline_5min",
        "15m": "market_kline_15min",
        "30m": "market_kline_30min",
        "1h": "market_kline_60min",
        "2h": "market_kline_2hour",
        "4h": "market_kline_4hour",
        "6h": "market_kline_6hour",
        "8h": "market_kline_8hour",
        "12h": "market_kline_12hour",
        "1d": "market_kline_1day",
        "3d": "market_kline_3day",
        "1w": "market_kline_1week",
        "1M": "market_kline_1month",
    }
    _INTERVAL_TO_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "8h": 28800,
        "12h": 43200,
        "1d": 86400,
        "3d": 259200,
        "1w": 604800,
        "1M": 2592000,
    }
    _CHANNEL_TO_INTERVAL = {value: key for key, value in _INTERVAL_TO_CHANNEL.items()}

    def __init__(
        self,
        symbols: Sequence[str],
        intervals: Sequence[str],
        *,
        symbol_intervals: dict[str, str] | None = None,
        on_candle_closed: CandleClosedCallback,
        on_candle_update: CandleUpdateCallback | None = None,
        on_log: LogCallback | None = None,
        on_reconnect: ReconnectCallback | None = None,
        reconnect_delay: float = 5.0,
        ping_interval: float = 20.0,
    ) -> None:
        if not symbols:
            raise ValueError("At least one symbol must be configured.")
        if not intervals:
            raise ValueError("At least one interval must be configured.")

        unsupported = [interval for interval in intervals if interval not in self._INTERVAL_TO_CHANNEL]
        if unsupported:
            raise ValueError(f"Unsupported websocket intervals: {', '.join(unsupported)}")

        self._symbols = list(dict.fromkeys(symbols))
        self._intervals = list(dict.fromkeys(intervals))
        self._symbol_intervals = self._build_symbol_intervals(symbol_intervals)
        self._symbol_lookup = {symbol.upper(): symbol for symbol in self._symbols}
        self._on_candle_closed = on_candle_closed
        self._on_candle_update = on_candle_update
        self._on_log = on_log
        self._on_reconnect = on_reconnect
        self._reconnect_delay = reconnect_delay
        self._ping_interval = ping_interval
        self._stop_event = asyncio.Event()
        self._latest_candles: dict[tuple[str, str], CandleRecord] = {}
        self._websocket: Any = None
        self._has_connected_once = False

    async def start(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_log(f"WebSocket error: {exc}")

            if not self._stop_event.is_set():
                await self._emit_log(
                    f"WebSocket disconnected. Reconnecting in {self._reconnect_delay:.1f}s."
                )
                await asyncio.sleep(self._reconnect_delay)

    async def stop(self) -> None:
        self._stop_event.set()
        websocket = self._websocket
        if websocket is not None:
            await websocket.close()

    async def _run_connection(self) -> None:
        await self._emit_log("Connecting to Bitunix WebSocket.")
        async with websockets.connect(
            self.PUBLIC_WS_URL,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=15,
        ) as websocket:
            self._websocket = websocket
            await self._subscribe(websocket)
            await self._emit_log("Bitunix WebSocket subscribed.")
            if self._has_connected_once:
                self._latest_candles.clear()
                await self._emit_log("WebSocket reconnected. Running incremental gap recovery.")
                await self._run_reconnect_hook()
            self._has_connected_once = True

            ping_task = asyncio.create_task(self._ping_loop(websocket))
            try:
                async for raw_message in websocket:
                    if self._stop_event.is_set():
                        break
                    await self._handle_raw_message(raw_message)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
                self._websocket = None

    async def _subscribe(self, websocket: Any) -> None:
        subscriptions = [
            {"symbol": symbol, "ch": self._INTERVAL_TO_CHANNEL[interval]}
            for symbol in self._symbols
            for interval in self._symbol_intervals.get(symbol, self._intervals)
        ]
        payload = {"op": "subscribe", "args": subscriptions}
        await websocket.send(json.dumps(payload))
        if subscriptions:
            formatted_subscriptions = ", ".join(
                f"{subscription['symbol']}->{self._CHANNEL_TO_INTERVAL.get(str(subscription['ch']), '?')} ({subscription['ch']})"
                for subscription in subscriptions
            )
            await self._emit_log(f"WebSocket streams active: {formatted_subscriptions}")

    async def _ping_loop(self, websocket: Any) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._ping_interval)
            if self._stop_event.is_set():
                break
            await websocket.send(json.dumps({"op": "ping", "ping": int(time.time())}))

    async def _handle_raw_message(self, raw_message: str | bytes) -> None:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        payload = json.loads(raw_message)
        if not isinstance(payload, dict):
            return

        if payload.get("op") == "pong" or payload.get("data") == "pong":
            return
        if payload.get("event") in {"subscribe", "subscribed"}:
            return

        channel = payload.get("ch")
        raw_symbol = payload.get("symbol")
        if raw_symbol is None:
            raw_symbol = payload.get("s")
        data = payload.get("data")

        if not isinstance(channel, str) or not isinstance(raw_symbol, str):
            return
        symbol = self._resolve_symbol(raw_symbol)
        if symbol is None:
            return
        if not isinstance(data, dict):
            return

        interval = self._CHANNEL_TO_INTERVAL.get(channel)
        if interval is None:
            return
        if interval not in self._symbol_intervals.get(symbol, ()):
            return

        timestamp_ms = self._extract_timestamp_ms(payload)
        candle = self._build_candle(symbol=symbol, interval=interval, timestamp_ms=timestamp_ms, data=data)
        await self._dispatch_candle(symbol, candle)

    async def _dispatch_candle(self, symbol: str, candle: CandleRecord) -> None:
        key = (symbol, candle.interval)
        previous_candle = self._latest_candles.get(key)
        self._latest_candles[key] = candle

        if self._on_candle_update is not None:
            update_result = self._on_candle_update(symbol, candle)
            if inspect.isawaitable(update_result):
                await update_result

        if previous_candle is None:
            return
        if previous_candle.open_time == candle.open_time:
            return

        callback_result = self._on_candle_closed(symbol, previous_candle)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def _run_reconnect_hook(self) -> None:
        if self._on_reconnect is None:
            return
        result = self._on_reconnect()
        if inspect.isawaitable(result):
            await result

    async def _emit_log(self, message: str) -> None:
        if self._on_log is None:
            return
        result = self._on_log(message)
        if inspect.isawaitable(result):
            await result

    def _build_symbol_intervals(
        self,
        symbol_intervals: dict[str, str] | None,
    ) -> dict[str, tuple[str, ...]]:
        if not symbol_intervals:
            return {symbol: tuple(self._intervals) for symbol in self._symbols}

        resolved: dict[str, tuple[str, ...]] = {}
        for symbol in self._symbols:
            interval = symbol_intervals.get(symbol)
            if interval is None:
                resolved[symbol] = tuple(self._intervals)
                continue
            if interval not in self._INTERVAL_TO_CHANNEL:
                raise ValueError(f"Unsupported websocket interval '{interval}' for symbol {symbol}.")
            resolved[symbol] = (interval,)
        return resolved

    def _resolve_symbol(self, symbol: str) -> str | None:
        normalized_symbol = symbol.strip()
        if not normalized_symbol:
            return None
        if normalized_symbol in self._symbol_intervals:
            return normalized_symbol
        upper_symbol = normalized_symbol.upper()
        resolved_symbol = self._symbol_lookup.get(upper_symbol)
        if resolved_symbol is not None:
            return resolved_symbol
        collapsed_symbol = upper_symbol.replace("-", "").replace("_", "").replace("/", "")
        return self._symbol_lookup.get(collapsed_symbol)

    @classmethod
    def _build_candle(
        cls,
        *,
        symbol: str,
        interval: str,
        timestamp_ms: int,
        data: dict[str, Any],
    ) -> CandleRecord:
        open_time = cls._bucket_open_time(timestamp_ms, interval)
        return CandleRecord(
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            open=float(cls._read_numeric(data, "o", "open")),
            high=float(cls._read_numeric(data, "h", "high")),
            low=float(cls._read_numeric(data, "l", "low")),
            close=float(cls._read_numeric(data, "c", "close")),
            volume=float(cls._read_numeric(data, "b", "baseVol", "volume")),
        )

    @classmethod
    def _bucket_open_time(cls, timestamp_ms: int, interval: str) -> datetime:
        candle_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        if interval == "1w":
            week_start = candle_time - timedelta(
                days=candle_time.weekday(),
                hours=candle_time.hour,
                minutes=candle_time.minute,
                seconds=candle_time.second,
                microseconds=candle_time.microsecond,
            )
            return week_start.replace(tzinfo=None)
        if interval == "1M":
            return candle_time.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=None,
            )

        interval_seconds = cls._INTERVAL_TO_SECONDS[interval]
        timestamp_seconds = timestamp_ms // 1000
        bucket_seconds = (timestamp_seconds // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(bucket_seconds, tz=UTC).replace(tzinfo=None)

    @staticmethod
    def _extract_timestamp_ms(payload: dict[str, Any]) -> int:
        for key in ("ts", "time", "timestamp"):
            value = payload.get(key)
            if value is not None:
                return int(value)
        raise ValueError("WebSocket message is missing a timestamp field.")

    @staticmethod
    def _read_numeric(data: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return float(value)
        raise ValueError(f"WebSocket candle is missing numeric field. Tried: {', '.join(keys)}")
