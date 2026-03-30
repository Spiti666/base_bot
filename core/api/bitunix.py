from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import requests

from config import settings
from core.data.db import CandleRecord


class BitunixAPIError(RuntimeError):
    """Raised when the Bitunix API request cannot be completed successfully."""


class BitunixClient:
    KLINES_PATH = "/api/v1/futures/market/kline"
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        min_retry_delay_seconds: float = 10.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be greater than or equal to 0.")
        if min_retry_delay_seconds < 0:
            raise ValueError("min_retry_delay_seconds must be greater than or equal to 0.")

        self._settings = settings.api
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", "base-bot/0.1")
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._min_retry_delay_seconds = min_retry_delay_seconds
        self._min_public_request_interval = 1.0 / float(
            self._settings.request_limits.public_requests_per_second
        )
        self._last_public_request_monotonic = 0.0

    def get_klines(
        self,
        symbol: str,
        interval: str,
        end_time: int | datetime | None = None,
        limit: int | None = None,
    ) -> list[CandleRecord]:
        self._validate_interval(interval)

        resolved_limit = (
            self._settings.request_limits.candles_per_request if limit is None else limit
        )
        if resolved_limit <= 0:
            raise ValueError("limit must be positive.")
        if resolved_limit > self._settings.request_limits.candles_per_request:
            raise ValueError(
                "limit exceeds configured candles_per_request "
                f"({self._settings.request_limits.candles_per_request})."
            )

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": resolved_limit,
        }
        resolved_end_time = self._to_milliseconds(end_time)
        if resolved_end_time is not None:
            params["endTime"] = resolved_end_time

        payload = self._request("GET", self.KLINES_PATH, params=params)
        return self._parse_klines_payload(payload=payload, symbol=symbol, interval=interval)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> BitunixClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._settings.base_url.rstrip('/')}{path}"
        timeout = self._settings.request_limits.request_timeout_seconds
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                self._respect_public_rate_limit()
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=timeout,
                )
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(
                        f"Retryable HTTP status {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()

                payload = response.json()
                if not isinstance(payload, dict):
                    raise BitunixAPIError("Bitunix API returned a non-object JSON response.")

                code = payload.get("code")
                if code not in (0, "0", None):
                    raise BitunixAPIError(
                        f"Bitunix API error {code}: {payload.get('msg') or payload.get('message') or 'unknown error'}"
                    )
                return payload
            except (requests.RequestException, ValueError, BitunixAPIError) as exc:
                last_error = exc
                if not self._should_retry(exc, attempt):
                    break
                time.sleep(self._retry_delay_seconds(exc, attempt))

        raise BitunixAPIError(f"Request to Bitunix failed after retries: {last_error}") from last_error

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self._max_retries:
            return False
        if isinstance(error, BitunixAPIError):
            return False
        if isinstance(error, requests.HTTPError):
            response = error.response
            return response is not None and response.status_code in self.RETRYABLE_STATUS_CODES
        return isinstance(error, requests.RequestException)

    def _backoff_delay(self, attempt: int) -> float:
        return self._backoff_factor * (2**attempt)

    def _retry_delay_seconds(self, error: Exception, attempt: int) -> float:
        backoff_delay = self._backoff_delay(attempt)
        if isinstance(error, requests.HTTPError):
            response = error.response
            if response is not None and response.status_code in self.RETRYABLE_STATUS_CODES:
                retry_after_delay = self._retry_after_delay(response)
                return max(
                    self._min_retry_delay_seconds,
                    backoff_delay,
                    retry_after_delay,
                )
        return max(backoff_delay, 0.1)

    @staticmethod
    def _retry_after_delay(response: requests.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return 0.0
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            return 0.0

    def _respect_public_rate_limit(self) -> None:
        now_monotonic = time.monotonic()
        elapsed = now_monotonic - self._last_public_request_monotonic
        remaining_delay = self._min_public_request_interval - elapsed
        if remaining_delay > 0:
            time.sleep(remaining_delay)
        self._last_public_request_monotonic = time.monotonic()

    def _parse_klines_payload(
        self,
        *,
        payload: dict[str, Any],
        symbol: str,
        interval: str,
    ) -> list[CandleRecord]:
        raw_candles = payload.get("data", [])
        if not isinstance(raw_candles, list):
            raise BitunixAPIError("Bitunix kline payload contains a non-list 'data' field.")

        candles = [
            CandleRecord(
                symbol=symbol,
                interval=interval,
                open_time=self._milliseconds_to_datetime(self._extract_int(item, "time", "openTime", "ts")),
                open=self._extract_float(item, "open"),
                high=self._extract_float(item, "high"),
                low=self._extract_float(item, "low"),
                close=self._extract_float(item, "close"),
                volume=self._extract_float(item, "baseVol", "volume"),
            )
            for item in raw_candles
        ]
        candles.sort(key=lambda candle: candle.open_time)
        return candles

    @staticmethod
    def _extract_float(item: Any, *keys: str) -> float:
        value = BitunixClient._extract_value(item, *keys)
        return float(value)

    @staticmethod
    def _extract_int(item: Any, *keys: str) -> int:
        value = BitunixClient._extract_value(item, *keys)
        return int(value)

    @staticmethod
    def _extract_value(item: Any, *keys: str) -> Any:
        if not isinstance(item, dict):
            raise BitunixAPIError("Each kline row must be an object.")

        for key in keys:
            if key in item and item[key] is not None:
                return item[key]
        raise BitunixAPIError(f"Missing expected kline field. Tried keys: {', '.join(keys)}")

    @staticmethod
    def _milliseconds_to_datetime(timestamp_ms: int) -> datetime:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).replace(tzinfo=None)

    @staticmethod
    def _to_milliseconds(value: int | datetime | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1000)

    def _validate_interval(self, interval: str) -> None:
        if interval not in self._settings.timeframes:
            raise ValueError(
                f"Unsupported interval '{interval}'. Allowed values: {', '.join(self._settings.timeframes)}"
            )
