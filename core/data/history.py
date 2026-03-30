from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Callable, Iterator, Sequence

from config import settings
from core.api.bitunix import BitunixClient
from core.data.db import CandleRecord, Database

HistoryProgressCallback = Callable[[int, str], None]
HistoryLogCallback = Callable[[str], None]


class HistoryManager:
    _CHUNK_MAX_RETRIES = 12
    _CHUNK_RETRY_BASE_DELAY_SECONDS = 1.0
    _CHUNK_RETRY_MIN_DELAY_SECONDS = 10.0
    _CHUNK_RETRY_MAX_DELAY_SECONDS = 180.0
    _MONTH_COVERAGE_TOLERANCE = 1
    _PROGRESS_LOG_PREFIX = "[HISTORY_PROGRESS]"

    def __init__(self, db: Database, client: BitunixClient | None = None) -> None:
        self._db = db
        self._client = client or BitunixClient()
        self._owns_client = client is None

    def sync_recent_candles(
        self,
        symbols: Sequence[str],
        *,
        interval: str = settings.trading.interval,
        candles_per_symbol: int = 1000,
        on_progress: HistoryProgressCallback | None = None,
    ) -> dict[str, int]:
        if candles_per_symbol <= 0:
            raise ValueError("candles_per_symbol must be positive.")

        saved_counts: dict[str, int] = {}
        total_symbols = max(len(symbols), 1)
        for symbol_index, symbol in enumerate(symbols):
            saved_counts[symbol] = self._download_recent_candles(
                symbol=symbol,
                interval=interval,
                candles_per_symbol=candles_per_symbol,
                on_progress=(
                    None
                    if on_progress is None
                    else lambda current_count, target_count, stage, current_symbol=symbol, current_index=symbol_index: self._emit_download_progress(
                        on_progress=on_progress,
                        symbol=current_symbol,
                        interval=interval,
                        symbol_index=current_index,
                        total_symbols=total_symbols,
                        current_count=current_count,
                        target_count=target_count,
                        stage=stage,
                    )
                ),
            )
        return saved_counts

    def sync_candles_since(
        self,
        symbols: Sequence[str],
        *,
        interval: str = settings.trading.interval,
        start_time: datetime,
        end_time: datetime | None = None,
        on_progress: HistoryProgressCallback | None = None,
        on_log: HistoryLogCallback | None = None,
    ) -> dict[str, int]:
        normalized_start = self._normalize_utc_datetime(start_time)
        normalized_end = self._normalize_utc_datetime(
            datetime.now(tz=UTC) if end_time is None else end_time
        )
        if normalized_end <= normalized_start:
            raise ValueError("end_time must be greater than start_time.")

        saved_counts: dict[str, int] = {}
        total_symbols = max(len(symbols), 1)
        for symbol_index, symbol in enumerate(symbols):
            saved_counts[symbol] = self._download_candles_since(
                symbol=symbol,
                interval=interval,
                start_time=normalized_start,
                end_time=normalized_end,
                symbol_index=symbol_index,
                total_symbols=total_symbols,
                on_progress=on_progress,
                on_log=on_log,
            )
        return saved_counts

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HistoryManager:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _download_candles_since(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        symbol_index: int,
        total_symbols: int,
        on_progress: HistoryProgressCallback | None = None,
        on_log: HistoryLogCallback | None = None,
    ) -> int:
        initial_count, _, _ = self._db.get_candle_stats(symbol, interval)
        interval_seconds = self._interval_to_seconds(interval)
        month_windows = list(self._iter_month_windows(start_time=start_time, end_time=end_time))
        total_months = max(len(month_windows), 1)
        ok_months = 0
        partial_months = 0

        for month_index, (month_start, month_end) in enumerate(month_windows):
            month_label = month_start.strftime("%b %Y")
            expected_count = self._expected_candle_count(
                start_time=month_start,
                end_time=month_end,
                interval_seconds=interval_seconds,
            )
            coverage_target = (
                0
                if expected_count == 0
                else max(expected_count - self._MONTH_COVERAGE_TOLERANCE, 1)
            )
            existing_count, existing_oldest_time, _existing_newest_time = self._db.get_candle_range_stats(
                symbol,
                interval,
                start_time=month_start,
                end_time=month_end,
            )

            if expected_count == 0 or existing_count >= coverage_target:
                ok_months += 1
                self._emit_month_progress_log(
                    on_log=on_log,
                    symbol=symbol,
                    month_label=month_label,
                    month_index=month_index + 1,
                    total_months=total_months,
                    action="Checking",
                    status="OK",
                    ok_months=ok_months,
                    partial_months=partial_months,
                )
                if on_progress is not None:
                    self._emit_range_progress(
                        on_progress=on_progress,
                        symbol=symbol,
                        interval=interval,
                        symbol_index=symbol_index,
                        total_symbols=total_symbols,
                        completed_months=month_index + 1,
                        total_months=total_months,
                        status_text=f"{symbol}: {month_label} [OK]",
                    )
                continue

            self._download_month_in_chunks(
                symbol=symbol,
                interval=interval,
                month_start=month_start,
                month_end=month_end,
                month_oldest_cached_time=existing_oldest_time,
                on_log=on_log,
            )

            refreshed_count = self._db.count_candles_in_range(
                symbol,
                interval,
                start_time=month_start,
                end_time=month_end,
            )
            if expected_count == 0 or refreshed_count >= coverage_target:
                status = "OK"
                ok_months += 1
            else:
                status = f"PARTIAL {refreshed_count}/{expected_count}"
                partial_months += 1
            self._emit_month_progress_log(
                on_log=on_log,
                symbol=symbol,
                month_label=month_label,
                month_index=month_index + 1,
                total_months=total_months,
                action="Syncing",
                status=status,
                ok_months=ok_months,
                partial_months=partial_months,
            )
            if on_progress is not None:
                self._emit_range_progress(
                    on_progress=on_progress,
                    symbol=symbol,
                    interval=interval,
                    symbol_index=symbol_index,
                    total_symbols=total_symbols,
                    completed_months=month_index + 1,
                    total_months=total_months,
                    status_text=f"{symbol}: {month_label} [{status}]",
                )

        if on_log is not None:
            on_log(
                f"{symbol}: History sync complete. Months={total_months}, OK={ok_months}, PARTIAL={partial_months}."
            )
        final_count, _, _ = self._db.get_candle_stats(symbol, interval)
        return max(final_count - initial_count, 0)

    def _download_month_in_chunks(
        self,
        *,
        symbol: str,
        interval: str,
        month_start: datetime,
        month_end: datetime,
        month_oldest_cached_time: datetime | None = None,
        on_log: HistoryLogCallback | None = None,
    ) -> None:
        batch_limit = settings.api.request_limits.candles_per_request
        month_start_ms = self._datetime_to_milliseconds(month_start)
        end_time_ms = self._datetime_to_milliseconds(month_end) - 1
        current_oldest_cached_time = month_oldest_cached_time
        front_fill_mode = (
            current_oldest_cached_time is not None
            and month_start < current_oldest_cached_time < month_end
        )
        if front_fill_mode and current_oldest_cached_time is not None:
            oldest_cached_ms = self._datetime_to_milliseconds(current_oldest_cached_time)
            end_time_ms = min(end_time_ms, oldest_cached_ms - 1)

        while end_time_ms >= month_start_ms:
            batch = self._fetch_klines_chunk_with_retry(
                symbol=symbol,
                interval=interval,
                end_time_ms=end_time_ms,
                limit=batch_limit,
                on_log=on_log,
            )
            if not batch:
                break

            in_window = [
                candle
                for candle in batch
                if month_start <= candle.open_time < month_end
            ]
            if front_fill_mode and current_oldest_cached_time is not None:
                candidate_candles = [
                    candle for candle in in_window if candle.open_time < current_oldest_cached_time
                ]
            else:
                candidate_candles = in_window

            if candidate_candles:
                # Persist each chunk immediately so progress survives interruptions.
                self._db.upsert_candles(candidate_candles)
                oldest_inserted_time = min(candle.open_time for candle in candidate_candles)
                if (
                    current_oldest_cached_time is None
                    or oldest_inserted_time < current_oldest_cached_time
                ):
                    current_oldest_cached_time = oldest_inserted_time
            elif front_fill_mode and in_window:
                # API returned only candles already cached in the month tail.
                # Break early to avoid repeatedly re-downloading identical chunks.
                break

            oldest_open_time = min(candle.open_time for candle in batch)
            oldest_open_time_ms = self._datetime_to_milliseconds(oldest_open_time)
            if oldest_open_time_ms <= month_start_ms:
                break
            next_end_time_ms = oldest_open_time_ms - 1
            if front_fill_mode and current_oldest_cached_time is not None:
                oldest_cached_ms = self._datetime_to_milliseconds(current_oldest_cached_time)
                next_end_time_ms = min(next_end_time_ms, oldest_cached_ms - 1)
            if next_end_time_ms >= end_time_ms:
                break
            end_time_ms = next_end_time_ms
            if len(batch) < batch_limit:
                break

    def _fetch_klines_chunk_with_retry(
        self,
        *,
        symbol: str,
        interval: str,
        end_time_ms: int,
        limit: int,
        on_log: HistoryLogCallback | None = None,
    ) -> list[CandleRecord]:
        for attempt in range(self._CHUNK_MAX_RETRIES + 1):
            try:
                return self._client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    end_time=end_time_ms,
                    limit=limit,
                )
            except Exception as exc:
                if attempt >= self._CHUNK_MAX_RETRIES:
                    raise
                retry_delay = min(
                    self._CHUNK_RETRY_MAX_DELAY_SECONDS,
                    max(
                        self._CHUNK_RETRY_MIN_DELAY_SECONDS,
                        self._CHUNK_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                    ),
                )
                if on_log is not None:
                    on_log(
                        f"{symbol}: retrying failed chunk ({attempt + 1}/{self._CHUNK_MAX_RETRIES}) "
                        f"after error '{exc}'. Waiting {retry_delay:.1f}s."
                    )
                time.sleep(retry_delay)
        return []

    def _download_recent_candles(
        self,
        *,
        symbol: str,
        interval: str,
        candles_per_symbol: int,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> int:
        batch_limit = settings.api.request_limits.candles_per_request
        initial_count, oldest_time, newest_time = self._db.get_candle_stats(symbol, interval)
        current_count = initial_count
        front_sync_boundary = newest_time

        if on_progress is not None:
            on_progress(min(current_count, candles_per_symbol), candles_per_symbol, "Checking cache")

        end_time: int | None = self._current_timestamp_ms()
        while True:
            request_limit = (
                batch_limit
                if front_sync_boundary is not None
                else min(batch_limit, max(candles_per_symbol - current_count, 1))
            )
            batch = self._client.get_klines(
                symbol=symbol,
                interval=interval,
                end_time=end_time,
                limit=request_limit,
            )
            if not batch:
                break

            new_candles = self._filter_newer_candles(batch, front_sync_boundary)
            if new_candles:
                self._db.upsert_candles(new_candles)
                current_count += len(new_candles)
                oldest_time = self._min_open_time(oldest_time, new_candles[0].open_time)
                newest_time = self._max_open_time(newest_time, new_candles[-1].open_time)
                if on_progress is not None:
                    on_progress(
                        min(current_count, candles_per_symbol),
                        candles_per_symbol,
                        "Syncing recent history",
                    )

            if front_sync_boundary is not None and any(
                candle.open_time <= front_sync_boundary for candle in batch
            ):
                break
            oldest_open_time = min(candle.open_time for candle in batch)
            end_time = self._datetime_to_milliseconds(oldest_open_time) - 1

            if front_sync_boundary is None and current_count >= candles_per_symbol:
                break
            if len(batch) < request_limit:
                break

        current_count, oldest_time, newest_time = self._db.get_candle_stats(symbol, interval)
        if current_count < candles_per_symbol:
            end_time = (
                None
                if oldest_time is None
                else self._datetime_to_milliseconds(oldest_time) - 1
            )
            while current_count < candles_per_symbol:
                remaining = candles_per_symbol - current_count
                request_limit = min(batch_limit, remaining)
                batch = self._client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    end_time=end_time,
                    limit=request_limit,
                )
                if not batch:
                    break

                older_candles = self._filter_older_candles(batch, oldest_time)
                if not older_candles:
                    break

                self._db.upsert_candles(older_candles)
                current_count += len(older_candles)
                oldest_time = self._min_open_time(oldest_time, older_candles[0].open_time)
                newest_time = self._max_open_time(newest_time, older_candles[-1].open_time)
                if on_progress is not None:
                    on_progress(
                        min(current_count, candles_per_symbol),
                        candles_per_symbol,
                        "Backfilling older history",
                    )

                oldest_open_time = min(candle.open_time for candle in older_candles)
                end_time = self._datetime_to_milliseconds(oldest_open_time) - 1
                if len(batch) < request_limit:
                    break

        final_count, _, _ = self._db.get_candle_stats(symbol, interval)
        return max(final_count - initial_count, 0)

    @staticmethod
    def _emit_download_progress(
        *,
        on_progress: HistoryProgressCallback,
        symbol: str,
        interval: str,
        symbol_index: int,
        total_symbols: int,
        current_count: int,
        target_count: int,
        stage: str,
    ) -> None:
        if target_count <= 0:
            progress_value = 0
        else:
            symbol_ratio = min(current_count / target_count, 1.0)
            overall_ratio = (symbol_index + symbol_ratio) / total_symbols
            progress_value = int(overall_ratio * 100)
        on_progress(
            progress_value,
            f"{stage} {symbol} {interval}... ({min(current_count, target_count)}/{target_count})",
        )

    @staticmethod
    def _emit_range_progress(
        *,
        on_progress: HistoryProgressCallback,
        symbol: str,
        interval: str,
        symbol_index: int,
        total_symbols: int,
        completed_months: int,
        total_months: int,
        status_text: str,
    ) -> None:
        month_ratio = min(completed_months / max(total_months, 1), 1.0)
        overall_ratio = (symbol_index + month_ratio) / max(total_symbols, 1)
        progress_value = int(max(0.0, min(overall_ratio, 1.0)) * 100)
        on_progress(
            progress_value,
            f"{status_text} [{symbol} {interval}]",
        )

    @classmethod
    def _emit_month_progress_log(
        cls,
        *,
        on_log: HistoryLogCallback | None,
        symbol: str,
        month_label: str,
        month_index: int,
        total_months: int,
        action: str,
        status: str,
        ok_months: int,
        partial_months: int,
    ) -> None:
        if on_log is None:
            return
        on_log(
            f"{cls._PROGRESS_LOG_PREFIX} "
            f"{symbol}: {action} {month_label}... [{status}] "
            f"({month_index}/{max(total_months, 1)}) "
            f"OK={ok_months} PARTIAL={partial_months}"
        )

    @staticmethod
    def _iter_month_windows(
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> Iterator[tuple[datetime, datetime]]:
        cursor = datetime(start_time.year, start_time.month, 1)
        while cursor < end_time:
            next_cursor = HistoryManager._next_month_start(cursor)
            window_start = max(start_time, cursor)
            window_end = min(end_time, next_cursor)
            if window_end > window_start:
                yield window_start, window_end
            cursor = next_cursor

    @staticmethod
    def _next_month_start(value: datetime) -> datetime:
        if value.month == 12:
            return datetime(value.year + 1, 1, 1)
        return datetime(value.year, value.month + 1, 1)

    @staticmethod
    def _expected_candle_count(
        *,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
    ) -> int:
        if end_time <= start_time:
            return 0
        total_seconds = int((end_time - start_time).total_seconds())
        return max(total_seconds // max(interval_seconds, 1), 0)

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        normalized = interval.strip()
        if normalized.endswith("M"):
            return int(normalized[:-1]) * 2592000
        normalized = normalized.lower()
        if normalized.endswith("m"):
            return int(normalized[:-1]) * 60
        if normalized.endswith("h"):
            return int(normalized[:-1]) * 3600
        if normalized.endswith("d"):
            return int(normalized[:-1]) * 86400
        if normalized.endswith("w"):
            return int(normalized[:-1]) * 604800
        if normalized == "1m":
            return 60
        raise ValueError(f"Unsupported interval for history sync: {interval}")

    @staticmethod
    def _normalize_utc_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    @staticmethod
    def _filter_newer_candles(
        candles: Sequence[CandleRecord],
        newest_time: datetime | None,
    ) -> list[CandleRecord]:
        if newest_time is None:
            return list(candles)
        return [candle for candle in candles if candle.open_time > newest_time]

    @staticmethod
    def _filter_older_candles(
        candles: Sequence[CandleRecord],
        oldest_time: datetime | None,
    ) -> list[CandleRecord]:
        if oldest_time is None:
            return list(candles)
        return [candle for candle in candles if candle.open_time < oldest_time]

    @staticmethod
    def _min_open_time(current_value: datetime | None, new_value: datetime) -> datetime:
        if current_value is None:
            return new_value
        return min(current_value, new_value)

    @staticmethod
    def _max_open_time(current_value: datetime | None, new_value: datetime) -> datetime:
        if current_value is None:
            return new_value
        return max(current_value, new_value)

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(datetime.now(tz=UTC).timestamp() * 1000)

    @staticmethod
    def _datetime_to_milliseconds(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1000)
