from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime
from threading import Lock
from typing import Callable, Iterator, Sequence

from config import settings
from core.api.bitunix import BitunixClient
from core.data.db import CandleRecord, Database

HistoryProgressCallback = Callable[[int, str], None]
HistoryLogCallback = Callable[[str], None]
HistoryAbortCallback = Callable[[], bool]


class _AdaptiveRateState:
    def __init__(
        self,
        *,
        max_requests_per_second: int,
        min_requests_per_second: float = 2.0,
        startup_ratio: float = 0.7,
        increase_step: float = 0.25,
        success_window: int = 40,
        backoff_factor: float = 0.8,
    ) -> None:
        self._max_rps = float(max(1, int(max_requests_per_second)))
        self._min_rps = max(1.0, min(float(min_requests_per_second), self._max_rps))
        startup_rps = self._max_rps * max(0.2, min(float(startup_ratio), 1.0))
        self._current_rps = max(self._min_rps, min(self._max_rps, startup_rps))
        self._increase_step = max(0.05, float(increase_step))
        self._success_window = max(1, int(success_window))
        self._backoff_factor = max(0.1, min(float(backoff_factor), 0.95))
        self._success_streak = 0
        self._lock = Lock()

    def current_interval_seconds(self) -> float:
        with self._lock:
            return 1.0 / float(max(self._current_rps, 1.0))

    def current_rps(self) -> float:
        with self._lock:
            return float(self._current_rps)

    def report_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self._success_streak < self._success_window:
                return
            self._success_streak = 0
            self._current_rps = min(self._max_rps, self._current_rps + self._increase_step)

    def report_throttle(self) -> None:
        with self._lock:
            self._success_streak = 0
            self._current_rps = max(self._min_rps, self._current_rps * self._backoff_factor)


class _AdaptiveAsyncRequestRateLimiter:
    def __init__(self, state: _AdaptiveRateState) -> None:
        self._state = state
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = loop.time()
            interval = self._state.current_interval_seconds()
            # Small randomized jitter reduces synchronized request bursts.
            jitter = random.uniform(0.0, min(0.02, interval * 0.15))
            self._next_allowed = max(self._next_allowed, now) + interval + jitter


class HistoryManager:
    _CHUNK_MAX_RETRIES = 12
    _CHUNK_RETRY_BASE_DELAY_SECONDS = 1.0
    _CHUNK_RETRY_MIN_DELAY_SECONDS = 10.0
    _CHUNK_RETRY_MAX_DELAY_SECONDS = 180.0
    _THROTTLE_RETRY_BASE_DELAY_SECONDS = 1.0
    _THROTTLE_RETRY_MAX_DELAY_SECONDS = 12.0
    _MONTH_COVERAGE_TOLERANCE = 1
    _PROGRESS_LOG_PREFIX = "[HISTORY_PROGRESS]"
    _PARALLEL_FETCH_WORKERS = 16
    _RAM_BULK_INSERT_CHUNK_SIZE = 100_000
    # Bitunix currently serves klines in pages of up to ~200 rows per request.
    # Keeping this at 200 avoids premature short-page termination and range gaps.
    _HISTORY_KLINE_REQUEST_LIMIT = 200
    _PARALLEL_FETCH_MAX_CONCURRENCY = 5
    _PARALLEL_TASK_START_STAGGER_SECONDS = 0.10
    _ADAPTIVE_RATE_STARTUP_RATIO = 0.45
    _ADAPTIVE_RATE_MIN_RPS = 1.5
    _ADAPTIVE_RATE_INCREASE_STEP = 0.15
    _ADAPTIVE_RATE_SUCCESS_WINDOW = 60
    _ADAPTIVE_RATE_BACKOFF_FACTOR = 0.65

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
        should_abort: HistoryAbortCallback | None = None,
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
                should_abort=should_abort,
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
        should_abort: HistoryAbortCallback | None = None,
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
            if should_abort is not None and should_abort():
                if on_log is not None:
                    on_log(f"{symbol}: History sync canceled before start.")
                break
            saved_counts[symbol] = self._download_candles_since(
                symbol=symbol,
                interval=interval,
                start_time=normalized_start,
                end_time=normalized_end,
                symbol_index=symbol_index,
                total_symbols=total_symbols,
                on_progress=on_progress,
                on_log=on_log,
                should_abort=should_abort,
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
        should_abort: HistoryAbortCallback | None = None,
    ) -> int:
        initial_count, _, _ = self._db.get_candle_stats(symbol, interval)
        interval_seconds = self._interval_to_seconds(interval)
        month_windows = list(self._iter_month_windows(start_time=start_time, end_time=end_time))
        total_months = max(len(month_windows), 1)
        ok_months = 0
        partial_months = 0

        for month_index, (month_start, month_end) in enumerate(month_windows):
            if should_abort is not None and should_abort():
                if on_log is not None:
                    on_log(f"{symbol}: History sync canceled during month scan.")
                break
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
                interval_seconds=interval_seconds,
                month_start=month_start,
                month_end=month_end,
                month_oldest_cached_time=existing_oldest_time,
                symbol_index=symbol_index,
                total_symbols=total_symbols,
                completed_months_base=float(month_index),
                total_months=total_months,
                month_label=month_label,
                on_progress=on_progress,
                on_log=on_log,
                should_abort=should_abort,
            )

            if should_abort is not None and should_abort():
                if on_log is not None:
                    on_log(f"{symbol}: History sync canceled after month chunk sync.")
                break

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
        interval_seconds: int,
        month_start: datetime,
        month_end: datetime,
        month_oldest_cached_time: datetime | None = None,
        symbol_index: int,
        total_symbols: int,
        completed_months_base: float,
        total_months: int,
        month_label: str,
        on_progress: HistoryProgressCallback | None = None,
        on_log: HistoryLogCallback | None = None,
        should_abort: HistoryAbortCallback | None = None,
    ) -> None:
        if should_abort is not None and should_abort():
            if on_log is not None:
                on_log(f"{symbol}: History sync canceled before month chunk sync.")
            return
        batch_limit = self._HISTORY_KLINE_REQUEST_LIMIT
        month_start_ms = self._datetime_to_milliseconds(month_start)
        month_end_ms = self._datetime_to_milliseconds(month_end) - 1
        current_oldest_cached_time = month_oldest_cached_time
        front_fill_mode = (
            current_oldest_cached_time is not None
            and month_start < current_oldest_cached_time < month_end
        )
        if front_fill_mode and current_oldest_cached_time is not None:
            oldest_cached_ms = self._datetime_to_milliseconds(current_oldest_cached_time)
            month_end_ms = min(month_end_ms, oldest_cached_ms - 1)
        if month_end_ms < month_start_ms:
            return

        chunk_end_times = self._build_month_chunk_end_times(
            month_start_ms=month_start_ms,
            month_end_ms=month_end_ms,
            interval_seconds=interval_seconds,
            batch_limit=batch_limit,
        )
        if not chunk_end_times:
            return

        if on_log is not None:
            on_log(
                f"{symbol}: Parallel chunk sync active for {month_label} "
                f"(chunks={len(chunk_end_times)}, workers={self._PARALLEL_FETCH_WORKERS}, limit={batch_limit})."
            )

        month_candles = asyncio.run(
            self._download_month_in_chunks_parallel_async(
                symbol=symbol,
                interval=interval,
                month_start=month_start,
                month_end=month_end,
                front_fill_cutoff=current_oldest_cached_time if front_fill_mode else None,
                chunk_end_times=chunk_end_times,
                symbol_index=symbol_index,
                total_symbols=total_symbols,
                completed_months_base=completed_months_base,
                total_months=total_months,
                month_label=month_label,
                on_progress=on_progress,
                on_log=on_log,
                should_abort=should_abort,
            )
        )
        if not month_candles:
            return
        deduplicated_candles = self._deduplicate_candles_by_open_time(month_candles)
        self._bulk_upsert_candles(deduplicated_candles)

    @staticmethod
    def _build_month_chunk_end_times(
        *,
        month_start_ms: int,
        month_end_ms: int,
        interval_seconds: int,
        batch_limit: int,
    ) -> list[int]:
        if month_end_ms < month_start_ms:
            return []
        interval_ms = max(interval_seconds, 1) * 1000
        step_ms = max(batch_limit, 1) * interval_ms
        chunk_end_times: list[int] = []
        current_end_ms = month_end_ms
        while current_end_ms >= month_start_ms:
            chunk_end_times.append(current_end_ms)
            next_end_ms = current_end_ms - step_ms
            if next_end_ms >= current_end_ms:
                break
            current_end_ms = next_end_ms
        return chunk_end_times

    async def _download_month_in_chunks_parallel_async(
        self,
        *,
        symbol: str,
        interval: str,
        month_start: datetime,
        month_end: datetime,
        front_fill_cutoff: datetime | None,
        chunk_end_times: Sequence[int],
        symbol_index: int,
        total_symbols: int,
        completed_months_base: float,
        total_months: int,
        month_label: str,
        on_progress: HistoryProgressCallback | None,
        on_log: HistoryLogCallback | None,
        should_abort: HistoryAbortCallback | None,
    ) -> list[CandleRecord]:
        total_chunks = len(chunk_end_times)
        if total_chunks == 0:
            return []

        worker_count = max(1, self._PARALLEL_FETCH_WORKERS)
        adaptive_rate_state = _AdaptiveRateState(
            max_requests_per_second=settings.api.request_limits.public_requests_per_second,
            min_requests_per_second=self._ADAPTIVE_RATE_MIN_RPS,
            startup_ratio=self._ADAPTIVE_RATE_STARTUP_RATIO,
            increase_step=self._ADAPTIVE_RATE_INCREASE_STEP,
            success_window=self._ADAPTIVE_RATE_SUCCESS_WINDOW,
            backoff_factor=self._ADAPTIVE_RATE_BACKOFF_FACTOR,
        )
        rate_limiter = _AdaptiveAsyncRequestRateLimiter(adaptive_rate_state)
        request_semaphore = asyncio.Semaphore(max(1, self._PARALLEL_FETCH_MAX_CONCURRENCY))
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="history-fetch",
        )
        clients = [
            BitunixClient(
                max_retries=6,
                backoff_factor=1.0,
                enforce_public_rate_limit=False,
            )
            for _ in range(worker_count)
        ]
        client_locks = [Lock() for _ in range(worker_count)]
        tasks: list[asyncio.Task[list[CandleRecord]]] = []
        downloaded_candles: list[CandleRecord] = []
        completed_chunks = 0
        month_sync_started_monotonic = time.monotonic()
        heartbeat_task: asyncio.Task[None] | None = None

        if on_log is not None:
            on_log(
                f"{symbol}: adaptive history rate-limit active "
                f"(start_rps={adaptive_rate_state.current_rps():.2f}, "
                f"max_rps={settings.api.request_limits.public_requests_per_second}, "
                f"concurrency={self._PARALLEL_FETCH_MAX_CONCURRENCY}, "
                f"stagger={self._PARALLEL_TASK_START_STAGGER_SECONDS:.2f}s)."
            )

        async def fetch_chunk(chunk_index: int, end_time_ms: int) -> list[CandleRecord]:
            async with request_semaphore:
                client_index = chunk_index % worker_count
                await rate_limiter.acquire()

                def run_fetch() -> list[CandleRecord]:
                    with client_locks[client_index]:
                        return self._fetch_klines_chunk_with_retry(
                            client=clients[client_index],
                            symbol=symbol,
                            interval=interval,
                            end_time_ms=end_time_ms,
                            limit=self._HISTORY_KLINE_REQUEST_LIMIT,
                            adaptive_rate_state=adaptive_rate_state,
                            on_log=on_log,
                        )

                return await loop.run_in_executor(executor, run_fetch)

        async def emit_chunk_sync_heartbeat() -> None:
            if on_log is None:
                return
            while True:
                await asyncio.sleep(10.0)
                if should_abort is not None and should_abort():
                    on_log(f"{symbol}: {month_label} sync canceled.")
                    return
                if completed_chunks >= total_chunks:
                    return
                elapsed_seconds = max(0.0, time.monotonic() - month_sync_started_monotonic)
                completed_ratio = completed_chunks / max(total_chunks, 1)
                on_log(
                    f"{symbol}: {month_label} sync heartbeat "
                    f"{completed_chunks}/{total_chunks} "
                    f"({completed_ratio * 100.0:.1f}%), "
                    f"elapsed={elapsed_seconds:.0f}s, "
                    f"adaptive_rps={adaptive_rate_state.current_rps():.2f}."
                )

        try:
            if should_abort is not None and should_abort():
                if on_log is not None:
                    on_log(f"{symbol}: {month_label} sync canceled before dispatch.")
                return []
            heartbeat_task = asyncio.create_task(emit_chunk_sync_heartbeat())
            for chunk_index, end_time_ms in enumerate(chunk_end_times):
                if should_abort is not None and should_abort():
                    if on_log is not None:
                        on_log(f"{symbol}: {month_label} sync canceled during dispatch.")
                    break
                tasks.append(asyncio.create_task(fetch_chunk(chunk_index, end_time_ms)))
                if chunk_index < (total_chunks - 1):
                    await asyncio.sleep(max(0.0, self._PARALLEL_TASK_START_STAGGER_SECONDS))
            for completed_task in asyncio.as_completed(tasks):
                if should_abort is not None and should_abort():
                    if on_log is not None:
                        on_log(f"{symbol}: {month_label} sync canceled while collecting chunks.")
                    break
                batch = await completed_task
                completed_chunks += 1
                if on_progress is not None:
                    self._emit_range_progress(
                        on_progress=on_progress,
                        symbol=symbol,
                        interval=interval,
                        symbol_index=symbol_index,
                        total_symbols=total_symbols,
                        completed_months=completed_months_base
                        + (completed_chunks / max(total_chunks, 1)),
                        total_months=total_months,
                        status_text=f"{symbol}: {month_label} [SYNC {completed_chunks}/{total_chunks}]",
                    )
                if not batch:
                    continue
                in_window = [
                    candle
                    for candle in batch
                    if month_start <= candle.open_time < month_end
                ]
                if front_fill_cutoff is not None:
                    in_window = [
                        candle for candle in in_window if candle.open_time < front_fill_cutoff
                    ]
                if in_window:
                    downloaded_candles.extend(in_window)
        finally:
            if heartbeat_task is not None and not heartbeat_task.done():
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            for task in tasks:
                if not task.done():
                    task.cancel()
            for client in clients:
                client.close()
            executor.shutdown(wait=True, cancel_futures=True)

        return downloaded_candles

    def _bulk_upsert_candles(self, candles: Sequence[CandleRecord]) -> None:
        if not candles:
            return
        chunk_size = max(1, self._RAM_BULK_INSERT_CHUNK_SIZE)
        for start_index in range(0, len(candles), chunk_size):
            end_index = start_index + chunk_size
            self._db.upsert_candles(candles[start_index:end_index])

    @classmethod
    def _deduplicate_candles_by_open_time(cls, candles: Sequence[CandleRecord]) -> list[CandleRecord]:
        if not candles:
            return []

        worker_count = max(1, cls._PARALLEL_FETCH_WORKERS)
        if len(candles) < 10_000 or worker_count == 1:
            deduplicated_map: dict[datetime, CandleRecord] = {}
            for candle in candles:
                deduplicated_map[candle.open_time] = candle
            return [deduplicated_map[key] for key in sorted(deduplicated_map.keys())]

        chunk_size = max(1, len(candles) // worker_count)
        chunks: list[Sequence[CandleRecord]] = [
            candles[index : index + chunk_size]
            for index in range(0, len(candles), chunk_size)
        ]

        def build_chunk_map(chunk: Sequence[CandleRecord]) -> dict[datetime, CandleRecord]:
            chunk_map: dict[datetime, CandleRecord] = {}
            for candle in chunk:
                chunk_map[candle.open_time] = candle
            return chunk_map

        partial_maps: list[dict[datetime, CandleRecord]]
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="history-merge") as executor:
            partial_maps = list(executor.map(build_chunk_map, chunks))

        merged_map: dict[datetime, CandleRecord] = {}
        for partial_map in partial_maps:
            merged_map.update(partial_map)
        return [merged_map[key] for key in sorted(merged_map.keys())]

    def _fetch_klines_chunk_with_retry(
        self,
        *,
        client: BitunixClient,
        symbol: str,
        interval: str,
        end_time_ms: int,
        limit: int,
        adaptive_rate_state: _AdaptiveRateState | None = None,
        on_log: HistoryLogCallback | None = None,
    ) -> list[CandleRecord]:
        for attempt in range(self._CHUNK_MAX_RETRIES + 1):
            try:
                candles = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    end_time=end_time_ms,
                    limit=limit,
                )
                if adaptive_rate_state is not None:
                    adaptive_rate_state.report_success()
                return candles
            except Exception as exc:
                is_throttle_error = self._is_throttle_error(exc)
                if is_throttle_error and adaptive_rate_state is not None:
                    adaptive_rate_state.report_throttle()
                if attempt >= self._CHUNK_MAX_RETRIES:
                    raise
                if is_throttle_error:
                    retry_delay = min(
                        self._THROTTLE_RETRY_MAX_DELAY_SECONDS,
                        max(
                            self._THROTTLE_RETRY_BASE_DELAY_SECONDS,
                            self._THROTTLE_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                        ),
                    )
                else:
                    retry_delay = min(
                        self._CHUNK_RETRY_MAX_DELAY_SECONDS,
                        max(
                            self._CHUNK_RETRY_MIN_DELAY_SECONDS,
                            self._CHUNK_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                        ),
                    )
                if on_log is not None:
                    rate_suffix = ""
                    if adaptive_rate_state is not None:
                        rate_suffix = f" adaptive_rps={adaptive_rate_state.current_rps():.2f}"
                    on_log(
                        f"{symbol}: retrying failed chunk ({attempt + 1}/{self._CHUNK_MAX_RETRIES}) "
                        f"after error '{exc}'. Waiting {retry_delay:.1f}s.{rate_suffix}"
                    )
                time.sleep(retry_delay)
        return []

    @staticmethod
    def _is_throttle_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return ("10006" in message) or ("too frequently" in message)

    def _download_recent_candles(
        self,
        *,
        symbol: str,
        interval: str,
        candles_per_symbol: int,
        should_abort: HistoryAbortCallback | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> int:
        batch_limit = self._HISTORY_KLINE_REQUEST_LIMIT
        initial_count, oldest_time, newest_time = self._db.get_candle_stats(symbol, interval)
        current_count = initial_count
        front_sync_boundary = newest_time

        def _is_aborted() -> bool:
            return should_abort is not None and bool(should_abort())

        if on_progress is not None:
            on_progress(min(current_count, candles_per_symbol), candles_per_symbol, "Checking cache")

        end_time: int | None = self._current_timestamp_ms()
        while True:
            if _is_aborted():
                if on_progress is not None:
                    on_progress(
                        min(current_count, candles_per_symbol),
                        candles_per_symbol,
                        "Sync canceled",
                    )
                break
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

        current_count, oldest_time, newest_time = self._db.get_candle_stats(symbol, interval)
        if current_count < candles_per_symbol:
            end_time = (
                None
                if oldest_time is None
                else self._datetime_to_milliseconds(oldest_time) - 1
            )
            while current_count < candles_per_symbol:
                if _is_aborted():
                    if on_progress is not None:
                        on_progress(
                            min(current_count, candles_per_symbol),
                            candles_per_symbol,
                            "Backfill canceled",
                        )
                    break
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
        completed_months: float,
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
