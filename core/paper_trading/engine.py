from __future__ import annotations

from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass
import math
import logging
from pathlib import Path
import traceback
from typing import Any, Callable
from datetime import UTC, datetime, timedelta

from config import settings
from core.data.db import Database, PaperTrade, PaperTradeCreate, PaperTradeUpdate
from core.paper_trading.persistence import load_paper_trades, save_paper_trades


@dataclass(frozen=True, slots=True)
class TradeRuntimeSettings:
    interval: str
    leverage: int
    take_profit_pct: float
    stop_loss_pct: float
    use_trailing_stop: bool
    trailing_activation_pct: float
    trailing_distance_pct: float
    breakeven_activation_pct: float
    breakeven_buffer_pct: float
    tight_trailing_activation_pct: float
    tight_trailing_distance_pct: float


class PaperTradingEngine:
    BREAKEVEN_ACTIVATION_PCT = 1.2
    BREAKEVEN_BUFFER_PCT = 0.05
    TIGHT_TRAILING_ACTIVATION_PCT = 0.0
    TIGHT_TRAILING_DISTANCE_PCT = 0.0
    BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE = 0.05
    BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE = BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE * 2.0
    INTRABAR_EXIT_STATUS_MAP: dict[str, str] = {
        "CLOSED_SL": "INTRABAR_SL",
        "CLOSED_TP": "INTRABAR_TP",
        "TRAILING_STOP": "INTRABAR_TRAILING",
        "TIGHT_TRAILING_STOP": "INTRABAR_TRAILING",
        "BREAKEVEN_STOP": "INTRABAR_BREAKEVEN",
    }

    def __init__(
        self,
        db: Database,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        leverage: int | None = None,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_activation_pct: float | None = None,
        trailing_distance_pct: float | None = None,
        breakeven_activation_pct: float | None = None,
        breakeven_buffer_pct: float | None = None,
        tight_trailing_activation_pct: float | None = None,
        tight_trailing_distance_pct: float | None = None,
        fee_pct_override: float | None = None,
        enable_persistence: bool = True,
        persistence_path: str | Path = "paper_trades.json",
        on_warning: Callable[[str, dict[str, Any]], None] | None = None,
        on_critical_state: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._db = db
        self._trading_settings = settings.trading
        self._symbol = symbol
        self._explicit_interval = interval
        self._explicit_leverage = leverage
        self._explicit_take_profit_pct = take_profit_pct
        self._explicit_stop_loss_pct = stop_loss_pct
        self._explicit_trailing_activation_pct = trailing_activation_pct
        self._explicit_trailing_distance_pct = trailing_distance_pct
        self._explicit_breakeven_activation_pct = breakeven_activation_pct
        self._explicit_breakeven_buffer_pct = breakeven_buffer_pct
        self._explicit_tight_trailing_activation_pct = tight_trailing_activation_pct
        self._explicit_tight_trailing_distance_pct = tight_trailing_distance_pct
        self._fee_pct_override = fee_pct_override
        self._enable_persistence = enable_persistence
        self._persistence_path = Path(persistence_path)
        self.active_trades: list[PaperTrade] = []
        self._restored_trade_count = 0
        self._dynamic_risk_overrides: dict[int, tuple[float | None, float | None]] = {}
        self._on_warning = on_warning
        self._on_critical_state = on_critical_state
        self._callback_failure_counts: Counter[str] = Counter()

        if self._fee_pct_override is not None and self._fee_pct_override < 0.0:
            raise ValueError("fee_pct_override must be greater than or equal to 0.")

        if self._enable_persistence:
            self.active_trades = load_paper_trades(self._persistence_path)
            self._restored_trade_count = self._restore_persisted_trades(self.active_trades)
            self._refresh_active_trade_storage()

        self._resolve_trade_settings(symbol)

    @property
    def restored_trade_count(self) -> int:
        return self._restored_trade_count

    @property
    def leverage(self) -> int:
        return self._resolve_trade_settings(self._symbol).leverage

    def get_runtime_settings(self, symbol: str | None = None) -> TradeRuntimeSettings:
        resolved_symbol = self._symbol if symbol is None else symbol
        return self._resolve_trade_settings(resolved_symbol)

    def get_effective_taker_fee_pct(self) -> float:
        if self._fee_pct_override is not None:
            return float(self._fee_pct_override)
        return float(self._trading_settings.taker_fee_pct)

    def _emit_warning(self, warning_code: str, payload: dict[str, Any]) -> None:
        warning_payload = dict(payload)
        warning_payload["warning_code"] = str(warning_code).strip()
        logging.getLogger("paper_engine").warning(
            "paper_engine warning code=%s payload=%s",
            warning_payload["warning_code"],
            warning_payload,
        )
        callback = self._on_warning
        if callback is None:
            return
        try:
            callback(warning_payload["warning_code"], warning_payload)
        except Exception as exc:
            self._callback_failure_counts["warning_callback"] += 1
            logging.getLogger("paper_engine").exception(
                "paper_engine warning callback failed for code=%s",
                warning_payload["warning_code"],
            )
            stacktrace_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()
            try:
                self._emit_critical_state(
                    "warning_callback_failed",
                    {
                        "reason": "warning_callback_failed",
                        "warning_code": warning_payload["warning_code"],
                        "warning_payload": warning_payload,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "exception_stacktrace": stacktrace_text,
                        "callback_failure_count": int(
                            self._callback_failure_counts["warning_callback"]
                        ),
                    },
                )
            except Exception:
                logging.getLogger("paper_engine").exception(
                    "paper_engine escalation failed after warning callback failure for code=%s",
                    warning_payload["warning_code"],
                )

    def _emit_critical_state(self, state_code: str, payload: dict[str, Any]) -> None:
        critical_payload = dict(payload)
        critical_payload["state_code"] = str(state_code).strip()
        logging.getLogger("paper_engine").error(
            "paper_engine critical state=%s payload=%s",
            critical_payload["state_code"],
            critical_payload,
        )
        callback = self._on_critical_state
        if callback is None:
            return
        try:
            callback(critical_payload["state_code"], critical_payload)
        except Exception as exc:
            self._callback_failure_counts["critical_callback"] += 1
            logging.getLogger("paper_engine").exception(
                "paper_engine critical callback failed for state=%s",
                critical_payload["state_code"],
            )
            raise RuntimeError(
                "critical_state_callback_failed:"
                f"{critical_payload['state_code']}:"
                f"{int(self._callback_failure_counts['critical_callback'])}"
            ) from exc

    def set_leverage(self, leverage: int | None) -> None:
        if leverage is None:
            self._explicit_leverage = None
            return
        if not self._trading_settings.min_leverage <= leverage <= self._trading_settings.max_leverage:
            raise ValueError("Configured leverage is outside the allowed range.")
        self._explicit_leverage = leverage

    def process_signal(
        self,
        symbol: str,
        current_price: float,
        signal_direction: int,
        *,
        strategy_name: str | None = None,
        leverage_scale: float = 1.0,
        risk_multiplier: float = 1.0,
        dynamic_stop_loss_pct: float | None = None,
        dynamic_take_profit_pct: float | None = None,
        timeframe: str | None = None,
        regime_label_at_entry: str | None = None,
        regime_confidence: float | None = None,
        session_label: str | None = None,
        signal_strength: float | None = None,
        confidence_score: float | None = None,
        atr_pct_at_entry: float | None = None,
        volume_ratio_at_entry: float | None = None,
        spread_estimate: float | None = None,
        move_already_extended_pct: float | None = None,
        entry_snapshot_json: str | dict[str, Any] | None = None,
        lifecycle_snapshot_json: str | dict[str, Any] | None = None,
        profile_version: str | None = None,
        review_status: str | None = None,
    ) -> int | None:
        self._validate_signal_direction(signal_direction)
        self._validate_price(current_price)
        if signal_direction == 0:
            return None
        if self._db.fetch_open_trades(symbol=symbol):
            return None
        if len(self._db.fetch_open_trades()) >= self._trading_settings.max_open_positions:
            return None
        if self._is_symbol_in_cooldown(symbol):
            return None

        current_equity = self._calculate_current_equity(
            current_symbol=symbol,
            current_price=current_price,
        )
        if current_equity <= 0:
            raise RuntimeError("Current equity must be positive before opening a trade.")
        free_balance = self._calculate_free_balance(current_equity=current_equity)
        if free_balance <= 0:
            return None

        trade_settings = self._resolve_trade_settings(symbol)
        resolved_risk_multiplier = (
            float(risk_multiplier)
            if math.isfinite(float(risk_multiplier)) and float(risk_multiplier) > 0.0
            else 1.0
        )
        resolved_leverage_scale = (
            float(leverage_scale)
            if math.isfinite(float(leverage_scale)) and float(leverage_scale) > 0.0
            else 1.0
        )
        margin_amount = (
            current_equity
            * (self._trading_settings.risk_per_trade_pct / 100.0)
            * resolved_risk_multiplier
        )
        if margin_amount <= 0:
            return None
        if margin_amount > free_balance:
            return None
        effective_leverage = max(
            1,
            int(round(float(trade_settings.leverage) * resolved_leverage_scale)),
        )
        position_size_usd = margin_amount * effective_leverage
        quantity = position_size_usd / current_price
        entry_fee = self._calculate_fee(position_size_usd)
        side = "LONG" if signal_direction > 0 else "SHORT"

        trade = PaperTradeCreate(
            symbol=symbol,
            side=side,
            entry_time=self._now(),
            entry_price=current_price,
            qty=quantity,
            leverage=effective_leverage,
            status="OPEN",
            total_fees=entry_fee,
            high_water_mark=current_price,
            strategy_name=strategy_name,
            timeframe=(
                str(timeframe).strip()
                if timeframe is not None and str(timeframe).strip()
                else str(trade_settings.interval)
            ),
            regime_label_at_entry=regime_label_at_entry,
            regime_confidence=regime_confidence,
            session_label=session_label,
            signal_strength=signal_strength,
            confidence_score=confidence_score,
            atr_pct_at_entry=atr_pct_at_entry,
            volume_ratio_at_entry=volume_ratio_at_entry,
            spread_estimate=spread_estimate,
            move_already_extended_pct=move_already_extended_pct,
            entry_snapshot_json=entry_snapshot_json,
            lifecycle_snapshot_json=lifecycle_snapshot_json,
            profile_version=profile_version,
            review_status=review_status,
        )
        try:
            trade_id = self._db.insert_trade(trade)
        except Exception as exc:
            stacktrace_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()
            self._emit_critical_state(
                "order_submit_failed",
                {
                    "symbol": str(symbol).strip().upper(),
                    "interval": str(trade.timeframe or trade_settings.interval),
                    "reason": "order_submit_failed",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                    "exception_stacktrace": stacktrace_text,
                    "state": "position_state_unknown",
                    "requires_reconcile": True,
                },
            )
            raise
        resolved_dynamic_stop_loss_pct = self._resolve_dynamic_live_pct(dynamic_stop_loss_pct)
        resolved_dynamic_take_profit_pct = self._resolve_dynamic_live_pct(dynamic_take_profit_pct)
        if (
            resolved_dynamic_stop_loss_pct is not None
            or resolved_dynamic_take_profit_pct is not None
        ):
            self._dynamic_risk_overrides[int(trade_id)] = (
                resolved_dynamic_stop_loss_pct,
                resolved_dynamic_take_profit_pct,
            )
        try:
            self._refresh_active_trade_storage()
        except Exception as exc:
            stacktrace_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()
            self._emit_critical_state(
                "reconcile_required",
                {
                    "symbol": str(symbol).strip().upper(),
                    "interval": str(trade.timeframe or trade_settings.interval),
                    "reason": "reconcile_required",
                    "message": (
                        "Trade persisted but runtime/persistence refresh failed after submit. "
                        "Manual reconciliation required."
                    ),
                    "trade_id": int(trade_id),
                    "exception_type": type(exc).__name__,
                    "exception_stacktrace": stacktrace_text,
                    "state": "position_state_unknown",
                    "requires_reconcile": True,
                },
            )
            raise
        return trade_id

    def update_positions(
        self,
        current_price: float,
        symbol: str | None = None,
        *,
        intrabar: bool = False,
    ) -> list[int]:
        self._validate_price(current_price)

        closed_trade_ids: list[int] = []
        exit_time = self._now()
        persistence_dirty = False
        for trade in self._db.fetch_open_trades(symbol=symbol):
            dynamic_stop_loss_pct = None
            dynamic_take_profit_pct = None
            if trade.id in self._dynamic_risk_overrides:
                dynamic_stop_loss_pct, dynamic_take_profit_pct = self._dynamic_risk_overrides[trade.id]
            new_high_water_mark = self._calculate_high_water_mark(trade, current_price)
            if new_high_water_mark != trade.high_water_mark:
                high_water_mark_updated = self._db.update_trade(
                    trade.id,
                    PaperTradeUpdate(high_water_mark=new_high_water_mark),
                )
                if not high_water_mark_updated:
                    message = (
                        "Failed to persist high_water_mark while trade is open; "
                        "position state may be inconsistent."
                    )
                    self._emit_critical_state(
                        "position_state_unknown",
                        {
                            "symbol": str(trade.symbol).strip().upper(),
                            "interval": str(self._resolve_trade_settings(trade.symbol).interval),
                            "reason": "position_state_unknown",
                            "message": message,
                            "trade_id": int(trade.id),
                            "state": "position_state_unknown",
                            "requires_reconcile": True,
                        },
                    )
                    raise RuntimeError(message)
                trade = self._replace_high_water_mark(trade, new_high_water_mark)
                persistence_dirty = True

            close_status, exit_price = self._resolve_live_exit(
                trade,
                current_price,
                dynamic_stop_loss_pct=dynamic_stop_loss_pct,
                dynamic_take_profit_pct=dynamic_take_profit_pct,
            )
            if close_status is None or exit_price is None:
                continue
            if intrabar:
                close_status = self._to_intrabar_exit_status(close_status)

            gross_pnl = self._calculate_gross_pnl(trade, exit_price)
            exit_fee = self._calculate_fee(trade.qty * exit_price)
            total_fees = trade.total_fees + exit_fee
            net_pnl = gross_pnl - total_fees
            was_updated = self._db.update_trade(
                trade.id,
                PaperTradeUpdate(
                    status=close_status,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    pnl=net_pnl,
                    total_fees=total_fees,
                ),
            )
            if was_updated:
                closed_trade_ids.append(trade.id)
                self._dynamic_risk_overrides.pop(trade.id, None)
                persistence_dirty = True
            else:
                message = (
                    f"Failed to persist exit for trade_id={int(trade.id)} "
                    f"symbol={trade.symbol} status={close_status}."
                )
                self._emit_critical_state(
                    "exit_persist_failed",
                    {
                        "symbol": str(trade.symbol).strip().upper(),
                        "interval": str(self._resolve_trade_settings(trade.symbol).interval),
                        "reason": "exit_persist_failed",
                        "message": message,
                        "trade_id": int(trade.id),
                        "exit_status": str(close_status),
                        "state": "position_state_unknown",
                        "requires_reconcile": True,
                    },
                )
                raise RuntimeError(message)
        if persistence_dirty:
            self._refresh_active_trade_storage()
        return closed_trade_ids

    def close_position_at_price(
        self,
        symbol: str,
        exit_price: float,
        *,
        status: str = "STRATEGY_EXIT",
    ) -> int | None:
        self._validate_price(exit_price)
        open_trades = self._db.fetch_open_trades(symbol=symbol)
        if not open_trades:
            return None

        trade = open_trades[0]
        exit_time = self._now()
        gross_pnl = self._calculate_gross_pnl(trade, exit_price)
        exit_fee = self._calculate_fee(trade.qty * exit_price)
        total_fees = trade.total_fees + exit_fee
        net_pnl = gross_pnl - total_fees
        was_updated = self._db.update_trade(
            trade.id,
            PaperTradeUpdate(
                status=status,
                exit_time=exit_time,
                exit_price=exit_price,
                pnl=net_pnl,
                total_fees=total_fees,
            ),
        )
        if was_updated:
            self._dynamic_risk_overrides.pop(trade.id, None)
            self._refresh_active_trade_storage()
            return trade.id
        message = (
            f"Failed to persist manual/strategy exit for trade_id={int(trade.id)} "
            f"symbol={trade.symbol} status={status}."
        )
        self._emit_critical_state(
            "exit_persist_failed",
            {
                "symbol": str(trade.symbol).strip().upper(),
                "interval": str(self._resolve_trade_settings(trade.symbol).interval),
                "reason": "exit_persist_failed",
                "message": message,
                "trade_id": int(trade.id),
                "exit_status": str(status),
                "state": "position_state_unknown",
                "requires_reconcile": True,
            },
        )
        raise RuntimeError(message)

    def run_historical_backtest(
        self,
        candles_df: Any | None,
        signals: list[int],
        strategy_exit_rule: Callable[[PaperTrade, dict[str, Any]], str | None] | None = None,
        candle_rows: list[dict[str, Any]] | None = None,
        dynamic_stop_loss_pcts: list[float] | None = None,
        dynamic_take_profit_pcts: list[float] | None = None,
        enable_chandelier_exit: bool = False,
        chandelier_period: int | None = None,
        chandelier_multiplier: float | None = None,
        strategy_exit_pre_take_profit: bool = False,
        max_bars_in_trade: int | None = None,
        early_stop_max_trades: int | None = None,
        early_stop_max_drawdown_pct: float | None = None,
        strategy_name: str | None = None,
    ) -> dict[str, Any]:
        if candle_rows is None:
            if candles_df is None:
                raise ValueError("candles_df must be provided when candle_rows is None.")
            candle_rows = self._extract_backtest_rows(candles_df)
        else:
            candle_rows = list(candle_rows)
        if len(candle_rows) != len(signals):
            raise ValueError("signals length must match the number of candles in candles_df.")
        if dynamic_stop_loss_pcts is not None and len(dynamic_stop_loss_pcts) != len(signals):
            raise ValueError("dynamic_stop_loss_pcts length must match signals length.")
        if dynamic_take_profit_pcts is not None and len(dynamic_take_profit_pcts) != len(signals):
            raise ValueError("dynamic_take_profit_pcts length must match signals length.")
        if not candle_rows:
            return {
                "total_pnl_usd": 0.0,
                "win_rate_pct": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "closed_trades": [],
                "real_leverage_avg": 0.0,
                "total_slippage_penalty_usd": 0.0,
                "slippage_penalty_pct_per_side": float(self.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
                "slippage_penalty_pct_per_trade": float(self.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE),
                "dynamic_stop_loss_overrides_applied": 0,
                "dynamic_take_profit_overrides_applied": 0,
                "time_stop_exits": 0,
            }
        resolved_max_bars_in_trade = None
        if max_bars_in_trade is not None:
            with suppress(Exception):
                candidate_max_bars = int(max_bars_in_trade)
                if 0 < candidate_max_bars < 10_000:
                    resolved_max_bars_in_trade = candidate_max_bars
        resolved_early_stop_max_trades = None
        if early_stop_max_trades is not None:
            with suppress(Exception):
                candidate_early_stop_trades = int(early_stop_max_trades)
                if candidate_early_stop_trades > 0:
                    resolved_early_stop_max_trades = candidate_early_stop_trades
        resolved_early_stop_max_drawdown_pct = None
        if early_stop_max_drawdown_pct is not None:
            with suppress(Exception):
                candidate_early_stop_drawdown_pct = float(early_stop_max_drawdown_pct)
                if math.isfinite(candidate_early_stop_drawdown_pct) and candidate_early_stop_drawdown_pct > 0.0:
                    resolved_early_stop_max_drawdown_pct = candidate_early_stop_drawdown_pct
        resolved_chandelier_period = (
            int(chandelier_period)
            if chandelier_period is not None
            else int(getattr(self._trading_settings, "chandelier_period", 22))
        )
        resolved_chandelier_multiplier = (
            float(chandelier_multiplier)
            if chandelier_multiplier is not None
            else float(getattr(self._trading_settings, "chandelier_multiplier", 3.0))
        )
        chandelier_active = bool(
            enable_chandelier_exit
            and resolved_chandelier_period > 1
            and math.isfinite(resolved_chandelier_multiplier)
            and resolved_chandelier_multiplier > 0.0
        )
        if chandelier_active:
            chandelier_long_levels, chandelier_short_levels = self._calculate_backtest_chandelier_levels(
                candle_rows,
                period=resolved_chandelier_period,
                multiplier=resolved_chandelier_multiplier,
            )
        else:
            chandelier_long_levels = [None] * len(candle_rows)
            chandelier_short_levels = [None] * len(candle_rows)
        start_capital = float(self._trading_settings.start_capital)
        current_capital = float(start_capital)
        running_peak_capital = float(start_capital)
        running_max_drawdown_pct = 0.0
        closed_trades: list[dict[str, Any]] = []
        open_trade: PaperTrade | None = None
        open_trade_entry_candle_index: int | None = None
        dynamic_risk_by_trade_id: dict[int, tuple[float | None, float | None]] = {}
        dynamic_stop_loss_overrides_applied = 0
        dynamic_take_profit_overrides_applied = 0
        time_stop_exits = 0
        next_trade_id = 1
        opened_trades_count = 0
        opened_trade_leverage_sum = 0.0
        # Determine first approved signal index so warmup candles (e.g., Setup Gate) are excluded
        first_signal_index: int | None = None
        try:
            first_signal_index = next(
                (i for i, s in enumerate(signals) if int(s) != 0),
                None,
            )
        except Exception:
            first_signal_index = None

        for candle_index, (candle_row, signal_direction) in enumerate(zip(candle_rows, signals)):
            raw_signal_direction = int(signal_direction)
            self._validate_signal_direction(raw_signal_direction)
            normalized_signal_direction = self._normalize_signal_direction(raw_signal_direction)

            if open_trade is not None:
                trade_dynamic_stop_loss_pct = None
                trade_dynamic_take_profit_pct = None
                if open_trade.id in dynamic_risk_by_trade_id:
                    trade_dynamic_stop_loss_pct, trade_dynamic_take_profit_pct = dynamic_risk_by_trade_id[
                        open_trade.id
                    ]
                open_trade = self._apply_backtest_high_water_mark(open_trade, candle_row)
                exit_status, exit_price = self._resolve_backtest_stop_only(
                    open_trade,
                    candle_row,
                    dynamic_stop_loss_pct=trade_dynamic_stop_loss_pct,
                )
                if (
                    exit_status is None
                    and exit_price is None
                    and chandelier_active
                ):
                    exit_status, exit_price = self._resolve_backtest_chandelier_exit(
                        open_trade,
                        candle_row,
                        chandelier_long=chandelier_long_levels[candle_index],
                        chandelier_short=chandelier_short_levels[candle_index],
                    )
                if (
                    exit_status is None
                    and exit_price is None
                    and strategy_exit_rule is not None
                    and strategy_exit_pre_take_profit
                ):
                    strategy_exit_status = strategy_exit_rule(open_trade, candle_row)
                    if strategy_exit_status is not None:
                        exit_status = strategy_exit_status
                        exit_price = float(candle_row["close"])
                if (
                    exit_status is None
                    and exit_price is None
                ):
                    exit_status, exit_price = self._resolve_backtest_take_profit_only(
                        open_trade,
                        candle_row,
                        dynamic_take_profit_pct=trade_dynamic_take_profit_pct,
                    )
                if (
                    exit_status is None
                    and exit_price is None
                    and strategy_exit_rule is not None
                    and not strategy_exit_pre_take_profit
                ):
                    strategy_exit_status = strategy_exit_rule(open_trade, candle_row)
                    if strategy_exit_status is not None:
                        exit_status = strategy_exit_status
                        exit_price = float(candle_row["close"])
                if (
                    exit_status is None
                    and exit_price is None
                    and resolved_max_bars_in_trade is not None
                    and open_trade_entry_candle_index is not None
                ):
                    bars_elapsed = candle_index - open_trade_entry_candle_index
                    if bars_elapsed >= resolved_max_bars_in_trade:
                        exit_status = "CLOSED_TIME_STOP"
                        exit_price = float(candle_row["close"])
                        time_stop_exits += 1
                if exit_status is not None and exit_price is not None:
                    closed_trade = self._close_backtest_trade(
                        open_trade,
                        exit_price=exit_price,
                        exit_time=self._resolve_backtest_time(candle_row),
                        status=exit_status,
                    )
                    current_capital += float(closed_trade["pnl"])
                    closed_trades.append(closed_trade)
                    dynamic_risk_by_trade_id.pop(open_trade.id, None)
                    open_trade = None
                    open_trade_entry_candle_index = None
                    if current_capital > running_peak_capital:
                        running_peak_capital = float(current_capital)
                    elif running_peak_capital > 0.0:
                        running_max_drawdown_pct = max(
                            running_max_drawdown_pct,
                            ((running_peak_capital - current_capital) / running_peak_capital) * 100.0,
                        )
                    if (
                        resolved_early_stop_max_trades is not None
                        and resolved_early_stop_max_drawdown_pct is not None
                        and len(closed_trades) >= resolved_early_stop_max_trades
                        and start_capital > 0.0
                    ):
                        capital_drop_from_start_pct = max(
                            0.0,
                            ((start_capital - current_capital) / start_capital) * 100.0,
                        )
                        if (
                            running_max_drawdown_pct > resolved_early_stop_max_drawdown_pct
                            or capital_drop_from_start_pct > resolved_early_stop_max_drawdown_pct
                        ):
                            break

            if open_trade is not None:
                continue
            if current_capital <= 0:
                break

            if raw_signal_direction == 0:
                continue

            normalized_entry_direction = normalized_signal_direction
            leverage_scale = 1.0

            dynamic_stop_loss_pct = self._resolve_dynamic_backtest_pct(
                dynamic_stop_loss_pcts,
                candle_index,
            )
            dynamic_take_profit_pct = self._resolve_dynamic_backtest_pct(
                dynamic_take_profit_pcts,
                candle_index,
            )

            entry_price = float(candle_row["close"])
            opened_trade = self._open_backtest_trade(
                trade_id=next_trade_id,
                candle_row=candle_row,
                signal_direction=int(normalized_entry_direction),
                entry_price=entry_price,
                current_capital=current_capital,
                strategy_name=strategy_name,
                leverage_scale=leverage_scale,
            )
            if opened_trade is None:
                continue
            open_trade = opened_trade
            opened_trades_count += 1
            opened_trade_leverage_sum += float(open_trade.leverage)
            if dynamic_stop_loss_pct is not None:
                dynamic_stop_loss_overrides_applied += 1
            if dynamic_take_profit_pct is not None:
                dynamic_take_profit_overrides_applied += 1
            if dynamic_stop_loss_pct is not None or dynamic_take_profit_pct is not None:
                dynamic_risk_by_trade_id[open_trade.id] = (
                    dynamic_stop_loss_pct,
                    dynamic_take_profit_pct,
                )
            open_trade_entry_candle_index = candle_index
            next_trade_id += 1

        if open_trade is not None:
            last_candle = candle_rows[-1]
            closed_trade = self._close_backtest_trade(
                open_trade,
                exit_price=float(last_candle["close"]),
                exit_time=self._resolve_backtest_time(last_candle),
                status="BACKTEST_EOD",
            )
            current_capital += float(closed_trade["pnl"])
            closed_trades.append(closed_trade)
            dynamic_risk_by_trade_id.pop(open_trade.id, None)
            open_trade_entry_candle_index = None

        total_trades = len(closed_trades)
        win_count = sum(1 for trade in closed_trades if float(trade["pnl"]) > 0)
        gross_profit = sum(float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) > 0)
        gross_loss = sum(-float(trade["pnl"]) for trade in closed_trades if float(trade["pnl"]) < 0)
        win_rate_pct = (win_count / total_trades) * 100.0 if total_trades else 0.0
        if gross_loss == 0.0:
            profit_factor = float("inf") if gross_profit > 0.0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        # Compute efficient running max-drawdown and longest consecutive losses
        # only considering trades that closed after the first approved signal
        # (excludes warmup). Use running variables to avoid replaying full
        # equity curves for performance.
        if first_signal_index is None:
            max_drawdown_pct = 0.0
            longest_consecutive_losses = 0
        else:
            start_time = self._resolve_backtest_time(candle_rows[first_signal_index])
            equity = float(self._trading_settings.start_capital)
            peak_equity = equity
            max_drawdown_pct = 0.0
            current_streak = 0
            longest_consecutive_losses = 0
            for tr in closed_trades:
                exit_time_val = tr.get("exit_time")
                try:
                    if isinstance(exit_time_val, str):
                        exit_dt = datetime.fromisoformat(exit_time_val)
                    else:
                        exit_dt = exit_time_val
                except Exception:
                    # If exit time can't be parsed, include the trade conservatively
                    exit_dt = None
                if exit_dt is None or exit_dt < start_time:
                    continue
                pnl = float(tr.get("pnl", 0.0))
                equity += pnl
                if equity > peak_equity:
                    peak_equity = equity
                else:
                    if peak_equity and peak_equity > 0:
                        dd = (peak_equity - equity) / peak_equity * 100.0
                        if dd > max_drawdown_pct:
                            max_drawdown_pct = dd
                if pnl < 0:
                    current_streak += 1
                    if current_streak > longest_consecutive_losses:
                        longest_consecutive_losses = current_streak
                else:
                    current_streak = 0

        breakeven_level_hits = sum(
            1
            for trade in closed_trades
            if str(trade.get("status")) in {"BREAKEVEN_STOP", "INTRABAR_BREAKEVEN"}
        )
        normal_trailing_level_hits = sum(
            1
            for trade in closed_trades
            if str(trade.get("status")) in {"TRAILING_STOP", "INTRABAR_TRAILING"}
        )
        tight_trailing_level_hits = sum(
            1 for trade in closed_trades if str(trade.get("status")) == "TIGHT_TRAILING_STOP"
        )
        total_slippage_penalty_usd = sum(
            float(trade.get("slippage_penalty_usd", 0.0) or 0.0)
            for trade in closed_trades
        )
        real_leverage_avg = (
            float(opened_trade_leverage_sum) / float(opened_trades_count)
            if opened_trades_count > 0
            else 0.0
        )

        return {
            "total_pnl_usd": current_capital - self._trading_settings.start_capital,
            "win_rate_pct": win_rate_pct,
            "total_trades": total_trades,
            "profit_factor": profit_factor,
            "closed_trades": closed_trades,
            "real_leverage_avg": float(real_leverage_avg),
            "max_drawdown_pct": max_drawdown_pct,
            "longest_consecutive_losses": longest_consecutive_losses,
            "breakeven_level_hits": int(breakeven_level_hits),
            "normal_trailing_level_hits": int(normal_trailing_level_hits),
            "tight_trailing_level_hits": int(tight_trailing_level_hits),
            "total_slippage_penalty_usd": float(total_slippage_penalty_usd),
            "slippage_penalty_pct_per_side": float(self.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE),
            "slippage_penalty_pct_per_trade": float(self.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_TRADE),
            "dynamic_stop_loss_overrides_applied": int(dynamic_stop_loss_overrides_applied),
            "dynamic_take_profit_overrides_applied": int(dynamic_take_profit_overrides_applied),
            "time_stop_exits": int(time_stop_exits),
        }

    def _calculate_current_equity(self, *, current_symbol: str, current_price: float) -> float:
        realized_pnl = self._db.fetch_realized_pnl()
        unrealized_pnl = 0.0
        for trade in self._db.fetch_open_trades():
            mark_price = self._resolve_mark_price_for_trade(
                trade,
                current_symbol=current_symbol,
                current_price=current_price,
            )
            # total_fees already contains the paid entry fee for open trades.
            unrealized_pnl += self._calculate_gross_pnl(trade, mark_price) - float(trade.total_fees)
        return self._trading_settings.start_capital + realized_pnl + unrealized_pnl

    def _calculate_used_margin(self) -> float:
        used_margin = 0.0
        for trade in self._db.fetch_open_trades():
            if trade.leverage <= 0:
                continue
            used_margin += (float(trade.qty) * float(trade.entry_price)) / float(trade.leverage)
        return used_margin

    def _calculate_free_balance(self, *, current_equity: float) -> float:
        return max(0.0, float(current_equity) - self._calculate_used_margin())

    def _resolve_mark_price_for_trade(
        self,
        trade: PaperTrade,
        *,
        current_symbol: str,
        current_price: float,
    ) -> float:
        if trade.symbol == current_symbol:
            return float(current_price)
        resolved_interval = str(self._resolve_trade_settings(trade.symbol).interval)
        try:
            candles = self._db.fetch_recent_candles(trade.symbol, resolved_interval, limit=1)
            if candles:
                return float(candles[-1].close)
            raise RuntimeError("missing_recent_mark_price")
        except Exception as exc:
            stacktrace_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()
            self._emit_critical_state(
                "position_state_unknown",
                {
                    "symbol": str(trade.symbol).strip().upper(),
                    "interval": resolved_interval,
                    "reason": "position_state_unknown",
                    "message": (
                        "Unable to resolve mark price for open trade during equity/risk calculation."
                    ),
                    "trade_id": int(trade.id),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "exception_stacktrace": stacktrace_text,
                    "state": "position_state_unknown",
                    "requires_reconcile": True,
                },
            )
            raise RuntimeError(
                f"position_state_unknown:mark_price_unavailable symbol={trade.symbol}"
            ) from exc

    def _is_symbol_in_cooldown(self, symbol: str) -> bool:
        cooldown_interval = self._interval_to_timedelta(self._resolve_trade_settings(symbol).interval)
        if cooldown_interval <= timedelta(0):
            return False

        last_closed_trade = self._db.fetch_last_closed_trade(symbol)
        if last_closed_trade is None or last_closed_trade.exit_time is None:
            return False

        return (self._now() - last_closed_trade.exit_time) < cooldown_interval

    def _resolve_live_exit(
        self,
        trade: PaperTrade,
        current_price: float,
        *,
        dynamic_stop_loss_pct: float | None = None,
        dynamic_take_profit_pct: float | None = None,
    ) -> tuple[str | None, float | None]:
        trade_settings = self._resolve_trade_settings(trade.symbol)
        stop_source, stop_price = self._get_protective_stop(
            trade,
            stop_loss_pct_override=dynamic_stop_loss_pct,
        )
        resolved_take_profit_pct = (
            float(dynamic_take_profit_pct)
            if (
                dynamic_take_profit_pct is not None
                and math.isfinite(float(dynamic_take_profit_pct))
                and float(dynamic_take_profit_pct) > 0.0
            )
            else float(trade_settings.take_profit_pct)
        )
        take_profit_ratio = resolved_take_profit_pct / 100.0

        if trade.side == "LONG":
            if stop_price is not None and current_price <= stop_price:
                return self._stop_source_to_status(stop_source), stop_price
            take_profit_price = trade.entry_price * (1.0 + take_profit_ratio)
            if current_price >= take_profit_price:
                return "CLOSED_TP", take_profit_price
            return None, None

        if trade.side == "SHORT":
            if stop_price is not None and current_price >= stop_price:
                return self._stop_source_to_status(stop_source), stop_price
            take_profit_price = trade.entry_price * (1.0 - take_profit_ratio)
            if current_price <= take_profit_price:
                return "CLOSED_TP", take_profit_price
            return None, None

        raise ValueError(f"Unsupported trade side '{trade.side}'.")

    @staticmethod
    def _calculate_gross_pnl(trade: PaperTrade, exit_price: float) -> float:
        if trade.side == "LONG":
            return (exit_price - trade.entry_price) * trade.qty
        if trade.side == "SHORT":
            return (trade.entry_price - exit_price) * trade.qty
        raise ValueError(f"Unsupported trade side '{trade.side}'.")

    def _calculate_fee(self, notional_usd: float) -> float:
        return notional_usd * (self.get_effective_taker_fee_pct() / 100.0)

    @classmethod
    def _calculate_backtest_slippage_penalty(
        cls,
        *,
        entry_notional_usd: float,
        exit_notional_usd: float,
    ) -> float:
        slippage_ratio_per_side = float(cls.BACKTEST_SLIPPAGE_PENALTY_PCT_PER_SIDE) / 100.0
        return (
            max(0.0, float(entry_notional_usd)) * slippage_ratio_per_side
            + max(0.0, float(exit_notional_usd)) * slippage_ratio_per_side
        )

    def _apply_backtest_high_water_mark(
        self,
        trade: PaperTrade,
        candle_row: dict[str, Any],
    ) -> PaperTrade:
        reference_price = float(candle_row["high"]) if trade.side == "LONG" else float(candle_row["low"])
        new_high_water_mark = self._calculate_high_water_mark(trade, reference_price)
        if new_high_water_mark == trade.high_water_mark:
            return trade
        return self._replace_high_water_mark(trade, new_high_water_mark)

    def _resolve_backtest_exit(
        self,
        trade: PaperTrade,
        candle_row: dict[str, Any],
        *,
        dynamic_stop_loss_pct: float | None = None,
        dynamic_take_profit_pct: float | None = None,
        include_take_profit: bool = True,
    ) -> tuple[str | None, float | None]:
        stop_status, stop_price = self._resolve_backtest_stop_only(
            trade,
            candle_row,
            dynamic_stop_loss_pct=dynamic_stop_loss_pct,
        )
        if stop_status is not None and stop_price is not None:
            return stop_status, stop_price
        if not include_take_profit:
            return None, None
        return self._resolve_backtest_take_profit_only(
            trade,
            candle_row,
            dynamic_take_profit_pct=dynamic_take_profit_pct,
        )

    def _resolve_backtest_stop_only(
        self,
        trade: PaperTrade,
        candle_row: dict[str, Any],
        *,
        dynamic_stop_loss_pct: float | None = None,
    ) -> tuple[str | None, float | None]:
        candle_high = float(candle_row["high"])
        candle_low = float(candle_row["low"])
        stop_source, stop_price = self._get_protective_stop(
            trade,
            stop_loss_pct_override=dynamic_stop_loss_pct,
        )
        if trade.side == "LONG":
            if stop_price is not None and candle_low <= stop_price:
                return self._stop_source_to_status(stop_source), stop_price
            return None, None

        if stop_price is not None and candle_high >= stop_price:
            return self._stop_source_to_status(stop_source), stop_price
        return None, None

    def _resolve_backtest_take_profit_only(
        self,
        trade: PaperTrade,
        candle_row: dict[str, Any],
        *,
        dynamic_take_profit_pct: float | None = None,
    ) -> tuple[str | None, float | None]:
        trade_settings = self._resolve_trade_settings(trade.symbol)
        resolved_take_profit_pct = (
            float(dynamic_take_profit_pct)
            if (
                dynamic_take_profit_pct is not None
                and math.isfinite(float(dynamic_take_profit_pct))
                and float(dynamic_take_profit_pct) > 0.0
            )
            else float(trade_settings.take_profit_pct)
        )
        take_profit_ratio = resolved_take_profit_pct / 100.0
        candle_high = float(candle_row["high"])
        candle_low = float(candle_row["low"])

        if trade.side == "LONG":
            take_profit_price = trade.entry_price * (1.0 + take_profit_ratio)
            if candle_high >= take_profit_price:
                return "CLOSED_TP", take_profit_price
            return None, None

        take_profit_price = trade.entry_price * (1.0 - take_profit_ratio)
        if candle_low <= take_profit_price:
            return "CLOSED_TP", take_profit_price
        return None, None

    def _resolve_backtest_chandelier_exit(
        self,
        trade: PaperTrade,
        candle_row: dict[str, Any],
        *,
        chandelier_long: float | None,
        chandelier_short: float | None,
    ) -> tuple[str | None, float | None]:
        close_price = float(candle_row["close"])
        if not math.isfinite(close_price):
            return None, None
        if trade.side == "LONG":
            if chandelier_long is not None and math.isfinite(float(chandelier_long)):
                if close_price < float(chandelier_long):
                    return "CLOSED_CHANDELIER", close_price
            return None, None
        if chandelier_short is not None and math.isfinite(float(chandelier_short)):
            if close_price > float(chandelier_short):
                return "CLOSED_CHANDELIER", close_price
        return None, None

    @staticmethod
    def _calculate_backtest_chandelier_levels(
        candle_rows: list[dict[str, Any]],
        *,
        period: int,
        multiplier: float,
    ) -> tuple[list[float | None], list[float | None]]:
        row_count = len(candle_rows)
        long_levels: list[float | None] = [None] * row_count
        short_levels: list[float | None] = [None] * row_count
        if row_count == 0:
            return long_levels, short_levels
        if period <= 1 or not math.isfinite(multiplier) or multiplier <= 0.0:
            return long_levels, short_levels

        tr_window: deque[float] = deque()
        high_window: deque[tuple[int, float]] = deque()
        low_window: deque[tuple[int, float]] = deque()
        tr_sum = 0.0
        previous_close: float | None = None

        for index, candle_row in enumerate(candle_rows):
            try:
                candle_high = float(candle_row["high"])
                candle_low = float(candle_row["low"])
                candle_close = float(candle_row["close"])
            except Exception:
                previous_close = None
                continue
            if (
                not math.isfinite(candle_high)
                or not math.isfinite(candle_low)
                or not math.isfinite(candle_close)
            ):
                previous_close = None
                continue

            if previous_close is None or not math.isfinite(previous_close):
                true_range = abs(candle_high - candle_low)
            else:
                true_range = max(
                    abs(candle_high - candle_low),
                    abs(candle_high - previous_close),
                    abs(candle_low - previous_close),
                )
            tr_window.append(float(true_range))
            tr_sum += float(true_range)
            if len(tr_window) > period:
                tr_sum -= float(tr_window.popleft())

            while high_window and float(high_window[-1][1]) <= candle_high:
                high_window.pop()
            high_window.append((index, candle_high))
            while high_window and int(high_window[0][0]) <= (index - period):
                high_window.popleft()

            while low_window and float(low_window[-1][1]) >= candle_low:
                low_window.pop()
            low_window.append((index, candle_low))
            while low_window and int(low_window[0][0]) <= (index - period):
                low_window.popleft()

            if len(tr_window) == period and index + 1 >= period:
                atr_value = tr_sum / float(period)
                highest_high = float(high_window[0][1])
                lowest_low = float(low_window[0][1])
                long_levels[index] = highest_high - (atr_value * float(multiplier))
                short_levels[index] = lowest_low + (atr_value * float(multiplier))

            previous_close = candle_close

        return long_levels, short_levels

    def _open_backtest_trade(
        self,
        *,
        trade_id: int,
        candle_row: dict[str, Any],
        signal_direction: int,
        entry_price: float,
        current_capital: float,
        strategy_name: str | None = None,
        leverage_scale: float = 1.0,
    ) -> PaperTrade | None:
        if signal_direction == 0:
            return None
        if not math.isfinite(entry_price) or entry_price <= 0.0:
            return None
        if not math.isfinite(current_capital) or current_capital <= 0.0:
            return None

        trade_settings = self._resolve_trade_settings(str(candle_row["symbol"]))
        risk_amount = current_capital * (self._trading_settings.risk_per_trade_pct / 100.0)
        if not math.isfinite(risk_amount) or risk_amount <= 0.0:
            return None
        normalized_leverage_scale = (
            float(leverage_scale)
            if math.isfinite(float(leverage_scale)) and float(leverage_scale) > 0.0
            else 1.0
        )
        effective_leverage = max(1, int(round(float(trade_settings.leverage) * normalized_leverage_scale)))
        position_size_usd = risk_amount * effective_leverage
        if not math.isfinite(position_size_usd) or position_size_usd <= 0.0:
            return None
        quantity = position_size_usd / entry_price
        if not math.isfinite(quantity) or quantity <= 0.0:
            return None

        entry_fee = self._calculate_fee(position_size_usd)
        side = "LONG" if signal_direction > 0 else "SHORT"
        return PaperTrade(
            id=trade_id,
            symbol=str(candle_row["symbol"]),
            side=side,
            entry_time=self._resolve_backtest_time(candle_row),
            entry_price=float(entry_price),
            qty=float(quantity),
            leverage=int(effective_leverage),
            status="OPEN",
            exit_time=None,
            exit_price=None,
            pnl=None,
            total_fees=float(entry_fee),
            high_water_mark=float(entry_price),
            strategy_name=(
                str(strategy_name).strip()
                if strategy_name is not None and str(strategy_name).strip()
                else None
            ),
            timeframe=str(candle_row.get("interval", trade_settings.interval)),
            review_status="PENDING",
        )

    def _close_backtest_trade(
        self,
        trade: PaperTrade,
        *,
        exit_price: float,
        exit_time: datetime,
        status: str,
    ) -> dict[str, Any]:
        gross_pnl = self._calculate_gross_pnl(trade, exit_price)
        exit_fee = self._calculate_fee(trade.qty * exit_price)
        total_fees = trade.total_fees + exit_fee
        net_pnl_before_slippage = gross_pnl - total_fees
        slippage_penalty_usd = self._calculate_backtest_slippage_penalty(
            entry_notional_usd=trade.qty * trade.entry_price,
            exit_notional_usd=trade.qty * exit_price,
        )
        net_pnl = net_pnl_before_slippage - slippage_penalty_usd
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "status": status,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "qty": trade.qty,
            "leverage": trade.leverage,
            "total_fees": total_fees,
            "pnl_before_slippage_usd": net_pnl_before_slippage,
            "slippage_penalty_usd": slippage_penalty_usd,
            "pnl": net_pnl,
        }

    def _get_normal_trailing_stop_price(self, trade: PaperTrade) -> float | None:
        trade_settings = self._resolve_trade_settings(trade.symbol)
        if not trade_settings.use_trailing_stop:
            return None
        if self._calculate_max_pnl_pct(trade) < trade_settings.trailing_activation_pct:
            return None

        trailing_distance_ratio = trade_settings.trailing_distance_pct / 100.0
        if trade.side == "LONG":
            return trade.high_water_mark * (1.0 - trailing_distance_ratio)
        return trade.high_water_mark * (1.0 + trailing_distance_ratio)

    def _get_tight_trailing_stop_price(self, trade: PaperTrade) -> float | None:
        trade_settings = self._resolve_trade_settings(trade.symbol)
        if not trade_settings.use_trailing_stop:
            return None
        if trade_settings.tight_trailing_activation_pct <= 0.0:
            return None
        if trade_settings.tight_trailing_distance_pct <= 0.0:
            return None
        if self._calculate_max_pnl_pct(trade) < trade_settings.tight_trailing_activation_pct:
            return None

        tight_trailing_distance_ratio = trade_settings.tight_trailing_distance_pct / 100.0
        if trade.side == "LONG":
            return trade.high_water_mark * (1.0 - tight_trailing_distance_ratio)
        return trade.high_water_mark * (1.0 + tight_trailing_distance_ratio)

    def _get_breakeven_stop_price(self, trade: PaperTrade) -> float | None:
        trade_settings = self._resolve_trade_settings(trade.symbol)
        if trade_settings.breakeven_activation_pct <= 0.0:
            return None
        if self._calculate_max_pnl_pct(trade) < trade_settings.breakeven_activation_pct:
            return None

        breakeven_buffer_ratio = trade_settings.breakeven_buffer_pct / 100.0
        if trade.side == "LONG":
            return trade.entry_price * (1.0 + breakeven_buffer_ratio)
        return trade.entry_price * (1.0 - breakeven_buffer_ratio)

    def _get_protective_stop(
        self,
        trade: PaperTrade,
        *,
        stop_loss_pct_override: float | None = None,
    ) -> tuple[str, float] | tuple[None, None]:
        candidates: list[tuple[str, float]] = [
            (
                "stop_loss",
                self._get_fixed_stop_price(
                    trade,
                    stop_loss_pct_override=stop_loss_pct_override,
                ),
            )
        ]

        breakeven_stop_price = self._get_breakeven_stop_price(trade)
        if breakeven_stop_price is not None:
            candidates.append(("breakeven", breakeven_stop_price))

        trailing_stop_price = self._get_normal_trailing_stop_price(trade)
        if trailing_stop_price is not None:
            candidates.append(("trailing", trailing_stop_price))

        tight_trailing_stop_price = self._get_tight_trailing_stop_price(trade)
        if tight_trailing_stop_price is not None:
            candidates.append(("tight_trailing", tight_trailing_stop_price))

        if not candidates:
            return None, None
        if trade.side == "LONG":
            return max(candidates, key=lambda candidate: candidate[1])
        return min(candidates, key=lambda candidate: candidate[1])

    def _get_fixed_stop_price(
        self,
        trade: PaperTrade,
        *,
        stop_loss_pct_override: float | None = None,
    ) -> float:
        if (
            stop_loss_pct_override is not None
            and math.isfinite(float(stop_loss_pct_override))
            and float(stop_loss_pct_override) > 0.0
        ):
            stop_loss_pct = float(stop_loss_pct_override)
        else:
            stop_loss_pct = float(self._resolve_trade_settings(trade.symbol).stop_loss_pct)
        stop_loss_ratio = stop_loss_pct / 100.0
        if trade.side == "LONG":
            return trade.entry_price * (1.0 - stop_loss_ratio)
        return trade.entry_price * (1.0 + stop_loss_ratio)

    @staticmethod
    def _resolve_dynamic_backtest_pct(
        values: list[float] | None,
        index: int,
    ) -> float | None:
        if values is None or index < 0 or index >= len(values):
            return None
        try:
            resolved_value = float(values[index])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(resolved_value) or resolved_value <= 0.0:
            return None
        return resolved_value

    @staticmethod
    def _resolve_dynamic_live_pct(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            resolved_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(resolved_value) or resolved_value <= 0.0:
            return None
        return resolved_value

    def _resolve_trade_settings(self, symbol: str | None) -> TradeRuntimeSettings:
        profile = None if symbol is None else self._trading_settings.coin_profiles.get(symbol)

        interval = self._trading_settings.interval
        leverage = self._trading_settings.default_leverage
        take_profit_pct = self._trading_settings.take_profit_pct
        stop_loss_pct = self._trading_settings.stop_loss_pct
        trailing_activation_pct = self._trading_settings.trailing_activation_pct
        trailing_distance_pct = self._trading_settings.trailing_distance_pct
        breakeven_activation_pct = self.BREAKEVEN_ACTIVATION_PCT
        breakeven_buffer_pct = self.BREAKEVEN_BUFFER_PCT
        tight_trailing_activation_pct = self.TIGHT_TRAILING_ACTIVATION_PCT
        tight_trailing_distance_pct = self.TIGHT_TRAILING_DISTANCE_PCT

        if profile is not None:
            if profile.interval is not None:
                interval = profile.interval
            if profile.default_leverage is not None:
                leverage = profile.default_leverage
            if profile.take_profit_pct is not None:
                take_profit_pct = profile.take_profit_pct
            if profile.stop_loss_pct is not None:
                stop_loss_pct = profile.stop_loss_pct
            if profile.trailing_activation_pct is not None:
                trailing_activation_pct = profile.trailing_activation_pct
            if profile.trailing_distance_pct is not None:
                trailing_distance_pct = profile.trailing_distance_pct
            if profile.breakeven_activation_pct is not None:
                breakeven_activation_pct = profile.breakeven_activation_pct
            if profile.breakeven_buffer_pct is not None:
                breakeven_buffer_pct = profile.breakeven_buffer_pct
            if profile.tight_trailing_activation_pct is not None:
                tight_trailing_activation_pct = profile.tight_trailing_activation_pct
            if profile.tight_trailing_distance_pct is not None:
                tight_trailing_distance_pct = profile.tight_trailing_distance_pct

        if self._explicit_interval is not None:
            interval = self._explicit_interval
        if self._explicit_leverage is not None:
            leverage = self._explicit_leverage
        if self._explicit_take_profit_pct is not None:
            take_profit_pct = self._explicit_take_profit_pct
        if self._explicit_stop_loss_pct is not None:
            stop_loss_pct = self._explicit_stop_loss_pct
        if self._explicit_trailing_activation_pct is not None:
            trailing_activation_pct = self._explicit_trailing_activation_pct
        if self._explicit_trailing_distance_pct is not None:
            trailing_distance_pct = self._explicit_trailing_distance_pct
        if self._explicit_breakeven_activation_pct is not None:
            breakeven_activation_pct = self._explicit_breakeven_activation_pct
        if self._explicit_breakeven_buffer_pct is not None:
            breakeven_buffer_pct = self._explicit_breakeven_buffer_pct
        if self._explicit_tight_trailing_activation_pct is not None:
            tight_trailing_activation_pct = self._explicit_tight_trailing_activation_pct
        if self._explicit_tight_trailing_distance_pct is not None:
            tight_trailing_distance_pct = self._explicit_tight_trailing_distance_pct

        if interval not in settings.api.timeframes:
            raise ValueError(f"Unsupported interval '{interval}'.")
        if not self._trading_settings.min_leverage <= leverage <= self._trading_settings.max_leverage:
            raise ValueError("Configured leverage is outside the allowed range.")
        if take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive.")
        if stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive.")
        if trailing_activation_pct < 0:
            raise ValueError("trailing_activation_pct must be greater than or equal to 0.")
        if trailing_distance_pct < 0:
            raise ValueError("trailing_distance_pct must be greater than or equal to 0.")
        if breakeven_activation_pct < 0:
            raise ValueError("breakeven_activation_pct must be greater than or equal to 0.")
        if breakeven_buffer_pct < 0:
            raise ValueError("breakeven_buffer_pct must be greater than or equal to 0.")
        if tight_trailing_activation_pct < 0:
            raise ValueError("tight_trailing_activation_pct must be greater than or equal to 0.")
        if tight_trailing_distance_pct < 0:
            raise ValueError("tight_trailing_distance_pct must be greater than or equal to 0.")

        return TradeRuntimeSettings(
            interval=interval,
            leverage=leverage,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            use_trailing_stop=self._trading_settings.use_trailing_stop,
            trailing_activation_pct=trailing_activation_pct,
            trailing_distance_pct=trailing_distance_pct,
            breakeven_activation_pct=breakeven_activation_pct,
            breakeven_buffer_pct=breakeven_buffer_pct,
            tight_trailing_activation_pct=tight_trailing_activation_pct,
            tight_trailing_distance_pct=tight_trailing_distance_pct,
        )

    @staticmethod
    def _stop_source_to_status(stop_source: str | None) -> str:
        mapping = {
            "stop_loss": "CLOSED_SL",
            "breakeven": "BREAKEVEN_STOP",
            "trailing": "TRAILING_STOP",
            "tight_trailing": "TIGHT_TRAILING_STOP",
        }
        if stop_source is None:
            return "CLOSED_SL"
        return mapping.get(stop_source, "CLOSED_SL")

    @classmethod
    def _to_intrabar_exit_status(cls, close_status: str) -> str:
        normalized_status = str(close_status).strip().upper()
        return str(cls.INTRABAR_EXIT_STATUS_MAP.get(normalized_status, normalized_status))

    def _calculate_max_pnl_pct(self, trade: PaperTrade) -> float:
        if trade.side == "LONG":
            return ((trade.high_water_mark / trade.entry_price) - 1.0) * 100.0
        return ((trade.entry_price / trade.high_water_mark) - 1.0) * 100.0

    @staticmethod
    def _calculate_high_water_mark(trade: PaperTrade, current_price: float) -> float:
        if trade.side == "LONG":
            return max(current_price, trade.high_water_mark)
        return min(current_price, trade.high_water_mark)

    @staticmethod
    def _replace_high_water_mark(trade: PaperTrade, high_water_mark: float) -> PaperTrade:
        return PaperTrade(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            qty=trade.qty,
            leverage=trade.leverage,
            status=trade.status,
            exit_time=trade.exit_time,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            total_fees=trade.total_fees,
            high_water_mark=high_water_mark,
            strategy_name=trade.strategy_name,
            timeframe=trade.timeframe,
            regime_label_at_entry=trade.regime_label_at_entry,
            regime_confidence=trade.regime_confidence,
            session_label=trade.session_label,
            signal_strength=trade.signal_strength,
            confidence_score=trade.confidence_score,
            atr_pct_at_entry=trade.atr_pct_at_entry,
            volume_ratio_at_entry=trade.volume_ratio_at_entry,
            spread_estimate=trade.spread_estimate,
            move_already_extended_pct=trade.move_already_extended_pct,
            entry_snapshot_json=trade.entry_snapshot_json,
            lifecycle_snapshot_json=trade.lifecycle_snapshot_json,
            profile_version=trade.profile_version,
            review_status=trade.review_status,
        )

    @staticmethod
    def _validate_price(price: float) -> None:
        if price <= 0:
            raise ValueError("current_price must be positive.")

    @staticmethod
    def _validate_signal_direction(signal_direction: int) -> None:
        if signal_direction not in (-2, -1, 0, 1, 2):
            raise ValueError("signal_direction must be one of -2, -1, 0, 1, or 2.")

    @staticmethod
    def _normalize_signal_direction(signal_direction: int) -> int:
        if signal_direction > 0:
            return 1
        if signal_direction < 0:
            return -1
        return 0

    @staticmethod
    def _extract_backtest_rows(candles_df: Any) -> list[dict[str, Any]]:
        if not hasattr(candles_df, "columns") or not hasattr(candles_df, "to_dict"):
            raise ValueError("candles_df must be a dataframe-like object with columns and to_dict().")

        required_columns = {"open_time", "open", "high", "low", "close", "volume"}
        missing_columns = required_columns.difference(set(candles_df.columns))
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"candles_df is missing required columns: {missing_list}.")

        rows = candles_df.to_dict("records")
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_rows.append(
                {
                    "symbol": str(row.get("symbol", "BACKTEST")),
                    "interval": str(row.get("interval", settings.trading.interval)),
                    "open_time": PaperTradingEngine._coerce_datetime(row["open_time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        return normalized_rows

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC).replace(tzinfo=None)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValueError("open_time values must be datetime-compatible.")

    def _resolve_backtest_time(self, candle_row: dict[str, Any]) -> datetime:
        return candle_row["open_time"] + self._interval_to_timedelta(str(candle_row["interval"]))

    def _restore_persisted_trades(self, persisted_trades: list[PaperTrade]) -> int:
        if not persisted_trades:
            return 0

        existing_symbols = {trade.symbol for trade in self._db.fetch_open_trades()}
        restored_count = 0
        for trade in persisted_trades:
            if trade.status != "OPEN" or trade.symbol in existing_symbols:
                continue
            self._db.insert_trade(
                PaperTradeCreate(
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_time=trade.entry_time,
                    entry_price=trade.entry_price,
                    qty=trade.qty,
                    leverage=trade.leverage,
                    status="OPEN",
                    total_fees=trade.total_fees,
                    high_water_mark=trade.high_water_mark,
                    strategy_name=trade.strategy_name,
                    timeframe=trade.timeframe,
                    regime_label_at_entry=trade.regime_label_at_entry,
                    regime_confidence=trade.regime_confidence,
                    session_label=trade.session_label,
                    signal_strength=trade.signal_strength,
                    confidence_score=trade.confidence_score,
                    atr_pct_at_entry=trade.atr_pct_at_entry,
                    volume_ratio_at_entry=trade.volume_ratio_at_entry,
                    spread_estimate=trade.spread_estimate,
                    move_already_extended_pct=trade.move_already_extended_pct,
                    entry_snapshot_json=trade.entry_snapshot_json,
                    lifecycle_snapshot_json=trade.lifecycle_snapshot_json,
                    profile_version=trade.profile_version,
                    review_status=trade.review_status,
                )
            )
            existing_symbols.add(trade.symbol)
            restored_count += 1
        return restored_count

    def _refresh_active_trade_storage(self) -> None:
        self.active_trades = self._db.fetch_open_trades()
        open_trade_ids = {int(trade.id) for trade in self.active_trades}
        self._dynamic_risk_overrides = {
            int(trade_id): (stop_loss_pct, take_profit_pct)
            for trade_id, (stop_loss_pct, take_profit_pct) in self._dynamic_risk_overrides.items()
            if int(trade_id) in open_trade_ids
        }
        if not self._enable_persistence:
            return
        save_paper_trades(self.active_trades, self._persistence_path)

    @staticmethod
    def _interval_to_timedelta(interval: str) -> timedelta:
        mapping = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "4h": timedelta(hours=4),
            "6h": timedelta(hours=6),
            "8h": timedelta(hours=8),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),
        }
        return mapping.get(interval, mapping.get(settings.trading.interval, timedelta(minutes=15)))
