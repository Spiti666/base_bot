from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def run_fast_backtest_loop(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    signal_arr: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    leverage: float = 25.0,
    taker_fee_pct: float = 0.0625,
    slippage_pct_per_side: float = 0.05,
    start_capital: float = 1000.0,
) -> tuple[float, float, float, int]:
    (
        total_pnl_usd,
        win_rate_pct,
        max_drawdown_pct,
        total_trades,
        _gross_profit,
        _gross_loss,
        _win_count,
        _loss_count,
        _long_trades,
        _short_trades,
        _longest_consecutive_losses,
        _total_fees_usd,
        _total_slippage_usd,
    ) = run_fast_backtest_loop_detailed(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        signal_arr,
        sl_pct,
        tp_pct,
        leverage,
        taker_fee_pct,
        slippage_pct_per_side,
        start_capital,
    )
    return total_pnl_usd, win_rate_pct, max_drawdown_pct, total_trades


@njit(cache=True)
def run_fast_backtest_loop_detailed(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    signal_arr: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    leverage: float = 25.0,
    taker_fee_pct: float = 0.0625,
    slippage_pct_per_side: float = 0.05,
    start_capital: float = 1000.0,
) -> tuple[
    float,
    float,
    float,
    int,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    early_stop_min_trades = 30
    early_stop_max_drawdown_pct = 40.0
    length = close_arr.shape[0]
    if length == 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0

    fee_ratio = taker_fee_pct / 100.0
    slippage_ratio = slippage_pct_per_side / 100.0

    equity = float(start_capital)
    peak_equity = float(start_capital)
    max_drawdown_pct = 0.0

    position_side = 0
    entry_price = 0.0
    qty = 0.0
    entry_fee_usd = 0.0
    entry_slippage_usd = 0.0

    total_pnl_usd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    win_count = 0
    loss_count = 0
    total_trades = 0
    long_trades = 0
    short_trades = 0
    total_fees_usd = 0.0
    total_slippage_usd = 0.0

    current_loss_streak = 0
    longest_consecutive_losses = 0

    for index in range(length):
        close_price = close_arr[index]
        high_price = high_arr[index]
        low_price = low_arr[index]
        if np.isnan(close_price) or close_price <= 0.0:
            continue
        if np.isnan(high_price) or high_price <= 0.0:
            high_price = close_price
        if np.isnan(low_price) or low_price <= 0.0:
            low_price = close_price
        if high_price < low_price:
            temp = high_price
            high_price = low_price
            low_price = temp

        if position_side != 0:
            stop_price = 0.0
            target_price = 0.0
            if position_side > 0:
                stop_price = entry_price * (1.0 - (sl_pct / 100.0))
                target_price = entry_price * (1.0 + (tp_pct / 100.0))
            else:
                stop_price = entry_price * (1.0 + (sl_pct / 100.0))
                target_price = entry_price * (1.0 - (tp_pct / 100.0))

            hit_stop = False
            hit_target = False
            if position_side > 0:
                hit_stop = low_price <= stop_price
                hit_target = high_price >= target_price
            else:
                hit_stop = high_price >= stop_price
                hit_target = low_price <= target_price

            if hit_stop or hit_target:
                exit_price = stop_price if hit_stop else target_price
                exit_notional = abs(qty * exit_price)
                exit_fee_usd = exit_notional * fee_ratio
                exit_slippage_usd = exit_notional * slippage_ratio
                gross_move = 0.0
                if position_side > 0:
                    gross_move = (exit_price - entry_price) * qty
                else:
                    gross_move = (entry_price - exit_price) * qty
                trade_pnl = gross_move - entry_fee_usd - entry_slippage_usd - exit_fee_usd - exit_slippage_usd

                total_pnl_usd += trade_pnl
                equity += trade_pnl
                total_fees_usd += entry_fee_usd + exit_fee_usd
                total_slippage_usd += entry_slippage_usd + exit_slippage_usd
                total_trades += 1

                if position_side > 0:
                    long_trades += 1
                else:
                    short_trades += 1

                if trade_pnl > 0.0:
                    win_count += 1
                    gross_profit += trade_pnl
                    current_loss_streak = 0
                elif trade_pnl < 0.0:
                    loss_count += 1
                    gross_loss += -trade_pnl
                    current_loss_streak += 1
                    if current_loss_streak > longest_consecutive_losses:
                        longest_consecutive_losses = current_loss_streak

                if equity > peak_equity:
                    peak_equity = equity
                elif peak_equity > 0.0:
                    current_drawdown = ((peak_equity - equity) / peak_equity) * 100.0
                    if current_drawdown > max_drawdown_pct:
                        max_drawdown_pct = current_drawdown
                if (
                    total_trades >= early_stop_min_trades
                    and max_drawdown_pct > early_stop_max_drawdown_pct
                ):
                    early_stop_win_rate_pct = 0.0
                    if total_trades > 0:
                        early_stop_win_rate_pct = (float(win_count) / float(total_trades)) * 100.0
                    return (
                        total_pnl_usd,
                        early_stop_win_rate_pct,
                        max_drawdown_pct,
                        total_trades,
                        gross_profit,
                        gross_loss,
                        win_count,
                        loss_count,
                        long_trades,
                        short_trades,
                        longest_consecutive_losses,
                        total_fees_usd,
                        total_slippage_usd,
                    )

                position_side = 0
                entry_price = 0.0
                qty = 0.0
                entry_fee_usd = 0.0
                entry_slippage_usd = 0.0

        if position_side != 0:
            continue

        signal = int(signal_arr[index])
        if signal == 0:
            continue
        side = 1 if signal > 0 else -1
        entry_notional = equity * leverage
        if entry_notional <= 0.0:
            continue
        quantity = entry_notional / close_price
        if quantity <= 0.0:
            continue
        position_side = side
        entry_price = close_price
        qty = quantity
        entry_fee_usd = entry_notional * fee_ratio
        entry_slippage_usd = entry_notional * slippage_ratio

    if position_side != 0:
        exit_price = close_arr[length - 1]
        if np.isnan(exit_price) or exit_price <= 0.0:
            exit_price = entry_price
        exit_notional = abs(qty * exit_price)
        exit_fee_usd = exit_notional * fee_ratio
        exit_slippage_usd = exit_notional * slippage_ratio
        gross_move = 0.0
        if position_side > 0:
            gross_move = (exit_price - entry_price) * qty
        else:
            gross_move = (entry_price - exit_price) * qty
        trade_pnl = gross_move - entry_fee_usd - entry_slippage_usd - exit_fee_usd - exit_slippage_usd

        total_pnl_usd += trade_pnl
        equity += trade_pnl
        total_fees_usd += entry_fee_usd + exit_fee_usd
        total_slippage_usd += entry_slippage_usd + exit_slippage_usd
        total_trades += 1
        if position_side > 0:
            long_trades += 1
        else:
            short_trades += 1

        if trade_pnl > 0.0:
            win_count += 1
            gross_profit += trade_pnl
            current_loss_streak = 0
        elif trade_pnl < 0.0:
            loss_count += 1
            gross_loss += -trade_pnl
            current_loss_streak += 1
            if current_loss_streak > longest_consecutive_losses:
                longest_consecutive_losses = current_loss_streak

        if equity > peak_equity:
            peak_equity = equity
        elif peak_equity > 0.0:
            current_drawdown = ((peak_equity - equity) / peak_equity) * 100.0
            if current_drawdown > max_drawdown_pct:
                max_drawdown_pct = current_drawdown

    win_rate_pct = 0.0
    if total_trades > 0:
        win_rate_pct = (float(win_count) / float(total_trades)) * 100.0
    return (
        total_pnl_usd,
        win_rate_pct,
        max_drawdown_pct,
        total_trades,
        gross_profit,
        gross_loss,
        win_count,
        loss_count,
        long_trades,
        short_trades,
        longest_consecutive_losses,
        total_fees_usd,
        total_slippage_usd,
    )
