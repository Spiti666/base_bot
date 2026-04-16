from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pprint import pformat

from config_sections.backtest_optimizer_defaults import (
    BACKTEST_HISTORY_START_UTC,
    DEFAULT_COIN_PROFILE_VALUES,
    OPTIMIZATION_DEFAULTS,
    OPTIMIZER_HISTORY_START_UTC,
    STRATEGY_GRID_DEFAULTS,
    TRADING_BASE_DEFAULTS,
    UNIVERSAL_RISK_GRID_DEFAULTS,
)
from config_sections.live_defaults import LIVE_DEFAULT_INTERVAL
from config_sections.production_registry import (
    PRODUCTION_PROFILE_INTERVAL,
    PRODUCTION_PROFILE_REGISTRY,
    PRODUCTION_STRATEGY_ALIASES,
)
from config_sections.strategy_defaults import (
    AVAILABLE_STRATEGIES,
    DEFAULT_SCAN_STRATEGIES,
    DEFAULT_COIN_STRATEGIES,
    DEFAULT_COIN_STRATEGY_PARAMS,
    DEFAULT_STRATEGY_NAME,
)

ACTIVE_COINS: list[str] = ['BTCUSDT',
 'ETHUSDT',
 'SOLUSDT',
 'XRPUSDT',
 'ADAUSDT',
 'DOGEUSDT',
 '1000PEPEUSDT',
 '1000SHIBUSDT',
 'BNBUSDT',
 'AVAXUSDT',
 'NEARUSDT',
 'DOTUSDT',
 'ARBUSDT',
 'SUIUSDT',
 'LTCUSDT',
 'BCHUSDT',
 'LINKUSDT',
 'TRXUSDT',
 'FILUSDT',
 'AAVEUSDT']

BACKTEST_ONLY_COINS: list[str] = ['TONUSDT',
 'WLDUSDT',
 'APTUSDT',
 'FETUSDT',
 'TAOUSDT',
 'ENAUSDT',
 'QNTUSDT',
 'ETCUSDT',
 'OPUSDT',
 'SEIUSDT',
 'INJUSDT',
 'ATOMUSDT',
 'POLUSDT',
 'TIAUSDT',
 'HBARUSDT',
 'UNIUSDT',
 'XLMUSDT',
 'ALGOUSDT',
 'ICPUSDT',
 'RENDERUSDT',
 'SANDUSDT',
 'GALAUSDT',
 'JUPUSDT',
 'PENDLEUSDT',
 'RUNEUSDT',
 'CRVUSDT',
 'ONDOUSDT',
 'KASUSDT',
 'IMXUSDT',
 'PYTHUSDT',
 '1000BONKUSDT',
 'JASMYUSDT',
 'LDOUSDT',
 'STXUSDT',
 'DYDXUSDT',
 'ARUSDT',
 'SUSHIUSDT',
 'EGLDUSDT',
 'COMPUSDT',
 'THETAUSDT']

# Dedicated backtest universe: live coins + backtest-only extensions.
BACKTEST_BATCH_SYMBOLS: tuple[str, ...] = tuple(
    dict.fromkeys((*ACTIVE_COINS, *BACKTEST_ONLY_COINS))
)
MAX_BACKTEST_CANDLES = 100_000

# Live production registry (overrides imported defaults from config_sections).
PRODUCTION_PROFILE_REGISTRY: dict[str, dict[str, object]] = {'1000PEPEUSDT': {'strategy_name': 'ema_band_rejection',
                  'interval': '1h',
                  'strategy_params': {'atr_length': 14.0,
                                      'atr_stop_buffer_mult': 0.5,
                                      'ema_fast': 5.0,
                                      'ema_mid': 10.0,
                                      'ema_slow': 20.0,
                                      'min_ema_spread_pct': 0.05,
                                      'min_slow_slope_pct': 0.0,
                                      'pullback_requires_outer_band_touch': 0.0,
                                      'rejection_body_min_ratio': 0.2,
                                      'rejection_wick_min_ratio': 0.35,
                                      'rsi_length': 14.0,
                                      'rsi_midline': 50.0,
                                      'rsi_midline_margin': 0.0,
                                      'signal_cooldown_bars': 3.0,
                                      'slope_lookback': 5.0,
                                      'use_atr_stop_buffer': 1.0,
                                      'use_rejection_quality_filter': 1.0,
                                      'use_rsi_cross_filter': 0.0,
                                      'use_rsi_filter': 1.0,
                                      'use_volume_filter': 0.0,
                                      'volume_ma_length': 20.0,
                                      'volume_multiplier': 1.0},
                  'default_leverage': 25,
                  'stop_loss_pct': 1.0,
                  'take_profit_pct': 5.0,
                  'trailing_activation_pct': 2.0,
                  'trailing_distance_pct': 0.1,
                  'breakeven_activation_pct': 2.0,
                  'breakeven_buffer_pct': 0.2},
 '1000SHIBUSDT': {'strategy_name': 'ema_band_rejection',
                  'interval': '1h',
                  'strategy_params': {'atr_length': 14.0,
                                      'atr_stop_buffer_mult': 0.5,
                                      'ema_fast': 5.0,
                                      'ema_mid': 10.0,
                                      'ema_slow': 20.0,
                                      'min_ema_spread_pct': 0.08,
                                      'min_slow_slope_pct': 0.0,
                                      'pullback_requires_outer_band_touch': 0.0,
                                      'rejection_body_min_ratio': 0.2,
                                      'rejection_wick_min_ratio': 0.35,
                                      'rsi_length': 14.0,
                                      'rsi_midline': 50.0,
                                      'rsi_midline_margin': 0.0,
                                      'signal_cooldown_bars': 3.0,
                                      'slope_lookback': 5.0,
                                      'use_atr_stop_buffer': 1.0,
                                      'use_rejection_quality_filter': 1.0,
                                      'use_rsi_cross_filter': 0.0,
                                      'use_rsi_filter': 1.0,
                                      'use_volume_filter': 0.0,
                                      'volume_ma_length': 20.0,
                                      'volume_multiplier': 1.0},
                  'default_leverage': 25,
                  'stop_loss_pct': 1.5,
                  'take_profit_pct': 5.0,
                  'trailing_activation_pct': 2.0,
                  'trailing_distance_pct': 0.1,
                  'breakeven_activation_pct': 2.0,
                  'breakeven_buffer_pct': 0.2},
 'AAVEUSDT': {'strategy_name': 'ema_band_rejection',
              'interval': '1h',
              'strategy_params': {'atr_length': 14.0,
                                  'atr_stop_buffer_mult': 0.5,
                                  'ema_fast': 5.0,
                                  'ema_mid': 10.0,
                                  'ema_slow': 20.0,
                                  'min_ema_spread_pct': 0.08,
                                  'min_slow_slope_pct': 0.02,
                                  'pullback_requires_outer_band_touch': 0.0,
                                  'rejection_body_min_ratio': 0.2,
                                  'rejection_wick_min_ratio': 0.35,
                                  'rsi_length': 14.0,
                                  'rsi_midline': 50.0,
                                  'rsi_midline_margin': 0.0,
                                  'signal_cooldown_bars': 0.0,
                                  'slope_lookback': 3.0,
                                  'use_atr_stop_buffer': 0.0,
                                  'use_rejection_quality_filter': 0.0,
                                  'use_rsi_cross_filter': 0.0,
                                  'use_rsi_filter': 1.0,
                                  'use_volume_filter': 1.0,
                                  'volume_ma_length': 20.0,
                                  'volume_multiplier': 1.25},
              'default_leverage': 25,
              'stop_loss_pct': 1.0,
              'take_profit_pct': 5.0,
              'trailing_activation_pct': 2.0,
              'trailing_distance_pct': 0.1,
              'breakeven_activation_pct': 2.0,
              'breakeven_buffer_pct': 0.2},
 'ADAUSDT': {'strategy_name': 'dual_thrust',
             'interval': '1h',
             'strategy_params': {'chandelier_multiplier': 1.5,
                                 'chandelier_period': 14.0,
                                 'dual_thrust_k1': 0.2,
                                 'dual_thrust_k2': 0.2,
                                 'dual_thrust_period': 89.0},
             'default_leverage': 25,
             'stop_loss_pct': 0.5,
             'take_profit_pct': 15.0,
             'trailing_activation_pct': 6.0,
             'trailing_distance_pct': 0.15,
             'breakeven_activation_pct': 3.5,
             'breakeven_buffer_pct': 0.2,
             'tight_trailing_activation_pct': 8.0,
             'tight_trailing_distance_pct': 0.3},
 'ARBUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.08,
                                 'min_slow_slope_pct': 0.02,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 1.0,
                                 'signal_cooldown_bars': 0.0,
                                 'slope_lookback': 5.0,
                                 'use_atr_stop_buffer': 1.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 1.0,
                                 'use_volume_filter': 1.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.0,
             'take_profit_pct': 3.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'AVAXUSDT': {'strategy_name': 'ema_band_rejection',
              'interval': '1h',
              'strategy_params': {'atr_length': 14.0,
                                  'atr_stop_buffer_mult': 0.5,
                                  'ema_fast': 5.0,
                                  'ema_mid': 10.0,
                                  'ema_slow': 20.0,
                                  'min_ema_spread_pct': 0.05,
                                  'min_slow_slope_pct': 0.02,
                                  'pullback_requires_outer_band_touch': 0.0,
                                  'rejection_body_min_ratio': 0.2,
                                  'rejection_wick_min_ratio': 0.35,
                                  'rsi_length': 14.0,
                                  'rsi_midline': 50.0,
                                  'rsi_midline_margin': 0.0,
                                  'signal_cooldown_bars': 3.0,
                                  'slope_lookback': 3.0,
                                  'use_atr_stop_buffer': 0.0,
                                  'use_rejection_quality_filter': 1.0,
                                  'use_rsi_cross_filter': 0.0,
                                  'use_rsi_filter': 0.0,
                                  'use_volume_filter': 0.0,
                                  'volume_ma_length': 20.0,
                                  'volume_multiplier': 1.0},
              'default_leverage': 25,
              'stop_loss_pct': 2.0,
              'take_profit_pct': 5.0,
              'trailing_activation_pct': 2.0,
              'trailing_distance_pct': 0.1,
              'breakeven_activation_pct': 2.0,
              'breakeven_buffer_pct': 0.2},
 'BCHUSDT': {'strategy_name': 'frama_cross',
             'interval': '1h',
             'strategy_params': {'chandelier_multiplier': 3.5,
                                 'chandelier_period': 14.0,
                                 'frama_fast_period': 13.0,
                                 'frama_slow_period': 60.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.5,
             'take_profit_pct': 3.5,
             'trailing_activation_pct': 5.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 5.0,
             'breakeven_buffer_pct': 0.2},
 'BNBUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.08,
                                 'min_slow_slope_pct': 0.02,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 0.0,
                                 'signal_cooldown_bars': 3.0,
                                 'slope_lookback': 3.0,
                                 'use_atr_stop_buffer': 0.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 0.0,
                                 'use_volume_filter': 0.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.5,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'BTCUSDT': {'strategy_name': 'frama_cross',
             'interval': '15m',
             'strategy_params': {'chandelier_multiplier': 2.5,
                                 'chandelier_period': 14.0,
                                 'frama_fast_period': 13.0,
                                 'frama_slow_period': 200.0,
                                 'volume_multiplier': 2.5},
             'default_leverage': 25,
             'stop_loss_pct': 3.0,
             'take_profit_pct': 7.5,
             'trailing_activation_pct': 7.5,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 5.0,
             'breakeven_buffer_pct': 0.2},
 'DOGEUSDT': {'strategy_name': 'dual_thrust',
              'interval': '1h',
              'strategy_params': {'chandelier_multiplier': 1.5,
                                  'chandelier_period': 14.0,
                                  'dual_thrust_k1': 0.5,
                                  'dual_thrust_k2': 0.3,
                                  'dual_thrust_period': 89.0},
              'default_leverage': 25,
              'stop_loss_pct': 1.5,
              'take_profit_pct': 3.0,
              'trailing_activation_pct': 6.0,
              'trailing_distance_pct': 0.3,
              'breakeven_activation_pct': 3.5,
              'breakeven_buffer_pct': 0.2,
              'tight_trailing_activation_pct': 8.0,
              'tight_trailing_distance_pct': 0.3},
 'DOTUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.05,
                                 'min_slow_slope_pct': 0.0,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 0.0,
                                 'signal_cooldown_bars': 3.0,
                                 'slope_lookback': 3.0,
                                 'use_atr_stop_buffer': 1.0,
                                 'use_rejection_quality_filter': 1.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 0.0,
                                 'use_volume_filter': 0.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.5,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'ETHUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.08,
                                 'min_slow_slope_pct': 0.0,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 1.0,
                                 'signal_cooldown_bars': 3.0,
                                 'slope_lookback': 3.0,
                                 'use_atr_stop_buffer': 0.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 1.0,
                                 'use_volume_filter': 1.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.0,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'FILUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.08,
                                 'min_slow_slope_pct': 0.02,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 0.0,
                                 'signal_cooldown_bars': 3.0,
                                 'slope_lookback': 3.0,
                                 'use_atr_stop_buffer': 0.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 0.0,
                                 'use_volume_filter': 1.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.0,
             'take_profit_pct': 3.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'LINKUSDT': {'strategy_name': 'ema_band_rejection',
              'interval': '1h',
              'strategy_params': {'atr_length': 14.0,
                                  'atr_stop_buffer_mult': 0.5,
                                  'ema_fast': 5.0,
                                  'ema_mid': 10.0,
                                  'ema_slow': 20.0,
                                  'min_ema_spread_pct': 0.05,
                                  'min_slow_slope_pct': 0.02,
                                  'pullback_requires_outer_band_touch': 0.0,
                                  'rejection_body_min_ratio': 0.2,
                                  'rejection_wick_min_ratio': 0.35,
                                  'rsi_length': 14.0,
                                  'rsi_midline': 50.0,
                                  'rsi_midline_margin': 0.0,
                                  'signal_cooldown_bars': 0.0,
                                  'slope_lookback': 3.0,
                                  'use_atr_stop_buffer': 1.0,
                                  'use_rejection_quality_filter': 1.0,
                                  'use_rsi_cross_filter': 0.0,
                                  'use_rsi_filter': 0.0,
                                  'use_volume_filter': 0.0,
                                  'volume_ma_length': 20.0,
                                  'volume_multiplier': 1.0},
              'default_leverage': 25,
              'stop_loss_pct': 2.0,
              'take_profit_pct': 3.0,
              'trailing_activation_pct': 2.0,
              'trailing_distance_pct': 0.1,
              'breakeven_activation_pct': 2.0,
              'breakeven_buffer_pct': 0.2},
 'LTCUSDT': {'strategy_name': 'frama_cross',
             'interval': '1h',
             'strategy_params': {'chandelier_multiplier': 3.5,
                                 'chandelier_period': 30.0,
                                 'frama_fast_period': 13.0,
                                 'frama_slow_period': 100.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.5,
             'take_profit_pct': 3.5,
             'trailing_activation_pct': 5.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 5.0,
             'breakeven_buffer_pct': 0.2},
 'NEARUSDT': {'strategy_name': 'frama_cross',
              'interval': '15m',
              'strategy_params': {'chandelier_multiplier': 2.5,
                                  'chandelier_period': 30.0,
                                  'frama_fast_period': 21.0,
                                  'frama_slow_period': 60.0,
                                  'volume_multiplier': 1.5},
              'default_leverage': 25,
              'stop_loss_pct': 3.0,
              'take_profit_pct': 5.0,
              'trailing_activation_pct': 5.0,
              'trailing_distance_pct': 0.1,
              'breakeven_activation_pct': 5.0,
              'breakeven_buffer_pct': 0.2},
 'SOLUSDT': {'strategy_name': 'frama_cross',
             'interval': '15m',
             'strategy_params': {'chandelier_multiplier': 2.5,
                                 'chandelier_period': 14.0,
                                 'frama_fast_period': 8.0,
                                 'frama_slow_period': 100.0,
                                 'volume_multiplier': 2.5},
             'default_leverage': 25,
             'stop_loss_pct': 3.0,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 5.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 3.5,
             'breakeven_buffer_pct': 0.2},
 'SUIUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.05,
                                 'min_slow_slope_pct': 0.02,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 1.0,
                                 'signal_cooldown_bars': 3.0,
                                 'slope_lookback': 3.0,
                                 'use_atr_stop_buffer': 0.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 1.0,
                                 'use_volume_filter': 1.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.5,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2},
 'TRXUSDT': {'strategy_name': 'frama_cross',
             'interval': '15m',
             'strategy_params': {'chandelier_multiplier': 2.5,
                                 'chandelier_period': 14.0,
                                 'frama_fast_period': 13.0,
                                 'frama_slow_period': 60.0,
                                 'volume_multiplier': 2.5},
             'default_leverage': 25,
             'stop_loss_pct': 0.5,
             'take_profit_pct': 3.5,
             'trailing_activation_pct': 5.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 5.0,
             'breakeven_buffer_pct': 0.2},
 'XRPUSDT': {'strategy_name': 'ema_band_rejection',
             'interval': '1h',
             'strategy_params': {'atr_length': 14.0,
                                 'atr_stop_buffer_mult': 0.5,
                                 'ema_fast': 5.0,
                                 'ema_mid': 10.0,
                                 'ema_slow': 20.0,
                                 'min_ema_spread_pct': 0.05,
                                 'min_slow_slope_pct': 0.02,
                                 'pullback_requires_outer_band_touch': 0.0,
                                 'rejection_body_min_ratio': 0.2,
                                 'rejection_wick_min_ratio': 0.35,
                                 'rsi_length': 14.0,
                                 'rsi_midline': 50.0,
                                 'rsi_midline_margin': 0.0,
                                 'signal_cooldown_bars': 0.0,
                                 'slope_lookback': 5.0,
                                 'use_atr_stop_buffer': 0.0,
                                 'use_rejection_quality_filter': 0.0,
                                 'use_rsi_cross_filter': 0.0,
                                 'use_rsi_filter': 0.0,
                                 'use_volume_filter': 1.0,
                                 'volume_ma_length': 20.0,
                                 'volume_multiplier': 1.0},
             'default_leverage': 25,
             'stop_loss_pct': 1.0,
             'take_profit_pct': 5.0,
             'trailing_activation_pct': 2.0,
             'trailing_distance_pct': 0.1,
             'breakeven_activation_pct': 2.0,
             'breakeven_buffer_pct': 0.2}}

EMA_BAND_REJECTION_1H_EXCLUDED_COINS: tuple[str, ...] = (
    "BNBUSDT",
    "LINKUSDT",
    "AAVEUSDT",
    "TRXUSDT",
    "LTCUSDT",
)


@dataclass(frozen=True, slots=True)
class CoinProfileSettings:
    interval: str | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    breakeven_activation_pct: float | None = None
    breakeven_buffer_pct: float | None = None
    trailing_activation_pct: float | None = None
    trailing_distance_pct: float | None = None
    tight_trailing_activation_pct: float | None = None
    tight_trailing_distance_pct: float | None = None
    default_leverage: int | None = None


def _build_default_coin_strategies() -> dict[str, str]:
    return dict(DEFAULT_COIN_STRATEGIES)


def _build_default_coin_strategy_params() -> dict[str, dict[str, float]]:
    return {
        symbol: dict(params)
        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items()
    }


def _build_default_coin_profiles() -> dict[str, CoinProfileSettings]:
    return {
        symbol: CoinProfileSettings(
            interval=(
                None if profile.get("interval") is None else str(profile["interval"])
            ),
            stop_loss_pct=(
                None if profile.get("stop_loss_pct") is None else float(profile["stop_loss_pct"])
            ),
            take_profit_pct=(
                None if profile.get("take_profit_pct") is None else float(profile["take_profit_pct"])
            ),
            breakeven_activation_pct=(
                None
                if profile.get("breakeven_activation_pct") is None
                else float(profile["breakeven_activation_pct"])
            ),
            breakeven_buffer_pct=(
                None
                if profile.get("breakeven_buffer_pct") is None
                else float(profile["breakeven_buffer_pct"])
            ),
            trailing_activation_pct=(
                None
                if profile.get("trailing_activation_pct") is None
                else float(profile["trailing_activation_pct"])
            ),
            trailing_distance_pct=(
                None
                if profile.get("trailing_distance_pct") is None
                else float(profile["trailing_distance_pct"])
            ),
            tight_trailing_activation_pct=(
                None
                if profile.get("tight_trailing_activation_pct") is None
                else float(profile["tight_trailing_activation_pct"])
            ),
            tight_trailing_distance_pct=(
                None
                if profile.get("tight_trailing_distance_pct") is None
                else float(profile["tight_trailing_distance_pct"])
            ),
            default_leverage=(
                None if profile.get("default_leverage") is None else int(profile["default_leverage"])
            ),
        )
        for symbol, profile in DEFAULT_COIN_PROFILE_VALUES.items()
    }


def _float_range_tuple(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("step must be positive.")
    values: list[float] = []
    index = 0
    current = float(start)
    epsilon = step * 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        index += 1
        current = start + step * index
    return tuple(values)


@dataclass(frozen=True, slots=True)
class RequestLimits:
    public_requests_per_second: int = 10
    private_requests_per_second: int = 10
    candles_per_request: int = 1000
    request_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.public_requests_per_second <= 0:
            raise ValueError("public_requests_per_second must be positive.")
        if self.private_requests_per_second <= 0:
            raise ValueError("private_requests_per_second must be positive.")
        if self.candles_per_request <= 0:
            raise ValueError("candles_per_request must be positive.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class APISettings:
    base_url: str = "https://fapi.bitunix.com"
    timeframes: tuple[str, ...] = (
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    )
    request_limits: RequestLimits = field(default_factory=RequestLimits)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty.")
        if not self.timeframes:
            raise ValueError("At least one timeframe must be configured.")


@dataclass(frozen=True, slots=True)
class LiveSettings:
    default_interval: str = LIVE_DEFAULT_INTERVAL
    available_symbols: tuple[str, ...] = tuple(ACTIVE_COINS)

    def __post_init__(self) -> None:
        if not self.default_interval:
            raise ValueError("default_interval must not be empty.")
        if not self.available_symbols:
            raise ValueError("available_symbols must not be empty.")


@dataclass(frozen=True, slots=True)
class StrategySettings:
    dual_thrust_period: int = 4
    dual_thrust_k1: float = 0.5
    dual_thrust_k2: float = 0.5
    dual_thrust_opt_periods: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["dual_thrust_opt_periods"]
    )
    dual_thrust_opt_k_values: tuple[float, ...] = tuple(
        float(value) for value in STRATEGY_GRID_DEFAULTS["dual_thrust_opt_k_values"]
    )
    dual_thrust_stop_loss_pct_options: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    dual_thrust_take_profit_pct_options: tuple[float, ...] = (5.0, 10.0, 15.0, 25.0)
    dual_thrust_trailing_activation_pct_options: tuple[float, ...] = (6.0, 8.0, 10.0, 15.0)
    dual_thrust_trailing_distance_pct_options: tuple[float, ...] = (
        0.15,
        0.3,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    )
    dual_thrust_symbol_optimization_grids: dict[str, dict[str, tuple[float, ...]]] = field(
        default_factory=dict
    )
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_fast_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["ema_fast_options"]
    )
    ema_slow_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["ema_slow_options"]
    )
    frama_fast_period: int = 16
    frama_slow_period: int = 55
    frama_fast_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["frama_fast_options"]
    )
    frama_slow_options: tuple[int, ...] = tuple(
        int(value) for value in STRATEGY_GRID_DEFAULTS["frama_slow_options"]
    )
    volume_sma_period: int = 20
    volume_multiplier: float = 1.0
    volume_multiplier_options: tuple[float, ...] = (1.0, 1.25, 1.5)
    use_late_entry_guard: bool = False
    late_entry_max_move_1_bar_pct: float = 0.0
    late_entry_max_move_2_bars_pct: float = 0.0
    late_entry_max_move_3_bars_pct: float = 0.0
    late_entry_max_distance_ref_pct: float = 0.0
    late_entry_max_distance_fast_ref_pct: float = 0.0
    late_entry_max_distance_mid_ref_pct: float = 0.0
    late_entry_max_atr_mult: float = 0.0
    use_pullback_reentry: bool = False
    pullback_reentry_min_touch: float = 0.0
    pullback_reentry_reconfirm_required: bool = False
    max_breakout_candle_body_pct: float = 0.0
    max_breakout_candle_range_atr_mult: float = 0.0
    available_strategies: tuple[str, ...] = AVAILABLE_STRATEGIES
    default_strategy_name: str = DEFAULT_STRATEGY_NAME
    coin_strategies: dict[str, str] = field(default_factory=_build_default_coin_strategies)
    coin_strategy_params: dict[str, dict[str, float]] = field(
        default_factory=_build_default_coin_strategy_params
    )

    def __post_init__(self) -> None:
        production_coin_strategies = dict(self.coin_strategies)
        production_coin_strategy_params = {
            symbol: dict(params)
            for symbol, params in self.coin_strategy_params.items()
        }
        for symbol, profile in PRODUCTION_PROFILE_REGISTRY.items():
            strategy_name = str(profile.get("strategy_name", profile.get("strategy", "")))
            strategy_name = PRODUCTION_STRATEGY_ALIASES.get(strategy_name, strategy_name)
            raw_strategy_params = profile.get("strategy_params", profile.get("params", {}))
            if not isinstance(raw_strategy_params, dict):
                raw_strategy_params = {}
            strategy_params = {
                str(param_name): float(param_value)
                for param_name, param_value in raw_strategy_params.items()
            }
            production_coin_strategies[symbol] = strategy_name
            production_coin_strategy_params[symbol] = strategy_params
        # Apply local optimizer overrides as defaults for symbols that are not pinned
        # in the production registry.
        for symbol, strategy_name in DEFAULT_COIN_STRATEGIES.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY:
                continue
            normalized_strategy_name = PRODUCTION_STRATEGY_ALIASES.get(
                str(strategy_name),
                str(strategy_name),
            )
            production_coin_strategies[symbol] = normalized_strategy_name
        for symbol, params in DEFAULT_COIN_STRATEGY_PARAMS.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY:
                continue
            production_coin_strategy_params[symbol] = {
                str(param_name): float(param_value)
                for param_name, param_value in params.items()
            }
        object.__setattr__(self, "coin_strategies", production_coin_strategies)
        object.__setattr__(self, "coin_strategy_params", production_coin_strategy_params)

        if self.dual_thrust_period <= 0:
            raise ValueError("dual_thrust_period must be positive.")
        if self.dual_thrust_k1 <= 0.0:
            raise ValueError("dual_thrust_k1 must be positive.")
        if self.dual_thrust_k2 <= 0.0:
            raise ValueError("dual_thrust_k2 must be positive.")
        if not self.dual_thrust_opt_periods:
            raise ValueError("dual_thrust_opt_periods must not be empty.")
        if any(value <= 0 for value in self.dual_thrust_opt_periods):
            raise ValueError("dual_thrust_opt_periods must contain only positive values.")
        if not self.dual_thrust_opt_k_values:
            raise ValueError("dual_thrust_opt_k_values must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_opt_k_values):
            raise ValueError("dual_thrust_opt_k_values must contain only positive values.")
        if not self.dual_thrust_stop_loss_pct_options:
            raise ValueError("dual_thrust_stop_loss_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_stop_loss_pct_options):
            raise ValueError("dual_thrust_stop_loss_pct_options must contain only positive values.")
        if not self.dual_thrust_take_profit_pct_options:
            raise ValueError("dual_thrust_take_profit_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_take_profit_pct_options):
            raise ValueError("dual_thrust_take_profit_pct_options must contain only positive values.")
        if not self.dual_thrust_trailing_activation_pct_options:
            raise ValueError("dual_thrust_trailing_activation_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_trailing_activation_pct_options):
            raise ValueError(
                "dual_thrust_trailing_activation_pct_options must contain only positive values."
            )
        if not self.dual_thrust_trailing_distance_pct_options:
            raise ValueError("dual_thrust_trailing_distance_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.dual_thrust_trailing_distance_pct_options):
            raise ValueError(
                "dual_thrust_trailing_distance_pct_options must contain only positive values."
            )
        if not any(
            float(trailing_distance_pct) < float(trailing_activation_pct)
            for trailing_activation_pct in self.dual_thrust_trailing_activation_pct_options
            for trailing_distance_pct in self.dual_thrust_trailing_distance_pct_options
        ):
            raise ValueError(
                "dual_thrust risk options must provide at least one trailing_distance_pct < trailing_activation_pct combination."
            )
        for symbol, symbol_grid in self.dual_thrust_symbol_optimization_grids.items():
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol:
                raise ValueError(
                    "dual_thrust_symbol_optimization_grids keys must be non-empty symbols."
                )
            stop_loss_options = tuple(
                float(value)
                for value in symbol_grid.get("stop_loss_pct_options", ())
            )
            take_profit_options = tuple(
                float(value)
                for value in symbol_grid.get("take_profit_pct_options", ())
            )
            if not stop_loss_options:
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}] must define stop_loss_pct_options."
                )
            if not take_profit_options:
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}] must define take_profit_pct_options."
                )
            if any(value <= 0.0 for value in stop_loss_options):
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}].stop_loss_pct_options must contain only positive values."
                )
            if any(value <= 0.0 for value in take_profit_options):
                raise ValueError(
                    f"dual_thrust_symbol_optimization_grids[{normalized_symbol}].take_profit_pct_options must contain only positive values."
                )
        if self.ema_fast_period <= 0:
            raise ValueError("ema_fast_period must be positive.")
        if self.ema_slow_period <= 0:
            raise ValueError("ema_slow_period must be positive.")
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be less than ema_slow_period.")
        if not self.ema_fast_options:
            raise ValueError("ema_fast_options must not be empty.")
        if any(value <= 0 for value in self.ema_fast_options):
            raise ValueError("ema_fast_options must contain only positive values.")
        if not self.ema_slow_options:
            raise ValueError("ema_slow_options must not be empty.")
        if any(value <= 0 for value in self.ema_slow_options):
            raise ValueError("ema_slow_options must contain only positive values.")
        if self.frama_fast_period <= 1:
            raise ValueError("frama_fast_period must be greater than 1.")
        if self.frama_slow_period <= 1:
            raise ValueError("frama_slow_period must be greater than 1.")
        if self.frama_fast_period >= self.frama_slow_period:
            raise ValueError("frama_fast_period must be less than frama_slow_period.")
        if not self.frama_fast_options:
            raise ValueError("frama_fast_options must not be empty.")
        if any(value <= 1 for value in self.frama_fast_options):
            raise ValueError("frama_fast_options must contain only values greater than 1.")
        if not self.frama_slow_options:
            raise ValueError("frama_slow_options must not be empty.")
        if any(value <= 1 for value in self.frama_slow_options):
            raise ValueError("frama_slow_options must contain only values greater than 1.")
        if self.volume_sma_period <= 0:
            raise ValueError("volume_sma_period must be positive.")
        if self.volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive.")
        if not self.volume_multiplier_options:
            raise ValueError("volume_multiplier_options must not be empty.")
        if any(value <= 0.0 for value in self.volume_multiplier_options):
            raise ValueError("volume_multiplier_options must contain only positive values.")
        if self.late_entry_max_move_1_bar_pct < 0.0:
            raise ValueError("late_entry_max_move_1_bar_pct must be greater than or equal to 0.")
        if self.late_entry_max_move_2_bars_pct < 0.0:
            raise ValueError("late_entry_max_move_2_bars_pct must be greater than or equal to 0.")
        if self.late_entry_max_move_3_bars_pct < 0.0:
            raise ValueError("late_entry_max_move_3_bars_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_fast_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_fast_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_distance_mid_ref_pct < 0.0:
            raise ValueError("late_entry_max_distance_mid_ref_pct must be greater than or equal to 0.")
        if self.late_entry_max_atr_mult < 0.0:
            raise ValueError("late_entry_max_atr_mult must be greater than or equal to 0.")
        if self.pullback_reentry_min_touch < 0.0:
            raise ValueError("pullback_reentry_min_touch must be greater than or equal to 0.")
        if self.max_breakout_candle_body_pct < 0.0:
            raise ValueError("max_breakout_candle_body_pct must be greater than or equal to 0.")
        if self.max_breakout_candle_range_atr_mult < 0.0:
            raise ValueError("max_breakout_candle_range_atr_mult must be greater than or equal to 0.")
        if not self.available_strategies:
            raise ValueError("available_strategies must not be empty.")
        if self.default_strategy_name not in self.available_strategies:
            raise ValueError("default_strategy_name must be one of available_strategies.")
        for symbol, strategy_name in self.coin_strategies.items():
            if not symbol:
                raise ValueError("coin_strategies keys must not be empty.")
            if strategy_name not in self.available_strategies:
                raise ValueError(
                    f"coin_strategies[{symbol}] must be one of available_strategies."
                )

@dataclass(frozen=True, slots=True)
class TradingSettings:
    interval: str = str(TRADING_BASE_DEFAULTS["interval"])
    backtest_history_start_utc: str = BACKTEST_HISTORY_START_UTC
    optimizer_history_start_utc: str = OPTIMIZER_HISTORY_START_UTC
    start_capital: float = float(TRADING_BASE_DEFAULTS["start_capital"])
    risk_per_trade_pct: float = 1.0
    taker_fee_pct: float = float(TRADING_BASE_DEFAULTS["taker_fee_pct"])
    use_setup_gate: bool = bool(TRADING_BASE_DEFAULTS["use_setup_gate"])
    use_hmm_regime_filter: bool = bool(TRADING_BASE_DEFAULTS["use_hmm_regime_filter"])
    hmm_allowed_regimes: tuple[str, ...] = tuple(TRADING_BASE_DEFAULTS["hmm_allowed_regimes"])
    min_confidence_pct: float = float(TRADING_BASE_DEFAULTS["min_confidence_pct"])
    take_profit_pct: float = float(TRADING_BASE_DEFAULTS["take_profit_pct"])
    stop_loss_pct: float = float(TRADING_BASE_DEFAULTS["stop_loss_pct"])
    chandelier_period: int = int(TRADING_BASE_DEFAULTS["chandelier_period"])
    chandelier_multiplier: float = float(TRADING_BASE_DEFAULTS["chandelier_multiplier"])
    use_trailing_stop: bool = bool(TRADING_BASE_DEFAULTS["use_trailing_stop"])
    trailing_activation_pct: float = float(TRADING_BASE_DEFAULTS["trailing_activation_pct"])
    trailing_distance_pct: float = float(TRADING_BASE_DEFAULTS["trailing_distance_pct"])
    min_leverage: int = int(TRADING_BASE_DEFAULTS["min_leverage"])
    max_leverage: int = int(TRADING_BASE_DEFAULTS["max_leverage"])
    default_leverage: int = int(TRADING_BASE_DEFAULTS["default_leverage"])
    max_open_positions: int = 5
    # Aggressive optimization defaults while preserving slot count for stable profile volume.
    optimization_take_profit_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_take_profit_pct_options"]
    )
    optimization_stop_loss_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_stop_loss_pct_options"]
    )
    optimization_trailing_activation_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_trailing_activation_pct_options"]
    )
    optimization_trailing_distance_pct_options: tuple[float, ...] = tuple(
        OPTIMIZATION_DEFAULTS["optimization_trailing_distance_pct_options"]
    )
    optimization_max_sample_profiles: int = int(
        OPTIMIZATION_DEFAULTS["optimization_max_sample_profiles"]
    )
    random_search_samples: int = int(
        OPTIMIZATION_DEFAULTS["random_search_samples"]
    )
    optimization_search_window_candles: int = int(
        OPTIMIZATION_DEFAULTS["optimization_search_window_candles"]
    )
    optimization_validation_top_n: int = int(
        OPTIMIZATION_DEFAULTS["optimization_validation_top_n"]
    )
    # Universal risk grid used for large automated scans
    # This is applied across strategies when running the universal scan
    universal_stop_loss_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_stop_loss_pct_options"]
    )
    universal_take_profit_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_take_profit_pct_options"]
    )
    universal_trailing_activation_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_trailing_activation_pct_options"]
    )
    universal_trailing_distance_pct_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_trailing_distance_pct_options"]
    )
    universal_chandelier_period_options: tuple[int, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_chandelier_period_options"]
    )
    universal_chandelier_multiplier_options: tuple[float, ...] = tuple(
        UNIVERSAL_RISK_GRID_DEFAULTS["universal_chandelier_multiplier_options"]
    )
    # Terminal I/O throttling for optimizer progress prints (every N profiles)
    optimizer_progress_print_modulo: int = 2500
    coin_profiles: dict[str, CoinProfileSettings] = field(default_factory=_build_default_coin_profiles)

    def __post_init__(self) -> None:
        production_coin_profiles = dict(self.coin_profiles)
        for symbol, profile in PRODUCTION_PROFILE_REGISTRY.items():
            resolved_interval = str(profile.get("interval", PRODUCTION_PROFILE_INTERVAL) or PRODUCTION_PROFILE_INTERVAL)
            production_coin_profiles[symbol] = CoinProfileSettings(
                interval=resolved_interval,
                take_profit_pct=float(profile["take_profit_pct"]),
                stop_loss_pct=float(profile["stop_loss_pct"]),
                breakeven_activation_pct=(
                    None
                    if profile.get("breakeven_activation_pct") is None
                    else float(profile["breakeven_activation_pct"])
                ),
                breakeven_buffer_pct=(
                    None
                    if profile.get("breakeven_buffer_pct") is None
                    else float(profile["breakeven_buffer_pct"])
                ),
                trailing_activation_pct=float(profile["trailing_activation_pct"]),
                trailing_distance_pct=float(profile["trailing_distance_pct"]),
                tight_trailing_activation_pct=(
                    None
                    if profile.get("tight_trailing_activation_pct") is None
                    else float(profile["tight_trailing_activation_pct"])
                ),
                tight_trailing_distance_pct=(
                    None
                    if profile.get("tight_trailing_distance_pct") is None
                    else float(profile["tight_trailing_distance_pct"])
                ),
                default_leverage=int(profile.get("default_leverage", 20) or 20),
            )
        # Apply local optimizer overrides as defaults for symbols that are not
        # pinned in the production registry.
        for symbol, profile in DEFAULT_COIN_PROFILE_VALUES.items():
            if symbol in PRODUCTION_PROFILE_REGISTRY:
                continue
            production_coin_profiles[symbol] = CoinProfileSettings(
                interval=(
                    None if profile.get("interval") is None else str(profile["interval"])
                ),
                stop_loss_pct=(
                    None if profile.get("stop_loss_pct") is None else float(profile["stop_loss_pct"])
                ),
                take_profit_pct=(
                    None if profile.get("take_profit_pct") is None else float(profile["take_profit_pct"])
                ),
                breakeven_activation_pct=(
                    None
                    if profile.get("breakeven_activation_pct") is None
                    else float(profile["breakeven_activation_pct"])
                ),
                breakeven_buffer_pct=(
                    None
                    if profile.get("breakeven_buffer_pct") is None
                    else float(profile["breakeven_buffer_pct"])
                ),
                trailing_activation_pct=(
                    None
                    if profile.get("trailing_activation_pct") is None
                    else float(profile["trailing_activation_pct"])
                ),
                trailing_distance_pct=(
                    None
                    if profile.get("trailing_distance_pct") is None
                    else float(profile["trailing_distance_pct"])
                ),
                tight_trailing_activation_pct=(
                    None
                    if profile.get("tight_trailing_activation_pct") is None
                    else float(profile["tight_trailing_activation_pct"])
                ),
                tight_trailing_distance_pct=(
                    None
                    if profile.get("tight_trailing_distance_pct") is None
                    else float(profile["tight_trailing_distance_pct"])
                ),
                default_leverage=(
                    None
                    if profile.get("default_leverage") is None
                    else int(profile["default_leverage"])
                ),
            )
        object.__setattr__(self, "coin_profiles", production_coin_profiles)
        if not self.interval:
            raise ValueError("interval must not be empty.")
        if not self.backtest_history_start_utc.strip():
            raise ValueError("backtest_history_start_utc must not be empty.")
        if not self.optimizer_history_start_utc.strip():
            raise ValueError("optimizer_history_start_utc must not be empty.")
        try:
            parsed_backtest_start = datetime.fromisoformat(
                self.backtest_history_start_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "backtest_history_start_utc must be a valid ISO-8601 datetime."
            ) from exc
        if parsed_backtest_start.tzinfo is None:
            parsed_backtest_start = parsed_backtest_start.replace(tzinfo=UTC)
        if parsed_backtest_start.astimezone(UTC) > datetime.now(tz=UTC):
            raise ValueError("backtest_history_start_utc must not be in the future.")
        try:
            parsed_optimizer_start = datetime.fromisoformat(
                self.optimizer_history_start_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "optimizer_history_start_utc must be a valid ISO-8601 datetime."
            ) from exc
        if parsed_optimizer_start.tzinfo is None:
            parsed_optimizer_start = parsed_optimizer_start.replace(tzinfo=UTC)
        if parsed_optimizer_start.astimezone(UTC) > datetime.now(tz=UTC):
            raise ValueError("optimizer_history_start_utc must not be in the future.")
        if self.start_capital <= 0:
            raise ValueError("start_capital must be positive.")
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be positive.")
        if self.taker_fee_pct < 0:
            raise ValueError("taker_fee_pct must be greater than or equal to 0.")
        if self.use_hmm_regime_filter and not self.hmm_allowed_regimes:
            raise ValueError(
                "hmm_allowed_regimes must not be empty when use_hmm_regime_filter is enabled."
            )
        if any(not str(regime).strip() for regime in self.hmm_allowed_regimes):
            raise ValueError("hmm_allowed_regimes must contain non-empty regime names.")
        if not 0.0 <= self.min_confidence_pct <= 100.0:
            raise ValueError("min_confidence_pct must be between 0 and 100.")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive.")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive.")
        if self.chandelier_period <= 1:
            raise ValueError("chandelier_period must be greater than 1.")
        if self.chandelier_multiplier <= 0.0:
            raise ValueError("chandelier_multiplier must be positive.")
        if self.trailing_activation_pct < 0:
            raise ValueError("trailing_activation_pct must be greater than or equal to 0.")
        if self.trailing_distance_pct < 0:
            raise ValueError("trailing_distance_pct must be greater than or equal to 0.")
        if self.min_leverage <= 0:
            raise ValueError("min_leverage must be positive.")
        if self.max_leverage < self.min_leverage:
            raise ValueError("max_leverage must be greater than or equal to min_leverage.")
        if not self.min_leverage <= self.default_leverage <= self.max_leverage:
            raise ValueError("default_leverage must be within the leverage range.")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive.")
        if not self.optimization_take_profit_pct_options:
            raise ValueError("optimization_take_profit_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.optimization_take_profit_pct_options):
            raise ValueError("optimization_take_profit_pct_options must contain only positive values.")
        if not self.optimization_stop_loss_pct_options:
            raise ValueError("optimization_stop_loss_pct_options must not be empty.")
        if any(value <= 0.0 for value in self.optimization_stop_loss_pct_options):
            raise ValueError("optimization_stop_loss_pct_options must contain only positive values.")
        if not self.optimization_trailing_activation_pct_options:
            raise ValueError("optimization_trailing_activation_pct_options must not be empty.")
        if any(value < 0.0 for value in self.optimization_trailing_activation_pct_options):
            raise ValueError(
                "optimization_trailing_activation_pct_options must contain only values >= 0."
            )
        if not self.optimization_trailing_distance_pct_options:
            raise ValueError("optimization_trailing_distance_pct_options must not be empty.")
        if any(value < 0.0 for value in self.optimization_trailing_distance_pct_options):
            raise ValueError(
                "optimization_trailing_distance_pct_options must contain only values >= 0."
            )
        if self.optimization_max_sample_profiles <= 0:
            raise ValueError("optimization_max_sample_profiles must be positive.")
        if self.random_search_samples <= 0:
            raise ValueError("random_search_samples must be positive.")
        if self.optimization_search_window_candles <= 0:
            raise ValueError("optimization_search_window_candles must be positive.")
        if self.optimization_validation_top_n <= 0:
            raise ValueError("optimization_validation_top_n must be positive.")
        if not self.universal_chandelier_period_options:
            raise ValueError("universal_chandelier_period_options must not be empty.")
        if any(value <= 1 for value in self.universal_chandelier_period_options):
            raise ValueError(
                "universal_chandelier_period_options must contain values greater than 1."
            )
        if not self.universal_chandelier_multiplier_options:
            raise ValueError("universal_chandelier_multiplier_options must not be empty.")
        if any(value <= 0.0 for value in self.universal_chandelier_multiplier_options):
            raise ValueError(
                "universal_chandelier_multiplier_options must contain only positive values."
            )
        if self.optimizer_progress_print_modulo <= 0:
            raise ValueError("optimizer_progress_print_modulo must be positive.")
        for symbol, profile in self.coin_profiles.items():
            if not symbol:
                raise ValueError("coin_profiles keys must not be empty.")
            if profile.stop_loss_pct is not None and profile.stop_loss_pct <= 0:
                raise ValueError(f"coin_profiles[{symbol}].stop_loss_pct must be positive.")
            if profile.take_profit_pct is not None and profile.take_profit_pct <= 0:
                raise ValueError(f"coin_profiles[{symbol}].take_profit_pct must be positive.")
            if (
                profile.trailing_activation_pct is not None
                and profile.trailing_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].trailing_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.trailing_distance_pct is not None
                and profile.trailing_distance_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].trailing_distance_pct must be greater than or equal to 0."
                )
            if (
                profile.breakeven_activation_pct is not None
                and profile.breakeven_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].breakeven_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.breakeven_buffer_pct is not None
                and profile.breakeven_buffer_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].breakeven_buffer_pct must be greater than or equal to 0."
                )
            if (
                profile.tight_trailing_activation_pct is not None
                and profile.tight_trailing_activation_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].tight_trailing_activation_pct must be greater than or equal to 0."
                )
            if (
                profile.tight_trailing_distance_pct is not None
                and profile.tight_trailing_distance_pct < 0
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].tight_trailing_distance_pct must be greater than or equal to 0."
                )
            if profile.default_leverage is not None and not (
                self.min_leverage <= profile.default_leverage <= self.max_leverage
            ):
                raise ValueError(
                    f"coin_profiles[{symbol}].default_leverage must be within the leverage range."
                )


@dataclass(frozen=True, slots=True)
class Settings:
    api: APISettings = field(default_factory=APISettings)
    live: LiveSettings = field(default_factory=LiveSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    trading: TradingSettings = field(default_factory=TradingSettings)

    def __post_init__(self) -> None:
        allowed_timeframes = set(self.api.timeframes)
        if self.live.default_interval not in allowed_timeframes:
            raise ValueError(
                f"live.default_interval must be one of api.timeframes: {self.live.default_interval}"
            )
        if self.trading.interval not in allowed_timeframes:
            raise ValueError(
                f"trading.interval must be one of api.timeframes: {self.trading.interval}"
            )
        for symbol, profile in self.trading.coin_profiles.items():
            if profile.interval is not None and profile.interval not in allowed_timeframes:
                raise ValueError(
                    f"coin_profiles[{symbol}].interval must be one of api.timeframes."
                )


def _replace_assignment_block(
    source: str,
    *,
    assignment_prefix: str,
    opening_char: str,
    closing_char: str,
    replacement_literal: str,
) -> str:
    assignment_start = source.find(assignment_prefix)
    if assignment_start < 0:
        raise ValueError(f"Could not locate assignment for {assignment_prefix!r}.")
    value_start = source.find(opening_char, assignment_start + len(assignment_prefix))
    if value_start < 0:
        raise ValueError(f"Could not locate opening delimiter for {assignment_prefix!r}.")

    depth = 0
    value_end = -1
    for index in range(value_start, len(source)):
        current_char = source[index]
        if current_char == opening_char:
            depth += 1
        elif current_char == closing_char:
            depth -= 1
            if depth == 0:
                value_end = index + 1
                break
    if value_end < 0:
        raise ValueError(f"Could not locate closing delimiter for {assignment_prefix!r}.")

    return (
        source[:assignment_start]
        + assignment_prefix
        + replacement_literal
        + source[value_end:]
    )


def _persist_runtime_coin_migration() -> None:
    config_path = Path(__file__).resolve()
    source_text = config_path.read_text(encoding="utf-8")
    formatter_kwargs = {"width": 120, "sort_dicts": False}

    source_text = _replace_assignment_block(
        source_text,
        assignment_prefix="ACTIVE_COINS: list[str] = ",
        opening_char="[",
        closing_char="]",
        replacement_literal=pformat(ACTIVE_COINS, **formatter_kwargs),
    )
    source_text = _replace_assignment_block(
        source_text,
        assignment_prefix="BACKTEST_ONLY_COINS: list[str] = ",
        opening_char="[",
        closing_char="]",
        replacement_literal=pformat(BACKTEST_ONLY_COINS, **formatter_kwargs),
    )
    source_text = _replace_assignment_block(
        source_text,
        assignment_prefix="DEFAULT_COIN_PROFILE_VALUES = ",
        opening_char="{",
        closing_char="}",
        replacement_literal=pformat(DEFAULT_COIN_PROFILE_VALUES, **formatter_kwargs),
    )
    source_text = _replace_assignment_block(
        source_text,
        assignment_prefix="DEFAULT_COIN_STRATEGIES = ",
        opening_char="{",
        closing_char="}",
        replacement_literal=pformat(DEFAULT_COIN_STRATEGIES, **formatter_kwargs),
    )
    source_text = _replace_assignment_block(
        source_text,
        assignment_prefix="DEFAULT_COIN_STRATEGY_PARAMS = ",
        opening_char="{",
        closing_char="}",
        replacement_literal=pformat(DEFAULT_COIN_STRATEGY_PARAMS, **formatter_kwargs),
    )
    config_path.write_text(source_text, encoding="utf-8")


def migrate_to_live(
    symbol: str,
    best_profile: dict[str, float] | None,
    *,
    min_confidence: float,
) -> None:
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must not be empty.")
    if not isinstance(best_profile, dict) or not best_profile:
        raise ValueError("best_profile must be a non-empty dict.")

    base_profile = dict(DEFAULT_COIN_PROFILE_VALUES.get("BTCUSDT", {}))
    base_profile.update(DEFAULT_COIN_PROFILE_VALUES.get(normalized_symbol, {}))
    base_strategy_params = dict(DEFAULT_COIN_STRATEGY_PARAMS.get(normalized_symbol, {}))

    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(fallback)

    take_profit_pct = _to_float(
        best_profile.get("take_profit_pct", base_profile.get("take_profit_pct", settings.trading.take_profit_pct)),
        settings.trading.take_profit_pct,
    )
    stop_loss_pct = _to_float(
        best_profile.get("stop_loss_pct", base_profile.get("stop_loss_pct", settings.trading.stop_loss_pct)),
        settings.trading.stop_loss_pct,
    )
    trailing_activation_pct = _to_float(
        best_profile.get(
            "trailing_activation_pct",
            base_profile.get("trailing_activation_pct", settings.trading.trailing_activation_pct),
        ),
        settings.trading.trailing_activation_pct,
    )
    trailing_distance_pct = _to_float(
        best_profile.get(
            "trailing_distance_pct",
            base_profile.get("trailing_distance_pct", settings.trading.trailing_distance_pct),
        ),
        settings.trading.trailing_distance_pct,
    )
    default_breakeven_activation = float(
        base_profile.get("breakeven_activation_pct")
        if base_profile.get("breakeven_activation_pct") is not None
        else 3.5
    )
    default_breakeven_buffer = float(
        base_profile.get("breakeven_buffer_pct")
        if base_profile.get("breakeven_buffer_pct") is not None
        else 0.2
    )
    default_tight_trailing_activation = float(
        base_profile.get("tight_trailing_activation_pct")
        if base_profile.get("tight_trailing_activation_pct") is not None
        else 8.0
    )
    default_tight_trailing_distance = float(
        base_profile.get("tight_trailing_distance_pct")
        if base_profile.get("tight_trailing_distance_pct") is not None
        else 0.3
    )
    breakeven_activation_pct = _to_float(
        best_profile.get(
            "breakeven_activation_pct",
            default_breakeven_activation,
        ),
        default_breakeven_activation,
    )
    breakeven_buffer_pct = _to_float(
        best_profile.get(
            "breakeven_buffer_pct",
            default_breakeven_buffer,
        ),
        default_breakeven_buffer,
    )
    tight_trailing_activation_pct = _to_float(
        best_profile.get(
            "tight_trailing_activation_pct",
            default_tight_trailing_activation,
        ),
        default_tight_trailing_activation,
    )
    tight_trailing_distance_pct = _to_float(
        best_profile.get(
            "tight_trailing_distance_pct",
            default_tight_trailing_distance,
        ),
        default_tight_trailing_distance,
    )

    strategy_name = str(base_strategy_name).strip() or str(settings.strategy.default_strategy_name)
    if any(key in best_profile for key in ("dual_thrust_period", "dual_thrust_k1", "dual_thrust_k2")):
        strategy_name = "dual_thrust"
    elif any(key in best_profile for key in ("frama_fast_period", "frama_slow_period")):
        strategy_name = "frama_cross"
    elif any(key in best_profile for key in ("ema_fast_period", "ema_slow_period")):
        strategy_name = "ema_cross_volume"

    if strategy_name == "dual_thrust":
        strategy_params = {
            "dual_thrust_period": _to_float(
                best_profile.get(
                    "dual_thrust_period",
                    base_strategy_params.get("dual_thrust_period", settings.strategy.dual_thrust_period),
                ),
                settings.strategy.dual_thrust_period,
            ),
            "dual_thrust_k1": _to_float(
                best_profile.get(
                    "dual_thrust_k1",
                    base_strategy_params.get("dual_thrust_k1", settings.strategy.dual_thrust_k1),
                ),
                settings.strategy.dual_thrust_k1,
            ),
            "dual_thrust_k2": _to_float(
                best_profile.get(
                    "dual_thrust_k2",
                    base_strategy_params.get("dual_thrust_k2", settings.strategy.dual_thrust_k2),
                ),
                settings.strategy.dual_thrust_k2,
            ),
        }
    elif strategy_name == "frama_cross":
        strategy_params = {
            "frama_fast_period": _to_float(
                best_profile.get(
                    "frama_fast_period",
                    base_strategy_params.get("frama_fast_period", settings.strategy.frama_fast_period),
                ),
                settings.strategy.frama_fast_period,
            ),
            "frama_slow_period": _to_float(
                best_profile.get(
                    "frama_slow_period",
                    base_strategy_params.get("frama_slow_period", settings.strategy.frama_slow_period),
                ),
                settings.strategy.frama_slow_period,
            ),
            "volume_multiplier": _to_float(
                best_profile.get(
                    "volume_multiplier",
                    base_strategy_params.get("volume_multiplier", settings.strategy.volume_multiplier),
                ),
                settings.strategy.volume_multiplier,
            ),
        }
    elif strategy_name == "ema_cross_volume":
        strategy_params = {
            "ema_fast_period": _to_float(
                best_profile.get(
                    "ema_fast_period",
                    base_strategy_params.get("ema_fast_period", settings.strategy.ema_fast_period),
                ),
                settings.strategy.ema_fast_period,
            ),
            "ema_slow_period": _to_float(
                best_profile.get(
                    "ema_slow_period",
                    base_strategy_params.get("ema_slow_period", settings.strategy.ema_slow_period),
                ),
                settings.strategy.ema_slow_period,
            ),
            "volume_multiplier": _to_float(
                best_profile.get(
                    "volume_multiplier",
                    base_strategy_params.get("volume_multiplier", settings.strategy.volume_multiplier),
                ),
                settings.strategy.volume_multiplier,
            ),
        }
    else:
        strategy_params = {
            str(key): float(value)
            for key, value in base_strategy_params.items()
            if str(key)
        }

    min_confidence_value = max(0.0, min(float(min_confidence), 100.0))
    deployed_profile = {
        "interval": str(base_profile.get("interval") or settings.trading.interval),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "breakeven_activation_pct": breakeven_activation_pct,
        "breakeven_buffer_pct": breakeven_buffer_pct,
        "trailing_activation_pct": trailing_activation_pct,
        "trailing_distance_pct": trailing_distance_pct,
        "tight_trailing_activation_pct": tight_trailing_activation_pct,
        "tight_trailing_distance_pct": tight_trailing_distance_pct,
        "default_leverage": int(
            base_profile.get("default_leverage", settings.trading.default_leverage)
            or settings.trading.default_leverage
        ),
        "min_confidence": min_confidence_value,
    }

    DEFAULT_COIN_PROFILE_VALUES[normalized_symbol] = deployed_profile
    DEFAULT_COIN_STRATEGIES[normalized_symbol] = strategy_name
    DEFAULT_COIN_STRATEGY_PARAMS[normalized_symbol] = dict(strategy_params)

    if normalized_symbol in BACKTEST_ONLY_COINS:
        BACKTEST_ONLY_COINS[:] = [item for item in BACKTEST_ONLY_COINS if item != normalized_symbol]
    if normalized_symbol not in ACTIVE_COINS:
        ACTIVE_COINS.append(normalized_symbol)
    ACTIVE_COINS[:] = list(dict.fromkeys(str(item).strip().upper() for item in ACTIVE_COINS if str(item).strip()))
    BACKTEST_ONLY_COINS[:] = list(
        dict.fromkeys(
            str(item).strip().upper()
            for item in BACKTEST_ONLY_COINS
            if str(item).strip() and str(item).strip().upper() not in ACTIVE_COINS
        )
    )

    global BACKTEST_BATCH_SYMBOLS
    BACKTEST_BATCH_SYMBOLS = tuple(dict.fromkeys((*ACTIVE_COINS, *BACKTEST_ONLY_COINS)))

    settings.strategy.coin_strategies[normalized_symbol] = strategy_name
    settings.strategy.coin_strategy_params[normalized_symbol] = dict(strategy_params)
    settings.trading.coin_profiles[normalized_symbol] = CoinProfileSettings(
        interval=str(deployed_profile["interval"]),
        stop_loss_pct=float(deployed_profile["stop_loss_pct"]),
        take_profit_pct=float(deployed_profile["take_profit_pct"]),
        breakeven_activation_pct=float(deployed_profile["breakeven_activation_pct"]),
        breakeven_buffer_pct=float(deployed_profile["breakeven_buffer_pct"]),
        trailing_activation_pct=float(deployed_profile["trailing_activation_pct"]),
        trailing_distance_pct=float(deployed_profile["trailing_distance_pct"]),
        tight_trailing_activation_pct=float(deployed_profile["tight_trailing_activation_pct"]),
        tight_trailing_distance_pct=float(deployed_profile["tight_trailing_distance_pct"]),
        default_leverage=int(deployed_profile["default_leverage"]),
    )
    object.__setattr__(settings.live, "available_symbols", tuple(ACTIVE_COINS))
    _persist_runtime_coin_migration()


settings = Settings()
USE_SETUP_GATE = settings.trading.use_setup_gate
COIN_STRATEGIES = settings.strategy.coin_strategies
COIN_STRATEGY_PARAMS = settings.strategy.coin_strategy_params

# Default scan strategies for automated multi-strategy runs
SCAN_STRATEGIES: tuple[str, ...] = tuple(DEFAULT_SCAN_STRATEGIES)

# Optional extra backtest-only strategies. Keep empty unless explicitly extended.
BACKTEST_ONLY_AVAILABLE_STRATEGIES: tuple[str, ...] = ()

# Optional optimizer strategy overrides per coin (research only).
BACKTEST_OPTIMIZER_COIN_STRATEGIES: dict[str, str] = {}

# Optional frozen winner presets for EMA Band Rejection 1h backtest mode.
EMA_BAND_REJECTION_1H_WINNERS: dict[str, dict[str, object]] = {}
