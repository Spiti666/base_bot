# Silent Failure Audit

Generated at: 2026-04-16T22:11:32

## Summary

| severity | count |
|---|---:|
| kritisch | 5 |
| mittel | 18 |
| niedrig | 141 |
| niedrig (legacy) | 40 |

## Remaining Critical (Updated)

| file | line | pattern | scope | code |
|---|---:|---|---|---|
| `main_engine.py` | 6317 | `with suppress(Exception)` | `def run` | `with suppress(Exception):` |
| `main_engine.py` | 6657 | `with suppress(Exception)` | `def _on_candle_closed` | `with suppress(Exception):` |
| `main_engine.py` | 8842 | `with suppress(Exception)` | `def resolve_market_regime` | `with suppress(Exception):` |
| `main_engine.py` | 9243 | `with suppress(Exception)` | `def build_entry_snapshot` | `with suppress(Exception):` |
| `core/paper_trading/engine.py` | 156 | `except Exception:` | `def _emit_warning` | `except Exception:` |

## Critical Split

### Echte verbleibende Risiken

_Keine._

### Formal kritisch, praktisch tolerierbar

| file | line | reason |
|---|---:|---|
| `main_engine.py` | 6317 | Best-effort Operational-Error-Register im fatalen Run-Abbruch; Kern stoppt bereits fail-closed. |
| `main_engine.py` | 6657 | Parameter-Coercion fuer required candle count; faellt auf sicheren Basiswert zurueck. |
| `main_engine.py` | 8842 | Observed-time Ableitung fuer Regime-Metadaten; bei Fehler fallback auf UTC now. |
| `main_engine.py` | 9243 | Snapshot-Scoring-Metrik (guard_component); betrifft Annotation, nicht Order-/Positionskonsistenz. |
| `core/paper_trading/engine.py` | 156 | Callback-Guard im Warning-Pfad; Fehler wird bereits mit Stacktrace geloggt und eskaliert. |

## Inventory

| severity | file | line | pattern | scope | code |
|---|---|---:|---|---|---|
| niedrig | `config.py` | 1430 | `except Exception:` | `def _to_float` | `except Exception:` |
| niedrig | `generate_summary_gesamt.py` | 195 | `except Exception:` | `def _parse_profile_dict` | `except Exception:` |
| niedrig | `gui.py` | 182 | `with suppress(Exception)` | `def _safe_int` | `with suppress(Exception):` |
| niedrig | `gui.py` | 434 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig | `gui.py` | 436 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig | `gui.py` | 438 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig | `gui.py` | 495 | `with suppress(Exception)` | `def _sys_excepthook` | `with suppress(Exception):` |
| niedrig | `gui.py` | 518 | `with suppress(Exception)` | `def _thread_excepthook` | `with suppress(Exception):` |
| niedrig | `gui.py` | 529 | `except Exception:` | `def _thread_excepthook` | `except Exception:` |
| niedrig | `gui.py` | 2513 | `with suppress(Exception)` | `def _parse_utc_date_only` | `with suppress(Exception):` |
| niedrig | `gui.py` | 2822 | `except Exception:` | `def _sanitize_backtest_strategy_name` | `except Exception:` |
| niedrig | `gui.py` | 2896 | `except Exception:` | `def _build_backtest_batch_queue` | `except Exception:` |
| niedrig | `gui.py` | 3606 | `except Exception:` | `def _flush_backtest_detail_log_buffer` | `except Exception:` |
| niedrig | `gui.py` | 4434 | `with suppress(Exception)` | `def _handle_backtest_finished` | `with suppress(Exception):` |
| niedrig | `gui.py` | 4919 | `with suppress(Exception)` | `def _resolve_report_leverage` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5330 | `with suppress(Exception)` | `def _diag_int` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5335 | `with suppress(Exception)` | `def _diag_float` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5498 | `with suppress(Exception)` | `def _record_backtest_report_entry` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5538 | `with suppress(Exception)` | `def _diag_int` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5543 | `with suppress(Exception)` | `def _diag_float` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7176 | `with suppress(Exception)` | `def _refresh_meta_reports_view` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7186 | `with suppress(Exception)` | `def _refresh_meta_reports_view` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7242 | `with suppress(Exception)` | `def _parse_meta_json` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7523 | `with suppress(Exception)` | `def _sync_backtest_symbol_universe` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7525 | `with suppress(Exception)` | `def _sync_backtest_symbol_universe` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7940 | `with suppress(Exception)` | `def _rebuild_backtest_symbol_grid` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7942 | `with suppress(Exception)` | `def _rebuild_backtest_symbol_grid` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8506 | `except Exception:` | `def _refresh_trade_readiness` | `except Exception:` |
| niedrig | `gui.py` | 8606 | `except Exception:` | `def _resolve_readiness_runtime_settings` | `except Exception:` |
| niedrig | `gui.py` | 8677 | `with suppress(Exception)` | `def _is_live_profile_loaded_for_symbol` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8690 | `with suppress(Exception)` | `def _is_live_leverage_confirmed_for_symbol` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8695 | `with suppress(Exception)` | `def _is_live_leverage_confirmed_for_symbol` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8835 | `with suppress(Exception)` | `def _param_int` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8840 | `with suppress(Exception)` | `def _param_float` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8848 | `with suppress(Exception)` | `def _param_flag` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8899 | `except Exception:` | `def _param_flag` | `except Exception:` |
| niedrig | `gui.py` | 8903 | `with suppress(Exception)` | `def _param_flag` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8927 | `with suppress(Exception)` | `def _param_flag` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8978 | `except Exception:` | `def _compute_frama_trade_readiness` | `except Exception:` |
| niedrig | `main_engine.py` | 538 | `with suppress(Exception)` | `def _summarize_profile_grid_bounds` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 600 | `with suppress(Exception)` | `def _history_seed_parse_utc_naive` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 612 | `with suppress(Exception)` | `def _history_seed_load_state` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 724 | `with suppress(Exception)` | `def _resolve_optimizer_min_breakeven_activation_pct_for_strategy` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 749 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 752 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 765 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 799 | `with suppress(Exception)` | `def _resolve_optimizer_min_total_trades_for_interval` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 801 | `with suppress(Exception)` | `def _resolve_optimizer_min_total_trades_for_interval` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 1881 | `except Exception:` | `def resolve_optimizer_strategy_for_symbol` | `except Exception:` |
| niedrig | `main_engine.py` | 1913 | `with suppress(Exception)` | `def resolve_ema_band_rejection_1h_winner_profile` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 1962 | `except Exception:` | `def _coerce_backtest_strategy_name` | `except Exception:` |
| niedrig | `main_engine.py` | 2270 | `with suppress(Exception)` | `def _clear_strategy_indicator_cache` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2326 | `with suppress(Exception)` | `def _available_memory_bytes` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2346 | `with suppress(Exception)` | `def _MemoryStatusEx` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2540 | `with suppress(Exception)` | `def _normalize_dynamic_pct_series` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2600 | `with suppress(Exception)` | `def _profile_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2605 | `with suppress(Exception)` | `def _profile_float` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2613 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2758 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2829 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2837 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2851 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2855 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2859 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2866 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2873 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2880 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2887 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3086 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3090 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3094 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3101 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3108 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3115 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3122 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3129 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3241 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3245 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3249 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3256 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3263 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3270 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3277 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3284 | `with suppress(Exception)` | `def _series_stats` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3404 | `except Exception:` | `def _temporary_strategy_profile` | `except Exception:` |
| niedrig | `main_engine.py` | 3603 | `with suppress(Exception)` | `def _safe_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3640 | `with suppress(Exception)` | `def _safe_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3722 | `with suppress(Exception)` | `def _cap_profit_factor_for_sorting` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3733 | `with suppress(Exception)` | `def _resolve_average_win_to_cost_ratio` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3969 | `with suppress(Exception)` | `def _resolve_random_search_sample_cap` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3985 | `with suppress(Exception)` | `def _resolve_optimizer_two_stage_strategy_names` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4002 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4004 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4009 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4025 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4036 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4042 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4535 | `except Exception:` | `def _candles_to_dataframe` | `except Exception:` |
| niedrig | `main_engine.py` | 4576 | `except Exception:` | `def _candles_dataframe_to_worker_payload` | `except Exception:` |
| niedrig | `main_engine.py` | 4669 | `except Exception:` | `def _extract_ohlcv_numpy_arrays_from_payload` | `except Exception:` |
| niedrig | `main_engine.py` | 4771 | `with suppress(Exception)` | `def _warmup_numba_runtime` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4799 | `with suppress(Exception)` | `def _profile_float` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4804 | `with suppress(Exception)` | `def _profile_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4812 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4890 | `with suppress(Exception)` | `def _profile_int` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4895 | `with suppress(Exception)` | `def _profile_float` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4903 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5107 | `with suppress(Exception)` | `def _compare_optimizer_final_summary_metrics` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5355 | `with suppress(Exception)` | `def _should_early_stop_profile` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5532 | `with suppress(Exception)` | `def _build_zero_fitness_summary` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5664 | `with suppress(Exception)` | `def _profile_or_default` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5894 | `with suppress(Exception)` | `def _finalize_profile_result` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 6317 | `with suppress(Exception)` | `def run` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 6657 | `with suppress(Exception)` | `def _on_candle_closed` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 8118 | `with suppress(Exception)` | `def _parse_json_payload` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8842 | `with suppress(Exception)` | `def resolve_market_regime` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9243 | `with suppress(Exception)` | `def build_entry_snapshot` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 9886 | `with suppress(Exception)` | `def recompute_strategy_health` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 10076 | `with suppress(Exception)` | `def recompute_strategy_health` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11166 | `with suppress(Exception)` | `def stop` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11505 | `except Exception:` | `def run_smart_multi_strategy_sweep` | `except Exception:` |
| mittel | `main_engine.py` | 11591 | `except Exception:` | `def run_smart_multi_strategy_sweep` | `except Exception:` |
| mittel | `main_engine.py` | 11594 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11597 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11600 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12002 | `with suppress(Exception)` | `def run` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12032 | `with suppress(Exception)` | `def run` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12577 | `except Exception:` | `def _build_equity_curve_events` | `except Exception:` |
| mittel | `main_engine.py` | 12696 | `except Exception:` | `def _build_regime_pnl_payload` | `except Exception:` |
| mittel | `main_engine.py` | 12724 | `except Exception:` | `def _build_regime_pnl_payload` | `except Exception:` |
| mittel | `main_engine.py` | 12892 | `with suppress(Exception)` | `def _profile_int` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12897 | `with suppress(Exception)` | `def _profile_float` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12905 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12961 | `with suppress(Exception)` | `def _profile_flag` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 13666 | `except Exception:` | `def _run_profile_optimization` | `except Exception:` |
| niedrig | `main_engine.py` | 13771 | `with suppress(Exception)` | `def _run_profile_optimization` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 13902 | `with suppress(Exception)` | `def _run_profile_optimization` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14038 | `except Exception:` | `def _run_profile_optimization` | `except Exception:` |
| niedrig | `main_engine.py` | 14582 | `except Exception:` | `def _build_profile_task` | `except Exception:` |
| niedrig | `main_engine.py` | 14598 | `with suppress(Exception)` | `def _build_profile_task` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14621 | `except Exception:` | `def _build_profile_task` | `except Exception:` |
| niedrig | `main_engine.py` | 14628 | `with suppress(Exception)` | `def _build_profile_task` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14633 | `with suppress(Exception)` | `def _build_profile_task` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14635 | `with suppress(Exception)` | `def _build_profile_task` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14638 | `with suppress(Exception)` | `def _build_profile_task` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14912 | `except Exception:` | `def _emit_signal_cache_progress` | `except Exception:` |
| niedrig | `main_engine.py` | 14927 | `except Exception:` | `def _emit_signal_cache_progress` | `except Exception:` |
| niedrig | `main_engine.py` | 14958 | `except Exception:` | `def _emit_optimization_progress` | `except Exception:` |
| niedrig | `main_engine.py` | 14973 | `except Exception:` | `def _emit_optimization_progress` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 118 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 120 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 122 | `with suppress(Exception)` | `def _close_faulthandler_file` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 179 | `with suppress(Exception)` | `def _sys_excepthook` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 202 | `with suppress(Exception)` | `def _thread_excepthook` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 213 | `except Exception:` | `def _thread_excepthook` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 1384 | `except Exception:` | `def _fallback_scan_strategy_for_symbol` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 1416 | `except Exception:` | `def _build_backtest_batch_queue` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2611 | `with suppress(Exception)` | `def _resolve_report_leverage` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2656 | `with suppress(Exception)` | `def _diag_int` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2661 | `with suppress(Exception)` | `def _diag_float` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2769 | `with suppress(Exception)` | `def _diag_int` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2774 | `with suppress(Exception)` | `def _diag_float` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 4135 | `except Exception:` | `def _refresh_trade_readiness` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 4274 | `except Exception:` | `def _compute_frama_trade_readiness` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 574 | `with suppress(Exception)` | `def _clear_strategy_indicator_cache` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 608 | `with suppress(Exception)` | `def _available_memory_bytes` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 628 | `with suppress(Exception)` | `def _MemoryStatusEx` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 836 | `except Exception:` | `def _temporary_strategy_profile` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 1499 | `with suppress(Exception)` | `def _run_optimization_profile_worker` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 1501 | `with suppress(Exception)` | `def _run_optimization_profile_worker` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2391 | `with suppress(Exception)` | `def stop` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2481 | `except Exception:` | `def run_smart_multi_strategy_sweep` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2550 | `except Exception:` | `def run_smart_multi_strategy_sweep` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2553 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2556 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2559 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2705 | `with suppress(Exception)` | `def run` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3245 | `except Exception:` | `def _run_profile_optimization` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3432 | `except Exception:` | `def _evaluate_optimization_phase` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3448 | `with suppress(Exception)` | `def _evaluate_optimization_phase` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3455 | `except Exception:` | `def _evaluate_optimization_phase` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3462 | `with suppress(Exception)` | `def _evaluate_optimization_phase` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3467 | `with suppress(Exception)` | `def _evaluate_optimization_phase` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3563 | `with suppress(Exception)` | `def _build_parallel_strategy_signal_cache` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3568 | `with suppress(Exception)` | `def _build_parallel_strategy_signal_cache` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3699 | `except Exception:` | `def _emit_signal_cache_progress` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3714 | `except Exception:` | `def _emit_signal_cache_progress` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3743 | `except Exception:` | `def _emit_optimization_progress` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3758 | `except Exception:` | `def _emit_optimization_progress` | `except Exception:` |
| mittel | `core/data/db.py` | 2014 | `with suppress(Exception)` | `def _to_optional_float` | `with suppress(Exception):` |
| kritisch | `core/paper_trading/engine.py` | 156 | `except Exception:` | `def _emit_warning` | `except Exception:` |
| niedrig | `core/paper_trading/engine.py` | 550 | `with suppress(Exception)` | `def run_historical_backtest` | `with suppress(Exception):` |
| niedrig | `core/paper_trading/engine.py` | 556 | `with suppress(Exception)` | `def run_historical_backtest` | `with suppress(Exception):` |
| niedrig | `core/paper_trading/engine.py` | 562 | `with suppress(Exception)` | `def run_historical_backtest` | `with suppress(Exception):` |
| niedrig | `core/paper_trading/engine.py` | 612 | `except Exception:` | `def run_historical_backtest` | `except Exception:` |
| niedrig | `core/paper_trading/engine.py` | 809 | `except Exception:` | `def run_historical_backtest` | `except Exception:` |
| niedrig | `core/paper_trading/engine.py` | 1150 | `except Exception:` | `def _calculate_backtest_chandelier_levels` | `except Exception:` |
| niedrig | `core/patterns/setup_gate.py` | 322 | `except Exception:` | `def _frame_cache_key` | `except Exception:` |
| niedrig | `core/patterns/setup_gate.py` | 326 | `except Exception:` | `def _frame_cache_key` | `except Exception:` |
| niedrig | `core/regime/hmm_regime_detector.py` | 23 | `except Exception:` | `def filter` | `except Exception:` |
| niedrig | `core/regime/hmm_regime_detector.py` | 188 | `except Exception:` | `def detect` | `except Exception:` |
| niedrig | `strategies/python/ema_band_rejection.py` | 196 | `except Exception:` | `def build_ema_band_rejection_signal_frame` | `except Exception:` |
| niedrig | `strategies/python/ema_band_rejection.py` | 664 | `except Exception:` | `def run_python_ema_band_rejection` | `except Exception:` |
| niedrig | `strategies/python/frama_cross.py` | 14 | `except Exception:` | `def -` | `except Exception:  # pragma: no cover - graceful fallback when numba is unavailable` |
