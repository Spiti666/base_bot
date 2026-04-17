# Silent Failure Audit

Generated at: 2026-04-16T21:51:48

## Summary

| severity | count |
|---|---:|
| kritisch | 29 |
| mittel | 18 |
| niedrig | 134 |
| niedrig (legacy) | 40 |

## Inventory

| severity | file | line | pattern | scope | code |
|---|---|---:|---|---|---|
| kritisch | `core/paper_trading/engine.py` | 131 | `except Exception:` | `def _emit_warning(self, warning_code: str, payload: dict[str, Any]) -> None:` | `except Exception:` |
| kritisch | `core/paper_trading/engine.py` | 150 | `except Exception:` | `def _emit_critical_state(self, state_code: str, payload: dict[str, Any]) -> None:` | `except Exception:` |
| kritisch | `core/paper_trading/engine.py` | 519 | `with suppress(Exception)` | `def run_historical_backtest(` | `with suppress(Exception):` |
| kritisch | `core/paper_trading/engine.py` | 525 | `with suppress(Exception)` | `def run_historical_backtest(` | `with suppress(Exception):` |
| kritisch | `core/paper_trading/engine.py` | 531 | `with suppress(Exception)` | `def run_historical_backtest(` | `with suppress(Exception):` |
| kritisch | `core/paper_trading/engine.py` | 581 | `except Exception:` | `def run_historical_backtest(` | `except Exception:` |
| kritisch | `core/paper_trading/engine.py` | 778 | `except Exception:` | `def run_historical_backtest(` | `except Exception:` |
| kritisch | `core/paper_trading/engine.py` | 1119 | `except Exception:` | `def _calculate_backtest_chandelier_levels(` | `except Exception:` |
| kritisch | `main_engine.py` | 6316 | `with suppress(Exception)` | `def run(self) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 6656 | `with suppress(Exception)` | `def _close_position_manually(self, symbol: str, exit_price: float) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 7139 | `with suppress(Exception)` | `def _on_ws_log(self, message: str) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 7569 | `with suppress(Exception)` | `def _fetch_recent_closed_trades(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 7837 | `with suppress(Exception)` | `def _evaluate_live_signal_direction(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 7847 | `with suppress(Exception)` | `def _evaluate_live_signal_direction(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 7863 | `with suppress(Exception)` | `def _evaluate_live_signal_direction(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8032 | `with suppress(Exception)` | `def _parse_json_payload(payload: object) -> dict[str, object]:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8634 | `with suppress(Exception)` | `def _initialize_meta_services(self) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8640 | `with suppress(Exception)` | `def _initialize_meta_services(self) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8646 | `with suppress(Exception)` | `def _initialize_meta_services(self) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8653 | `with suppress(Exception)` | `def _initialize_meta_services(self) -> None:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8685 | `with suppress(Exception)` | `def resolve_market_regime(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8771 | `with suppress(Exception)` | `def resolve_market_regime(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 8971 | `with suppress(Exception)` | `def resolve_market_regime(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9052 | `with suppress(Exception)` | `def build_entry_snapshot(` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9142 | `with suppress(Exception)` | `def _resolve_configured_take_profit_pct(self, symbol: str) -> float:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9148 | `with suppress(Exception)` | `def _resolve_configured_take_profit_pct(self, symbol: str) -> float:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9158 | `with suppress(Exception)` | `def _resolve_configured_stop_loss_pct(self, symbol: str) -> float:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9164 | `with suppress(Exception)` | `def _resolve_configured_stop_loss_pct(self, symbol: str) -> float:` | `with suppress(Exception):` |
| kritisch | `main_engine.py` | 9280 | `with suppress(Exception)` | `def _ensure_lifecycle_snapshot(self, trade: PaperTrade) -> dict[str, object]:` | `with suppress(Exception):` |
| mittel | `core/data/db.py` | 2014 | `with suppress(Exception)` | `def _to_optional_float(value: Any) -> float \| None:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 9593 | `with suppress(Exception)` | `def recompute_strategy_health(` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 9783 | `with suppress(Exception)` | `def recompute_strategy_health(` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 10873 | `with suppress(Exception)` | `def stop(self) -> None:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11212 | `except Exception:` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `except Exception:` |
| mittel | `main_engine.py` | 11298 | `except Exception:` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `except Exception:` |
| mittel | `main_engine.py` | 11301 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11304 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11307 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11709 | `with suppress(Exception)` | `def run(self) -> None:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 11739 | `with suppress(Exception)` | `def run(self) -> None:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12284 | `except Exception:` | `def _build_equity_curve_events(` | `except Exception:` |
| mittel | `main_engine.py` | 12403 | `except Exception:` | `def _build_regime_pnl_payload(` | `except Exception:` |
| mittel | `main_engine.py` | 12431 | `except Exception:` | `def _build_regime_pnl_payload(` | `except Exception:` |
| mittel | `main_engine.py` | 12599 | `with suppress(Exception)` | `def _profile_int(field_name: str, default_value: int) -> int:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12604 | `with suppress(Exception)` | `def _profile_float(field_name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12612 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| mittel | `main_engine.py` | 12668 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `config.py` | 1430 | `except Exception:` | `def _to_float(value: object, fallback: float) -> float:` | `except Exception:` |
| niedrig | `core/patterns/setup_gate.py` | 322 | `except Exception:` | `def _frame_cache_key(candles_df: Any) -> tuple[int, int, object, float \| None]:` | `except Exception:` |
| niedrig | `core/patterns/setup_gate.py` | 326 | `except Exception:` | `def _frame_cache_key(candles_df: Any) -> tuple[int, int, object, float \| None]:` | `except Exception:` |
| niedrig | `core/regime/hmm_regime_detector.py` | 23 | `except Exception:` | `def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401` | `except Exception:` |
| niedrig | `core/regime/hmm_regime_detector.py` | 188 | `except Exception:` | `def detect(self, candles_dataframe: pd.DataFrame) -> HMMRegimeDetectionResult:` | `except Exception:` |
| niedrig | `generate_summary_gesamt.py` | 195 | `except Exception:` | `def _parse_profile_dict(payload: str) -> dict[str, float]:` | `except Exception:` |
| niedrig | `gui.py` | 182 | `with suppress(Exception)` | `def _safe_int(value: object, default: int = 0) -> int:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 434 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 436 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 438 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 495 | `with suppress(Exception)` | `def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 518 | `with suppress(Exception)` | `def _thread_excepthook(args) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 529 | `except Exception:` | `def _thread_excepthook(args) -> None:` | `except Exception:` |
| niedrig | `gui.py` | 2513 | `with suppress(Exception)` | `def _parse_utc_date_only(value: object, fallback: QDate) -> QDate:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 2822 | `except Exception:` | `def _sanitize_backtest_strategy_name(symbol: str, strategy_name: str) -> str:` | `except Exception:` |
| niedrig | `gui.py` | 2896 | `except Exception:` | `def _build_backtest_batch_queue(` | `except Exception:` |
| niedrig | `gui.py` | 3606 | `except Exception:` | `def _flush_backtest_detail_log_buffer(self) -> None:` | `except Exception:` |
| niedrig | `gui.py` | 4434 | `with suppress(Exception)` | `def _handle_backtest_finished(self, result: dict) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 4919 | `with suppress(Exception)` | `def _resolve_report_leverage(self, symbol: str, value: object \| None) -> int:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5330 | `with suppress(Exception)` | `def _diag_int(field_name: str) -> int:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5335 | `with suppress(Exception)` | `def _diag_float(field_name: str) -> float:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5498 | `with suppress(Exception)` | `def _record_backtest_report_entry(self, result: dict) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5538 | `with suppress(Exception)` | `def _diag_int(field_name: str) -> int:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 5543 | `with suppress(Exception)` | `def _diag_float(field_name: str) -> float:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7176 | `with suppress(Exception)` | `def _refresh_meta_reports_view(` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7186 | `with suppress(Exception)` | `def _refresh_meta_reports_view(` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7242 | `with suppress(Exception)` | `def _parse_meta_json(raw_payload: str) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7523 | `with suppress(Exception)` | `def _sync_backtest_symbol_universe(self) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7525 | `with suppress(Exception)` | `def _sync_backtest_symbol_universe(self) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7940 | `with suppress(Exception)` | `def _rebuild_backtest_symbol_grid(self) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 7942 | `with suppress(Exception)` | `def _rebuild_backtest_symbol_grid(self) -> None:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8506 | `except Exception:` | `def _refresh_trade_readiness(self, *, force: bool = False) -> None:` | `except Exception:` |
| niedrig | `gui.py` | 8606 | `except Exception:` | `def _resolve_readiness_runtime_settings(self):` | `except Exception:` |
| niedrig | `gui.py` | 8677 | `with suppress(Exception)` | `def _is_live_profile_loaded_for_symbol(` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8690 | `with suppress(Exception)` | `def _is_live_leverage_confirmed_for_symbol(` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8695 | `with suppress(Exception)` | `def _is_live_leverage_confirmed_for_symbol(` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8835 | `with suppress(Exception)` | `def _param_int(name: str, default_value: int) -> int:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8840 | `with suppress(Exception)` | `def _param_float(name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8848 | `with suppress(Exception)` | `def _param_flag(name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8899 | `except Exception:` | `def _param_flag(name: str, default_value: bool = False) -> bool:` | `except Exception:` |
| niedrig | `gui.py` | 8903 | `with suppress(Exception)` | `def _param_flag(name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8927 | `with suppress(Exception)` | `def _param_flag(name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `gui.py` | 8978 | `except Exception:` | `def _compute_frama_trade_readiness(` | `except Exception:` |
| niedrig | `main_engine.py` | 538 | `with suppress(Exception)` | `def _summarize_profile_grid_bounds(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 600 | `with suppress(Exception)` | `def _history_seed_parse_utc_naive(value: object) -> datetime \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 612 | `with suppress(Exception)` | `def _history_seed_load_state() -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 724 | `with suppress(Exception)` | `def _resolve_optimizer_min_breakeven_activation_pct_for_strategy(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 749 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 752 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 765 | `with suppress(Exception)` | `def _profile_meets_breakeven_constraint(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 799 | `with suppress(Exception)` | `def _resolve_optimizer_min_total_trades_for_interval(interval: str) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 801 | `with suppress(Exception)` | `def _resolve_optimizer_min_total_trades_for_interval(interval: str) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 1881 | `except Exception:` | `def resolve_optimizer_strategy_for_symbol(` | `except Exception:` |
| niedrig | `main_engine.py` | 1913 | `with suppress(Exception)` | `def resolve_ema_band_rejection_1h_winner_profile(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 1962 | `except Exception:` | `def _coerce_backtest_strategy_name(` | `except Exception:` |
| niedrig | `main_engine.py` | 2270 | `with suppress(Exception)` | `def _clear_strategy_indicator_cache(candles_dataframe: pd.DataFrame \| None) -> None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2326 | `with suppress(Exception)` | `def _available_memory_bytes() -> int \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2346 | `with suppress(Exception)` | `class _MemoryStatusEx(ctypes.Structure):` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2540 | `with suppress(Exception)` | `def _normalize_dynamic_pct_series(values: object) -> list[float] \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2600 | `with suppress(Exception)` | `def _profile_int(field_name: str, default_value: int) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2605 | `with suppress(Exception)` | `def _profile_float(field_name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2613 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2758 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2829 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2837 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2851 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2855 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2859 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2866 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2873 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2880 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 2887 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3086 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3090 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3094 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3101 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3108 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3115 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3122 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3129 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3241 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3245 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3249 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3256 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3263 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3270 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3277 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3284 | `with suppress(Exception)` | `def _series_stats(series: pd.Series) -> tuple[float, float, float, int]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3404 | `except Exception:` | `def _temporary_strategy_profile(` | `except Exception:` |
| niedrig | `main_engine.py` | 3603 | `with suppress(Exception)` | `def _safe_int(value: object, default: int = -1) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3640 | `with suppress(Exception)` | `def _safe_int(value: object, default: int = -1) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3722 | `with suppress(Exception)` | `def _cap_profit_factor_for_sorting(value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3733 | `with suppress(Exception)` | `def _resolve_average_win_to_cost_ratio(summary: dict[str, float]) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3969 | `with suppress(Exception)` | `def _resolve_random_search_sample_cap() -> int \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 3985 | `with suppress(Exception)` | `def _resolve_optimizer_two_stage_strategy_names() -> set[str]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4002 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy(strategy_name: str) -> int \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4004 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy(strategy_name: str) -> int \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4009 | `with suppress(Exception)` | `def _resolve_optimizer_worker_cap_for_strategy(strategy_name: str) -> int \| None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4025 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4036 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4042 | `with suppress(Exception)` | `def _resolve_strategy_specific_positive_int(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4535 | `except Exception:` | `def _candles_to_dataframe(candles: Sequence[CandleRecord]) -> pd.DataFrame:` | `except Exception:` |
| niedrig | `main_engine.py` | 4576 | `except Exception:` | `def _candles_dataframe_to_worker_payload(` | `except Exception:` |
| niedrig | `main_engine.py` | 4669 | `except Exception:` | `def _extract_ohlcv_numpy_arrays_from_payload(` | `except Exception:` |
| niedrig | `main_engine.py` | 4771 | `with suppress(Exception)` | `def _warmup_numba_runtime(log_callback: Callable[[str], None] \| None = None) -> None:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4799 | `with suppress(Exception)` | `def _profile_float(field_name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4804 | `with suppress(Exception)` | `def _profile_int(field_name: str, default_value: int) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4812 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4890 | `with suppress(Exception)` | `def _profile_int(field_name: str, default_value: int) -> int:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4895 | `with suppress(Exception)` | `def _profile_float(field_name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 4903 | `with suppress(Exception)` | `def _profile_flag(field_name: str, default_value: bool = False) -> bool:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5107 | `with suppress(Exception)` | `def _compare_optimizer_final_summary_metrics(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5355 | `with suppress(Exception)` | `def _should_early_stop_profile(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5532 | `with suppress(Exception)` | `def _build_zero_fitness_summary(strategy_profile: OptimizationProfile) -> dict[str, float]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5664 | `with suppress(Exception)` | `def _profile_or_default(field_name: str, default_value: float) -> float:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 5894 | `with suppress(Exception)` | `def _finalize_profile_result(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 13373 | `except Exception:` | `def _run_profile_optimization(` | `except Exception:` |
| niedrig | `main_engine.py` | 13478 | `with suppress(Exception)` | `def _run_profile_optimization(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 13609 | `with suppress(Exception)` | `def _run_profile_optimization(` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 13745 | `except Exception:` | `def _run_profile_optimization(` | `except Exception:` |
| niedrig | `main_engine.py` | 14289 | `except Exception:` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `except Exception:` |
| niedrig | `main_engine.py` | 14305 | `with suppress(Exception)` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14328 | `except Exception:` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `except Exception:` |
| niedrig | `main_engine.py` | 14335 | `with suppress(Exception)` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14340 | `with suppress(Exception)` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14342 | `with suppress(Exception)` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14345 | `with suppress(Exception)` | `def _build_profile_task(profile: OptimizationProfile) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig | `main_engine.py` | 14619 | `except Exception:` | `def _emit_signal_cache_progress(` | `except Exception:` |
| niedrig | `main_engine.py` | 14634 | `except Exception:` | `def _emit_signal_cache_progress(` | `except Exception:` |
| niedrig | `main_engine.py` | 14665 | `except Exception:` | `def _emit_optimization_progress(` | `except Exception:` |
| niedrig | `main_engine.py` | 14680 | `except Exception:` | `def _emit_optimization_progress(` | `except Exception:` |
| niedrig | `strategies/python/ema_band_rejection.py` | 196 | `except Exception:` | `def build_ema_band_rejection_signal_frame(` | `except Exception:` |
| niedrig | `strategies/python/ema_band_rejection.py` | 664 | `except Exception:` | `def run_python_ema_band_rejection(candles_dataframe: Any) -> int:` | `except Exception:` |
| niedrig | `strategies/python/frama_cross.py` | 14 | `except Exception:` | `-` | `except Exception:  # pragma: no cover - graceful fallback when numba is unavailable` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 118 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 120 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 122 | `with suppress(Exception)` | `def _close_faulthandler_file() -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 179 | `with suppress(Exception)` | `def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 202 | `with suppress(Exception)` | `def _thread_excepthook(args) -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 213 | `except Exception:` | `def _thread_excepthook(args) -> None:` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 1384 | `except Exception:` | `def _fallback_scan_strategy_for_symbol(symbol: str) -> str \| None:` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 1416 | `except Exception:` | `def _build_backtest_batch_queue(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2611 | `with suppress(Exception)` | `def _resolve_report_leverage(self, symbol: str, value: object \| None) -> int:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2656 | `with suppress(Exception)` | `def _diag_int(field_name: str) -> int:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2661 | `with suppress(Exception)` | `def _diag_float(field_name: str) -> float:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2769 | `with suppress(Exception)` | `def _diag_int(field_name: str) -> int:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 2774 | `with suppress(Exception)` | `def _diag_float(field_name: str) -> float:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 4135 | `except Exception:` | `def _refresh_trade_readiness(self, *, force: bool = False) -> None:` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/gui.py` | 4274 | `except Exception:` | `def _compute_frama_trade_readiness(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 574 | `with suppress(Exception)` | `def _clear_strategy_indicator_cache(candles_dataframe: pd.DataFrame \| None) -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 608 | `with suppress(Exception)` | `def _available_memory_bytes() -> int \| None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 628 | `with suppress(Exception)` | `class _MemoryStatusEx(ctypes.Structure):` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 836 | `except Exception:` | `def _temporary_strategy_profile(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 1499 | `with suppress(Exception)` | `def _run_optimization_profile_worker(task: dict[str, object]) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 1501 | `with suppress(Exception)` | `def _run_optimization_profile_worker(task: dict[str, object]) -> dict[str, object]:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2391 | `with suppress(Exception)` | `def stop(self) -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2481 | `except Exception:` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2550 | `except Exception:` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2553 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2556 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2559 | `with suppress(Exception)` | `def run_smart_multi_strategy_sweep(self, *, coins: Sequence[str]):` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 2705 | `with suppress(Exception)` | `def run(self) -> None:` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3245 | `except Exception:` | `def _run_profile_optimization(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3432 | `except Exception:` | `def _evaluate_optimization_phase(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3448 | `with suppress(Exception)` | `def _evaluate_optimization_phase(` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3455 | `except Exception:` | `def _evaluate_optimization_phase(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3462 | `with suppress(Exception)` | `def _evaluate_optimization_phase(` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3467 | `with suppress(Exception)` | `def _evaluate_optimization_phase(` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3563 | `with suppress(Exception)` | `def _build_parallel_strategy_signal_cache(` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3568 | `with suppress(Exception)` | `def _build_parallel_strategy_signal_cache(` | `with suppress(Exception):` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3699 | `except Exception:` | `def _emit_signal_cache_progress(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3714 | `except Exception:` | `def _emit_signal_cache_progress(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3743 | `except Exception:` | `def _emit_optimization_progress(` | `except Exception:` |
| niedrig (legacy) | `archive/tools_legacy/backup_legacy/main_engine.py` | 3758 | `except Exception:` | `def _emit_optimization_progress(` | `except Exception:` |
