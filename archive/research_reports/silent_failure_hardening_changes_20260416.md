# Silent Failure Hardening Changes (2026-04-16)

## Changed (Critical)

- `main_engine.py`
  - `run(...)`: uncaught runtime thread exceptions now log full context + traceback and are registered as operational errors.
  - `_on_candle_closed(...)`: critical sections (`update_positions`, `close_position_at_price`, `process_signal`) now:
    - log full context + traceback,
    - register operational error,
    - set symbol to `reconcile_required` with `position_state_unknown`,
    - fail-closed return (no blind continuation).
  - `_on_candle_update(...)`: intrabar `update_positions` critical failure now uses same fail-closed + reconcile flow.
  - `_close_position_manually(...)`: close persist errors now trigger full critical flow and do not fail silently.
  - `_evaluate_global_meta_guards(...)`: new symbol-level hard guard:
    - blocks entries with `block_reason=reconcile_required:<reason>`,
    - clears reconcile lock only after runtime/db trade-id consistency check passes.
  - Added explicit reconcile/state management:
    - `_set_symbol_reconcile_required(...)`
    - `_clear_symbol_reconcile_required(...)`
  - Added exception context logger:
    - `_emit_exception_context_log(...)` with stacktrace.
  - `_register_operational_error(...)`:
    - accepts `exception_stacktrace`,
    - tracks source/signature counters,
    - emits periodic top-error overview.
- `core/paper_trading/engine.py`
  - Added warning/critical callbacks and structured emits:
    - `_emit_warning(...)`
    - `_emit_critical_state(...)`
  - `process_signal(...)`:
    - trade insert failures trigger `order_submit_failed`,
    - post-insert refresh failures trigger `reconcile_required`.
  - `update_positions(...)`:
    - high-water-mark persist failures trigger `position_state_unknown` and raise.
    - exit persist failures trigger `exit_persist_failed` and raise.
  - `close_position_at_price(...)`:
    - persist failure now triggers `exit_persist_failed` and raises (no silent `None` success illusion).
  - `_resolve_mark_price_for_trade(...)`:
    - removed silent fallback (`except ...: pass` + `entry_price` fallback),
    - now emits critical `position_state_unknown` and raises fail-closed.
- `core/paper_trading/persistence.py`
  - `load_paper_trades(...)` now logs malformed/corrupt payload situations (no silent drops).
- `gui.py`
  - hard-block classifier extended with `reconcile_required:` so overview reflects real emergency blocks.

## Changed (Medium)

- `main_engine.py`
  - `_review_and_recompute_after_close(...)` no longer suppresses review/health/meta/report failures silently.
  - each service error now:
    - logs full context + traceback,
    - marks service degraded via `_mark_meta_service_degraded(...)`,
    - writes adaptation events.
  - service recovery clears degraded state with `_clear_meta_service_degraded(...)`.
  - `evaluate_meta_policy(...)` now includes degraded meta-service signal and can downgrade policy to `degraded` (without hard-blocking core execution).
  - `_insert_adaptation_event(...)` no longer silently swallows write failures; warning + traceback emitted.

## New Explicit Runtime States / Reasons

- `order_submit_failed`
- `position_state_unknown`
- `exit_persist_failed`
- `reconcile_required`

These are now propagated into adaptation/meta payloads and policy blocking reasons.

## Tolerated (Intentional, Non-Critical)

- Remaining `suppress(Exception)`/broad catches in:
  - optimizer/backtest orchestration paths,
  - analytics/report assembly helpers,
  - GUI-only presentation/clipboard/log-flush paths,
  - websocket task-cancel cleanup.
- These are tracked in the full audit inventory and are not used to confirm order/position state transitions.

## Full Inventory

- Full pattern inventory with classification is generated in:
  - `archive/research_reports/silent_failure_audit_20260416_215148.md`
