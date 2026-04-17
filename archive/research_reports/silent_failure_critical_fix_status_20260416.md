# Critical Silent-Failure Fix Status (2026-04-16)

Referenz: `archive/research_reports/silent_failure_critical_split_20260416.md`

## Ergebnis

- Von den 17 als "echte verbleibende Risiken" markierten Stellen wurden alle 17 gehärtet.
- Schwerpunkt blieb auf Fehlerbehandlung, Transparenz und Zustandskonsistenz.
- Keine fachliche Strategie-Logik (Signalregeln/Parameter) wurde geändert.

## Geschlossene Risiko-Stellen

1. `core/paper_trading/engine.py:131`
   - Warning-Callback-Fehler werden nicht nur geloggt, sondern zu `warning_callback_failed` als Critical-State eskaliert.
2. `core/paper_trading/engine.py:150`
   - Critical-Callback-Fehler laufen fail-closed mit `RuntimeError(critical_state_callback_failed:...)`.
3. `main_engine.py:7569`
   - `fetch_recent_closed_trades` ohne stilles Verschlucken; Degraded-Markierung + Sichtbarkeit.
4. `main_engine.py:7837`
   - Signal-Richtung aus Payload mit explizitem Exception-Handling und Warn-Kontext.
5. `main_engine.py:7847`
   - Dynamic-SL-Parsing mit explizitem Exception-Handling und Warn-Kontext.
6. `main_engine.py:7863`
   - Dynamic-TP-Parsing mit explizitem Exception-Handling und Warn-Kontext.
7. `main_engine.py:8634`
   - Meta-Init Health-Row-Load ohne `suppress`; Degraded-State bei Fehler.
8. `main_engine.py:8640`
   - Meta-Init Health-Recompute ohne `suppress`; Degraded-State bei Fehler.
9. `main_engine.py:8646`
   - Meta-Init Policy-Eval ohne `suppress`; Degraded-State bei Fehler.
10. `main_engine.py:8653`
    - Meta-Init Report-Generierung ohne `suppress`; Degraded-State bei Fehler.
11. `main_engine.py:8771`
    - HMM-Detection ohne stilles Verschlucken; Fehler sichtbar + Degraded-Markierung + stabiler Fallbackpfad.
12. `main_engine.py:8971`
    - Regime-Observation-Persistenz ohne `suppress`; Fehler sichtbar + Degraded-Markierung.
13. `main_engine.py:9142`
    - Production `take_profit_pct` Parse-Fehler sichtbar; kein lautloser Blind-Fallback.
14. `main_engine.py:9148`
    - Coin-Profile `take_profit_pct` Parse-Fehler sichtbar; kein lautloser Blind-Fallback.
15. `main_engine.py:9158`
    - Production `stop_loss_pct` Parse-Fehler sichtbar; kein lautloser Blind-Fallback.
16. `main_engine.py:9164`
    - Coin-Profile `stop_loss_pct` Parse-Fehler sichtbar; kein lautloser Blind-Fallback.
17. `main_engine.py:9280`
    - Lifecycle-Snapshot-Persistenz ohne `suppress`; Fehler sichtbar + Degraded-Markierung.

## Zusätzliche Transparenz/Robustheit

- Heartbeat-Report-Trigger (`_perform_heartbeat_check`) ebenfalls ohne stilles `suppress`, mit Degraded-Signal.
- Einmal-Warnungen (`_emit_warning_once`) für wiederkehrende invalid-config Fälle, um Log-Flut zu vermeiden.

## Checks

- `python -m py_compile main_engine.py core/paper_trading/engine.py core/paper_trading/persistence.py gui.py` erfolgreich.
- Callback-Verhalten smoke-getestet:
  - warning callback failure -> `warning_callback_failed` eskaliert.
  - critical callback failure -> fail-closed `RuntimeError`.
