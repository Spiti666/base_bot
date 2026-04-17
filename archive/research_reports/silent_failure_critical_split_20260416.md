# Critical Silent-Failure Split (2026-04-16)

Basis: `archive/research_reports/silent_failure_audit_20260416_215148.md` (29 als `kritisch` markierte Fundstellen).

## Gruppe A — Echte verbleibende Risiken (17)

| file:line | Einstufung | Warum echtes Risiko |
|---|---|---|
| `core/paper_trading/engine.py:131` | Risiko | Callback-Fehler bei Warnungen kann Upstream-Handling aushebeln; aktuell nur Log, kein Retry/Fallback-Pfad. |
| `core/paper_trading/engine.py:150` | Risiko | Callback-Fehler bei Critical-State kann Reconcile-/Schutzreaktionen upstream verlieren; nur Log. |
| `main_engine.py:7569` | Risiko | DB-Read-Fehler bei closed trades wird zu `[]`; Candidate-/Policy-Metriken können stillschweigend verfälscht werden. |
| `main_engine.py:7837` | Risiko | Fehler in Signal-Payload-Parsing wird geschluckt; kann still auf Fallback-Signalpfad wechseln. |
| `main_engine.py:7847` | Risiko | Dynamic-SL aus Payload kann still ausfallen; Risiko-Parameter laufen dann implizit mit Defaults. |
| `main_engine.py:7863` | Risiko | Dynamic-TP aus Payload kann still ausfallen; Exit-Risikoprofil kann unbemerkt abweichen. |
| `main_engine.py:8634` | Risiko | Fehler beim Einlesen bestehender Health-Ziele beim Start werden unterdrückt; Meta-Initialzustand kann unvollständig bleiben. |
| `main_engine.py:8640` | Risiko | Fehler bei initialem `recompute_strategy_health` werden verschluckt; Health kann stale bleiben ohne klare Sichtbarkeit. |
| `main_engine.py:8646` | Risiko | Fehler bei initialem `evaluate_meta_policy` werden verschluckt; Policies können stale/inkonsistent starten. |
| `main_engine.py:8653` | Risiko | Fehler bei initialer Report-Generierung werden unterdrückt; Transparenz/Operator-Sicht leidet still. |
| `main_engine.py:8771` | Risiko | HMM-Detect-Ausnahmen werden komplett unterdrückt; Regime-Qualität sinkt ohne explizites Error-Signal. |
| `main_engine.py:8971` | Risiko | Persistenz von Regime-Observations kann still scheitern; Lern-/Health-Datenlücken ohne sofortige Sichtbarkeit. |
| `main_engine.py:9142` | Risiko | Invalides `take_profit_pct` im Production-Profil fällt lautlos auf Default zurück. |
| `main_engine.py:9148` | Risiko | Invalides `take_profit_pct` im Coin-Profil fällt lautlos auf Default zurück. |
| `main_engine.py:9158` | Risiko | Invalides `stop_loss_pct` im Production-Profil fällt lautlos auf Default zurück. |
| `main_engine.py:9164` | Risiko | Invalides `stop_loss_pct` im Coin-Profil fällt lautlos auf Default zurück. |
| `main_engine.py:9280` | Risiko | Persistenz von Lifecycle-Snapshot kann still fehlschlagen; Reviews/Health basieren dann auf partiellen Daten. |

## Gruppe B — Formal kritisch markiert, praktisch tolerierbar (12)

| file:line | Einstufung | Warum tolerierbar |
|---|---|---|
| `core/paper_trading/engine.py:519` | Tolerierbar | Backtest-Only Parameter-Coercion (`max_bars_in_trade`); bei Parse-Fehler bleibt sicherer `None`-Fallback. |
| `core/paper_trading/engine.py:525` | Tolerierbar | Backtest-Only Parameter-Coercion (`early_stop_max_trades`); kein Live-Positionskonsistenz-Risiko. |
| `core/paper_trading/engine.py:531` | Tolerierbar | Backtest-Only Parameter-Coercion (`early_stop_max_drawdown_pct`); kein Live-Risiko. |
| `core/paper_trading/engine.py:581` | Tolerierbar | Backtest-Index-Helfer (`first_signal_index`) mit konservativem Fallback `None`. |
| `core/paper_trading/engine.py:778` | Tolerierbar | Backtest-Statistik: unparsebare `exit_time` wird konservativ behandelt; kein Live-State betroffen. |
| `core/paper_trading/engine.py:1119` | Tolerierbar | Chandelier-Berechnung überspringt defekte Candle-Zeilen; robust gegen Datenmüll, kein stiller Live-Order-Fehlerpfad. |
| `main_engine.py:6316` | Tolerierbar | Bereits im Fatal-Exception-Pfad; zusätzlicher Operational-Error-Write ist Best-Effort, Thread stoppt ohnehin fail-closed. |
| `main_engine.py:6656` | Tolerierbar | Required-candle Rechenhilfe mit sicherem Fallback auf Strategy-Basiswert; keine State-Inkonsistenz. |
| `main_engine.py:7139` | Tolerierbar | Heartbeat ruft Report-Generator best-effort auf; Kern-Tradeflow bleibt getrennt. |
| `main_engine.py:8032` | Tolerierbar | JSON-Parse-Helfer mit defensivem `{}`-Fallback; dient Stabilität bei optionalen Payloads. |
| `main_engine.py:8685` | Tolerierbar | `observed_at`-Ableitung für Regime mit robustem Zeitfallback (`now`); kein Konsistenzbruch. |
| `main_engine.py:9052` | Tolerierbar | Snapshot-Metrik (`guard_component`) fallbackt auf 0.0; betrifft primär Annotation/Score, nicht Kern-Orderkonsistenz. |

## Kurzfazit

- Von 29 formal kritischen Fundstellen sind **17 echte verbleibende Risiken**.
- **12 sind praktisch tolerierbar**, weil sie Backtest-/Best-Effort-/Annotation-Pfade betreffen oder bereits fail-closed eingebettet sind.

## Priorisierte Restarbeit

1. Gruppe-A-Risiken in dieser Reihenfolge härten:
   1) Config-Risk-Fallbacks (`9142/9148/9158/9164`),
   2) Meta-Init-Suppressions (`8634/8640/8646/8653`),
   3) Regime-Detect/Persist (`8771/8971`),
   4) Dynamic-SL/TP-Parsing (`7837/7847/7863`),
   5) Callback-/Lifecycle-/closed-trade-Pfade (`131/150/7569/9280`).
