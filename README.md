# Base Bot

Stand: 2026-04-16

Trading-Workspace mit zwei operativen Bereichen:

1. Live-Runtime (Live-Marktdaten + Paper-Execution)
2. Analytics/Backtest (Simulation, Optimizer, Live-vs-Challenger, Meta-Reports)

Zentraler Einstieg ist die GUI (`gui.py`).

## TL;DR

- Produktive Konfigurationswahrheit ist `config.py`.
- Live handelt Paper-Trades auf Live-Marktdaten (keine echten Exchange-Orders).
- Meta-Bot laeuft in Modus B: beobachten/bewerten/warnen, nur weiche Eingriffe im Normalbetrieb.
- Harte Blockaden nur bei echten Notfaellen und harten Global-Guards.
- Exits (SL/TP/Trailing/Breakeven) laufen intrabar, Entries bleiben candle-basiert.

## Schnellstart

Voraussetzungen:

- Python 3.10+
- `PyQt6`, `numpy`, `pandas`, `duckdb`, `requests`, `websockets`, `hmmlearn`, `numba`
- optional: `polars`

Installation:

```bash
pip install PyQt6 numpy pandas duckdb requests websockets hmmlearn numba polars
```

Start:

```bash
python gui.py
```

## Konfigurationswahrheit

Nur `config.py` ist produktiv verbindlich fuer Routing, Strategien, Intervalle, Profile und Backtest-Universum.

Wichtige Schluessel:

- `ACTIVE_COINS` (Live-Universum)
- `BACKTEST_ONLY_COINS` (nur fuer Backtest)
- `BACKTEST_BATCH_SYMBOLS` (Union aus Live + Backtest-only)
- `PRODUCTION_PROFILE_REGISTRY` (Strategie/Intervall/Risk/Params pro Live-Coin)
- `MAX_BACKTEST_CANDLES` (Tail-Window-Grenze)

Hinweis:

- `config.py` kann aus `config_sections/*` initialisieren, ueberschreibt aber die finale produktive Wahrheit im laufenden Projekt.
- Legacy-/Archivdateien sind keine Runtime-Wahrheit.

## Aktueller Produktionsstand (aus `config.py`)

Live-Coins: 20

- `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `1000PEPEUSDT`, `1000SHIBUSDT`, `BNBUSDT`, `AVAXUSDT`, `NEARUSDT`, `DOTUSDT`, `ARBUSDT`, `SUIUSDT`, `LTCUSDT`, `BCHUSDT`, `LINKUSDT`, `TRXUSDT`, `FILUSDT`, `AAVEUSDT`

Backtest-only Coins: 40

- `TONUSDT`, `WLDUSDT`, `APTUSDT`, `FETUSDT`, `TAOUSDT`, `ENAUSDT`, `QNTUSDT`, `ETCUSDT`, `OPUSDT`, `SEIUSDT`, `INJUSDT`, `ATOMUSDT`, `POLUSDT`, `TIAUSDT`, `HBARUSDT`, `UNIUSDT`, `XLMUSDT`, `ALGOUSDT`, `ICPUSDT`, `RENDERUSDT`, `SANDUSDT`, `GALAUSDT`, `JUPUSDT`, `PENDLEUSDT`, `RUNEUSDT`, `CRVUSDT`, `ONDOUSDT`, `KASUSDT`, `IMXUSDT`, `PYTHUSDT`, `1000BONKUSDT`, `JASMYUSDT`, `LDOUSDT`, `STXUSDT`, `DYDXUSDT`, `ARUSDT`, `SUSHIUSDT`, `EGLDUSDT`, `COMPUSDT`, `THETAUSDT`

Backtest-Batch-Universum: 60

Live-Registry-Verteilung (aktuell):

- Strategien: `ema_band_rejection=12`, `frama_cross=6`, `dual_thrust=2`
- Intervalle: `1h=16`, `15m=4`

Recompute-Check:

```bash
python -c "from collections import Counter;import config;reg=config.PRODUCTION_PROFILE_REGISTRY;print('live',len(config.ACTIVE_COINS));print('backtest_only',len(config.BACKTEST_ONLY_COINS));print('batch',len(config.BACKTEST_BATCH_SYMBOLS));print('strategies',dict(Counter(v.get('strategy_name','') for v in reg.values())));print('intervals',dict(Counter(v.get('interval','') for v in reg.values())))"
```

Backtest-Universum als Liste ausgeben:

```bash
python -c "import config; print('\n'.join(config.BACKTEST_BATCH_SYMBOLS))"
```

## Strategie-Set

Im Kern verfuegbar (je nach Modus/Profil):

- `frama_cross`
- `dual_thrust`
- `ema_band_rejection`
- `ema_cross_volume`

## Live-Runtime Ablauf

Live-Pfad in `main_engine.py`:

1. Candle-Update (`_on_candle_update`) aktualisiert Markpreis und prueft intrabar sofortige Risk-Exits.
2. Candle-Close (`_on_candle_closed`) verarbeitet Strategie-Logik, Setup-Gate, Meta-Policy und Entry.
3. Entry wird als `paper_trades` Datensatz mit vollstaendigem Entry-Snapshot persistiert.
4. Geschlossene Trades triggern automatisch Review + Health-Recompute.

Kernkomponenten:

- WebSocket: `core/api/websocket.py`
- REST/History: `core/api/bitunix.py`, `core/data/history.py`
- Regime: `core/regime/hmm_regime_detector.py`
- Execution: `core/paper_trading/engine.py`
- Persistenz: `core/data/db.py`

## Runtime Startreihenfolge

Beim Start folgt die Runtime diesem Schema:

1. DB initialisieren und Migrationen aus `core/data/db.py` anwenden
2. Engine/Services starten (Regime, Review, Health, Meta)
3. GUI-Refresh-Timer und Runtime-Hooks aktivieren
4. History preload/sync und danach Live-WebSocket-Loop

## Intrabar-Exits (wichtig)

Ab sofort werden harte Risk-Exits intrabar auf normalen Updates ausgefuehrt, nicht erst bei Candle-Close:

- `INTRABAR_SL`
- `INTRABAR_TP`
- `INTRABAR_TRAILING`
- `INTRABAR_BREAKEVEN`

Candle-basierte Exits bleiben separat:

- z. B. `STRATEGY_EXIT`

Design:

- Entries bleiben candle-basiert.
- Intrabar-Exit hat Vorrang.
- Keine Doppel-Closings, da nur `OPEN`-Trades geschlossen werden.

## Datenbank und Migrationen

DB: lokale DuckDB (`data/paper_trading.duckdb` + `.wal`/`.shm`)

Migration in `core/data/db.py`:

- `CREATE TABLE IF NOT EXISTS`
- `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
- bestehende Daten bleiben erhalten

Haupttabellen:

- `candles`
- `paper_trades`
- `live_signals`
- `backtest_runs`
- `trade_reviews`
- `strategy_health`
- `regime_observations`
- `adaptation_log`

`paper_trades` wurde erweitert um:

- `timeframe`
- `regime_label_at_entry`
- `regime_confidence`
- `session_label`
- `signal_strength`
- `confidence_score`
- `atr_pct_at_entry`
- `volume_ratio_at_entry`
- `spread_estimate`
- `move_already_extended_pct`
- `entry_snapshot_json`
- `lifecycle_snapshot_json`
- `profile_version`
- `review_status`

## Entry Snapshot Engine

`build_entry_snapshot(...)` in `main_engine.py` erzeugt strukturierte Entry-Snapshots pro Trade.

Snapshot enthält u. a.:

- `symbol`, `strategy_name`, `timeframe`, `side`
- `entry_time`, `entry_price`, `leverage`, `profile_version`
- `regime_label_at_entry`, `regime_confidence`, `session_label`
- `signal_strength`, `confidence_score`
- `atr_pct_at_entry`, `volume_ratio_at_entry`, `spread_estimate`
- `move_already_extended_pct`
- `ema_distance_metrics`
- `recent_range_metrics`
- `setup_reason_code`
- `veto_flags`
- `feature_snapshot_version`

## Regime Engine

`resolve_market_regime(...)` liefert pro Symbol/Timeframe:

- `regime_label`
- `regime_confidence`
- `trend_bias`
- `volatility_state`
- `expansion_state`
- `liquidity_state`
- `session_label`
- `regime_features_json`

Label-Universum:

- `trend_clean_up`
- `trend_clean_down`
- `trend_exhausted`
- `range_balanced`
- `range_volatile`
- `breakout_transition`
- `compression_pre_breakout`
- `panic_expansion`
- `illiquid_noise`

Persistenz:

- Jede Beobachtung wird in `regime_observations` gespeichert.

HMM-Verhalten:

- Non-convergence wird nicht hart uebernommen.
- Bei Non-convergence wird letzter stabiler Regime-Zustand weiterverwendet (wenn vorhanden).
- Ohne stabilen Cache: heuristischer Fallback.
- `transmat_`-Warnungen werden mit Symbol/Timeframe/Window-Kontext geloggt.

## Trade Review Engine

Beim Wechsel von `OPEN` auf geschlossenen Status:

- `review_closed_trade(trade_id)` erstellt automatischen Review.
- Review wird in `trade_reviews` persistiert.
- Trade bekommt `review_status=REVIEWED_AUTO`.

Lifecycle-Snapshot (falls fehlt, wird vorher erzeugt):

- `max_favorable_excursion`
- `max_adverse_excursion`
- `bars_in_trade`
- `exit_trigger_category`
- `did_reach_partial_tp`
- `best_unrealized_pnl`
- `worst_unrealized_pnl`

Fehlerkatalog:

- `late_entry_after_expansion`
- `entry_into_exhaustion`
- `entry_against_regime`
- `entry_without_sufficient_confirmation`
- `stop_too_tight`
- `stop_too_wide`
- `tp_too_conservative`
- `tp_too_ambitious`
- `strategy_regime_mismatch`
- `strategy_coin_mismatch`
- `should_not_have_traded`

## Strategy Health Layer

`recompute_strategy_health(...)` aggregiert pro `symbol + strategy + timeframe`:

- `trades_count`, `pnl_sum`, `winrate`, `avg_pnl`, `avg_fees`
- `late_entry_rate`, `regime_mismatch_rate`, `avoidable_loss_rate`, `error_rate`
- `max_loss_streak`
- `health_score`
- `risk_multiplier`
- `state`
- `last_review_at`
- `window_size`

Status:

- `healthy`
- `degraded`
- `watchlist`
- `paused` (nur ueber harte Guards/Failsafe im Policy-Layer)

Low-Sample Verhalten (Modus B):

- `low_sample_observe_only` ist Info-Flag.
- Low-Sample alleine setzt nicht automatisch `watchlist`.
- Ohne negative Evidenz bleibt Zustand `healthy`, `allow_trade=yes`, `risk_multiplier=1.0`.

## Meta-Bot (Modus B)

Zentrale Policy:

- `evaluate_meta_policy(symbol, strategy_name, interval, current_context)`

Rueckgabe:

- `allow_trade`
- `risk_multiplier`
- `state`
- `block_reason`
- `warning_reason`
- `meta_flags`
- `effective_policy_json`

Modus-B Regeln im Normalbetrieb:

- `healthy`: normal handeln
- `watchlist`: normal handeln, markieren/warnen
- `degraded`: normal handeln, optional leicht reduziertes Risiko
- `paused`: nur bei echten Notfaellen oder harten Guards

`paper_only`:

- wird intern auf `watchlist` abgebildet und hat keine eigene Blockierwirkung.

`allow_trade`:

- Standardfall `yes`
- `no` nur bei harten aktiven Guards/Failsafe

## Harte Guards und Fail-Safe

Globale Leitplanken in `main_engine.py`:

- `META_MAX_DAILY_LOSS_USD` (5% Startkapital)
- `META_MAX_SYMBOL_LOSS_STREAK` (5)
- `META_MAX_STRATEGY_DRAWDOWN_PCT` (12%)
- `META_MAX_OPEN_POSITIONS_GUARD`
- `META_MAX_CORRELATED_RISK` (4)
- `META_MAX_PAUSE_EVENTS_PER_DAY` (12)
- `META_COOLDOWN_AFTER_PAUSE_MINUTES` (90)

Harte Fail-Safe Gruende:

- `positions_desync`
- `duplicate_entries_detected`
- `missing_exit_control`
- `repeated_operational_error` (nur echte Runtime-Exceptions, nicht Heartbeat)

Operational Error Kette:

- Dedup: `META_OPERATIONAL_ERROR_DEDUP_SECONDS=120`
- Window: `META_OPERATIONAL_ERROR_WINDOW_MINUTES=30`
- Threshold: `META_OPERATIONAL_ERROR_FAILSAFE_THRESHOLD=5`
- Breitenbedingung: mind. 2 Symbole oder 2 Signaturen
- Auto-Recovery fuer `repeated_operational_error`: `META_FAILSAFE_AUTO_RECOVER_MINUTES=60`

Heartbeat:

- Heartbeat-Hinweise sind warning-only und eskalieren nicht direkt in Fail-Safe.

## Meta Reports

Funktionen:

- `build_daily_meta_report(...)`
- `build_weekly_meta_report(...)`
- `build_learning_log_entry(...)`

Ausgabe:

- JSON + Markdown + TXT
- Zielordner: `data/meta_reports/`

Dateimuster:

- `daily_meta_report_*.json|md|txt`
- `weekly_meta_report_*.json|md|txt`

## GUI (PyQt6)

Bestehende Tabs bleiben erhalten:

- Trading
- Analytics/Backtest
- Live-vs-Challenger
- Positions/Logs

Neuer Meta-Bot Bereich:

- Meta Overview
- Strategy Health
- Trade Reviews
- Adaptation Log
- Learning Reports

Backtest-Coin-Selection Verbesserungen:

- Vertikal scrollbar
- Horizontaler Scroll deaktiviert
- Reihen-Selektion per farbigem Button (`R1`, `R2`, ...)
- Row-Toggle selektiert/deselektiert komplette Reihe
- Splitter-Logik bevorzugt Breite der Coin-Selektion, Log-Panel wird bei Bedarf schmaler
- Auch bei laufendem Backtest bleibt Scroll in der Selection nutzbar

Meta Overview:

- `active_blocks` zaehlt nur echte harte Blockgruende
- `global_risk_state` basiert auf denselben Policy-Zustaenden wie die Policy-Tabelle

## Backtest und History-Sync

History-Laden ueber `core/data/history.py`:

- REST Page-Limit auf `200` abgestimmt (`_HISTORY_KLINE_REQUEST_LIMIT=200`)
- vermeidet fruehes Stoppen bei kurzen API-Pages
- Abort-Callback (`should_abort`) wird durch den Sync-Pfad propagiert

Backtest-History Modi:

- Fester Date-Range (`date_range_window`) wenn konfiguriert
- Sonst Tail-Window (`latest_tail_window`, bis `MAX_BACKTEST_CANDLES`)
- Loader-Modus und effektives Fenster werden im Log ausgegeben

Hinweis:

- Exchange-Historie kann je Coin/Intervall begrenzt sein; fehlende aeltere Daten sind dann API-seitig.

## Reports / Hilfsskripte

- `archive/backtest_compact_summaries/backtest_compact_summary_*.txt`
- `archive/research_reports/live_vs_challenger_summary_*.txt`
- `summary_gesamt.txt` (abgeleitet)
- `generate_summary_gesamt.py`
- `tools/meta_bot_smoke_checks.py`

Smoke-Checks:

```bash
python tools/meta_bot_smoke_checks.py
```

## Troubleshooting

`Model is not converging...`:

- HMM-Fit hat in einem Window keine Verbesserung erreicht.
- Runtime nutzt dann stabilen Regime-Fallback statt harter Neubewertung.

`Some rows of transmat_ have zero sum...`:

- Mindestens ein HMM-State hat im Trainingsfenster keine beobachteten Transitionen.
- Das ist ein struktureller Warnhinweis, kein sofortiger Runtime-Abbruch.
- Der Bot loggt dazu Symbol/Timeframe/Window-Kontext.

Wenn Trades unerwartet blockiert erscheinen:

- `Meta Bot -> Current Meta Policy` und `Adaptation Log` pruefen.
- Besonders `block_reason` und `source` auswerten.
- `active_blocks` zaehlt nur harte aktive Sperren.

## Projektstruktur (Kurzuebersicht)

- `gui.py` - GUI Entrypoint + komplette Desktop-Oberflaeche
- `main_engine.py` - Live/Backtest Runtime, Regime, Review, Health, Meta Policy, Reporting
- `config.py` - produktive Runtime-Konfiguration
- `core/data/db.py` - DuckDB + Migrationen + Dataclasses + Fetch/Upsert APIs
- `core/data/history.py` - Historien-Sync
- `core/paper_trading/engine.py` - Entry/Exit-Execution, Intrabar-Exit-Logik
- `core/regime/hmm_regime_detector.py` - HMM-Regime-Erkennung inkl. Warn-/Convergence-Infos
- `strategies/python/` - Strategien
- `engine/` - Backtest-Kern

## Wichtige Hinweise

- Live-Runtime ist Paper-Trading auf Live-Marktdaten.
- Keine echte Exchange-Orderplatzierung in diesem Flow.
- `summary_gesamt.txt` ist abgeleitet und sollte ueber `generate_summary_gesamt.py` erzeugt werden.
