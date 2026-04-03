# Base Bot

Stand: 2026-04-03

Trading-Workspace mit zwei operativen Bereichen:

1. Live-Bot (Live-Marktdaten + Paper-Execution)
2. Analytics/Backtest (Simulation, Optimizer, Live-vs-Challenger-Vergleich)

Zentraler Einstieg ist die GUI (`gui.py`).

## Aktueller Produktionsstand

Die produktive Wahrheit liegt in `config.py`:

- `ACTIVE_COINS` (aktuell 20 Coins)
- `PRODUCTION_PROFILE_REGISTRY` (Strategie, Intervall, Strategie-Parameter, Risk je Coin)

Aktive Live-Coins (20):

- `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `1000PEPEUSDT`, `1000SHIBUSDT`, `BNBUSDT`, `AVAXUSDT`, `NEARUSDT`, `DOTUSDT`, `ARBUSDT`, `SUIUSDT`, `LTCUSDT`, `BCHUSDT`, `LINKUSDT`, `TRXUSDT`, `FILUSDT`, `AAVEUSDT`

Strategieverteilung in der Live-Registry:

- `ema_band_rejection`: 8
- `dual_thrust`: 7
- `frama_cross`: 4
- `ema_cross_volume`: 1

Intervallverteilung:

- `1h`: 16
- `15m`: 2
- `5m`: 2

## Aktive Strategien

Aktiv genutzter Kern (Live + Backtest-Ausfuehrung):

- `frama_cross`
- `dual_thrust`
- `ema_band_rejection`
- `ema_cross_volume`

Hinweis:

- `supertrend_ema` existiert weiterhin im Code (`strategies/python/supertrend_ema.py`) und im Runner-Mapping, ist aber nicht Teil der aktiven Backtest-Ausfuehrungsmenge in der GUI.

## Live-Bot (Runtime)

Was live passiert:

- WebSocket-Streaming via Bitunix (`core/api/websocket.py`)
- Candle-History-Sync via REST (`core/api/bitunix.py`, `core/data/history.py`)
- Strategieauswertung pro Coin/Intervall (`main_engine.py` + `strategies/python/*`)
- Setup-Gate und optionales HMM-Regime-Filtering (`core/patterns/setup_gate.py`, `core/regime/hmm_regime_detector.py`)
- Paper-Trading-Execution inkl. Fees, Trailing, Breakeven (`core/paper_trading/engine.py`)
- Persistenz in DuckDB + JSON (`core/data/db.py`, `paper_trades.json`)

GUI-Anzeigen:

- Coin-Kacheln/Radar
- Trade-Readiness
- Offene/geschlossene Trades
- Live-Logs und Backtest-Logs

## Analytics/Backtest

Backtest-Engine:

- Thread-Klasse: `BacktestThread`
- Kanonischer Importpfad: `core/engine/backtest_engine.py` (Implementierung in `main_engine.py`)

Backtest kann:

- Historische Candles laden/synchronisieren
- Signale berechnen und mit Risk-/Fee-Logik simulieren
- Single, Selected-Batch und All-Coins-Batch laufen lassen
- Optimizer-Profillaeufe ausfuehren (Sampling/Full-Scan)
- Kompaktberichte schreiben nach `archive/backtest_compact_summaries/`

## GUI-Modi im Backtest-Tab

### Standard

- Frei waehlbar: Coin(s), Strategie(n), Timeframe(s), Leverage
- Enthalten: `Auto (from Config)`

`Auto (from Config)` bedeutet:

- Strategie wird coin-spezifisch aus der aktuellen Konfiguration aufgeloest.
- Intervall kann coin-spezifisch aus Konfig-/Registry-Daten aufgeloest werden.
- In Standard-Backtests koennen strategie-spezifische Defaults (z. B. EMA-Band-Winner-Preset fuer non-optimization) greifen.

### Live vs Challenger

Ziel:

- Pro aktivem Live-Coin: Live-Baseline gegen Challenger-Strategien vergleichen.

Logik:

- Baseline = aktuelles Coin-Profil aus `PRODUCTION_PROFILE_REGISTRY`
- Challenger-Menge = erlaubte 4 aktiven Strategien (`frama_cross`, `dual_thrust`, `ema_band_rejection`, `ema_cross_volume`) ohne die jeweilige Baseline-Strategie
- Intervall-Fairness-Lock aktiv: Baseline-Intervall wird fuer alle Challenger dieses Coins erzwungen
- Strategie-/Timeframe-Selector der GUI wird in diesem Modus fuer die Queue ignoriert

Export:

- `archive/research_reports/live_vs_challenger_summary_YYYYMMDD_HHMMSS.txt`

Urteile pro Coin:

- `BLEIBT`
- `ERSETZEN`
- `BEOBACHTEN`

## Reports und abgeleitete Dateien

- `archive/backtest_compact_summaries/backtest_compact_summary_*.txt`
  - Kompaktberichte aus Batch-/Backtest-Sessions
- `archive/research_reports/live_vs_challenger_summary_*.txt`
  - Vergleichsberichte fuer Live-vs-Challenger
- `summary_gesamt.txt`
  - abgeleitete Statusdatei (Live-Registry gegen Backtest-Compact-Dateien gematcht)
- `generate_summary_gesamt.py`
  - Generator fuer `summary_gesamt.txt` aus `config.PRODUCTION_PROFILE_REGISTRY`
  - sucht Backtest-Compact-Dateien in:
    - Root: `backtest_compact_summary_*.txt`
    - Archiv: `archive/backtest_compact_summaries/backtest_compact_summary_*.txt`
  - ohne Treffer: kein Crash, stattdessen Hinweis im Report

Aufruf:

```bash
python generate_summary_gesamt.py
```

Optional:

```bash
python generate_summary_gesamt.py --out summary_gesamt.txt
```

## Projektstruktur (aktuell)

Wichtige aktive Dateien:

- `gui.py` - GUI-Entrypoint und komplette Desktop-Oberflaeche
- `main_engine.py` - Live- und Backtest-Thread-Implementierung, Runner, Optimizer
- `config.py` - zentrale Runtime-Konfiguration inkl. Live-Registry
- `generate_summary_gesamt.py` - Summary-Generator

Aktive Pakete:

- `core/api/` - REST/WebSocket
- `core/data/` - DB-Zugriff und History
- `core/paper_trading/` - Paper-Execution + Persistenz
- `core/patterns/` - Setup-Gate
- `core/regime/` - HMM-Regime-Filter
- `core/engine/` - kanonische Wrapper-Importpfade
- `engine/` - numba-optimierte Backtest-Loops
- `strategies/python/` - Strategie-Implementierungen
- `strategies/jit_indicators.py` - JIT-Indikator- und Signalhilfen
- `config_sections/` - Default-/Legacy-Quellsektionen, die in `config.py` uebersteuert werden koennen

Archiv/Legacy:

- `archive/tools_legacy/backup_legacy/` - historische Backup-Dateien (nicht produktiver Kern)
- `archive/research_reports/` - Research- und Vergleichsreports
- `archive/backtest_compact_summaries/` - erzeugte Kompaktberichte

Lokale Laufzeitdaten:

- `data/paper_trading.duckdb` (+ WAL/SHM)
- `paper_trades.json`
- `logs/*.log`

## Datenbank (aktuell)

Verwendet wird eine lokale eingebettete `DuckDB`-Datei:

- DB-Typ: `DuckDB` (file-based, keine separate Server-Instanz)
- Hauptdatei: `data/paper_trading.duckdb`
- Laufzeitdateien: `data/paper_trading.duckdb.wal`, `data/paper_trading.duckdb.shm`

Wichtige Tabellen:

- `candles` (historische OHLCV-Daten)
- `paper_trades` (Paper-Positionen und Exits)
- `live_signals` (geloggte Live-Signale)
- `backtest_runs` (Backtest-/Optimizer-Ergebnisse)

## Setup und Start

### Voraussetzungen

- Python 3.10+
- Pakete:
  - `PyQt6`
  - `numpy`
  - `pandas`
  - `duckdb`
  - `requests`
  - `websockets`
  - `hmmlearn`
  - `numba`
  - optional: `polars` (nur fuer `PolarsDataLoader`-Workflows)

Installation (Beispiel):

```bash
pip install PyQt6 numpy pandas duckdb requests websockets hmmlearn numba polars
```

Start:

```bash
python gui.py
```

## Wichtige Hinweise

- Live-Runtime arbeitet als Paper-Trading auf Live-Marktdaten (keine echte Exchange-Orderplatzierung in diesem Flow).
- `PRODUCTION_PROFILE_REGISTRY` + `ACTIVE_COINS` sind die produktive Konfigurationswahrheit.
- Der Backtest-Deploy-Flow kann Konfigurationsdaten in `config.py` persistieren.
- `summary_gesamt.txt` ist eine abgeleitete Datei und sollte ueber `generate_summary_gesamt.py` erzeugt werden.
