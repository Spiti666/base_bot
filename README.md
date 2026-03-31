# Base Bot

Trading-Workspace mit **zwei Kernbereichen**:

1. **Live-Bot** (Live-Marktdaten + Paper-Execution)
2. **Backtest/Optimizer** (historische Simulation + Profilsuche)

Die Bedienung läuft zentral über die GUI.

## Live-Bot

### Was der Live-Bot macht
- Streamt Candles über WebSocket (Bitunix Public Feed).
- Routet pro Coin das konfigurierte Intervall und die konfigurierte Strategie.
- Berechnet Signale und Setup-Freigaben.
- Führt Trades in der lokalen Paper-Engine aus (inkl. Gebühren-/Risk-Logik).
- Zeigt in der GUI: Coin-Radar, Trade-Readiness, Open Positions, Live-Logs und Runtime-Profilstatus.

### Live-Strategien (Routing)
- `frama_cross`
- `ema_cross_volume`
- `dual_thrust`

### Live-Konfiguration (wichtig)
Die zentralen Live-Parameter liegen in `config.py`:
- `ACTIVE_COINS`
- `PRODUCTION_PROFILE_REGISTRY` (Strategie + Intervall + Parameter + Risk je Coin)
- `DEFAULT_COIN_STRATEGIES`
- `settings.live.default_interval`
- `settings.trading` (Leverage, Fees, Risk, Gate/HMM etc.)

## Backtest / Optimizer

### Was der Backtest macht
- Lädt historische Candles aus der lokalen DB.
- Simuliert Trades inkl. Gebühren, Slippage-Penalty, Trailing, Breakeven, Exit-Regeln.
- Schreibt Ergebnisse in die DB (`backtest_runs`) und in kompakte TXT-Reports (`backtest_compact_summary_YYYYMMDD_HHMMSS.txt`).
- Unterstützt Batch-Läufe über mehrere Coins/Strategien/Timeframes.

### Optimizer-Fokus
- Grid-basierte Profilsuche pro Strategie-Familie.
- Multiprocessing (bis 16 Worker).
- Sampling/Full-Scan je nach Konfiguration.
- Validierungslogik mit Robustheitsfiltern (z. B. Trades, PF, Netto-Metriken).

### Backtest-Strategien (ausführbar)
- `frama_cross`
- `ema_cross_volume`
- `dual_thrust`

## Start

### Voraussetzungen
- Python 3.10+ (empfohlen: aktuelles 3.x)
- Installierte Pakete: `PyQt6`, `numpy`, `pandas`, `duckdb`, `requests`, `websockets`, `hmmlearn`

### Installation (Beispiel)
```bash
pip install PyQt6 numpy pandas duckdb requests websockets hmmlearn
```

### Start
```bash
python gui.py
```

## Typischer Workflow

### Live-Bot
1. Profile in `config.py` prüfen (`PRODUCTION_PROFILE_REGISTRY` / `ACTIVE_COINS`).
2. GUI starten.
3. Live-Bot starten.
4. Coin-Radar, Readiness und Open Positions überwachen.

### Backtest
1. Coins, Strategien und Timeframes im Backtest-Tab wählen.
2. `Run Backtest` oder Batch-Lauf starten.
3. Ergebnis in Kacheln/Logs prüfen.
4. `backtest_compact_summary_*.txt` für Vergleich und Dokumentation nutzen.

## Daten & Artefakte

- `data/paper_trading.duckdb` - Candle-Historie, Paper-Trades, Backtest-Runs  
  Hinweis: Diese Datei ist **nicht im Repository enthalten** (zu groß) und wird lokal erzeugt/gefüllt.
- `paper_trades.json` - Persistenz offener Paper-Positionen
- `backtest_compact_summary_*.txt` - kompakter Laufreport inkl. Aggregates
- `logs/` - GUI- und Runtime-Logs

## Hinweis

Der Live-Bereich arbeitet in dieser Codebasis als **Paper-Trading-Ausführung** mit Live-Marktdaten (keine echte Orderplatzierung an einer Börse im aktuellen Flow).
