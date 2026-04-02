from __future__ import annotations

from collections import Counter
from contextlib import suppress
import atexit
import importlib
import faulthandler
import html
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import statistics
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from traceback import format_exception
from typing import Sequence

import numpy as np
import pandas as pd
from PyQt6.QtCore import QDate, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import config as config_module
from config import BACKTEST_ONLY_COINS, settings
from core.data.db import Database
from core.engine.backtest_engine import (
    BacktestThread,
    generate_compact_summary,
)
from core.engine.live_engine import BotEngineThread
from core.paper_trading.engine import PaperTradingEngine
from main_engine import (
    generate_optimization_grid,
    get_strategy_badge,
    resolve_optimizer_strategy_for_symbol,
    resolve_interval_for_symbol,
    resolve_strategy_for_symbol,
)
from strategies.python.frama_cross import _calculate_frama_series as calculate_frama_series


STRATEGY_LABELS = {
    "__auto__": "Auto (from Config)",
    "ema_cross_volume": "EMA Cross + Volume",
    "ema_band_rejection": "EMA Band Rejection",
    "frama_cross": "FRAMA Cross",
    "dual_thrust": "Dual Thrust",
}
BACKTEST_AUTO_STRATEGY = "__auto__"
BACKTEST_STRATEGY_NAMES = (
    BACKTEST_AUTO_STRATEGY,
    "ema_cross_volume",
    "ema_band_rejection",
    "frama_cross",
    "dual_thrust",
)
BACKTEST_INTERVAL_OPTIONS = tuple(
    interval
    for interval in dict.fromkeys(("1m", "5m", "15m", "1h", "4h", str(settings.trading.interval)))
    if interval in settings.api.timeframes
)
BACKTEST_SYMBOL_GRID_COLUMNS = 5
BACKTEST_INTERVAL_GRID_COLUMNS = 4
BACKTEST_TRADE_WARNING_THRESHOLD = 20
BACKTEST_DEPLOY_MIN_TRADE_COUNT = 25
BACKTEST_DEPLOY_MIN_WIN_RATE_PCT = 55.0
BACKTEST_DEPLOY_MIN_ROBUST_PF = 1.2
BACKTEST_REPORT_LOW_SAMPLE_TRADES_THRESHOLD = 25
BACKTEST_REPORT_STRONG_SAMPLE_TRADES_THRESHOLD = 100
GUI_LOG_DIRECTORY = Path("logs")
GUI_RUNTIME_LOG_PATH = GUI_LOG_DIRECTORY / "gui_runtime.log"
GUI_CRASH_LOG_PATH = GUI_LOG_DIRECTORY / "gui_crash.log"
GUI_FAULTHANDLER_LOG_PATH = GUI_LOG_DIRECTORY / "gui_faulthandler.log"
GUI_BACKTEST_DEBUG_LOG_PATH = GUI_LOG_DIRECTORY / "backtest_debug.log"
_GUI_LOGGING_INITIALIZED = False
_FAULTHANDLER_FILE_HANDLE = None
BacktestBatchItem = tuple[str, str, str | None]


def _configured_backtest_symbols() -> tuple[str, ...]:
    configured = tuple(
        symbol
        for symbol in dict.fromkeys(
            str(raw_symbol).strip().upper()
            for raw_symbol in (*settings.live.available_symbols, *BACKTEST_ONLY_COINS)
        )
        if symbol
    )
    if configured:
        return configured
    return tuple(settings.live.available_symbols)


def _safe_numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return float(default)
        return numeric
    text = str(value).strip().upper() if value is not None else ""
    if not text:
        return float(default)
    if text == "MAX":
        return float("inf")
    if text in {"INF", "+INF", "INFINITY", "+INFINITY"}:
        return float("inf")
    if text in {"-INF", "-INFINITY"}:
        return float("-inf")
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(numeric):
        return float(default)
    return numeric


def calculate_fitness_score(summary_data: object) -> float:
    if not isinstance(summary_data, dict):
        return float("-inf")

    net_pnl_usd = _safe_numeric(
        summary_data.get("net_pnl_usd", summary_data.get("total_pnl_usd", 0.0)),
        0.0,
    )
    trade_count = int(max(0.0, _safe_numeric(
        summary_data.get("trade_count", summary_data.get("total_trades", 0)),
        0.0,
    )))
    win_rate_pct = _safe_numeric(summary_data.get("win_rate_pct", 0.0), 0.0)
    # Parse robust PF defensively ("MAX"/"inf") to avoid edge-case crashes.
    _ = _safe_numeric(
        summary_data.get(
            "robust_profit_factor",
            summary_data.get(
                "robust_profit_factor_display",
                summary_data.get("profit_factor", 0.0),
            ),
        ),
        0.0,
    )

    deploy_ready = (
        trade_count >= int(BACKTEST_DEPLOY_MIN_TRADE_COUNT)
        and win_rate_pct >= float(BACKTEST_DEPLOY_MIN_WIN_RATE_PCT)
        and net_pnl_usd > 0.0
    )
    if not deploy_ready:
        return float(net_pnl_usd - 10000.0)

    score = float(net_pnl_usd * (win_rate_pct / 100.0))
    if math.isnan(score):
        return float("-inf")
    if score == float("inf"):
        return 1e18
    if score == float("-inf"):
        return -1e18
    return score


def _close_faulthandler_file() -> None:
    global _FAULTHANDLER_FILE_HANDLE
    file_handle = _FAULTHANDLER_FILE_HANDLE
    if file_handle is None:
        return
    with suppress(Exception):
        faulthandler.disable()
    with suppress(Exception):
        file_handle.flush()
    with suppress(Exception):
        file_handle.close()
    _FAULTHANDLER_FILE_HANDLE = None


def _append_exception_to_crash_log(
    context: str,
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback,
) -> None:
    GUI_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    traceback_text = "".join(format_exception(exc_type, exc_value, exc_traceback))
    with GUI_CRASH_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {context}\n")
        handle.write(traceback_text)
        if not traceback_text.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def _setup_gui_file_logging() -> None:
    global _GUI_LOGGING_INITIALIZED
    global _FAULTHANDLER_FILE_HANDLE
    if _GUI_LOGGING_INITIALIZED:
        return

    GUI_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gui")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            GUI_RUNTIME_LOG_PATH,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
        )
        logger.addHandler(handler)

    previous_sys_excepthook = sys.excepthook

    def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:
        _append_exception_to_crash_log(
            "Uncaught exception (sys.excepthook)",
            exc_type,
            exc_value,
            exc_traceback,
        )
        logger.error(
            "Uncaught exception (sys.excepthook)",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        with suppress(Exception):
            previous_sys_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _sys_excepthook

    previous_thread_excepthook = getattr(threading, "excepthook", None)

    def _thread_excepthook(args) -> None:
        thread_name = "unknown"
        if getattr(args, "thread", None) is not None:
            thread_name = str(getattr(args.thread, "name", "unknown"))
        _append_exception_to_crash_log(
            f"Uncaught thread exception ({thread_name})",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )
        logger.error(
            "Uncaught thread exception (%s)",
            thread_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        if previous_thread_excepthook is not None:
            with suppress(Exception):
                previous_thread_excepthook(args)

    if previous_thread_excepthook is not None:
        threading.excepthook = _thread_excepthook

    if _FAULTHANDLER_FILE_HANDLE is None:
        try:
            _FAULTHANDLER_FILE_HANDLE = GUI_FAULTHANDLER_LOG_PATH.open("a", encoding="utf-8")
            faulthandler.enable(file=_FAULTHANDLER_FILE_HANDLE, all_threads=True)
            atexit.register(_close_faulthandler_file)
        except Exception:
            logger.exception("Failed to initialize faulthandler log.")

    logger.info(
        "GUI file logging initialized. runtime=%s crash=%s faulthandler=%s",
        GUI_RUNTIME_LOG_PATH.as_posix(),
        GUI_CRASH_LOG_PATH.as_posix(),
        GUI_FAULTHANDLER_LOG_PATH.as_posix(),
    )
    _GUI_LOGGING_INITIALIZED = True

APP_BACKGROUND = "#0a0a0a"
PANEL_BACKGROUND = "#161616"
SURFACE_BACKGROUND = "#101010"
TEXT_PRIMARY = "#e0e0e0"
TEXT_MUTED = "#9e9e9e"
ACCENT_PRIMARY = "#0078d4"
ACCENT_PROFIT = "#2e7d32"
ACCENT_DANGER = "#c62828"
ACCENT_WARNING = "#ef6c00"
BORDER_COLOR = "#2b2b2b"

BUTTON_STYLE_GREEN = (
    f"QPushButton {{ background-color: {ACCENT_PROFIT}; color: white; border: 1px solid #1b5e20; border-radius: 6px; padding: 4px 12px; }}"
)
BUTTON_STYLE_RED = (
    f"QPushButton {{ background-color: {ACCENT_DANGER}; color: white; border: 1px solid #8e0000; border-radius: 6px; padding: 4px 12px; }}"
)
BUTTON_STYLE_BLUE = (
    f"QPushButton {{ background-color: {ACCENT_PRIMARY}; color: white; border: 1px solid #2f49d1; border-radius: 6px; padding: 4px 12px; }}"
)
BUTTON_STYLE_DISABLED = (
    f"QPushButton {{ background-color: #2a2a2a; color: #6e6e6e; border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 4px 12px; }}"
)


class MetricCard(QFrame):
    def __init__(self, title: str, accent_color: str) -> None:
        super().__init__()
        self._accent_color = accent_color
        self.setObjectName("MetricCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricCardTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricCardValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self._apply_style(accent_color)

    def set_value(self, value: str, color: str | None = None) -> None:
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color or TEXT_PRIMARY};")

    def _apply_style(self, accent_color: str) -> None:
        self.setStyleSheet(
            f"""
            QFrame#MetricCard {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-left: 4px solid {accent_color};
                border-radius: 6px;
            }}
            QLabel#MetricCardTitle {{
                color: {TEXT_MUTED};
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
            }}
            QLabel#MetricCardValue {{
                color: {TEXT_PRIMARY};
                font-size: 24px;
                font-weight: 700;
            }}
            """
        )


class LiveSymbolCard(QFrame):
    clicked = pyqtSignal(str)
    deploy_requested = pyqtSignal(str)

    def __init__(
        self,
        symbol: str,
        *,
        compact: bool = False,
        show_deploy_button: bool = False,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self._compact = compact
        self._show_deploy_button = show_deploy_button
        self._is_selected = False
        self._strategy_badge_text = "TREND"
        self._strategy_name_text = "-"
        self._status_text = "IDLE"
        self._status_color = TEXT_MUTED
        self._profit_factor: float | None = None
        self._win_rate: float | None = None
        self._trade_count: int | None = None
        self._compact_summary: dict[str, object] | None = None
        self.setObjectName("LiveSymbolCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if compact and show_deploy_button:
            self.setMinimumHeight(102)
            self.setMaximumHeight(126)
        else:
            self.setMinimumHeight(86 if compact else 102)

        root_layout = QVBoxLayout(self)
        if compact and show_deploy_button:
            root_layout.setContentsMargins(8, 6, 8, 6)
            root_layout.setSpacing(2)
        elif compact:
            root_layout.setContentsMargins(10, 8, 10, 8)
            root_layout.setSpacing(4)
        else:
            root_layout.setContentsMargins(14, 12, 14, 12)
            root_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.symbol_label = QLabel(symbol)
        self.symbol_label.setObjectName("SymbolName")
        header_layout.addWidget(self.symbol_label)
        self.strategy_name_label = QLabel(self._strategy_name_text)
        self.strategy_name_label.setObjectName("StrategyNameLabel")
        self.strategy_name_label.setWordWrap(False)
        header_layout.addWidget(self.strategy_name_label)
        header_layout.addStretch(1)
        root_layout.addLayout(header_layout)

        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(6)
        self.strategy_badge_label = QLabel(self._strategy_badge_text)
        self.strategy_badge_label.setObjectName("StrategyBadge")
        meta_layout.addWidget(self.strategy_badge_label)
        meta_layout.addStretch(1)
        self.status_label = QLabel(self._status_text)
        self.status_label.setObjectName("CoinStatusBadge")
        meta_layout.addWidget(self.status_label)
        root_layout.addLayout(meta_layout)

        self.metrics_label = QLabel()
        self.metrics_label.setObjectName("ProfitFactorLabel")
        self.metrics_label.hide()
        root_layout.addWidget(self.metrics_label)

        self.summary_row_1 = QLabel()
        self.summary_row_1.setObjectName("BacktestSummaryRow")
        self.summary_row_1.hide()
        root_layout.addWidget(self.summary_row_1)
        self.summary_row_2 = QLabel()
        self.summary_row_2.setObjectName("BacktestSummaryRow")
        self.summary_row_2.hide()
        root_layout.addWidget(self.summary_row_2)
        self.summary_row_3 = QLabel()
        self.summary_row_3.setObjectName("BacktestSummaryRow")
        self.summary_row_3.hide()
        root_layout.addWidget(self.summary_row_3)

        self.btn_deploy_live = QPushButton("Pending Optimization")
        self.btn_deploy_live.setVisible(self._show_deploy_button)
        if self._show_deploy_button:
            self.btn_deploy_live.setMaximumHeight(22)
        self.btn_deploy_live.clicked.connect(
            lambda _checked=False, current_symbol=self.symbol: self.deploy_requested.emit(current_symbol)
        )
        if self._show_deploy_button:
            root_layout.addWidget(self.btn_deploy_live)

        self.set_profit_factor(None)
        self.set_win_rate(None)
        self.set_trade_count(None)
        self.set_compact_summary(None)
        self.set_strategy_badge("TREND")
        self.set_status("IDLE", TEXT_MUTED)
        self.set_deploy_state("pending")
        self.set_selected(False)

    def set_profit_factor(self, value: float | None) -> None:
        self._profit_factor = value
        self._update_metric_label()

    def set_win_rate(self, value: float | None) -> None:
        self._win_rate = value
        self._update_metric_label()

    def set_trade_count(self, value: int | None) -> None:
        self._trade_count = value
        self._update_metric_label()

    def _update_metric_label(self) -> None:
        if self._show_deploy_button:
            self.metrics_label.hide()
            summary = self._compact_summary or {}
            if not summary:
                for row_label in (self.summary_row_1, self.summary_row_2, self.summary_row_3):
                    row_label.hide()
                return

            net_pnl_usd = float(summary.get("net_pnl_usd", 0.0) or 0.0)
            win_rate_pct = float(summary.get("win_rate_pct", 0.0) or 0.0)
            robust_pf_display = str(summary.get("robust_profit_factor_display", "0.00") or "0.00")
            trade_count = int(summary.get("trade_count", 0) or 0)
            max_drawdown_pct = float(summary.get("max_drawdown_pct", 0.0) or 0.0)
            avg_win_usd = float(summary.get("avg_win_usd", 0.0) or 0.0)

            pnl_color = ACCENT_PROFIT if net_pnl_usd > 0.0 else ACCENT_DANGER if net_pnl_usd < 0.0 else TEXT_MUTED
            trades_color = ACCENT_WARNING if trade_count < BACKTEST_TRADE_WARNING_THRESHOLD else TEXT_PRIMARY

            self.summary_row_1.setText(
                f"PnL: <span style='color:{pnl_color};'>${net_pnl_usd:,.2f}</span> | "
                f"Win-Rate: {win_rate_pct:.2f}%"
            )
            self.summary_row_2.setText(
                f"PF: {robust_pf_display} | "
                f"Trades: <span style='color:{trades_color};'>{trade_count}</span>"
            )
            self.summary_row_3.setText(
                f"Max DD: {max_drawdown_pct:.2f}% | "
                f"Avg Win: ${avg_win_usd:,.2f}"
            )
            for row_label in (self.summary_row_1, self.summary_row_2, self.summary_row_3):
                row_label.show()
            return

        if self._profit_factor is None and self._win_rate is None and self._trade_count is None:
            self.metrics_label.hide()
            return
        parts: list[str] = []
        if self._profit_factor is not None:
            parts.append(f"PF: {self._profit_factor:.2f}")
        if self._win_rate is not None:
            parts.append(f"WR: {self._win_rate:.1f}%")
        if self._trade_count is not None:
            parts.append(f"TR: {self._trade_count}")
        color = (
            ACCENT_PROFIT
            if self._profit_factor is not None and self._profit_factor > 1.0
            else ACCENT_WARNING
            if self._profit_factor is not None and self._profit_factor >= 0.9
            else TEXT_MUTED
        )
        self.metrics_label.setText(" | ".join(parts))
        self.metrics_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.metrics_label.show()

    def set_compact_summary(self, summary: dict[str, object] | None) -> None:
        self._compact_summary = None if summary is None else dict(summary)
        self._update_metric_label()

    def set_strategy_badge(
        self,
        badge_text: str,
        strategy_name: str | None = None,
        interval: str | None = None,
    ) -> None:
        self._strategy_badge_text = badge_text
        self.strategy_badge_label.setText(badge_text)
        if strategy_name:
            resolved_strategy_name = str(strategy_name).strip()
            display_name = STRATEGY_LABELS.get(
                resolved_strategy_name,
                resolved_strategy_name.replace("_", " ").title(),
            )
            resolved_interval = str(interval).strip() if interval is not None else ""
            if resolved_interval:
                display_name = f"{display_name} ({resolved_interval})"
            self._set_strategy_name_text(display_name)
        else:
            self._set_strategy_name_text("-")
        if strategy_name:
            if interval:
                self.strategy_badge_label.setToolTip(f"{strategy_name} @ {interval}")
            else:
                self.strategy_badge_label.setToolTip(strategy_name)
        self._apply_card_style()

    def _set_strategy_name_text(self, strategy_name_text: str) -> None:
        self._strategy_name_text = strategy_name_text
        metrics: QFontMetrics = self.strategy_name_label.fontMetrics()
        max_width = 120 if self._compact else 160
        elided_text = metrics.elidedText(
            strategy_name_text,
            Qt.TextElideMode.ElideRight,
            max_width,
        )
        if self._show_deploy_button and strategy_name_text.strip() and strategy_name_text.strip() != "-":
            self.strategy_name_label.setText(f"[Winner: {elided_text}]")
        else:
            self.strategy_name_label.setText(elided_text)
        if elided_text != strategy_name_text:
            self.strategy_name_label.setToolTip(strategy_name_text)
        else:
            self.strategy_name_label.setToolTip("")

    def set_status(self, status_text: str, color: str) -> None:
        self._status_text = status_text
        self._status_color = color
        self.status_label.setText(status_text)
        self._apply_card_style()

    def set_selected(self, selected: bool) -> None:
        self._is_selected = selected
        self._apply_card_style()

    def set_deploy_state(self, state: str) -> None:
        if not self._show_deploy_button:
            return
        normalized_state = str(state).strip().lower()
        if normalized_state == "ready":
            self.btn_deploy_live.setEnabled(True)
            self.btn_deploy_live.setText("🚀 Deploy to Live")
            self.btn_deploy_live.setStyleSheet(
                "QPushButton {"
                "background-color: #1f5f2e;"
                "color: #ecfff0;"
                "border: 1px solid #2e7d42;"
                "border-radius: 6px;"
                "padding: 4px 10px;"
                "font-weight: 700;"
                "}"
                "QPushButton:hover { background-color: #287b3b; }"
            )
            return
        if normalized_state == "deployed":
            self.btn_deploy_live.setEnabled(False)
            self.btn_deploy_live.setText("LIVE ✅")
            self.btn_deploy_live.setStyleSheet(
                "QPushButton {"
                "background-color: #205d33;"
                "color: #e6ffe9;"
                "border: 1px solid #2c7a46;"
                "border-radius: 6px;"
                "padding: 4px 10px;"
                "font-weight: 700;"
                "}"
            )
            return
        if normalized_state == "blocked":
            self.btn_deploy_live.setEnabled(False)
            self.btn_deploy_live.setText("Needs Robustness")
            self.btn_deploy_live.setStyleSheet(
                "QPushButton {"
                "background-color: #3a2b18;"
                "color: #ffcc80;"
                "border: 1px solid #7a5a2e;"
                "border-radius: 6px;"
                "padding: 4px 10px;"
                "font-weight: 700;"
                "}"
            )
            return
        self.btn_deploy_live.setEnabled(False)
        self.btn_deploy_live.setText("Pending Optimization")
        self.btn_deploy_live.setStyleSheet(
            "QPushButton {"
            "background-color: #2a2a2a;"
            "color: #8a8a8a;"
            "border: 1px solid #3a3a3a;"
            "border-radius: 6px;"
            "padding: 4px 10px;"
            "font-weight: 600;"
            "}"
        )

    def _apply_card_style(self) -> None:
        border_color = ACCENT_PRIMARY if self._is_selected else BORDER_COLOR
        symbol_font_size = "12px" if self._compact else "14px"
        badge_font_size = "9px" if self._compact else "10px"
        metrics_font_size = "10px" if self._compact else "11px"
        self.setStyleSheet(
            f"""
            QFrame#LiveSymbolCard {{
                background-color: {PANEL_BACKGROUND};
                border: 2px solid {border_color};
                border-radius: 6px;
            }}
            QLabel#SymbolName {{
                background-color: transparent;
                color: {TEXT_PRIMARY};
                font-weight: 700;
                font-size: {symbol_font_size};
            }}
            QLabel#StrategyBadge {{
                background-color: transparent;
                color: {ACCENT_WARNING};
                border: none;
                padding: 0px;
                font-size: {badge_font_size};
                font-weight: 700;
            }}
            QLabel#StrategyNameLabel {{
                background-color: transparent;
                color: {ACCENT_WARNING};
                font-size: {badge_font_size};
                font-weight: 700;
            }}
            QLabel#CoinStatusBadge {{
                background-color: rgba(255, 255, 255, 0.04);
                color: {self._status_color};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: {badge_font_size};
                font-weight: 700;
            }}
            QLabel#ProfitFactorLabel {{
                color: {TEXT_MUTED};
                font-size: {metrics_font_size};
            }}
            QLabel#BacktestSummaryRow {{
                color: {TEXT_MUTED};
                font-size: {metrics_font_size};
                font-weight: 600;
            }}
            """
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.clicked.emit(self.symbol)
        super().mousePressEvent(event)


class TradeReadinessBars(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self._items: list[dict[str, object]] = []
        self.setMinimumHeight(126)
        self.setObjectName("TradeReadinessBars")
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(450)
        self._blink_timer.timeout.connect(self._on_blink_tick)
        self._blink_timer.start()
        self.setStyleSheet(
            f"""
            QFrame#TradeReadinessBars {{
                background-color: {SURFACE_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
            """
        )

    def set_items(self, items: list[dict[str, object]]) -> None:
        self._items = list(items)
        self.update()

    def clear_items(self) -> None:
        self._items = []
        self.update()

    def _on_blink_tick(self) -> None:
        if any(bool(item.get("eta_blink", False)) for item in self._items):
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        draw_rect = self.rect().adjusted(8, 8, -8, -8)
        if not self._items:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(draw_rect, Qt.AlignmentFlag.AlignCenter, "Readiness unavailable")
            painter.end()
            return

        item_count = max(len(self._items), 1)
        eta_height = 12
        label_height = 16
        value_height = 12
        bar_area = draw_rect.adjusted(0, value_height, 0, -(label_height + eta_height))
        if bar_area.height() <= 0:
            painter.end()
            return

        slot_width = bar_area.width() / float(item_count)
        bar_width = max(6.0, slot_width * 0.72)

        for index, item in enumerate(self._items):
            symbol = str(item.get("symbol", "-"))
            score = max(0.0, min(100.0, float(item.get("score", 0.0) or 0.0)))
            color_hex = str(item.get("color", ACCENT_PRIMARY))
            eta_text = str(item.get("eta_text", "--"))
            eta_color_hex = str(item.get("eta_color", TEXT_MUTED))
            eta_blink = bool(item.get("eta_blink", False))
            value_text = f"{int(round(score)):d}%"

            center_x = bar_area.left() + (index + 0.5) * slot_width
            bar_left = int(center_x - (bar_width / 2.0))
            bar_right = int(center_x + (bar_width / 2.0))
            bar_rect = bar_area.adjusted(
                bar_left - bar_area.left(),
                0,
                -(bar_area.right() - bar_right),
                0,
            )
            if bar_rect.width() <= 0:
                continue

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1b1b1b"))
            painter.drawRoundedRect(bar_rect, 3, 3)

            fill_height = int((bar_rect.height() * score) / 100.0)
            if fill_height > 0:
                fill_rect = bar_rect.adjusted(
                    0,
                    bar_rect.height() - fill_height,
                    0,
                    0,
                )
                painter.setBrush(QColor(color_hex))
                painter.drawRoundedRect(fill_rect, 3, 3)

            slot_left = int(bar_area.left() + index * slot_width)
            slot_right = int(bar_area.left() + (index + 1) * slot_width)
            text_rect_top = draw_rect.adjusted(
                slot_left - draw_rect.left(),
                0,
                -(draw_rect.right() - slot_right),
                -(draw_rect.height() - value_height),
            )
            text_rect_eta = draw_rect.adjusted(
                slot_left - draw_rect.left(),
                draw_rect.height() - (label_height + eta_height),
                -(draw_rect.right() - slot_right),
                -label_height,
            )
            text_rect_bottom = draw_rect.adjusted(
                slot_left - draw_rect.left(),
                draw_rect.height() - label_height,
                -(draw_rect.right() - slot_right),
                0,
            )

            painter.setPen(QColor(TEXT_PRIMARY))
            painter.drawText(
                text_rect_top,
                int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter),
                value_text,
            )

            eta_pen_color = eta_color_hex
            if eta_blink:
                blink_on = int(time.monotonic() * 2.0) % 2 == 0
                eta_pen_color = "#39ff88" if blink_on else "#1f5d3a"
            painter.setPen(QColor(eta_pen_color))
            painter.drawText(
                text_rect_eta,
                int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter),
                eta_text,
            )

            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(
                text_rect_bottom,
                int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter),
                symbol.replace("USDT", ""),
            )

        painter.end()


class TradingTerminalWindow(QMainWindow):
    _HISTORY_PROGRESS_PREFIX = "[HISTORY_PROGRESS]"
    _SIGNAL_CACHE_PROGRESS_PREFIX = "[SIGNAL_CACHE_PROGRESS]"
    _BACKTEST_LOG_MODE_QUIET = "quiet"
    _BACKTEST_LOG_MODE_STANDARD = "standard"
    _BACKTEST_LOG_MODE_DEBUG = "debug"
    _MAX_LOG_LINES = 4000
    _BACKTEST_LOG_WRAP_COLUMNS = 120
    _MAX_BACKTEST_RESULT_TRADES = 5000
    _MAX_BACKTEST_TABLE_ROWS = 5000
    _MAX_BACKTEST_OPTIMIZATION_RESULTS = 200

    def __init__(self) -> None:
        super().__init__()
        self._db_path = Path("data/paper_trading.duckdb")
        self._live_interval = settings.live.default_interval
        self._bot_thread: BotEngineThread | None = None
        self._backtest_thread: BacktestThread | None = None
        self._latest_prices: dict[str, float] = {}
        self._price_rows: dict[str, int] = {}
        self._positions: dict[int, dict] = {}
        self._live_profit_factors: dict[str, float] = {}
        self._live_win_rates: dict[str, float] = {}
        self._live_trade_counts: dict[str, int] = {}
        self._live_runtime_profiles: dict[str, dict[str, object]] = {}
        self._backtest_profit_factors: dict[str, float] = {}
        self._backtest_win_rates: dict[str, float] = {}
        self._backtest_trade_counts: dict[str, int] = {}
        self._backtest_compact_summaries: dict[str, dict[str, object]] = {}
        self._backtest_best_profiles: dict[str, dict[str, float]] = {}
        self._backtest_deployed_symbols: set[str] = set()
        self._coin_cards: dict[str, LiveSymbolCard] = {}
        self._selected_live_symbols: set[str] = set()
        self._backtest_coin_cards: dict[str, LiveSymbolCard] = {}
        self._selected_backtest_symbols: set[str] = set()
        self._backtest_strategy_buttons: dict[str, QPushButton] = {}
        self._selected_backtest_strategies: set[str] = set()
        self._backtest_interval_buttons: dict[str, QPushButton] = {}
        self._selected_backtest_intervals: set[str] = set()
        self._heartbeat_snapshot: dict[str, dict[str, object]] = {}
        self._realized_pnl_total = 0.0
        self._realized_pnl_today = 0.0
        self._equity_history: list[float] = []
        self._readiness_last_refresh_ts = 0.0
        self._readiness_refresh_interval_seconds = 2.0
        self._readiness_runtime_settings = settings
        self._readiness_config_file = Path(getattr(config_module, "__file__", "config.py")).resolve()
        self._readiness_config_mtime_ns = 0
        self._readiness_config_reload_error_mtime_ns: int | None = None
        with suppress(OSError):
            self._readiness_config_mtime_ns = self._readiness_config_file.stat().st_mtime_ns
        self._heartbeat_generation = 0
        self._bot_stop_requested = False
        self._backtest_stop_requested = False
        self._progress_generation = 0
        self._batch_queue: list[BacktestBatchItem] = []
        self._batch_auto_strategy_flags: list[bool] = []
        self._batch_active = False
        self._batch_mode = "standard"
        self._batch_total_symbols = 0
        self._batch_started_symbols: set[str] = set()
        self._batch_completed_symbols: set[str] = set()
        self._batch_symbol_remaining_runs: dict[str, int] = {}
        self._backtest_report_started_at: datetime | None = None
        self._backtest_report_entries: list[dict[str, object]] = []
        self._backtest_report_errors: list[str] = []
        self._backtest_report_pending = False
        self._backtest_report_mode = "single"
        self._last_backtest_result: dict[str, object] | None = None
        self._analytics_splitter: QSplitter | None = None
        self._backtest_progress_log_active = False
        self._active_backtest_log_mode = self._BACKTEST_LOG_MODE_STANDARD
        self._active_backtest_log_context: dict[str, int] = {"coins": 1, "strategies": 1, "timeframes": 1}
        self.backtest_table = None
        self._backtest_period_user_modified = False
        self._pending_close_request = False

        self.setWindowTitle("Bitunix Quant Station")
        self.resize(2048, 1152)
        self.setMinimumSize(1600, 900)

        self._build_ui()
        self._apply_theme()
        self._load_startup_state()
        self._sync_positions_table()
        self._update_status_label()
        self._apply_button_states()
        self._start_candle_timer()
        self.statusBar().showMessage("Ready")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        bot_running = self._bot_thread is not None and self._bot_thread.isRunning()
        backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()

        if bot_running:
            self._stop_bot()
        if backtest_running or self._batch_active:
            self._stop_backtest()

        bot_running = self._bot_thread is not None and self._bot_thread.isRunning()
        backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
        if bot_running or backtest_running:
            self._pending_close_request = True
            self.statusBar().showMessage(
                "Waiting for running threads to stop before closing...",
                5000,
            )
            self._append_backtest_log(
                "Close requested while a worker thread is still running. "
                "Waiting for graceful shutdown to prevent thread destruction."
            )
            event.ignore()
            return

        self._pending_close_request = False
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(10)
        self.setCentralWidget(central)

        performance_header = self._build_performance_header()
        central_layout.addWidget(performance_header)

        candle_timer = self._build_candle_timer_panel()
        central_layout.addWidget(candle_timer)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        central_layout.addWidget(self.tabs)

        self.trading_tab = QWidget()
        trading_layout = QVBoxLayout(self.trading_tab)
        trading_layout.setContentsMargins(0, 0, 0, 0)
        trading_layout.setSpacing(10)
        self.tabs.addTab(self.trading_tab, "Trading & Portfolio")

        controls = self._build_controls_bar()
        trading_layout.addWidget(controls)

        self.feed_table = self._build_feed_table()
        dashboard_splitter = QSplitter(Qt.Orientation.Horizontal)
        dashboard_splitter.setChildrenCollapsible(False)
        dashboard_splitter.setHandleWidth(2)
        dashboard_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        trading_layout.addWidget(dashboard_splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(2)
        left_layout.addWidget(left_splitter, 1)

        positions_panel = self._build_positions_panel()
        log_panel = self._build_log_panel()
        left_splitter.addWidget(positions_panel)
        left_splitter.addWidget(log_panel)
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 3)
        left_splitter.setSizes([650, 350])

        right_panel = self._build_coin_sidebar()
        dashboard_splitter.addWidget(left_panel)
        dashboard_splitter.addWidget(right_panel)
        dashboard_splitter.setStretchFactor(0, 4)
        dashboard_splitter.setStretchFactor(1, 1)
        dashboard_splitter.setSizes([1638, 410])

        self.analytics_tab = self._build_analytics_tab()
        self.tabs.addTab(self.analytics_tab, "Analytics & Backtest")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_performance_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("PerformanceHeader")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.balance_card = MetricCard("Guthaben", ACCENT_PRIMARY)
        self.open_trades_card = MetricCard("Offene Trades", "#00acc1")
        self.day_pnl_card = MetricCard("Tages-PnL", ACCENT_PROFIT)

        layout.addWidget(self.balance_card, 1)
        layout.addWidget(self.open_trades_card, 1)
        layout.addWidget(self.day_pnl_card, 1)
        return frame

    def _build_controls_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("DashboardPanel")
        frame.setFrameShape(QFrame.Shape.NoFrame)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        symbol_label = QLabel("Backtest Coin")
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(list(settings.live.available_symbols))
        self.symbol_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.symbol_combo.currentTextChanged.connect(self._on_header_backtest_coin_changed)

        leverage_label = QLabel("Leverage")
        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(
            settings.trading.min_leverage,
            settings.trading.max_leverage,
        )
        self.leverage_spin.setValue(settings.trading.default_leverage)
        self.leverage_spin.setSuffix("x")
        self.leverage_spin.setFixedWidth(92)
        self.leverage_spin.valueChanged.connect(self._on_leverage_changed)

        confidence_label = QLabel("Min Confidence")
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setSingleStep(5)
        self.confidence_spin.setValue(int(settings.trading.min_confidence_pct))
        self.confidence_spin.setSuffix("%")
        self.confidence_spin.setFixedWidth(96)

        strategy_label = QLabel("Fallback Strategy")
        self.strategy_combo = QComboBox()
        self._populate_strategy_combo(self.strategy_combo)
        self.strategy_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.strategy_combo.currentIndexChanged.connect(lambda _index: self._update_status_label())
        self.strategy_combo.currentIndexChanged.connect(lambda _index: self._refresh_live_symbol_cards())

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_bot)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_bot)
        self.stop_button.setEnabled(False)

        self.market_pulse_label = QLabel()
        self.market_pulse_label.setFixedSize(16, 16)
        self.market_pulse_label.setToolTip("Market Pulse")
        self._fade_market_pulse(0)

        self.sync_status_label = QLabel("Sync: -")
        self.sync_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.sync_status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700;")

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        clock_font = QFont("Courier New", 10)
        self.utc_time_label = QLabel("UTC: --:--:--")
        self.utc_time_label.setFont(clock_font)
        self.utc_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.utc_time_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")

        self.local_time_label = QLabel("Local: --:--:--")
        self.local_time_label.setFont(clock_font)
        self.local_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.local_time_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")

        layout.addWidget(symbol_label)
        layout.addWidget(self.symbol_combo)
        layout.addSpacing(8)
        layout.addWidget(leverage_label)
        layout.addWidget(self.leverage_spin)
        layout.addSpacing(8)
        layout.addWidget(confidence_label)
        layout.addWidget(self.confidence_spin)
        layout.addSpacing(8)
        layout.addWidget(strategy_label)
        layout.addWidget(self.strategy_combo)
        layout.addSpacing(8)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.market_pulse_label)
        layout.addWidget(self.sync_status_label)
        layout.addWidget(self.status_label, 1)
        layout.addSpacing(10)
        layout.addWidget(self.utc_time_label)
        layout.addSpacing(6)
        layout.addWidget(self.local_time_label)
        return frame

    def _build_log_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("Execution Log")
        title.setStyleSheet("font-weight: 700;")
        subtitle = QLabel("Live events, fills and strategy routing")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        header.addWidget(title)
        header.addSpacing(6)
        header.addWidget(subtitle)
        header.addStretch(1)
        layout.addLayout(header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_output, 1)
        return panel

    def _build_coin_sidebar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        self.coin_radar_title_label = QLabel("Coin Radar")
        self.coin_radar_title_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.coin_radar_count_label = QLabel("(0)")
        self.coin_radar_count_label.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {ACCENT_PRIMARY};"
        )
        title_row.addWidget(self.coin_radar_title_label)
        title_row.addWidget(self.coin_radar_count_label)
        title_row.addStretch(1)
        subtitle = QLabel("Strategy-aware live universe")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addLayout(title_row)
        layout.addWidget(subtitle)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        self.live_search_input = QLineEdit()
        self.live_search_input.setPlaceholderText("Search symbols...")
        self.live_search_input.textChanged.connect(self._filter_live_symbols)
        actions_row.addWidget(self.live_search_input, 1)
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self._select_all_symbols)
        actions_row.addWidget(self.select_all_button)
        self.select_winners_button = QPushButton("Select Winners")
        self.select_winners_button.clicked.connect(self._select_winner_symbols)
        actions_row.addWidget(self.select_winners_button)
        layout.addLayout(actions_row)

        self.live_symbols_scroll = QScrollArea()
        self.live_symbols_scroll.setWidgetResizable(True)
        self.live_symbols_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.live_symbols_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.live_symbols_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.live_symbols_scroll.setObjectName("CoinCardScroll")

        self.live_symbols_container = QWidget()
        self.live_symbols_layout = QVBoxLayout(self.live_symbols_container)
        self.live_symbols_layout.setContentsMargins(0, 0, 0, 0)
        self.live_symbols_layout.setSpacing(4)
        for symbol in settings.live.available_symbols:
            card = LiveSymbolCard(symbol)
            card.clicked.connect(self._toggle_live_symbol_selection)
            self._coin_cards[symbol] = card
            self.live_symbols_layout.addWidget(card)
        self.live_symbols_layout.addStretch(1)
        self.live_symbols_scroll.setWidget(self.live_symbols_container)
        layout.addWidget(self.live_symbols_scroll, 1)

        self._update_coin_radar_count()
        self._select_default_live_symbols()
        return panel

    def _build_candle_timer_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("DashboardPanel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.candle_timer_label = QLabel("Next Candle Check")
        self.candle_timer_label.setStyleSheet("font-weight: 700;")
        self.candle_timer_progress = QProgressBar()
        self.candle_timer_progress.setRange(0, 100)
        self.candle_timer_progress.setValue(0)
        self.candle_timer_progress.setTextVisible(False)
        self.candle_timer_progress.setFixedHeight(16)
        self.candle_timer_remaining_label = QLabel("--:--")
        self.candle_timer_remaining_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")

        layout.addWidget(self.candle_timer_label)
        layout.addWidget(self.candle_timer_progress, 1)
        layout.addWidget(self.candle_timer_remaining_label)
        return frame

    def _build_positions_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_row = QFrame()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        header_layout.addWidget(QLabel("Open Positions"))
        header_layout.addStretch(1)

        self.close_position_button = QPushButton("Close Selected")
        self.close_position_button.clicked.connect(self._close_selected_position)
        header_layout.addWidget(self.close_position_button)
        self.copy_positions_button = QPushButton("Copy Positions")
        self.copy_positions_button.clicked.connect(self._copy_positions_table_to_clipboard)
        header_layout.addWidget(self.copy_positions_button)
        layout.addWidget(header_row)

        self.positions_table = self._build_positions_table()
        layout.addWidget(self.positions_table, 1)

        equity_row = QFrame()
        equity_layout = QVBoxLayout(equity_row)
        equity_layout.setContentsMargins(0, 6, 0, 0)
        equity_layout.setSpacing(6)
        equity_title = QLabel("Trade Readiness")
        equity_title.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")
        self.readiness_bars = TradeReadinessBars()
        equity_layout.addWidget(equity_title)
        equity_layout.addWidget(self.readiness_bars)
        layout.addWidget(equity_row)
        return panel

    def _build_analytics_tab(self) -> QWidget:
        analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(analytics_tab)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(0)

        analytics_splitter = QSplitter(Qt.Orientation.Horizontal)
        analytics_splitter.setChildrenCollapsible(False)
        analytics_splitter.setHandleWidth(2)
        self._analytics_splitter = analytics_splitter

        analytics_main = QWidget()
        analytics_main_layout = QVBoxLayout(analytics_main)
        analytics_main_layout.setContentsMargins(0, 0, 10, 0)
        analytics_main_layout.setSpacing(10)

        button_row = QFrame()
        button_row.setObjectName("DashboardPanel")
        button_layout = QVBoxLayout(button_row)
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setSpacing(8)
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.backtest_period_bar = self._build_backtest_period_bar()
        button_layout.addWidget(self.backtest_period_bar)
        button_layout.addLayout(filter_layout)
        button_layout.addLayout(action_layout)

        # Hidden compatibility controls: keep internal state wiring for existing
        # backtest flows, but remove the visible header fields.
        self.backtest_coin_combo = QComboBox()
        backtest_symbols = list(_configured_backtest_symbols())
        self.backtest_coin_combo.addItems(backtest_symbols)
        self.backtest_coin_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if backtest_symbols:
            self.backtest_coin_combo.setCurrentText(backtest_symbols[0])
        self.backtest_coin_combo.currentTextChanged.connect(self._on_backtest_coin_changed)

        self.backtest_strategy_combo = QComboBox()
        self._populate_strategy_combo(
            self.backtest_strategy_combo,
            strategy_names=BACKTEST_STRATEGY_NAMES,
        )
        self.backtest_strategy_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.backtest_strategy_combo.currentIndexChanged.connect(self._on_backtest_strategy_changed)
        self.backtest_strategy_combo.currentIndexChanged.connect(lambda _index: self._refresh_backtest_symbol_cards())

        backtest_interval_label = QLabel("Timeframe")
        self.backtest_interval_combo = QComboBox()
        self.backtest_interval_combo.addItems(list(BACKTEST_INTERVAL_OPTIONS))
        self.backtest_interval_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.backtest_interval_combo.setCurrentText(settings.trading.interval)
        self.backtest_interval_combo.currentTextChanged.connect(self._on_backtest_interval_changed)
        filter_layout.addWidget(backtest_interval_label)
        filter_layout.addWidget(self.backtest_interval_combo)
        filter_layout.addSpacing(8)

        backtest_leverage_label = QLabel("Backtest Leverage")
        self.backtest_leverage_slider = QSlider(Qt.Orientation.Horizontal)
        self.backtest_leverage_slider.setRange(
            settings.trading.min_leverage,
            settings.trading.max_leverage,
        )
        self.backtest_leverage_slider.setSingleStep(1)
        self.backtest_leverage_slider.setPageStep(5)
        self.backtest_leverage_slider.setTickInterval(5)
        self.backtest_leverage_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        initial_backtest_symbol = backtest_symbols[0] if backtest_symbols else None
        initial_backtest_leverage = settings.trading.default_leverage
        if initial_backtest_symbol:
            profile = settings.trading.coin_profiles.get(initial_backtest_symbol)
            if profile is not None and profile.default_leverage is not None:
                initial_backtest_leverage = int(profile.default_leverage)
        self.backtest_leverage_slider.setValue(
            max(
                settings.trading.min_leverage,
                min(initial_backtest_leverage, settings.trading.max_leverage),
            )
        )
        self.backtest_leverage_slider.setFixedWidth(220)
        self.backtest_leverage_slider.valueChanged.connect(self._on_backtest_leverage_changed)

        self.backtest_leverage_value_label = QLabel(f"{self.backtest_leverage_slider.value()}x")
        self.backtest_leverage_value_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.backtest_leverage_value_label.setFixedWidth(42)
        self.backtest_leverage_value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-weight: 700; font-family: Consolas, 'Courier New', monospace;"
        )

        filter_layout.addWidget(backtest_leverage_label)
        filter_layout.addWidget(self.backtest_leverage_slider)
        filter_layout.addWidget(self.backtest_leverage_value_label)
        filter_layout.addStretch(1)

        self.backtest_button = QPushButton("Run Backtest")
        self.backtest_button.clicked.connect(self._start_backtest)
        action_layout.addWidget(self.backtest_button)
        self.optimize_button = QPushButton("Optimize Profile")
        self.optimize_button.setCheckable(True)
        self.optimize_button.toggled.connect(lambda _checked: self._update_status_label())
        self.optimize_button.toggled.connect(lambda _checked: self._apply_button_states())
        self.optimize_button.toggled.connect(lambda _checked: self._refresh_backtest_period_visuals())
        action_layout.addWidget(self.optimize_button)
        self.run_selected_button = QPushButton("Run Selected Coins")
        self.run_selected_button.clicked.connect(self._on_run_selected_clicked)
        action_layout.addWidget(self.run_selected_button)
        self.run_all_button = QPushButton("Run All Coins")
        self.run_all_button.clicked.connect(self._on_run_all_clicked)
        action_layout.addWidget(self.run_all_button)
        self.stop_backtest_button = QPushButton("Stop Backtest")
        self.stop_backtest_button.clicked.connect(self._stop_backtest)
        action_layout.addWidget(self.stop_backtest_button)
        action_layout.addStretch(1)
        for action_button in (
            self.backtest_button,
            self.optimize_button,
            self.run_selected_button,
            self.run_all_button,
            self.stop_backtest_button,
        ):
            action_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            action_button.setMinimumWidth(max(action_button.sizeHint().width() + 14, 136))
        self._refresh_backtest_period_visuals()
        analytics_main_layout.addWidget(button_row)

        selection_row = QHBoxLayout()
        selection_row.setContentsMargins(0, 0, 0, 0)
        selection_row.setSpacing(10)
        timeframe_selector_panel = self._build_backtest_interval_selector()
        strategy_selector_panel = self._build_backtest_strategy_selector()
        selection_row.addWidget(timeframe_selector_panel, 1)
        selection_row.addWidget(strategy_selector_panel, 1)
        analytics_main_layout.addLayout(selection_row)
        selector_panel = self._build_backtest_symbol_selector()
        analytics_main_layout.addWidget(selector_panel, 1)

        metrics_row = QFrame()
        metrics_row.setObjectName("DashboardPanel")
        metrics_layout = QHBoxLayout(metrics_row)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(10)

        self.backtest_total_pnl_label = self._build_metric_label("Total PnL", "-")
        self.backtest_win_rate_label = self._build_metric_label("Win Rate", "-")
        self.backtest_total_trades_label = self._build_metric_label("Total Trades", "-")
        metrics_layout.addWidget(self.backtest_total_pnl_label, 1)
        metrics_layout.addWidget(self.backtest_win_rate_label, 1)
        metrics_layout.addWidget(self.backtest_total_trades_label, 1)
        analytics_main_layout.addWidget(metrics_row)

        backtest_log_panel = self._build_backtest_log_panel()
        analytics_splitter.addWidget(analytics_main)
        analytics_splitter.addWidget(backtest_log_panel)
        analytics_splitter.setStretchFactor(0, 6)
        analytics_splitter.setStretchFactor(1, 4)
        analytics_main.setMinimumWidth(640)
        backtest_log_panel.setMinimumWidth(420)
        analytics_layout.addWidget(analytics_splitter, 1)
        QTimer.singleShot(0, self._apply_backtest_splitter_sizes)
        return analytics_tab

    def _build_backtest_log_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("Backtest Log")
        title.setStyleSheet("font-weight: 700;")
        subtitle = QLabel("Strategy routing, history sync and optimizer progress")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        header.addWidget(title)
        header.addSpacing(6)
        header.addWidget(subtitle)
        header.addStretch(1)
        layout.addLayout(header)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        action_hint = QLabel("Log Actions")
        action_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        action_row.addWidget(action_hint)
        action_row.addStretch(1)
        self.copy_backtest_log_button = QPushButton("Copy Log")
        self.copy_backtest_log_button.clicked.connect(self._copy_backtest_log_to_clipboard)
        action_row.addWidget(self.copy_backtest_log_button)
        self.clear_backtest_log_button = QPushButton("Clear Log")
        self.clear_backtest_log_button.clicked.connect(self._clear_backtest_log)
        action_row.addWidget(self.clear_backtest_log_button)
        layout.addLayout(action_row)

        self.backtest_log_output = QTextEdit()
        self.backtest_log_output.setReadOnly(True)
        self.backtest_log_output.setLineWrapMode(QTextEdit.LineWrapMode.FixedColumnWidth)
        self.backtest_log_output.setLineWrapColumnOrWidth(self._BACKTEST_LOG_WRAP_COLUMNS)
        self.backtest_log_output.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.backtest_log_output.setFont(QFont("Consolas", 10))
        self.backtest_log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.backtest_log_output, 1)
        return panel

    def _build_backtest_symbol_selector(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title = QLabel("Backtest Coin Selection")
        title.setStyleSheet("font-weight: 700;")
        subtitle = QLabel("Select multiple coins for sequential backtests")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        header_layout.addWidget(title)
        header_layout.addSpacing(6)
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)

        self.backtest_select_all_button = QPushButton("Select All")
        self.backtest_select_all_button.clicked.connect(self._select_all_backtest_symbols)
        header_layout.addWidget(self.backtest_select_all_button)
        self.backtest_clear_cards_button = QPushButton("Clear Cards")
        self.backtest_clear_cards_button.clicked.connect(self._clear_backtest_coin_cards)
        header_layout.addWidget(self.backtest_clear_cards_button)
        layout.addLayout(header_layout)

        self.backtest_symbols_scroll = QScrollArea()
        self.backtest_symbols_scroll.setWidgetResizable(True)
        self.backtest_symbols_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.backtest_symbols_scroll.setObjectName("CoinCardScroll")
        self.backtest_symbols_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.backtest_symbols_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.backtest_symbols_container = QWidget()
        self.backtest_symbols_layout = QGridLayout(self.backtest_symbols_container)
        self.backtest_symbols_layout.setContentsMargins(0, 0, 0, 0)
        self.backtest_symbols_layout.setSpacing(4)
        for symbol in _configured_backtest_symbols():
            card = LiveSymbolCard(symbol, compact=True, show_deploy_button=True)
            card.clicked.connect(self._toggle_backtest_symbol_selection)
            card.deploy_requested.connect(self._deploy_backtest_symbol_to_live)
            self._backtest_coin_cards[symbol] = card
        self._rebuild_backtest_symbol_grid()
        self.backtest_symbols_scroll.setWidget(self.backtest_symbols_container)
        layout.addWidget(self.backtest_symbols_scroll, 1)
        QTimer.singleShot(0, self._sync_backtest_symbol_selector_viewport)

        self._select_default_backtest_symbols()
        self._sync_backtest_symbol_universe()
        return panel

    def _build_backtest_strategy_selector(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title = QLabel("Batch Strategy Selection")
        title.setStyleSheet("font-weight: 700;")
        subtitle = QLabel("Select multiple strategies for sequential runs")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        header_layout.addWidget(title)
        header_layout.addSpacing(6)
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        strategy_grid = QGridLayout()
        strategy_grid.setContentsMargins(0, 0, 0, 0)
        strategy_grid.setHorizontalSpacing(8)
        strategy_grid.setVerticalSpacing(8)
        strategy_buttons: list[QPushButton] = []
        for index, strategy_name in enumerate(BACKTEST_STRATEGY_NAMES):
            button = QPushButton(STRATEGY_LABELS.get(strategy_name, strategy_name))
            button.setCheckable(True)
            button.setFixedHeight(28)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.toggled.connect(
                lambda checked, name=strategy_name: self._toggle_backtest_strategy_selection(name, checked)
            )
            self._backtest_strategy_buttons[strategy_name] = button
            strategy_buttons.append(button)
            row = index // 2
            column = index % 2
            strategy_grid.addWidget(button, row, column)
        if strategy_buttons:
            # Add safety padding so long labels do not clip with theme/border rendering.
            strategy_width = max(button.sizeHint().width() for button in strategy_buttons) + 26
            for button in strategy_buttons:
                button.setFixedWidth(strategy_width)
        strategy_grid.setColumnStretch(0, 1)
        strategy_grid.setColumnStretch(1, 1)
        layout.addLayout(strategy_grid)

        self._select_default_backtest_strategies()
        return panel

    def _build_backtest_interval_selector(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("DashboardPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title = QLabel("Batch Timeframe Selection")
        title.setStyleSheet("font-weight: 700;")
        subtitle = QLabel("Select one or multiple timeframes for sequential runs")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        header_layout.addWidget(title)
        header_layout.addSpacing(6)
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)

        self.backtest_select_all_intervals_button = QPushButton("All TF")
        self.backtest_select_all_intervals_button.clicked.connect(self._select_all_backtest_intervals)
        header_layout.addWidget(self.backtest_select_all_intervals_button)

        self.backtest_single_interval_button = QPushButton("Primary TF")
        self.backtest_single_interval_button.clicked.connect(self._select_primary_backtest_interval)
        header_layout.addWidget(self.backtest_single_interval_button)

        layout.addLayout(header_layout)

        interval_grid = QGridLayout()
        interval_grid.setContentsMargins(0, 0, 0, 0)
        interval_grid.setHorizontalSpacing(6)
        interval_grid.setVerticalSpacing(0)
        interval_buttons: list[QPushButton] = []
        for index, interval in enumerate(BACKTEST_INTERVAL_OPTIONS):
            button = QPushButton(interval)
            button.setCheckable(True)
            button.setFixedHeight(28)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.toggled.connect(
                lambda checked, value=interval: self._toggle_backtest_interval_selection(value, checked)
            )
            self._backtest_interval_buttons[interval] = button
            interval_buttons.append(button)
            interval_grid.addWidget(button, 0, index, Qt.AlignmentFlag.AlignLeft)
        if interval_buttons:
            interval_width = max(button.sizeHint().width() for button in interval_buttons) + 12
            for button in interval_buttons:
                button.setFixedWidth(interval_width)
        layout.addLayout(interval_grid)

        self._select_default_backtest_intervals()
        return panel

    @staticmethod
    def _parse_utc_date_only(value: object, fallback: QDate) -> QDate:
        text = str(value or "").strip()
        if not text:
            return fallback
        with suppress(Exception):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(UTC).replace(tzinfo=None)
            return QDate(parsed.year, parsed.month, parsed.day)
        return fallback

    def _default_backtest_period_start_date(self, *, optimize_mode: bool) -> QDate:
        fallback = QDate.currentDate().addYears(-1)
        configured_value = (
            settings.trading.optimizer_history_start_utc
            if optimize_mode
            else settings.trading.backtest_history_start_utc
        )
        return self._parse_utc_date_only(configured_value, fallback)

    def _build_backtest_period_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("BacktestPeriodBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("Period (UTC)")
        title.setStyleSheet("font-weight: 700;")
        layout.addWidget(title)

        today = QDate.currentDate()
        min_date = QDate(2018, 1, 1)
        max_date = today.addYears(1)

        self.backtest_period_start_edit = QDateEdit()
        self.backtest_period_start_edit.setCalendarPopup(True)
        self.backtest_period_start_edit.setDisplayFormat("yyyy-MM-dd")
        self.backtest_period_start_edit.setMinimumDate(min_date)
        self.backtest_period_start_edit.setMaximumDate(max_date)
        self.backtest_period_start_edit.setDate(self._default_backtest_period_start_date(optimize_mode=False))
        self.backtest_period_start_edit.dateChanged.connect(self._on_backtest_period_changed)
        self.backtest_period_start_edit.setFixedWidth(124)
        layout.addWidget(self.backtest_period_start_edit)

        arrow_label = QLabel("->")
        arrow_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700;")
        layout.addWidget(arrow_label)

        self.backtest_period_end_edit = QDateEdit()
        self.backtest_period_end_edit.setCalendarPopup(True)
        self.backtest_period_end_edit.setDisplayFormat("yyyy-MM-dd")
        self.backtest_period_end_edit.setMinimumDate(min_date)
        self.backtest_period_end_edit.setMaximumDate(max_date)
        self.backtest_period_end_edit.setDate(today)
        self.backtest_period_end_edit.dateChanged.connect(self._on_backtest_period_changed)
        self.backtest_period_end_edit.setFixedWidth(124)
        layout.addWidget(self.backtest_period_end_edit)

        quick_presets = (
            ("2024-Full", "2024"),
            ("2025-Full", "2025"),
            ("YTD", "ytd"),
        )
        for label_text, preset_key in quick_presets:
            button = QPushButton(label_text)
            button.setFixedHeight(24)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, preset=preset_key: self._select_backtest_period_preset(preset))
            layout.addWidget(button)

        self.backtest_period_span_label = QLabel("")
        self.backtest_period_span_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")
        layout.addWidget(self.backtest_period_span_label)

        self.backtest_walkforward_strip = QFrame()
        self.backtest_walkforward_strip.setVisible(False)
        strip_layout = QHBoxLayout(self.backtest_walkforward_strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(0)
        self.backtest_walkforward_in_sample = QLabel("In-Sample")
        self.backtest_walkforward_in_sample.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.backtest_walkforward_out_sample = QLabel("Out-of-Sample")
        self.backtest_walkforward_out_sample.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.backtest_walkforward_in_sample.setStyleSheet(
            "background:#1f5f7a; color:#e7f5ff; border:1px solid #2f7ea4; border-top-left-radius:5px; border-bottom-left-radius:5px; padding:1px 6px; font-size:10px; font-weight:700;"
        )
        self.backtest_walkforward_out_sample.setStyleSheet(
            "background:#4e6b29; color:#e8ffe0; border:1px solid #67863a; border-left:none; border-top-right-radius:5px; border-bottom-right-radius:5px; padding:1px 6px; font-size:10px; font-weight:700;"
        )
        strip_layout.addWidget(self.backtest_walkforward_in_sample, 7)
        strip_layout.addWidget(self.backtest_walkforward_out_sample, 3)
        layout.addWidget(self.backtest_walkforward_strip)

        layout.addStretch(1)
        self._refresh_backtest_period_visuals()
        return bar

    def _on_backtest_period_changed(self, _value: QDate) -> None:
        self._backtest_period_user_modified = True
        start_date = self.backtest_period_start_edit.date()
        end_date = self.backtest_period_end_edit.date()
        if end_date < start_date:
            if self.sender() is self.backtest_period_start_edit:
                self.backtest_period_end_edit.blockSignals(True)
                self.backtest_period_end_edit.setDate(start_date)
                self.backtest_period_end_edit.blockSignals(False)
            else:
                self.backtest_period_start_edit.blockSignals(True)
                self.backtest_period_start_edit.setDate(end_date)
                self.backtest_period_start_edit.blockSignals(False)
        self._refresh_backtest_period_visuals()

    def _select_backtest_period_preset(self, preset: str) -> None:
        today = datetime.now(UTC).date()
        if preset == "2024":
            start_date = QDate(2024, 1, 1)
            end_date = QDate(2024, 12, 31)
        elif preset == "2025":
            start_date = QDate(2025, 1, 1)
            end_date = QDate(2025, 12, 31)
        else:
            start_date = QDate(today.year, 1, 1)
            end_date = QDate(today.year, today.month, today.day)
        self.backtest_period_start_edit.blockSignals(True)
        self.backtest_period_end_edit.blockSignals(True)
        self.backtest_period_start_edit.setDate(start_date)
        self.backtest_period_end_edit.setDate(end_date)
        self.backtest_period_start_edit.blockSignals(False)
        self.backtest_period_end_edit.blockSignals(False)
        self._backtest_period_user_modified = True
        self._refresh_backtest_period_visuals()

    def _current_backtest_history_range(self) -> tuple[datetime | None, datetime | None]:
        if not hasattr(self, "backtest_period_start_edit") or not hasattr(self, "backtest_period_end_edit"):
            return None, None
        start_date = self.backtest_period_start_edit.date()
        end_date = self.backtest_period_end_edit.date()
        if not start_date.isValid() or not end_date.isValid():
            return None, None
        if end_date < start_date:
            end_date = start_date
        start_utc = datetime(start_date.year(), start_date.month(), start_date.day())
        end_utc_exclusive = (
            datetime(end_date.year(), end_date.month(), end_date.day())
            + timedelta(days=1)
        )
        return start_utc, end_utc_exclusive

    def _refresh_backtest_period_visuals(self) -> None:
        if not hasattr(self, "backtest_period_start_edit") or not hasattr(self, "backtest_period_end_edit"):
            return
        start_date = self.backtest_period_start_edit.date()
        end_date = self.backtest_period_end_edit.date()
        span_days = max(1, start_date.daysTo(end_date) + 1)
        self.backtest_period_span_label.setText(f"Window: {span_days}d")
        walk_forward_active = bool(
            getattr(settings.trading, "use_hmm_regime_filter", False)
            and hasattr(self, "optimize_button")
            and self.optimize_button.isChecked()
        )
        self.backtest_walkforward_strip.setVisible(walk_forward_active)

    def _set_backtest_period_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self, "backtest_period_start_edit"):
            self.backtest_period_start_edit.setEnabled(enabled)
        if hasattr(self, "backtest_period_end_edit"):
            self.backtest_period_end_edit.setEnabled(enabled)
        if hasattr(self, "backtest_period_bar"):
            for button in self.backtest_period_bar.findChildren(QPushButton):
                button.setEnabled(enabled)

    @staticmethod
    def _populate_strategy_combo(
        combo_box: QComboBox,
        *,
        strategy_names: tuple[str, ...] | None = None,
        select_default: bool = False,
    ) -> None:
        combo_box.clear()
        available_names = (
            settings.strategy.available_strategies
            if strategy_names is None
            else strategy_names
        )
        for strategy_name in available_names:
            combo_box.addItem(
                STRATEGY_LABELS.get(strategy_name, strategy_name),
                strategy_name,
            )
        combo_box.setPlaceholderText("Strategie wählen")
        if not select_default:
            combo_box.setCurrentIndex(-1)
            return
        if BACKTEST_AUTO_STRATEGY in available_names:
            default_strategy_index = combo_box.findData(BACKTEST_AUTO_STRATEGY)
        else:
            default_strategy_index = combo_box.findData(settings.strategy.default_strategy_name)
        if default_strategy_index >= 0:
            combo_box.setCurrentIndex(default_strategy_index)
        else:
            combo_box.setCurrentIndex(-1)

    def _current_live_strategy_name(self) -> str:
        return str(
            self.strategy_combo.currentData() or settings.strategy.default_strategy_name
        )

    def _current_backtest_strategy_name(self) -> str:
        return str(
            self.backtest_strategy_combo.currentData() or BACKTEST_AUTO_STRATEGY
        )

    def _is_backtest_auto_strategy(
        self,
        strategy_name: str | None = None,
    ) -> bool:
        resolved_name = self._current_backtest_strategy_name() if strategy_name is None else strategy_name
        return resolved_name == BACKTEST_AUTO_STRATEGY

    @staticmethod
    def _sanitize_backtest_strategy_name(symbol: str, strategy_name: str) -> str:
        allowed_strategy_names = {
            str(name)
            for name in BACKTEST_STRATEGY_NAMES
            if str(name) != BACKTEST_AUTO_STRATEGY
        }
        if strategy_name in allowed_strategy_names:
            return strategy_name
        try:
            fallback_name = resolve_optimizer_strategy_for_symbol(
                symbol,
                settings.strategy.default_strategy_name,
                apply_backtest_optimizer_overrides=False,
            )
            if fallback_name in allowed_strategy_names:
                return fallback_name
        except Exception:
            pass
        return "frama_cross"

    def _resolve_backtest_strategy_for_symbol(
        self,
        symbol: str,
        strategy_name: str | None = None,
    ) -> str:
        selected_strategy = self._current_backtest_strategy_name() if strategy_name is None else strategy_name
        if self._is_backtest_auto_strategy(selected_strategy):
            # Auto (from Config) always follows the current live-config mapping.
            resolved_strategy = resolve_strategy_for_symbol(
                symbol,
                settings.strategy.default_strategy_name,
                use_coin_override=True,
            )
            return self._sanitize_backtest_strategy_name(symbol, resolved_strategy)
        resolved_strategy = resolve_strategy_for_symbol(
            symbol,
            selected_strategy,
            use_coin_override=False,
        )
        return self._sanitize_backtest_strategy_name(symbol, resolved_strategy)

    @staticmethod
    def _fallback_scan_strategy_for_symbol(symbol: str) -> str | None:
        try:
            return resolve_optimizer_strategy_for_symbol(
                symbol,
                settings.strategy.default_strategy_name,
                apply_backtest_optimizer_overrides=False,
            )
        except Exception:
            return None

    def _build_backtest_batch_queue(
        self,
        symbols: list[str],
        *,
        interval_overrides: Sequence[str] | None = None,
    ) -> tuple[list[BacktestBatchItem], list[str], str]:
        selected_strategy_names = self._get_selected_backtest_strategy_names()
        if not selected_strategy_names:
            selected_strategy_names = [self._current_backtest_strategy_name()]

        if interval_overrides is None:
            selected_intervals = self._get_selected_backtest_intervals()
        else:
            selected_intervals = [
                interval
                for interval in dict.fromkeys(str(interval) for interval in interval_overrides)
                if interval in BACKTEST_INTERVAL_OPTIONS
            ]
        if not selected_intervals:
            selected_intervals = list(BACKTEST_INTERVAL_OPTIONS)
        interval_label = (
            selected_intervals[0]
            if len(selected_intervals) == 1
            else f"{len(selected_intervals)} timeframes"
        )

        base_queue: list[BacktestBatchItem] = []
        base_auto_flags: list[bool] = []
        has_auto = BACKTEST_AUTO_STRATEGY in selected_strategy_names
        if has_auto:
            for symbol in symbols:
                resolved_strategy = self._resolve_backtest_strategy_for_symbol(
                    symbol,
                    BACKTEST_AUTO_STRATEGY,
                )
                batch_item: BacktestBatchItem = (symbol, resolved_strategy, None)
                base_queue.append(batch_item)
                base_auto_flags.append(True)

        for symbol in symbols:
            for strategy_name in selected_strategy_names:
                if strategy_name == BACKTEST_AUTO_STRATEGY:
                    continue
                try:
                    resolved_strategy = resolve_strategy_for_symbol(
                        symbol,
                        strategy_name,
                        use_coin_override=False,
                    )
                except Exception:
                    continue
                base_queue.append((symbol, resolved_strategy, None))
                base_auto_flags.append(False)

        queue: list[BacktestBatchItem] = []
        auto_flags: list[bool] = []
        for (symbol, strategy_name, interval_override), is_auto in zip(
            base_queue,
            base_auto_flags,
            strict=False,
        ):
            if interval_override is not None:
                queue.append((symbol, strategy_name, interval_override))
                auto_flags.append(bool(is_auto))
                continue
            for interval in selected_intervals:
                queue.append((symbol, strategy_name, interval))
                auto_flags.append(bool(is_auto))
        self._batch_auto_strategy_flags = auto_flags

        effective_selected_strategies = list(
            dict.fromkeys(strategy_name for _symbol, strategy_name, _interval in queue)
        )
        if has_auto and len(selected_strategy_names) == 1:
            mode_label = f"{STRATEGY_LABELS[BACKTEST_AUTO_STRATEGY]} / {interval_label}"
        elif has_auto:
            mode_label = (
                f"{len(selected_strategy_names)} strategies "
                f"(incl. {STRATEGY_LABELS[BACKTEST_AUTO_STRATEGY]}) / {interval_label}"
            )
        else:
            mode_label = f"{len(effective_selected_strategies)} strategies / {interval_label}"
        return queue, effective_selected_strategies, mode_label

    @staticmethod
    def _expand_batch_queue_with_intervals(
        base_queue: list[BacktestBatchItem],
        intervals: tuple[str, ...],
    ) -> list[BacktestBatchItem]:
        expanded: list[BacktestBatchItem] = []
        for symbol, strategy_name, interval_override in base_queue:
            if interval_override is not None:
                expanded.append((symbol, strategy_name, interval_override))
                continue
            for interval in intervals:
                expanded.append((symbol, strategy_name, interval))
        return expanded

    def _current_backtest_interval(self) -> str:
        if hasattr(self, "backtest_interval_combo"):
            return self.backtest_interval_combo.currentText() or settings.trading.interval
        return settings.trading.interval

    def _current_backtest_symbol(self) -> str:
        return self.backtest_coin_combo.currentText()

    def _current_backtest_leverage(self) -> int:
        return int(self.backtest_leverage_slider.value())

    def _set_current_backtest_symbol(self, symbol: str, *, ensure_selected: bool = True) -> None:
        if self.symbol_combo.findText(symbol) >= 0 and self.symbol_combo.currentText() != symbol:
            was_blocked = self.symbol_combo.blockSignals(True)
            self.symbol_combo.setCurrentText(symbol)
            self.symbol_combo.blockSignals(was_blocked)
        if self.backtest_coin_combo.currentText() != symbol:
            was_blocked = self.backtest_coin_combo.blockSignals(True)
            self.backtest_coin_combo.setCurrentText(symbol)
            self.backtest_coin_combo.blockSignals(was_blocked)
        self._sync_backtest_interval_for_symbol(symbol)
        if ensure_selected:
            self._selected_backtest_symbols.add(symbol)
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _set_current_backtest_strategy(self, strategy_name: str) -> None:
        strategy_index = self.backtest_strategy_combo.findData(strategy_name)
        if strategy_index >= 0 and self.backtest_strategy_combo.currentIndex() != strategy_index:
            was_blocked = self.backtest_strategy_combo.blockSignals(True)
            self.backtest_strategy_combo.setCurrentIndex(strategy_index)
            self.backtest_strategy_combo.blockSignals(was_blocked)
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _on_header_backtest_coin_changed(self, symbol: str) -> None:
        if hasattr(self, "backtest_coin_combo") and self.backtest_coin_combo.currentText() != symbol:
            was_blocked = self.backtest_coin_combo.blockSignals(True)
            self.backtest_coin_combo.setCurrentText(symbol)
            self.backtest_coin_combo.blockSignals(was_blocked)
        self._update_status_label()

    def _on_backtest_coin_changed(self, symbol: str) -> None:
        if self.symbol_combo.findText(symbol) >= 0 and self.symbol_combo.currentText() != symbol:
            was_blocked = self.symbol_combo.blockSignals(True)
            self.symbol_combo.setCurrentText(symbol)
            self.symbol_combo.blockSignals(was_blocked)
        self._sync_backtest_interval_for_symbol(symbol)
        self._selected_backtest_symbols.add(symbol)
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _sync_backtest_interval_for_symbol(self, symbol: str) -> None:
        if not hasattr(self, "backtest_interval_combo"):
            return
        # Keep backtest timeframe fully controlled by the GUI selection.
        current_interval = self._current_backtest_interval()
        if current_interval in BACKTEST_INTERVAL_OPTIONS:
            return
        fallback_interval = settings.trading.interval
        if fallback_interval not in BACKTEST_INTERVAL_OPTIONS:
            fallback_interval = BACKTEST_INTERVAL_OPTIONS[0]
        was_blocked = self.backtest_interval_combo.blockSignals(True)
        self.backtest_interval_combo.setCurrentText(fallback_interval)
        self.backtest_interval_combo.blockSignals(was_blocked)

    def _on_backtest_strategy_changed(self, _index: int) -> None:
        selected_strategy_name = self.backtest_strategy_combo.currentData()
        if selected_strategy_name is None:
            self._selected_backtest_strategies = set()
        else:
            self._selected_backtest_strategies = {str(selected_strategy_name)}
        self._refresh_backtest_strategy_buttons()
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _on_backtest_interval_changed(self, _interval: str) -> None:
        current_interval = self._current_backtest_interval()
        if current_interval in BACKTEST_INTERVAL_OPTIONS:
            selected_intervals = self._get_selected_backtest_intervals()
            if len(selected_intervals) <= 1:
                self._selected_backtest_intervals = {current_interval}
                self._refresh_backtest_interval_buttons()
        self._update_status_label()

    def _summarize_live_resolved_strategies(self) -> str:
        selected_symbols = self._get_selected_live_symbols()
        if not selected_symbols:
            return "-"

        unique_badges: list[str] = []
        for symbol in selected_symbols:
            badge = get_strategy_badge(symbol, self._current_live_strategy_name())
            if badge not in unique_badges:
                unique_badges.append(badge)

        if len(unique_badges) == 1:
            return unique_badges[0]
        return "MIXED (" + " / ".join(unique_badges) + ")"

    def _summarize_backtest_intervals(self) -> str:
        selected_intervals = self._get_selected_backtest_intervals()
        if not selected_intervals:
            return self._current_backtest_interval()
        if len(selected_intervals) == 1:
            return selected_intervals[0]
        if len(selected_intervals) <= 3:
            return ", ".join(selected_intervals)
        preview = ", ".join(selected_intervals[:3])
        return f"{len(selected_intervals)} TF ({preview} +{len(selected_intervals) - 3})"

    def _format_live_strategy_resolution(self, symbols: list[str], fallback_strategy_name: str) -> str:
        parts: list[str] = []
        for symbol in symbols:
            resolved_strategy = resolve_strategy_for_symbol(symbol, fallback_strategy_name)
            parts.append(
                f"{symbol}={STRATEGY_LABELS.get(resolved_strategy, resolved_strategy)}"
            )
        return ", ".join(parts)

    def _collect_live_symbol_intervals(self, symbols: list[str]) -> dict[str, str]:
        return {
            symbol: resolve_interval_for_symbol(symbol, self._live_interval)
            for symbol in symbols
        }

    def _sorted_live_intervals(self, intervals: Sequence[str]) -> list[str]:
        unique_intervals = [
            str(interval)
            for interval in dict.fromkeys(str(interval) for interval in intervals)
            if str(interval).strip()
        ]
        return sorted(
            unique_intervals,
            key=lambda interval: (
                self._interval_total_seconds(interval),
                interval.lower(),
            ),
        )

    def _summarize_live_intervals(self) -> str:
        selected_symbols = self._get_selected_live_symbols()
        if not selected_symbols:
            return "-"
        unique_intervals = self._sorted_live_intervals(
            resolve_interval_for_symbol(symbol, self._live_interval)
            for symbol in selected_symbols
        )
        if len(unique_intervals) == 1:
            return unique_intervals[0]
        return "MIXED (" + " / ".join(unique_intervals) + ")"

    def _format_live_interval_resolution(self, symbols: list[str]) -> str:
        parts: list[str] = []
        for symbol in symbols:
            parts.append(f"{symbol}={resolve_interval_for_symbol(symbol, self._live_interval)}")
        return ", ".join(parts)

    def _build_feed_table(self) -> QTableWidget:
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Symbol", "Last Price", "Updated"])
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    def _reset_feed_table(self) -> None:
        self.feed_table.setRowCount(0)
        self._price_rows.clear()
        self._latest_prices.clear()
        self._heartbeat_snapshot = {}

    def _build_positions_table(self) -> QTableWidget:
        table = QTableWidget(0, 12)
        table.setHorizontalHeaderLabels(
            [
                "ID",
                "Symbol",
                "Side",
                "Entry",
                "Last",
                "Qty",
                "Leverage",
                "Einsatz (USDT)",
                "Fees Paid",
                "Unrealized Net",
                "Health",
                "Opened (Local)",
            ]
        )
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.ResizeToContents)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    def _build_backtest_table(self) -> QTableWidget:
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(
            [
                "Entry Time (Local)",
                "Side",
                "Entry Price",
                "Exit Price",
                "PnL",
            ]
        )
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    @staticmethod
    def _build_metric_label(title: str, value: str) -> QLabel:
        label = QLabel(f"{title}: {value}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFrameShape(QFrame.Shape.StyledPanel)
        label.setStyleSheet(
            f"background: {PANEL_BACKGROUND}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 6px; font-size: 13px; font-weight: 700; padding: 10px;"
        )
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return label

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")
        font = QFont()
        font.setPointSize(10)
        app.setFont(font)

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(APP_BACKGROUND))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE_BACKGROUND))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL_BACKGROUND))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL_BACKGROUND))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Button, QColor(PANEL_BACKGROUND))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_PRIMARY))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)

        app.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {APP_BACKGROUND};
                color: {TEXT_PRIMARY};
            }}
            QTabWidget::pane {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background: {APP_BACKGROUND};
            }}
            QTabBar::tab {{
                background: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 12px;
                margin-right: 4px;
                color: {TEXT_MUTED};
            }}
            QTabBar::tab:selected {{
                background: {SURFACE_BACKGROUND};
                color: {TEXT_PRIMARY};
            }}
            QFrame#DashboardPanel, QFrame#PerformanceHeader {{
                background: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
            QComboBox, QSpinBox, QPushButton, QTextEdit, QTableWidget, QScrollArea, QLineEdit {{
                background: {SURFACE_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 4px 6px;
                color: {TEXT_PRIMARY};
            }}
            QHeaderView::section {{
                background: {PANEL_BACKGROUND};
                border: 0;
                border-right: 1px solid {BORDER_COLOR};
                border-bottom: 1px solid {BORDER_COLOR};
                padding: 6px;
                color: {TEXT_MUTED};
                font-weight: 700;
            }}
            QTableWidget {{
                gridline-color: transparent;
                selection-background-color: rgba(0, 120, 212, 0.18);
                alternate-background-color: {PANEL_BACKGROUND};
            }}
            QTextEdit {{
                font-family: Consolas, "Courier New", monospace;
                background: {PANEL_BACKGROUND};
            }}
            QScrollArea#CoinCardScroll {{
                background: transparent;
                border: none;
            }}
            QLineEdit {{
                min-height: 28px;
            }}
            QProgressBar {{
                background: {SURFACE_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                text-align: center;
                color: {TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: {ACCENT_PRIMARY};
                border-radius: 6px;
            }}
            QSlider::groove:horizontal {{
                background: {SURFACE_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                height: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT_PRIMARY};
                border-radius: 6px;
            }}
            QSlider::add-page:horizontal {{
                background: #1c1c1c;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                border: 1px solid {ACCENT_PRIMARY};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            """
        )

    def _start_bot(self) -> None:
        if self._bot_thread is not None and self._bot_thread.isRunning():
            return

        symbols = self._get_selected_live_symbols()
        if not symbols:
            self._append_live_log("Select at least one live symbol before starting the bot.")
            return
        leverage = self.leverage_spin.value()
        min_confidence_pct = float(self.confidence_spin.value())
        strategy_name = self._current_live_strategy_name()
        strategy_label = self.strategy_combo.currentText()
        live_strategy_summary = self._summarize_live_resolved_strategies()
        live_symbol_intervals = self._collect_live_symbol_intervals(symbols)
        live_intervals = self._sorted_live_intervals(live_symbol_intervals.values())
        self._append_live_log(f"--- LIVE PAPER TRADING STARTED FOR: {', '.join(symbols)} ---")
        self._append_live_log(
            f"Starting bot for {', '.join(symbols)} with configured leverage {leverage}x "
            f"(see runtime log for per-symbol leverage), fallback strategy {strategy_label}, "
            f"and min confidence {min_confidence_pct:.0f}%."
        )
        self._append_live_log(
            f"Resolved live strategies: {live_strategy_summary} | "
            f"{self._format_live_strategy_resolution(symbols, strategy_name)}."
        )
        self._append_live_log(
            f"Resolved live intervals: {', '.join(live_intervals)} | "
            f"{self._format_live_interval_resolution(symbols)}."
        )
        self._append_live_log(
            "Live stream plan: "
            + ", ".join(
                f"{symbol}->{interval}"
                for symbol, interval in live_symbol_intervals.items()
            )
        )
        self._reset_feed_table()
        self._live_runtime_profiles.clear()
        self._bot_stop_requested = False

        self._bot_thread = BotEngineThread(
            symbols,
            strategy_name=strategy_name,
            intervals=live_intervals,
            symbol_intervals=live_symbol_intervals,
            leverage=leverage,
            min_confidence_pct=min_confidence_pct,
            db_path=self._db_path,
        )
        self._bot_thread.log_message.connect(self._append_live_log)
        self._bot_thread.progress_update.connect(self._update_progress)
        self._bot_thread.trade_opened.connect(self._handle_trade_opened)
        self._bot_thread.positions_updated.connect(self._handle_positions_updated)
        self._bot_thread.price_update.connect(self._handle_price_update)
        self._bot_thread.heartbeat_status.connect(self._update_heartbeat_display)
        self._bot_thread.runtime_profile_update.connect(self._handle_runtime_profile_update)
        self._bot_thread.finished.connect(self._handle_thread_finished)
        self._bot_thread.start()

        self.symbol_combo.setEnabled(False)
        self._set_live_symbol_controls_enabled(False)
        self.confidence_spin.setEnabled(False)
        self.strategy_combo.setEnabled(False)
        self.sync_status_label.setText("Sync: Starting...")
        self.sync_status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700;")
        self._apply_button_states()
        self._update_status_label()
        self._refresh_live_symbol_cards()
        self._refresh_trade_readiness(force=True)

    @staticmethod
    def _normalize_backtest_log_mode(raw_mode: str | None) -> str:
        normalized = str(raw_mode or "").strip().lower()
        if normalized in {
            TradingTerminalWindow._BACKTEST_LOG_MODE_QUIET,
            TradingTerminalWindow._BACKTEST_LOG_MODE_STANDARD,
            TradingTerminalWindow._BACKTEST_LOG_MODE_DEBUG,
        }:
            return normalized
        return TradingTerminalWindow._BACKTEST_LOG_MODE_STANDARD

    def _resolve_backtest_log_mode_for_run(
        self,
        *,
        symbol_override: str | None,
        strategy_override: str | None,
        interval_override: str | None,
    ) -> tuple[str, dict[str, int]]:
        env_mode = self._normalize_backtest_log_mode(os.getenv("BACKTEST_LOG_MODE"))
        if env_mode != self._BACKTEST_LOG_MODE_STANDARD:
            context = {"coins": 1, "strategies": 1, "timeframes": 1}
            return env_mode, context

        configured_mode = self._normalize_backtest_log_mode(
            getattr(settings.trading, "backtest_log_mode", None)
        )
        if configured_mode != self._BACKTEST_LOG_MODE_STANDARD:
            context = {"coins": 1, "strategies": 1, "timeframes": 1}
            return configured_mode, context

        if self._batch_active:
            context = {
                "coins": max(1, self._batch_total_symbols),
                "strategies": max(1, len(self._get_selected_backtest_strategy_names())),
                "timeframes": max(1, len(self._get_selected_backtest_intervals())),
            }
            active_thread_runs = (
                1
                if self._backtest_thread is not None and self._backtest_thread.isRunning()
                else 0
            )
            remaining_runs = len(self._batch_queue) + active_thread_runs
            if (
                context["coins"] == 1
                and context["strategies"] == 1
                and context["timeframes"] == 1
                and remaining_runs <= 1
            ):
                return self._BACKTEST_LOG_MODE_QUIET, context
            return self._BACKTEST_LOG_MODE_STANDARD, context

        selected_symbols = self._get_selected_backtest_symbols()
        selected_strategies = self._get_selected_backtest_strategy_names()
        selected_intervals = self._get_selected_backtest_intervals()
        context = {
            "coins": 1 if symbol_override is not None else max(1, len(selected_symbols)),
            "strategies": 1 if strategy_override is not None else max(1, len(selected_strategies)),
            "timeframes": 1 if interval_override is not None else max(1, len(selected_intervals)),
        }
        if context["coins"] == 1 and context["strategies"] == 1 and context["timeframes"] == 1:
            return self._BACKTEST_LOG_MODE_QUIET, context
        return self._BACKTEST_LOG_MODE_STANDARD, context

    def _set_active_backtest_log_mode(self, mode: str, context: dict[str, int]) -> None:
        self._active_backtest_log_mode = self._normalize_backtest_log_mode(mode)
        self._active_backtest_log_context = {
            "coins": int(context.get("coins", 1) or 1),
            "strategies": int(context.get("strategies", 1) or 1),
            "timeframes": int(context.get("timeframes", 1) or 1),
        }

    def _write_backtest_detail_log(self, message: str) -> None:
        try:
            GUI_BACKTEST_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with GUI_BACKTEST_DEBUG_LOG_PATH.open("a", encoding="utf-8") as detail_file:
                detail_file.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass

    @staticmethod
    def _summarize_profile_blob(profile_blob: str) -> str:
        pairs = {
            key: value.strip()
            for key, value in re.findall(r"'([^']+)':\s*([^,}]+)", profile_blob)
        }
        if not pairs:
            return "profile"

        parts: list[str] = []
        if {"ema_fast", "ema_mid", "ema_slow"}.issubset(pairs):
            parts.append(f"EMA{pairs['ema_fast']}/{pairs['ema_mid']}/{pairs['ema_slow']}")
        elif {"ema_fast_period", "ema_slow_period"}.issubset(pairs):
            parts.append(f"EMA{pairs['ema_fast_period']}/{pairs['ema_slow_period']}")
        elif {"frama_fast_period", "frama_slow_period"}.issubset(pairs):
            parts.append(f"FRAMA{pairs['frama_fast_period']}/{pairs['frama_slow_period']}")
        elif {"dual_thrust_period", "dual_thrust_k1", "dual_thrust_k2"}.issubset(pairs):
            parts.append(
                "DT "
                f"p={pairs['dual_thrust_period']} "
                f"k1={pairs['dual_thrust_k1']} "
                f"k2={pairs['dual_thrust_k2']}"
            )
        for key, label in (
            ("slope_lookback", "slope"),
            ("min_ema_spread_pct", "spread"),
            ("volume_multiplier", "vol"),
            ("rsi_length", "rsi"),
            ("volume_ma_length", "volMA"),
            ("atr_stop_buffer_mult", "atrBuf"),
            ("stop_loss_pct", "SL"),
            ("take_profit_pct", "TP"),
            ("trailing_activation_pct", "TrailOn"),
            ("trailing_distance_pct", "TrailDist"),
            ("breakeven_activation_pct", "BE"),
        ):
            if key in pairs:
                parts.append(f"{label}={pairs[key]}")
        if "use_rsi_filter" in pairs:
            parts.append("RSI=on" if int(float(pairs["use_rsi_filter"])) > 0 else "RSI=off")
        if "use_volume_filter" in pairs:
            parts.append("VOL=on" if int(float(pairs["use_volume_filter"])) > 0 else "VOL=off")
        if "use_atr_stop_buffer" in pairs:
            parts.append("ATR=on" if int(float(pairs["use_atr_stop_buffer"])) > 0 else "ATR=off")
        return " | ".join(parts[:8]) if parts else "profile"

    @classmethod
    def _compact_profile_preview(cls, profile: object) -> str:
        if not isinstance(profile, dict) or not profile:
            return "{}"
        blob = ", ".join(f"'{key}': {value}" for key, value in profile.items())
        return cls._summarize_profile_blob(blob)

    def _compact_backtest_thread_log_message(self, message: str) -> str | None:
        mode = self._active_backtest_log_mode
        if mode == self._BACKTEST_LOG_MODE_DEBUG:
            return message

        if message.startswith("BACKTEST_TRACE|"):
            return None

        suppressed_in_standard = (
            "top profile diagnostics",
            "Stage 1 #",
            "Stage 2 #",
            "Backtest runtime leverage",
            "Backtest fee stress active",
            "Backtest slippage penalty active",
            "Tiered trailing summary:",
            "Trailing Level 1",
            "Trailing Level 2",
            "Trailing Level 3",
            "Dynamic SL/TP overrides applied",
            "Time-Stop exits applied",
            "EMA Band Rejection JIT alignment check passed",
            "EMA Band Rejection end-to-end consistency probe passed",
            "Numba JIT path active for",
            "Python vector path active for",
            "Intermediate optimizer checkpoint cache restored",
        )
        if any(token in message for token in suppressed_in_standard):
            return None

        if mode != self._BACKTEST_LOG_MODE_QUIET:
            return message

        quiet_allow_tokens = (
            "Backtest strategy selected:",
            "Backtest history window",
            "Historical data range loaded:",
            "Backtest window in use:",
            "optimizer mode:",
            "Stage 1: evaluating",
            "Stage 2: verifying",
            "Stage 1 fail-reason summary:",
            "Stage 2 fail-reason summary:",
            "Stage 2 Verified Candidate:",
            "Stage 2 Low-Edge Candidate:",
            "Best Optimizer Candidate",
            "Low-Edge Optimizer Candidate",
            "Final Verified Result",
            "Optimizer verdict for",
            "Sampling mode active:",
            "Full Scan active:",
            "No eligible profile under strict avg_profit gate;",
            "No optimization profile passed stability constraints",
            "Stage 2 verification produced no eligible profile",
            "Optimization did not produce any result.",
            "Backtest result saved to DB",
            "Optimization for ",
            "HMM regime detector failed",
            "JIT alignment check failed",
            "consistency probe failed",
            "consistency probe error",
            "EMA Band Rejection audit",
        )
        if not any(token in message for token in quiet_allow_tokens):
            return None

        if "profile={" in message:
            message = re.sub(
                r"profile=\{([^}]*)\}",
                lambda match: f"profile={self._summarize_profile_blob(match.group(1))}",
                message,
                count=1,
            )
        return message

    def _handle_backtest_thread_log(self, message: str) -> None:
        self._write_backtest_detail_log(message)
        compact_message = self._compact_backtest_thread_log_message(message)
        if compact_message is None:
            return
        self._append_backtest_log(compact_message)

    def _start_backtest(
        self,
        *,
        symbol_override: str | None = None,
        strategy_override: str | None = None,
        auto_from_config: bool | None = None,
        interval_override: str | None = None,
        allow_auto_interval_override: bool | None = None,
    ) -> None:
        self._sync_backtest_symbol_universe()
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            return
        live_running = self._bot_thread is not None and self._bot_thread.isRunning()
        self._backtest_stop_requested = False
        if not self._batch_active:
            self._begin_backtest_report_session(mode="single")

        if symbol_override is None:
            selected_symbols = self._get_selected_backtest_symbols()
            if len(selected_symbols) > 1:
                self._append_backtest_log(
                    f"Multiple coins selected ({len(selected_symbols)}). Starting selected batch instead of single-coin backtest."
                )
                self._on_run_selected_clicked()
                return
            selected_strategies = self._get_selected_backtest_strategy_names()
            if len(selected_strategies) > 1:
                self._append_backtest_log(
                    f"Multiple strategies selected ({len(selected_strategies)}). Starting selected batch instead of single-coin backtest."
                )
                self._on_run_selected_clicked()
                return
            symbol = selected_symbols[0] if len(selected_symbols) == 1 else self._current_backtest_symbol()
        else:
            symbol = symbol_override
        selected_backtest_interval = interval_override or self._current_backtest_interval()
        selected_strategy_name = strategy_override or self._current_backtest_strategy_name()
        auto_strategy_mode = (
            self._is_backtest_auto_strategy(selected_strategy_name)
            if auto_from_config is None
            else bool(auto_from_config)
        )
        if interval_override is not None:
            backtest_interval = resolve_interval_for_symbol(
                symbol,
                selected_backtest_interval,
                use_coin_override=False,
            )
            interval_source = "batch interval override"
        elif auto_strategy_mode:
            backtest_interval = resolve_interval_for_symbol(
                symbol,
                selected_backtest_interval,
                use_coin_override=True,
            )
            interval_source = "coin profile (Auto from Config)"
        else:
            backtest_interval = resolve_interval_for_symbol(
                symbol,
                selected_backtest_interval,
                use_coin_override=False,
            )
            interval_source = "GUI selection"
        leverage = self._current_backtest_leverage()
        min_confidence_pct = float(self.confidence_spin.value())
        strategy_name = self._resolve_backtest_strategy_for_symbol(symbol, selected_strategy_name)
        selected_strategy_label = (
            STRATEGY_LABELS[BACKTEST_AUTO_STRATEGY]
            if auto_strategy_mode
            else STRATEGY_LABELS.get(selected_strategy_name, selected_strategy_name)
        )
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        optimize_profile = self._is_optimization_mode_active()
        if optimize_profile:
            grid_size = len(generate_optimization_grid(strategy_name, symbol=symbol))
            strategy_log_label = (
                f"{selected_strategy_label} -> {strategy_label}"
                if auto_strategy_mode
                else strategy_label
            )
            self._append_backtest_log(
                f"Optimizing profile for {symbol} {backtest_interval} with selected strategy "
                f"{strategy_log_label}, "
                f"configured leverage {leverage}x, min confidence {min_confidence_pct:.0f}%, "
                f"and {grid_size} parameter combinations."
            )
        else:
            strategy_log_label = (
                f"{selected_strategy_label} -> {strategy_label}"
                if auto_strategy_mode
                else strategy_label
            )
            self._append_backtest_log(
                f"Running backtest for {symbol} {backtest_interval} with selected strategy {strategy_log_label} "
                f"with configured leverage {leverage}x (see runtime log for effective leverage) "
                f"and min confidence {min_confidence_pct:.0f}%."
            )
        self._append_backtest_log(
            f"Backtest profile defaults active for {symbol}: interval={backtest_interval} "
            f"(from {interval_source}; "
            "risk settings loaded from coin_profiles at runtime)."
        )
        if live_running:
            self._append_backtest_log(
                "Live bot is active. Backtest will use an isolated temp DB so both can run in parallel."
            )
        history_requested_start_utc: datetime | None = None
        history_end_utc: datetime | None = None
        if self._backtest_period_user_modified:
            history_requested_start_utc, history_end_utc = self._current_backtest_history_range()
        if history_requested_start_utc is not None:
            requested_end_text = (
                history_end_utc.strftime("%Y-%m-%d %H:%M:%S")
                if history_end_utc is not None
                else "latest available"
            )
            self._append_backtest_log(
                "Backtest period selection: "
                f"{history_requested_start_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC -> "
                f"{requested_end_text} UTC."
            )

        log_mode, log_context = self._resolve_backtest_log_mode_for_run(
            symbol_override=symbol_override,
            strategy_override=strategy_override,
            interval_override=interval_override,
        )
        self._set_active_backtest_log_mode(log_mode, log_context)
        self._append_backtest_log(
            "Backtest log mode: "
            f"{log_mode.upper()} "
            f"(coins={log_context['coins']}, strategies={log_context['strategies']}, timeframes={log_context['timeframes']})."
        )

        self.backtest_button.setEnabled(False)
        backtest_thread = BacktestThread(
            symbol=symbol,
            interval=backtest_interval,
            strategy_name=strategy_name,
            auto_from_config=auto_strategy_mode,
            allow_auto_interval_override=(
                bool(allow_auto_interval_override)
                if allow_auto_interval_override is not None
                else interval_override is None
            ),
            leverage=leverage,
            min_confidence_pct=min_confidence_pct,
            optimize_profile=optimize_profile,
            isolated_db=live_running,
            db_path=self._db_path,
            history_requested_start_utc=history_requested_start_utc,
            history_end_utc=history_end_utc,
            log_mode=log_mode,
        )
        self._backtest_thread = backtest_thread
        backtest_thread.log_message.connect(self._handle_backtest_thread_log)
        backtest_thread.progress_update.connect(self._update_progress)
        backtest_thread.backtest_finished.connect(self._handle_backtest_finished)
        backtest_thread.backtest_error.connect(self._handle_backtest_error)
        backtest_thread.finished.connect(
            lambda finished_thread=backtest_thread: self._handle_backtest_thread_finished(
                finished_thread
            )
        )
        backtest_thread.start()
        self.symbol_combo.setEnabled(False)
        self.backtest_coin_combo.setEnabled(False)
        self.backtest_strategy_combo.setEnabled(False)
        self.backtest_interval_combo.setEnabled(False)
        self.backtest_leverage_slider.setEnabled(False)
        self._set_backtest_period_controls_enabled(False)
        self._set_backtest_symbol_controls_enabled(False)
        self.confidence_spin.setEnabled(False)
        self._apply_button_states()
        self._refresh_live_symbol_cards()
        self._refresh_backtest_symbol_cards()

    def _on_run_all_clicked(self) -> None:
        self._sync_backtest_symbol_universe()
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            return

        symbols = list(_configured_backtest_symbols())
        if not symbols:
            self._append_backtest_log("No symbols available for batch backtest.")
            return

        optimize_profile = self._is_optimization_mode_active()
        self._backtest_stop_requested = False
        self._begin_backtest_report_session(mode="batch")
        self._batch_queue, selected_strategies, strategy_mode_label = self._build_backtest_batch_queue(symbols)
        self._batch_mode = "standard"
        self._batch_active = True
        self._initialize_batch_tracking(
            [symbol for symbol, _strategy_name, _interval_override in self._batch_queue]
        )
        if optimize_profile:
            self._append_backtest_log(
                f"Starting batch profile optimization for {len(symbols)} coins using "
                f"{strategy_mode_label} ({len(self._batch_queue)} runs)."
            )
        else:
            self._append_backtest_log(
                f"Starting batch backtest for {len(symbols)} coins using "
                f"{strategy_mode_label} ({len(self._batch_queue)} runs)."
            )
        if self._bot_thread is not None and self._bot_thread.isRunning():
            self._append_backtest_log(
                "Live bot is active. Batch runs will use isolated temp DBs to avoid lock contention."
            )
        self._apply_button_states()
        self._process_next_in_batch()

    def _on_run_selected_clicked(self) -> None:
        self._sync_backtest_symbol_universe()
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            return

        symbols = self._get_selected_backtest_symbols()
        if not symbols:
            self._append_backtest_log("Select at least one coin in the backtest selector first.")
            return

        optimize_profile = self._is_optimization_mode_active()
        self._backtest_stop_requested = False
        self._begin_backtest_report_session(mode="batch")
        self._batch_queue, selected_strategies, strategy_mode_label = self._build_backtest_batch_queue(symbols)
        self._batch_mode = "standard"
        self._batch_active = True
        self._initialize_batch_tracking(
            [symbol for symbol, _strategy_name, _interval_override in self._batch_queue]
        )
        if optimize_profile:
            self._append_backtest_log(
                f"Starting selected profile optimization for {len(symbols)} coins using "
                f"{strategy_mode_label} ({len(self._batch_queue)} runs)."
            )
        else:
            self._append_backtest_log(
                f"Starting selected backtest for {len(symbols)} coins using "
                f"{strategy_mode_label} ({len(self._batch_queue)} runs)."
            )
        if self._bot_thread is not None and self._bot_thread.isRunning():
            self._append_backtest_log(
                "Live bot is active. Selected runs will use isolated temp DBs to avoid lock contention."
            )
        self._apply_button_states()
        self._process_next_in_batch()

    def _process_next_in_batch(self) -> None:
        if not self._batch_active:
            return
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            QTimer.singleShot(50, self._process_next_in_batch)
            return
        if not self._batch_queue:
            self._batch_active = False
            self._batch_mode = "standard"
            self._batch_auto_strategy_flags.clear()
            self._batch_total_symbols = 0
            self._batch_started_symbols.clear()
            self._batch_completed_symbols.clear()
            self._batch_symbol_remaining_runs.clear()
            self._set_active_backtest_log_mode(
                self._BACKTEST_LOG_MODE_STANDARD,
                {"coins": 1, "strategies": 1, "timeframes": 1},
            )
            self._append_backtest_log("Batch Backtest completed!")
            self._finalize_backtest_report_session(status="completed")
            self.backtest_coin_combo.setEnabled(True)
            self.backtest_strategy_combo.setEnabled(True)
            self.backtest_interval_combo.setEnabled(True)
            self.backtest_leverage_slider.setEnabled(True)
            self._set_backtest_period_controls_enabled(True)
            self._set_backtest_symbol_controls_enabled(True)
            if self._bot_thread is None or not self._bot_thread.isRunning():
                self.symbol_combo.setEnabled(True)
                self._set_live_symbol_controls_enabled(True)
                self.confidence_spin.setEnabled(True)
            self._apply_button_states()
            self._refresh_backtest_symbol_cards()
            return

        symbol, strategy_name, interval_override = self._batch_queue.pop(0)
        if symbol not in self._batch_started_symbols:
            self._batch_started_symbols.add(symbol)
            batch_progress_message = (
                f"Coin {len(self._batch_started_symbols)}/{max(self._batch_total_symbols, 1)} starting: {symbol}"
            )
            self._append_backtest_log(batch_progress_message)
            self.statusBar().showMessage(batch_progress_message, 4000)
        self._set_current_backtest_symbol(symbol)
        if interval_override is not None and self.backtest_interval_combo.currentText() != interval_override:
            was_blocked = self.backtest_interval_combo.blockSignals(True)
            self.backtest_interval_combo.setCurrentText(interval_override)
            self.backtest_interval_combo.blockSignals(was_blocked)
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        auto_from_config_for_run = (
            bool(self._batch_auto_strategy_flags.pop(0))
            if self._batch_auto_strategy_flags
            else False
        )
        if not auto_from_config_for_run:
            self._set_current_backtest_strategy(strategy_name)
        else:
            strategy_label = f"{strategy_label} [Auto]"
        strategy_context_label = (
            f"{strategy_label} @ {interval_override}"
            if interval_override is not None
            else strategy_label
        )
        if self._is_optimization_mode_active():
            self._append_backtest_log(
                f"--- Starting next optimization: {symbol} / {strategy_context_label} ---"
            )
        else:
            self._append_backtest_log(
                f"--- Starting next backtest: {symbol} / {strategy_context_label} ---"
            )
        self._start_backtest(
            symbol_override=symbol,
            strategy_override=strategy_name,
            auto_from_config=auto_from_config_for_run,
            interval_override=interval_override,
            allow_auto_interval_override=False,
        )

    def _stop_bot(self) -> None:
        if self._bot_thread is None:
            return
        if self._bot_thread.isRunning():
            self._append_live_log("Stopping bot.")
            self._bot_thread.stop()
            self._bot_stop_requested = True
            self._apply_button_states()
            self.status_label.setText("Stopping...")
            return
        self._handle_thread_finished()

    def _stop_backtest(self) -> None:
        backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
        batch_active = self._batch_active
        if not backtest_running and not batch_active:
            return

        if batch_active:
            remaining_runs = len(self._batch_queue) + (1 if backtest_running else 0)
            self._batch_active = False
            self._batch_mode = "standard"
            self._batch_queue.clear()
            self._batch_auto_strategy_flags.clear()
            self._batch_total_symbols = 0
            self._batch_started_symbols.clear()
            self._batch_completed_symbols.clear()
            self._batch_symbol_remaining_runs.clear()
            self._set_active_backtest_log_mode(
                self._BACKTEST_LOG_MODE_STANDARD,
                {"coins": 1, "strategies": 1, "timeframes": 1},
            )
            self._append_backtest_log(
                f"Stopping batch backtest ({remaining_runs} remaining run(s) canceled)."
            )
        else:
            self._append_backtest_log("Stopping backtest.")

        if backtest_running and self._backtest_thread is not None:
            self._backtest_thread.stop()
            self._backtest_stop_requested = True
            self._apply_button_states()
            return

        self._backtest_stop_requested = False
        self._handle_backtest_thread_finished()

    def _handle_thread_finished(self) -> None:
        if self._bot_thread is not None:
            self._bot_thread.deleteLater()
        self._bot_thread = None
        self._bot_stop_requested = False
        self._live_runtime_profiles.clear()
        with Database(self._db_path) as db:
            self._realized_pnl_total = db.fetch_realized_pnl()
            self._realized_pnl_today = db.fetch_realized_pnl_since(self._today_start())
        self._reset_progress("Live bot stopped.")
        self.symbol_combo.setEnabled(True)
        self._set_live_symbol_controls_enabled(True)
        self.leverage_spin.setEnabled(True)
        self.confidence_spin.setEnabled(self._backtest_thread is None or not self._backtest_thread.isRunning())
        self.strategy_combo.setEnabled(True)
        self.backtest_coin_combo.setEnabled(True)
        self.backtest_strategy_combo.setEnabled(True)
        self.backtest_interval_combo.setEnabled(True)
        self._set_backtest_period_controls_enabled(True)
        self._heartbeat_snapshot = {}
        self.sync_status_label.setText("Sync: -")
        self.sync_status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700;")
        self._apply_button_states()
        self._update_status_label()
        self._refresh_live_symbol_cards()
        self._refresh_backtest_symbol_cards()
        if hasattr(self, "readiness_bars"):
            self.readiness_bars.clear_items()
        if self._pending_close_request:
            bot_running = self._bot_thread is not None and self._bot_thread.isRunning()
            backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
            if not bot_running and not backtest_running:
                self._pending_close_request = False
                QTimer.singleShot(0, self.close)

    def _handle_backtest_finished(self, result: dict) -> None:
        (
            compact_result,
            trimmed_trades,
            trimmed_optimization_rows,
        ) = self._compact_backtest_result_for_gui(result)
        compact_summary = generate_compact_summary(result)
        compact_result["compact_summary"] = dict(compact_summary)
        self._last_backtest_result = compact_result
        try:
            self._record_backtest_report_entry(result)
        except Exception as exc:
            logging.getLogger("gui").exception("Backtest report entry failed")
            self._append_backtest_log(
                f"Backtest report warning: failed to record summary entry ({exc})."
            )
            self._backtest_report_errors.append(str(exc))
        symbol = str(result.get("symbol") or self._current_backtest_symbol())
        strategy_name = self._sanitize_backtest_strategy_name(
            symbol,
            str(result.get("strategy_name") or self._current_backtest_strategy_name()),
        )
        interval = str(result.get("interval") or self._current_backtest_interval())
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        profit_factor = float(compact_summary.get("robust_profit_factor", 0.0) or 0.0)
        profit_factor_display = str(compact_summary.get("robust_profit_factor_display", f"{profit_factor:.2f}") or f"{profit_factor:.2f}")
        total_pnl_usd = float(compact_summary.get("net_pnl_usd", 0.0) or 0.0)
        win_rate_pct = float(compact_summary.get("win_rate_pct", 0.0) or 0.0)
        total_trades = int(compact_summary.get("trade_count", 0) or 0)
        long_trades = int(result.get("long_trades", 0) or 0)
        short_trades = int(result.get("short_trades", 0) or 0)
        if (long_trades + short_trades) <= 0 and total_trades > 0:
            trade_rows = compact_result.get("closed_trades", [])
            if isinstance(trade_rows, list):
                long_trades = sum(
                    1
                    for trade in trade_rows
                    if str(trade.get("side", "")).strip().upper() == "LONG"
                )
                short_trades = sum(
                    1
                    for trade in trade_rows
                    if str(trade.get("side", "")).strip().upper() == "SHORT"
                )
        existing_summary = self._backtest_compact_summaries.get(symbol)
        incoming_fitness_score = calculate_fitness_score(compact_summary)
        existing_fitness_score = (
            calculate_fitness_score(existing_summary)
            if isinstance(existing_summary, dict)
            else float("-inf")
        )
        incoming_is_winner = (
            not isinstance(existing_summary, dict)
            or incoming_fitness_score > existing_fitness_score
        )

        best_profile = result.get("best_profile", {})
        normalized_profile: dict[str, float] = {}
        if isinstance(best_profile, dict) and best_profile:
            for key, value in best_profile.items():
                with suppress(Exception):
                    normalized_profile[str(key)] = float(value)

        if incoming_is_winner:
            compact_summary["strategy_name"] = strategy_name
            self._backtest_profit_factors[symbol] = profit_factor
            self._backtest_win_rates[symbol] = win_rate_pct
            self._backtest_trade_counts[symbol] = total_trades
            self._backtest_compact_summaries[symbol] = dict(compact_summary)
            if normalized_profile:
                self._backtest_best_profiles[symbol] = dict(normalized_profile)
            else:
                self._backtest_best_profiles.pop(symbol, None)
        else:
            existing_strategy_name = "-"
            if isinstance(existing_summary, dict):
                existing_strategy_name = self._sanitize_backtest_strategy_name(
                    symbol,
                    str(existing_summary.get("strategy_name", "-") or "-"),
                )
            self._append_backtest_log(
                "King of the Hill hold: "
                f"{symbol} kept {existing_strategy_name} "
                f"(score={existing_fitness_score:.2f}) over {strategy_label} "
                f"(score={incoming_fitness_score:.2f})."
            )
        self._update_live_symbol_card(symbol)
        winner_tag = "[WINNER] " if profit_factor > 1.0 else ""
        verdict = "PASS" if (total_pnl_usd > 0.0 and profit_factor >= 1.0) else "WARN"
        self.backtest_total_pnl_label.setText(f"Total PnL: {total_pnl_usd:,.2f} USD")
        self.backtest_win_rate_label.setText(f"Win Rate: {win_rate_pct:,.2f}%")
        self.backtest_total_trades_label.setText(f"Total Trades: {total_trades}")
        if trimmed_trades > 0:
            self._append_backtest_log(
                f"GUI memory guard: stored only the latest {self._MAX_BACKTEST_RESULT_TRADES} "
                f"closed trades ({trimmed_trades} trimmed from UI payload)."
            )
        if trimmed_optimization_rows > 0:
            self._append_backtest_log(
                f"GUI memory guard: kept top {self._MAX_BACKTEST_OPTIMIZATION_RESULTS} optimization rows "
                f"({trimmed_optimization_rows} trimmed from UI payload)."
            )
        if result.get("optimization_mode"):
            sample_note = ""
            if result.get("sampled_profiles") is not None and result.get("evaluated_profiles") is not None:
                sample_note = (
                    f" sampled={int(result.get('sampled_profiles', 0) or 0)}/"
                    f"{int(result.get('evaluated_profiles', 0) or 0)}"
                )
            if result.get("validated_profiles") is not None:
                sample_note += f" validated={int(result.get('validated_profiles', 0) or 0)}"
            optimizer_quality_status = str(
                compact_summary.get("optimizer_quality_status")
                or result.get("optimizer_quality_status")
                or ""
            ).strip().upper()
            if optimizer_quality_status:
                sample_note += f" quality={optimizer_quality_status}"
            best_profile_preview = (
                self._compact_profile_preview(best_profile)
                if self._active_backtest_log_mode == self._BACKTEST_LOG_MODE_QUIET
                else str(best_profile)
            )
            self._append_backtest_log(
                "Optimization finished: "
                f"{'[KOTH WIN] ' if incoming_is_winner else '[KOTH HOLD] '}"
                f"{winner_tag}{symbol} "
                f"[{strategy_label} {interval}] "
                f"best_profile={best_profile_preview} "
                f"pnl={total_pnl_usd:,.2f} "
                f"win_rate={win_rate_pct:,.2f}% "
                f"trades={total_trades} (L: {long_trades} / S: {short_trades}) "
                f"profit_factor={profit_factor_display} "
                f"max_dd={float(result.get('max_drawdown_pct', 0.0)):.2f}% "
                f"longest_losses={int(result.get('longest_consecutive_losses', 0) or 0)}"
                f" verdict={verdict}"
                f"{sample_note}"
            )
        else:
            self._append_backtest_log(
                "Backtest finished: "
                f"{'[KOTH WIN] ' if incoming_is_winner else '[KOTH HOLD] '}"
                f"{winner_tag}{symbol} "
                f"[{strategy_label} {interval}] "
                f"pnl={total_pnl_usd:,.2f} "
                f"win_rate={win_rate_pct:,.2f}% "
                f"trades={total_trades} (L: {long_trades} / S: {short_trades}) "
                f"profit_factor={profit_factor_display} "
                f"verdict={verdict}"
            )
        self._mark_batch_symbol_progress(symbol, success=True)
        if self._batch_active:
            QTimer.singleShot(0, self._process_next_in_batch)
        else:
            final_status = "stopped" if self._backtest_stop_requested else "completed"
            self._finalize_backtest_report_session(status=final_status)
            self._refresh_live_symbol_cards()
        self._refresh_backtest_symbol_cards()

    def _handle_backtest_error(self, message: str) -> None:
        logging.getLogger("gui").error("Backtest failed: %s", message)
        self._reset_progress(f"Backtest failed: {message}")
        self._append_backtest_log(f"Backtest failed: {message}")
        self._backtest_report_errors.append(str(message))
        self._mark_batch_symbol_progress(self._current_backtest_symbol(), success=False)
        if self._batch_active:
            QTimer.singleShot(0, self._process_next_in_batch)

    def _initialize_batch_tracking(self, run_symbols: list[str]) -> None:
        self._batch_total_symbols = len(dict.fromkeys(run_symbols))
        self._batch_started_symbols.clear()
        self._batch_completed_symbols.clear()
        self._batch_symbol_remaining_runs = dict(Counter(run_symbols))

    def _mark_batch_symbol_progress(self, symbol: str, *, success: bool) -> None:
        if not self._batch_active:
            return
        remaining_runs = self._batch_symbol_remaining_runs.get(symbol)
        if remaining_runs is None:
            return
        remaining_runs -= 1
        if remaining_runs > 0:
            self._batch_symbol_remaining_runs[symbol] = remaining_runs
            return
        self._batch_symbol_remaining_runs.pop(symbol, None)
        self._batch_completed_symbols.add(symbol)
        status_text = "completed" if success else "finished with errors"
        progress_message = (
            f"Coin {len(self._batch_completed_symbols)}/{max(self._batch_total_symbols, 1)} "
            f"{status_text}: {symbol}"
        )
        self._append_backtest_log(progress_message)
        self.statusBar().showMessage(progress_message, 5000)

    def _handle_backtest_thread_finished(
        self,
        finished_thread: BacktestThread | None = None,
    ) -> None:
        if finished_thread is not None:
            finished_thread.deleteLater()
            if finished_thread is self._backtest_thread:
                self._backtest_thread = None
        elif self._backtest_thread is not None:
            self._backtest_thread.deleteLater()
            self._backtest_thread = None

        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            return

        if not self._batch_active and self._backtest_report_pending:
            final_status = "stopped" if self._backtest_stop_requested else "completed"
            self._finalize_backtest_report_session(status=final_status)
        self._backtest_stop_requested = False
        if self._batch_active:
            return
        self._set_active_backtest_log_mode(
            self._BACKTEST_LOG_MODE_STANDARD,
            {"coins": 1, "strategies": 1, "timeframes": 1},
        )
        self.backtest_coin_combo.setEnabled(True)
        self.backtest_strategy_combo.setEnabled(True)
        self.backtest_interval_combo.setEnabled(True)
        self.backtest_leverage_slider.setEnabled(True)
        self._set_backtest_period_controls_enabled(True)
        self._set_backtest_symbol_controls_enabled(True)
        if self._bot_thread is None or not self._bot_thread.isRunning():
            self.symbol_combo.setEnabled(True)
            self._set_live_symbol_controls_enabled(True)
            self.confidence_spin.setEnabled(True)
        self._apply_button_states()
        self._refresh_live_symbol_cards()
        self._refresh_backtest_symbol_cards()
        if self._pending_close_request:
            bot_running = self._bot_thread is not None and self._bot_thread.isRunning()
            backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
            if not bot_running and not backtest_running:
                self._pending_close_request = False
                QTimer.singleShot(0, self.close)

    def _on_leverage_changed(self, value: int) -> None:
        if self._bot_thread is not None:
            self._bot_thread.set_leverage(value)
        self._update_status_label()

    def _on_backtest_leverage_changed(self, value: int) -> None:
        self.backtest_leverage_value_label.setText(f"{int(value)}x")
        self._update_status_label()

    def _handle_trade_opened(self, trade_payload: dict) -> None:
        normalized_trade = dict(trade_payload)
        normalized_trade["symbol"] = str(normalized_trade.get("symbol", "")).strip().upper()
        self._positions[int(normalized_trade["id"])] = normalized_trade
        self._sync_positions_table()
        self._refresh_live_symbol_cards()

    def _handle_positions_updated(self, positions: list) -> None:
        normalized_positions: dict[int, dict] = {}
        for position in positions:
            normalized_position = dict(position)
            normalized_position["symbol"] = str(normalized_position.get("symbol", "")).strip().upper()
            normalized_positions[int(normalized_position["id"])] = normalized_position
        self._positions = normalized_positions
        self._sync_positions_table()
        self._refresh_live_symbol_cards()

    def _handle_price_update(self, symbol: str, price: float) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return
        self._latest_prices[normalized_symbol] = price
        self._upsert_feed_row(normalized_symbol, price)
        self._flash_market_pulse()
        self._sync_positions_table()
        self._refresh_live_symbol_cards()

    def _handle_runtime_profile_update(self, payload: dict) -> None:
        normalized_symbol = str(payload.get("symbol", "")).strip().upper()
        if not normalized_symbol:
            return
        self._live_runtime_profiles[normalized_symbol] = dict(payload)
        self._refresh_live_symbol_cards()
        self._refresh_trade_readiness(force=True)

    def _update_heartbeat_display(self, status_dict: dict) -> None:
        self._heartbeat_snapshot = {
            str(symbol).strip().upper(): dict(info) for symbol, info in status_dict.items()
        }
        lagging_symbols = [
            symbol
            for symbol, info in self._heartbeat_snapshot.items()
            if str(info.get("status", "OK")).upper() == "LAGGING"
        ]

        if lagging_symbols:
            self.sync_status_label.setText(
                f"Sync: LAGGING ({len(lagging_symbols)}/{len(self._heartbeat_snapshot)})"
            )
            self.sync_status_label.setStyleSheet(
                f"color: {ACCENT_WARNING}; font-weight: 700;"
            )
        elif self._heartbeat_snapshot:
            self.sync_status_label.setText(f"Sync: OK ({len(self._heartbeat_snapshot)})")
            self.sync_status_label.setStyleSheet(
                f"color: {ACCENT_PROFIT}; font-weight: 700;"
            )
        else:
            self.sync_status_label.setText("Sync: -")
            self.sync_status_label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-weight: 700;"
            )

        for symbol in self._heartbeat_snapshot:
            self._apply_feed_row_heartbeat_style(symbol)
        self._refresh_live_symbol_cards()

    def _upsert_feed_row(self, symbol: str, price: float) -> None:
        row = self._price_rows.get(symbol)
        if row is None:
            row = self.feed_table.rowCount()
            self.feed_table.insertRow(row)
            self._price_rows[symbol] = row
            self.feed_table.setItem(row, 0, QTableWidgetItem(symbol))

        self.feed_table.setItem(row, 1, self._make_item(f"{price:,.4f}", align_right=True))
        self.feed_table.setItem(row, 2, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self._apply_feed_row_heartbeat_style(symbol)

    def _apply_feed_row_heartbeat_style(self, symbol: str) -> None:
        row = self._price_rows.get(symbol)
        if row is None:
            return

        heartbeat_info = self._heartbeat_snapshot.get(symbol, {})
        is_lagging = str(heartbeat_info.get("status", "OK")).upper() == "LAGGING"
        background = QColor("#5d4037") if is_lagging else QColor(PANEL_BACKGROUND)
        foreground = QColor("#ffb74d") if is_lagging else QColor(TEXT_PRIMARY)

        for column in range(self.feed_table.columnCount()):
            item = self.feed_table.item(row, column)
            if item is None:
                continue
            item.setBackground(background)
            item.setForeground(foreground)

    def _sync_positions_table(self) -> None:
        self.positions_table.setRowCount(0)
        total_unrealized_pnl = 0.0
        for row, trade_id in enumerate(sorted(self._positions)):
            trade = self._positions[trade_id]
            last_price = self._latest_prices.get(trade["symbol"], float(trade["entry_price"]))
            unrealized_pnl = self._calculate_unrealized_net_pnl(trade, last_price)
            total_unrealized_pnl += unrealized_pnl
            fees_paid = float(trade.get("total_fees", 0.0) or 0.0)
            entry_price = float(trade["entry_price"])
            quantity = float(trade["qty"])
            leverage = max(float(trade.get("leverage", 1) or 1), 1.0)
            invested_margin = (entry_price * quantity) / leverage

            self.positions_table.insertRow(row)
            self.positions_table.setItem(row, 0, self._make_item(str(trade["id"])))
            self.positions_table.setItem(row, 1, self._make_item(str(trade["symbol"])))
            self.positions_table.setItem(row, 2, self._make_item(str(trade["side"])))
            self.positions_table.setItem(row, 3, self._make_item(f"{entry_price:,.4f}", True))
            self.positions_table.setItem(row, 4, self._make_item(f"{last_price:,.4f}", True))
            self.positions_table.setItem(row, 5, self._make_item(f"{quantity:,.6f}", True))
            self.positions_table.setItem(row, 6, self._make_item(f"{int(leverage)}x", True))
            self.positions_table.setItem(row, 7, self._make_item(f"{invested_margin:,.2f}", True))
            self.positions_table.setItem(row, 8, self._make_item(f"{fees_paid:,.4f}", True))
            self.positions_table.setItem(row, 9, self._make_pnl_item(unrealized_pnl))
            self.positions_table.setCellWidget(row, 10, self._build_trade_health_bar(trade, last_price))
            self.positions_table.setItem(row, 11, self._make_item(self._format_timestamp(trade["entry_time"])))
        self._update_performance_header(total_unrealized_pnl)

    def _populate_backtest_table(self, closed_trades: list[dict]) -> None:
        if self.backtest_table is None:
            return
        self.backtest_table.setRowCount(0)
        trades_for_table = closed_trades
        if len(trades_for_table) > self._MAX_BACKTEST_TABLE_ROWS:
            trades_for_table = list(trades_for_table[-self._MAX_BACKTEST_TABLE_ROWS :])
        for row, trade in enumerate(trades_for_table):
            self.backtest_table.insertRow(row)
            self.backtest_table.setItem(row, 0, self._make_item(self._format_timestamp(trade.get("entry_time"))))
            self.backtest_table.setItem(row, 1, self._make_item(str(trade.get("side", "-"))))
            self.backtest_table.setItem(
                row,
                2,
                self._make_item(f"{float(trade.get('entry_price', 0.0)):,.4f}", True),
            )
            self.backtest_table.setItem(
                row,
                3,
                self._make_item(f"{float(trade.get('exit_price', 0.0)):,.4f}", True),
            )
            self.backtest_table.setItem(row, 4, self._make_pnl_item(float(trade.get("pnl", 0.0))))

    def _copy_positions_table_to_clipboard(self) -> None:
        self._copy_table_to_clipboard(self.positions_table, "Positions table")

    def _copy_backtest_table_to_clipboard(self) -> None:
        summary_text = self._build_backtest_summary_text().strip()
        if not summary_text:
            self.statusBar().showMessage("Backtest result is empty.", 3000)
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(summary_text)
        self.statusBar().showMessage("Backtest summary copied to clipboard.", 3000)
        self._append_backtest_log("Backtest summary copied to clipboard.")

    def _copy_backtest_summary_to_clipboard(self) -> None:
        summary_text = self._build_backtest_summary_text().strip()
        if not summary_text:
            self.statusBar().showMessage("Backtest summary is empty.", 3000)
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(summary_text)
        self.statusBar().showMessage("Backtest summary copied to clipboard.", 3000)
        self._append_backtest_log("Backtest summary copied to clipboard.")

    def _copy_backtest_log_to_clipboard(self) -> None:
        log_text = self.backtest_log_output.toPlainText().strip()
        if not log_text:
            self.statusBar().showMessage("Backtest log is empty.", 3000)
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(log_text)
        self.statusBar().showMessage("Backtest log copied to clipboard.", 3000)
        self._append_backtest_log("Backtest log copied to clipboard.")

    def _clear_backtest_log(self) -> None:
        self.backtest_log_output.clear()
        self.statusBar().showMessage("Backtest log cleared.", 3000)

    def _compact_backtest_result_for_gui(self, result: dict) -> tuple[dict, int, int]:
        compact = dict(result)
        trimmed_trades = 0
        trimmed_optimization_rows = 0

        closed_trades = compact.get("closed_trades")
        if isinstance(closed_trades, list) and len(closed_trades) > self._MAX_BACKTEST_RESULT_TRADES:
            trimmed_trades = len(closed_trades) - self._MAX_BACKTEST_RESULT_TRADES
            compact["closed_trades"] = list(closed_trades[-self._MAX_BACKTEST_RESULT_TRADES :])

        optimization_results = compact.get("optimization_results")
        if (
            isinstance(optimization_results, list)
            and len(optimization_results) > self._MAX_BACKTEST_OPTIMIZATION_RESULTS
        ):
            trimmed_optimization_rows = (
                len(optimization_results) - self._MAX_BACKTEST_OPTIMIZATION_RESULTS
            )
            compact["optimization_results"] = list(
                optimization_results[: self._MAX_BACKTEST_OPTIMIZATION_RESULTS]
            )

        return compact, trimmed_trades, trimmed_optimization_rows

    @staticmethod
    def _format_report_profit_factor(value: float) -> str:
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.2f}"

    @staticmethod
    def _classify_sample_quality(total_trades: int) -> tuple[bool, str]:
        if total_trades <= BACKTEST_REPORT_LOW_SAMPLE_TRADES_THRESHOLD:
            return True, "weak"
        if total_trades < BACKTEST_REPORT_STRONG_SAMPLE_TRADES_THRESHOLD:
            return False, "medium"
        return False, "strong"

    @staticmethod
    def _normalize_utc_text(value: object | None) -> str:
        if value is None:
            return "-"
        if isinstance(value, datetime):
            resolved = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return resolved.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        text = str(value).strip()
        if not text:
            return "-"
        return text.replace("T", " ").replace("Z", "")

    def _resolve_report_leverage(self, symbol: str, value: object | None) -> int:
        if value is not None:
            with suppress(Exception):
                parsed = int(value)
                if parsed > 0:
                    return parsed
        profile = settings.trading.coin_profiles.get(symbol)
        if profile is not None and profile.default_leverage is not None:
            return int(profile.default_leverage)
        return int(settings.trading.default_leverage)

    def _build_backtest_summary_text(self) -> str:
        result = self._last_backtest_result
        if not result:
            return ""
        compact_summary = (
            dict(result.get("compact_summary"))
            if isinstance(result.get("compact_summary"), dict)
            else generate_compact_summary(result)
        )

        symbol = str(result.get("symbol") or self._current_backtest_symbol())
        strategy_name = self._sanitize_backtest_strategy_name(
            symbol,
            str(result.get("strategy_name") or self._current_backtest_strategy_name()),
        )
        interval = str(result.get("interval") or self._current_backtest_interval())
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        total_trades = int(compact_summary.get("trade_count", result.get("total_trades", 0)) or 0)
        low_sample, sample_quality = self._classify_sample_quality(total_trades)
        effective_leverage = self._resolve_report_leverage(symbol, result.get("effective_leverage"))
        history_start_text = self._normalize_utc_text(
            result.get("history_start_utc") or result.get("history_requested_start_utc")
        )
        history_end_text = self._normalize_utc_text(result.get("history_end_utc"))
        history_candles = int(
            result.get("history_candles", result.get("search_window_candles", 0)) or 0
        )
        total_signals = int(result.get("total_signals", 0) or 0)
        approved_signals = int(result.get("approved_signals", 0) or 0)
        blocked_signals = int(result.get("blocked_signals", 0) or 0)
        gate_pass_rate_pct = (
            (float(approved_signals) / float(total_signals) * 100.0)
            if total_signals > 0
            else 0.0
        )
        strategy_diagnostics = (
            dict(result.get("strategy_diagnostics"))
            if isinstance(result.get("strategy_diagnostics"), dict)
            else {}
        )
        best_profile = result.get("best_profile")
        if not isinstance(best_profile, dict):
            best_profile = {}
        def _diag_int(field_name: str) -> int:
            with suppress(Exception):
                return int(strategy_diagnostics.get(field_name, 0) or 0)
            return 0

        def _diag_float(field_name: str) -> float:
            with suppress(Exception):
                return float(strategy_diagnostics.get(field_name, 0.0) or 0.0)
            return 0.0
        profit_factor = float(result.get("profit_factor", 0.0) or 0.0)
        robust_pf_display = str(compact_summary.get("robust_profit_factor_display", "0.00") or "0.00")
        avg_win_fee_ratio_display = str(compact_summary.get("avg_win_fee_ratio_display", "0.00") or "0.00")
        net_pnl_usd = float(compact_summary.get("net_pnl_usd", result.get("total_pnl_usd", 0.0)) or 0.0)
        lines = [
            "Backtest Summary",
            f"Symbol\t{symbol}",
            f"Strategy\t{strategy_label}",
            f"Interval\t{interval}",
            f"Net PnL USD\t{net_pnl_usd:.2f}",
            f"Win Rate %\t{float(result.get('win_rate_pct', 0.0) or 0.0):.2f}",
            f"Robust PF\t{robust_pf_display}",
            f"Profit Factor\t{self._format_report_profit_factor(profit_factor)}",
            f"Total Trades\t{total_trades}",
            f"Avg Win/Fee Ratio\t{avg_win_fee_ratio_display}",
            f"Average Win USD\t{float(result.get('average_win_usd', 0.0) or 0.0):.2f}",
            f"Average Loss USD\t{float(result.get('average_loss_usd', 0.0) or 0.0):.2f}",
            f"Real RRR\t{float(result.get('real_rrr', 0.0) or 0.0):.2f}",
            f"Max Drawdown %\t{float(result.get('max_drawdown_pct', 0.0) or 0.0):.2f}",
            f"Longest Losses\t{int(result.get('longest_consecutive_losses', 0) or 0)}",
            f"Total Signals\t{total_signals}",
            f"Approved / Blocked\t{approved_signals} / {blocked_signals}",
            f"Gate Pass Rate %\t{gate_pass_rate_pct:.2f}",
            f"Validated Profiles\t{int(result.get('validated_profiles', 0) or 0)}",
            f"Leverage\t{effective_leverage}x",
            "Leverage Mode\t"
            f"configured={int(result.get('configured_leverage', settings.trading.default_leverage) or settings.trading.default_leverage)}x "
            f"effective={effective_leverage}x",
            "HF Metrics\t"
            f"real_lev_avg={float(result.get('real_leverage_avg', 0.0) or 0.0):.2f}x",
            f"Stop / Take %\t{float(result.get('stop_loss_pct', 0.0) or 0.0):.2f} / {float(result.get('take_profit_pct', 0.0) or 0.0):.2f}",
            "Trailing Act / Dist %\t"
            f"{float(result.get('trailing_activation_pct', 0.0) or 0.0):.2f} / {float(result.get('trailing_distance_pct', 0.0) or 0.0):.2f}",
            f"Fee Side %\t{float(result.get('effective_taker_fee_pct', 0.0) or 0.0):.4f}",
            "Warmup / Required Candles\t"
            f"{int(result.get('required_warmup_candles', 0) or 0)} / {int(result.get('required_candles', 0) or 0)}",
            "Min Confidence %\t"
            + (
                f"{float(result.get('min_confidence_pct')):.2f}"
                if result.get("min_confidence_pct") is not None
                else "n/a"
            ),
            f"History Window UTC\t{history_start_text} -> {history_end_text}",
            f"History Candles\t{history_candles}",
            f"low_sample\t{str(low_sample).lower()}",
            f"sample_quality\t{sample_quality}",
        ]
        if strategy_diagnostics:
            lines.extend(
                [
                    f"Cluster Count\t{_diag_int('cluster_count')}",
                    f"Min Cluster Size\t{_diag_int('min_cluster_size')}",
                    f"Band Points\t{_diag_float('band_points'):.2f}",
                    f"Band Pct\t{_diag_float('band_pct'):.2f}",
                    f"RSI Period\t{_diag_int('rsi_period')}",
                    f"RSI SMA Period\t{_diag_int('rsi_sma_period')}",
                    f"Valid Cluster Events\t{_diag_int('valid_cluster_events')}",
                    f"Rejected Out-Of-Band Values\t{_diag_int('rejected_out_of_band_values')}",
                    f"Reset Count\t{_diag_int('reset_count')}",
                    f"Long Entry Count\t{_diag_int('long_entry_count')}",
                    f"Short Entry Count\t{_diag_int('short_entry_count')}",
                ]
            )
        if result.get("optimization_mode"):
            lines.append(f"Evaluated Profiles\t{int(result.get('evaluated_profiles', 0) or 0)}")
            lines.append(f"Validated Profiles\t{int(result.get('validated_profiles', 0) or 0)}")
            lines.append(f"Sampled Profiles\t{int(result.get('sampled_profiles', 0) or 0)}")
            lines.append(f"Theoretical Profiles\t{int(result.get('theoretical_profiles', 0) or 0)}")
            lines.append(f"Sampling Coverage %\t{float(result.get('sampling_coverage_pct', 0.0) or 0.0):.2f}")
            lines.append(f"Sampling Mode\t{str(result.get('sampling_mode', '') or 'n/a')}")
            optimizer_quality_status = str(
                result.get("optimizer_quality_status")
                or compact_summary.get("optimizer_quality_status")
                or ""
            ).strip().upper()
            if optimizer_quality_status:
                lines.append(f"Optimizer Verdict\t{optimizer_quality_status}")
            lines.append(f"Search Window Candles\t{int(result.get('search_window_candles', 0) or 0)}")
            lines.append(f"Optimizer Workers\t{int(result.get('optimizer_worker_processes', 0) or 0)}")
            lines.append(
                "Full History Optimization\t"
                + ("yes" if bool(result.get("full_history_optimization")) else "no")
            )
            lines.append(
                "Force Full Scan\t"
                + ("yes" if bool(result.get("optimizer_force_full_scan")) else "no")
            )
            session_leaderboard = (
                result.get("session_leaderboard")
                if result.get("session_leaderboard") is not None
                else result.get("session_day_breakdown")
            )
            if isinstance(session_leaderboard, list) and session_leaderboard:
                top_row = session_leaderboard[0]
                if isinstance(top_row, dict):
                    lines.append(
                        "Top Session\t"
                        f"{str(top_row.get('session', '') or '').strip().lower()} "
                        f"(Avg PF {float(top_row.get('avg_pf', 0.0) or 0.0):.2f}, "
                        f"Avg WR {float(top_row.get('avg_win_rate_pct', 0.0) or 0.0):.2f}%, "
                        f"Runs {int(top_row.get('runs', 0) or 0)})"
                    )
            best_profile = result.get("best_profile", {})
            lines.append(
                "Best Profile\t"
                + json.dumps(best_profile, ensure_ascii=True, sort_keys=True)
            )
        return "\n".join(lines)

    def _begin_backtest_report_session(self, *, mode: str) -> None:
        self._backtest_report_started_at = datetime.now(UTC)
        self._backtest_report_entries.clear()
        self._backtest_report_errors.clear()
        self._backtest_report_pending = True
        self._backtest_report_mode = mode

    def _record_backtest_report_entry(self, result: dict) -> None:
        if not self._backtest_report_pending:
            self._begin_backtest_report_session(mode="single")
        compact_summary = generate_compact_summary(result)
        symbol = str(result.get("symbol") or self._current_backtest_symbol())
        total_trades = int(compact_summary.get("trade_count", result.get("total_trades", 0)) or 0)
        total_signals = int(result.get("total_signals", 0) or 0)
        approved_signals = int(result.get("approved_signals", 0) or 0)
        blocked_signals = int(result.get("blocked_signals", 0) or 0)
        gate_pass_rate_pct = (
            (float(approved_signals) / float(total_signals) * 100.0)
            if total_signals > 0
            else 0.0
        )
        low_sample, sample_quality = self._classify_sample_quality(total_trades)
        optimization_mode = bool(result.get("optimization_mode"))
        default_history_start = (
            settings.trading.optimizer_history_start_utc
            if optimization_mode
            else settings.trading.backtest_history_start_utc
        )
        strategy_diagnostics = (
            dict(result.get("strategy_diagnostics"))
            if isinstance(result.get("strategy_diagnostics"), dict)
            else {}
        )
        best_profile = (
            dict(result.get("best_profile", {}))
            if isinstance(result.get("best_profile"), dict)
            else {}
        )
        raw_session_leaderboard = (
            result.get("session_leaderboard")
            if result.get("session_leaderboard") is not None
            else result.get("session_day_breakdown")
        )
        session_leaderboard: list[dict[str, object]] = []
        if isinstance(raw_session_leaderboard, list):
            for raw_row in raw_session_leaderboard:
                if not isinstance(raw_row, dict):
                    continue
                session_label = str(raw_row.get("session", "") or "").strip().lower()
                if not session_label:
                    continue
                with suppress(Exception):
                    session_leaderboard.append(
                        {
                            "session": session_label,
                            "avg_pf": float(raw_row.get("avg_pf", 0.0) or 0.0),
                            "avg_win_rate_pct": float(raw_row.get("avg_win_rate_pct", 0.0) or 0.0),
                            "avg_pnl_usd": float(raw_row.get("avg_pnl_usd", 0.0) or 0.0),
                            "runs": int(raw_row.get("runs", 0) or 0),
                            "top_runs": int(raw_row.get("top_runs", 0) or 0),
                        }
                    )
        session_top = session_leaderboard[0] if session_leaderboard else {}

        def _diag_int(field_name: str) -> int:
            with suppress(Exception):
                return int(strategy_diagnostics.get(field_name, 0) or 0)
            return 0

        def _diag_float(field_name: str) -> float:
            with suppress(Exception):
                return float(strategy_diagnostics.get(field_name, 0.0) or 0.0)
            return 0.0

        entry: dict[str, object] = {
            "symbol": symbol,
            "strategy_name": self._sanitize_backtest_strategy_name(
                symbol,
                str(result.get("strategy_name") or self._current_backtest_strategy_name()),
            ),
            "interval": str(result.get("interval") or self._current_backtest_interval()),
            "optimization_mode": optimization_mode,
            "total_pnl_usd": float(compact_summary.get("net_pnl_usd", result.get("total_pnl_usd", 0.0)) or 0.0),
            "profit_factor": float(compact_summary.get("robust_profit_factor", result.get("profit_factor", 0.0)) or 0.0),
            "robust_profit_factor_display": str(compact_summary.get("robust_profit_factor_display", "0.00") or "0.00"),
            "win_rate_pct": float(compact_summary.get("win_rate_pct", result.get("win_rate_pct", 0.0)) or 0.0),
            "total_trades": total_trades,
            "max_drawdown_pct": float(compact_summary.get("max_drawdown_pct", result.get("max_drawdown_pct", 0.0)) or 0.0),
            "avg_win_fee_ratio": float(compact_summary.get("avg_win_fee_ratio", 0.0) or 0.0),
            "avg_win_fee_ratio_display": str(compact_summary.get("avg_win_fee_ratio_display", "0.00") or "0.00"),
            "avg_win_usd": float(compact_summary.get("avg_win_usd", result.get("average_win_usd", 0.0)) or 0.0),
            "real_leverage_avg": float(compact_summary.get("real_leverage_avg", result.get("real_leverage_avg", 0.0)) or 0.0),
            "longest_consecutive_losses": int(result.get("longest_consecutive_losses", 0) or 0),
            "real_rrr": float(result.get("real_rrr", 0.0) or 0.0),
            "effective_leverage": self._resolve_report_leverage(
                symbol,
                result.get("effective_leverage"),
            ),
            "configured_leverage": int(
                result.get(
                    "configured_leverage",
                    settings.trading.default_leverage,
                )
                or settings.trading.default_leverage
            ),
            "history_requested_start_utc": self._normalize_utc_text(
                result.get("history_requested_start_utc")
                or default_history_start
            ),
            "history_start_utc": self._normalize_utc_text(
                result.get("history_start_utc")
                or result.get("history_requested_start_utc")
                or default_history_start
            ),
            "history_end_utc": self._normalize_utc_text(result.get("history_end_utc")),
            "history_candles": int(
                result.get("history_candles", result.get("search_window_candles", 0)) or 0
            ),
            "history_loader_mode": str(result.get("history_loader_mode", "") or "tail_window_latest"),
            "tail_window_limit_candles": int(result.get("tail_window_limit_candles", 0) or 0),
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "blocked_signals": blocked_signals,
            "gate_pass_rate_pct": float(gate_pass_rate_pct),
            "effective_taker_fee_pct": float(result.get("effective_taker_fee_pct", 0.0) or 0.0),
            "min_confidence_pct": (
                float(result.get("min_confidence_pct"))
                if result.get("min_confidence_pct") is not None
                else None
            ),
            "stop_loss_pct": float(result.get("stop_loss_pct", 0.0) or 0.0),
            "take_profit_pct": float(result.get("take_profit_pct", 0.0) or 0.0),
            "trailing_activation_pct": float(result.get("trailing_activation_pct", 0.0) or 0.0),
            "trailing_distance_pct": float(result.get("trailing_distance_pct", 0.0) or 0.0),
            "required_warmup_candles": int(result.get("required_warmup_candles", 0) or 0),
            "required_candles": int(result.get("required_candles", 0) or 0),
            "strategy_diagnostics": strategy_diagnostics,
            "cluster_count": _diag_int("cluster_count"),
            "min_cluster_size": _diag_int("min_cluster_size"),
            "band_points": _diag_float("band_points"),
            "band_pct": _diag_float("band_pct"),
            "rsi_period": _diag_int("rsi_period"),
            "rsi_sma_period": _diag_int("rsi_sma_period"),
            "valid_cluster_events": _diag_int("valid_cluster_events"),
            "rejected_out_of_band_values": _diag_int("rejected_out_of_band_values"),
            "reset_count": _diag_int("reset_count"),
            "long_entry_count": _diag_int("long_entry_count"),
            "short_entry_count": _diag_int("short_entry_count"),
            "low_sample": bool(low_sample),
            "sample_quality": sample_quality,
            "best_profile": best_profile,
            "compact_summary": dict(compact_summary),
            "sampled_profiles": int(result.get("sampled_profiles", 0) or 0),
            "evaluated_profiles": int(result.get("evaluated_profiles", 0) or 0),
            "validated_profiles": int(result.get("validated_profiles", 0) or 0),
            "theoretical_profiles": int(result.get("theoretical_profiles", 0) or 0),
            "sampling_coverage_pct": float(result.get("sampling_coverage_pct", 0.0) or 0.0),
            "sampling_random_seed": int(result.get("sampling_random_seed", 0) or 0),
            "sampling_mode": str(result.get("sampling_mode", "") or "n/a"),
            "optimizer_quality_status": str(
                compact_summary.get("optimizer_quality_status")
                or result.get("optimizer_quality_status")
                or ""
            ).strip().upper(),
            "search_window_candles": int(result.get("search_window_candles", 0) or 0),
            "full_history_optimization": bool(result.get("full_history_optimization")),
            "optimizer_worker_processes": int(result.get("optimizer_worker_processes", 0) or 0),
            "optimizer_max_sample_profiles": int(result.get("optimizer_max_sample_profiles", 0) or 0),
            "optimizer_force_full_scan": bool(result.get("optimizer_force_full_scan")),
            "session_leaderboard": [dict(row) for row in session_leaderboard[:8]],
            "session_top_session": str(session_top.get("session", "") or ""),
            "session_top_avg_pf": float(session_top.get("avg_pf", 0.0) or 0.0),
            "session_top_avg_win_rate_pct": float(session_top.get("avg_win_rate_pct", 0.0) or 0.0),
            "session_top_runs": int(session_top.get("runs", 0) or 0),
        }
        self._backtest_report_entries.append(entry)

    def _finalize_backtest_report_session(self, *, status: str) -> None:
        if not self._backtest_report_pending:
            return
        end_at = datetime.now(UTC)
        started_at = self._backtest_report_started_at or end_at
        duration_seconds = int(max(0.0, (end_at - started_at).total_seconds()))
        entries = list(self._backtest_report_entries)
        errors = list(self._backtest_report_errors)

        total_runs = len(entries)
        def _is_interval_mismatch_reinfaller(entry: dict[str, object]) -> bool:
            symbol_text = str(entry.get("symbol", "")).strip().upper()
            interval_text = str(entry.get("interval", "")).strip().lower()
            return symbol_text in {"DOTUSDT", "DOGEUSDT"} and interval_text == "15m"

        entries_for_verdict = [
            entry
            for entry in entries
            if not _is_interval_mismatch_reinfaller(entry)
        ]
        ignored_reinfaller_runs = max(0, len(entries) - len(entries_for_verdict))
        if not entries_for_verdict:
            entries_for_verdict = list(entries)
            ignored_reinfaller_runs = 0

        profitable_runs = sum(1 for entry in entries if float(entry.get("total_pnl_usd", 0.0) or 0.0) > 0.0)
        negative_runs = sum(1 for entry in entries if float(entry.get("total_pnl_usd", 0.0) or 0.0) < 0.0)
        total_pnl = sum(
            float(entry.get("total_pnl_usd", 0.0) or 0.0)
            for entry in entries_for_verdict
        )
        low_sample_runs = sum(1 for entry in entries if bool(entry.get("low_sample")))
        infinite_pf_runs = sum(
            1
            for entry in entries
            if math.isinf(float(entry.get("profit_factor", 0.0) or 0.0))
        )
        finite_pf_values = [
            float(entry.get("profit_factor", 0.0) or 0.0)
            for entry in entries
            if math.isfinite(float(entry.get("profit_factor", 0.0) or 0.0))
        ]
        avg_pf_finite = (
            sum(finite_pf_values) / len(finite_pf_values)
            if finite_pf_values
            else None
        )
        median_pf_finite = (
            statistics.median(finite_pf_values)
            if finite_pf_values
            else None
        )
        finite_pf_without_low_sample = [
            float(entry.get("profit_factor", 0.0) or 0.0)
            for entry in entries
            if (
                not bool(entry.get("low_sample"))
                and math.isfinite(float(entry.get("profit_factor", 0.0) or 0.0))
            )
        ]
        avg_pf_without_low_sample = (
            sum(finite_pf_without_low_sample) / len(finite_pf_without_low_sample)
            if finite_pf_without_low_sample
            else None
        )
        total_trades_all = sum(
            int(entry.get("total_trades", 0) or 0)
            for entry in entries_for_verdict
        )
        weighted_won_trades = sum(
            (
                float(entry.get("win_rate_pct", 0.0) or 0.0)
                / 100.0
                * int(entry.get("total_trades", 0) or 0)
            )
            for entry in entries_for_verdict
        )
        overall_hit_rate_pct = (
            (weighted_won_trades / float(total_trades_all)) * 100.0
            if total_trades_all > 0
            else 0.0
        )
        max_drawdown_batch_pct = max(
            (
                float(entry.get("max_drawdown_pct", 0.0) or 0.0)
                for entry in entries_for_verdict
            ),
            default=0.0,
        )
        longest_losing_streak = max(
            (
                int(entry.get("longest_consecutive_losses", 0) or 0)
                for entry in entries_for_verdict
            ),
            default=0,
        )
        start_capital = float(settings.trading.start_capital)
        total_profit_pct = (
            sum(
                (
                    float(entry.get("total_pnl_usd", 0.0) or 0.0)
                    / start_capital
                    * 100.0
                )
                for entry in entries_for_verdict
            )
            if start_capital > 0.0
            else 0.0
        )
        average_gain_per_trade_pct = (
            total_profit_pct / float(total_trades_all)
            if total_trades_all > 0
            else 0.0
        )
        rating_profit_factor = (
            float(avg_pf_without_low_sample)
            if avg_pf_without_low_sample is not None
            else (
                float(avg_pf_finite)
                if avg_pf_finite is not None
                else 0.0
            )
        )
        optimization_entries = [
            entry for entry in entries if bool(entry.get("optimization_mode"))
        ]
        sampled_profiles_total = sum(
            int(entry.get("sampled_profiles", 0) or 0)
            for entry in optimization_entries
        )
        theoretical_profiles_total = sum(
            int(entry.get("theoretical_profiles", 0) or 0)
            for entry in optimization_entries
        )
        if theoretical_profiles_total > 0:
            grid_coverage_pct = (
                float(sampled_profiles_total)
                / float(theoretical_profiles_total)
                * 100.0
            )
        elif optimization_entries:
            grid_coverage_pct = (
                sum(
                    float(entry.get("sampling_coverage_pct", 0.0) or 0.0)
                    for entry in optimization_entries
                )
                / float(len(optimization_entries))
            )
        else:
            grid_coverage_pct = 0.0
        seed_values = sorted(
            {
                int(entry.get("sampling_random_seed", 0) or 0)
                for entry in optimization_entries
                if int(entry.get("sampling_random_seed", 0) or 0) > 0
            }
        )
        seed_display = (
            ",".join(str(seed) for seed in seed_values)
            if seed_values
            else "n/a"
        )
        evaluated_profiles_total = sum(
            int(entry.get("evaluated_profiles", 0) or 0)
            for entry in optimization_entries
        )
        validated_profiles_total = sum(
            int(entry.get("validated_profiles", 0) or 0)
            for entry in optimization_entries
        )
        force_full_scan_runs = sum(
            1 for entry in optimization_entries if bool(entry.get("optimizer_force_full_scan"))
        )
        optimizer_workers_display = "n/a"
        sampling_mode_display = "n/a"
        search_window_candles_display = "n/a"
        full_history_optimization_display = "n/a"
        session_top_counter = Counter(
            str(entry.get("session_top_session", "") or "").strip().lower()
            for entry in optimization_entries
            if str(entry.get("session_top_session", "") or "").strip()
        )
        session_top_distribution = (
            ", ".join(
                f"{session}:{count}"
                for session, count in sorted(
                    session_top_counter.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )
            )
            if session_top_counter
            else "n/a"
        )
        is_win_rate_robust = overall_hit_rate_pct >= float(BACKTEST_DEPLOY_MIN_WIN_RATE_PCT)
        is_pf_robust = rating_profit_factor >= float(BACKTEST_DEPLOY_MIN_ROBUST_PF)
        if is_win_rate_robust and is_pf_robust:
            final_verdict = "BESTANDEN"
            verdict_reason = (
                "Durchschnittliche Trefferquote und Profit-Faktor erfüllen die Robustheits-Grenze."
            )
        else:
            final_verdict = "WARNUNG"
            verdict_reason = (
                "Durchschnittliche Trefferquote oder Profit-Faktor unterhalb der Robustheits-Grenze."
            )
        mode_label = "Batch" if self._backtest_report_mode == "batch" else "Single"

        def _display_single_or_mixed(
            values: list[object],
            *,
            empty_label: str = "n/a",
            mixed_prefix: str = "mixed",
        ) -> str:
            normalized_values = [
                str(value).strip()
                for value in values
                if value is not None and str(value).strip()
            ]
            if not normalized_values:
                return empty_label
            unique_values = sorted(set(normalized_values))
            if len(unique_values) == 1:
                return unique_values[0]
            return f"{mixed_prefix}({', '.join(unique_values)})"

        requested_history_start_display = _display_single_or_mixed(
            [entry.get("history_requested_start_utc") for entry in entries]
        )
        history_loader_mode_display = _display_single_or_mixed(
            [entry.get("history_loader_mode") for entry in entries]
        )
        required_warmup_display = _display_single_or_mixed(
            [entry.get("required_warmup_candles") for entry in entries],
            empty_label="0",
        )
        required_candles_display = _display_single_or_mixed(
            [entry.get("required_candles") for entry in entries],
            empty_label="0",
        )
        min_confidence_values = [
            f"{float(entry.get('min_confidence_pct')):.2f}"
            for entry in entries
            if entry.get("min_confidence_pct") is not None
        ]
        min_confidence_display = _display_single_or_mixed(
            list(min_confidence_values),
            empty_label="n/a",
        )
        fee_side_display = _display_single_or_mixed(
            [
                f"{float(entry.get('effective_taker_fee_pct', 0.0) or 0.0):.4f}"
                for entry in entries
            ],
            empty_label="0.0000",
        )
        leverage_pairs = {
            (
                int(entry.get("configured_leverage", 0) or 0),
                int(entry.get("effective_leverage", 0) or 0),
            )
            for entry in entries
        }
        if not leverage_pairs:
            leverage_mode_display = "n/a"
        elif all(configured == effective for configured, effective in leverage_pairs):
            leverage_mode_display = "configured=effective"
        elif all(configured != effective for configured, effective in leverage_pairs):
            leverage_mode_display = "configured!=effective"
        else:
            leverage_mode_display = "mixed"
        optimization_mode_display = _display_single_or_mixed(
            ["yes" if bool(entry.get("optimization_mode")) else "no" for entry in entries]
        )
        optimizer_workers_display = _display_single_or_mixed(
            [entry.get("optimizer_worker_processes") for entry in optimization_entries],
            empty_label="n/a",
        )
        sampling_mode_display = _display_single_or_mixed(
            [entry.get("sampling_mode") for entry in optimization_entries],
            empty_label="n/a",
        )
        search_window_candles_display = _display_single_or_mixed(
            [entry.get("search_window_candles") for entry in optimization_entries],
            empty_label="n/a",
        )
        full_history_optimization_display = _display_single_or_mixed(
            [
                "yes" if bool(entry.get("full_history_optimization")) else "no"
                for entry in optimization_entries
            ],
            empty_label="n/a",
        )

        lines = [
            "Backtest Compact Summary",
            f"Mode: {mode_label}",
            f"Status: {status}",
            f"Started (UTC): {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Finished (UTC): {end_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration (s): {duration_seconds}",
            f"Runs: {total_runs}",
            f"Profitable Runs: {profitable_runs}",
            f"Negative Runs: {negative_runs}",
            (
                "Verdict Exclusions: "
                f"{ignored_reinfaller_runs} run(s) (DOT/DOGE 15m ignored for final verdict)"
            ),
            f"Low-Sample Runs (<= {BACKTEST_REPORT_LOW_SAMPLE_TRADES_THRESHOLD} trades): {low_sample_runs}",
            f"Infinite PF Runs: {infinite_pf_runs}",
            f"Total PnL USD: {total_pnl:.2f}",
            "Average Profit Factor (finite runs): "
            + (
                self._format_report_profit_factor(float(avg_pf_finite))
                if avg_pf_finite is not None
                else "n/a"
            ),
            "Median Profit Factor (finite runs): "
            + (
                self._format_report_profit_factor(float(median_pf_finite))
                if median_pf_finite is not None
                else "n/a"
            ),
            "Average Profit Factor ex Low-Sample (finite runs): "
            + (
                self._format_report_profit_factor(float(avg_pf_without_low_sample))
                if avg_pf_without_low_sample is not None
                else "n/a"
            ),
            "",
            "Run Environment",
            f"Requested History Start UTC: {requested_history_start_display}",
            f"History Loader Mode: {history_loader_mode_display}",
            f"Required Warmup Candles: {required_warmup_display}",
            f"Required Candles: {required_candles_display}",
            f"Min Confidence %: {min_confidence_display}",
            f"Fee Stress Side %: {fee_side_display}",
            f"Leverage Mode: {leverage_mode_display}",
            f"Optimization Mode: {optimization_mode_display}",
            f"Optimizer Workers: {optimizer_workers_display}",
            f"Sampling Mode: {sampling_mode_display}",
            f"Search Window Candles: {search_window_candles_display}",
            f"Full-History Optimization: {full_history_optimization_display}",
            "",
            "Run Details",
            "symbol | strategy | interval | leverage | real_leverage_avg | period_utc | candles | total_signals | approved | blocked | gate_pass_rate_pct | stop_loss_pct | take_profit_pct | trailing_activation_pct | trailing_distance_pct | fee_pct_side | warmup_candles | min_confidence_pct | cluster_count | min_cluster_size | band_points | band_pct | rsi_period | rsi_sma_period | valid_cluster_events | rejected_out_of_band_values | reset_count | pnl_usd | robust_pf | win_rate_pct | trades | max_dd_pct | avg_win_fee_ratio | avg_win_usd | real_rrr | low_sample | sample_quality | optimization | evaluated_profiles | validated_profiles | sampled_profiles | theoretical_profiles | sampling_mode | sampling_coverage_pct | search_window_candles | optimizer_workers | full_history_optimization | force_full_scan | session_top",
        ]
        for entry in entries:
            strategy_name = str(entry.get("strategy_name", ""))
            strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
            history_start = str(entry.get("history_start_utc", "-") or "-")
            history_end = str(entry.get("history_end_utc", "-") or "-")
            if history_end == "-":
                period_utc = history_start
            else:
                period_utc = f"{history_start} -> {history_end}"
            profit_factor = float(entry.get("profit_factor", 0.0) or 0.0)
            lines.append(
                " | ".join(
                    [
                        str(entry.get("symbol", "-")),
                        strategy_label,
                        str(entry.get("interval", "-")),
                        f"{int(entry.get('effective_leverage', 0) or 0)}x",
                        f"{float(entry.get('real_leverage_avg', 0.0) or 0.0):.2f}x",
                        period_utc,
                        str(int(entry.get("history_candles", 0) or 0)),
                        str(int(entry.get("total_signals", 0) or 0)),
                        str(int(entry.get("approved_signals", 0) or 0)),
                        str(int(entry.get("blocked_signals", 0) or 0)),
                        f"{float(entry.get('gate_pass_rate_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('stop_loss_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('take_profit_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('trailing_activation_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('trailing_distance_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('effective_taker_fee_pct', 0.0) or 0.0):.4f}",
                        str(int(entry.get("required_warmup_candles", 0) or 0)),
                        (
                            f"{float(entry.get('min_confidence_pct')):.2f}"
                            if entry.get("min_confidence_pct") is not None
                            else "n/a"
                        ),
                        str(int(entry.get("cluster_count", 0) or 0)),
                        str(int(entry.get("min_cluster_size", 0) or 0)),
                        f"{float(entry.get('band_points', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('band_pct', 0.0) or 0.0):.2f}",
                        str(int(entry.get("rsi_period", 0) or 0)),
                        str(int(entry.get("rsi_sma_period", 0) or 0)),
                        str(int(entry.get("valid_cluster_events", 0) or 0)),
                        str(int(entry.get("rejected_out_of_band_values", 0) or 0)),
                        str(int(entry.get("reset_count", 0) or 0)),
                        f"{float(entry.get('total_pnl_usd', 0.0) or 0.0):.2f}",
                        self._format_report_profit_factor(profit_factor),
                        f"{float(entry.get('win_rate_pct', 0.0) or 0.0):.2f}",
                        str(int(entry.get("total_trades", 0) or 0)),
                        f"{float(entry.get('max_drawdown_pct', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('avg_win_fee_ratio', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('avg_win_usd', 0.0) or 0.0):.2f}",
                        f"{float(entry.get('real_rrr', 0.0) or 0.0):.2f}",
                        str(bool(entry.get("low_sample"))).lower(),
                        str(entry.get("sample_quality", "-")),
                        "yes" if bool(entry.get("optimization_mode")) else "no",
                        str(int(entry.get("evaluated_profiles", 0) or 0)),
                        str(int(entry.get("validated_profiles", 0) or 0)),
                        str(int(entry.get("sampled_profiles", 0) or 0)),
                        str(int(entry.get("theoretical_profiles", 0) or 0)),
                        str(entry.get("sampling_mode", "n/a") or "n/a"),
                        f"{float(entry.get('sampling_coverage_pct', 0.0) or 0.0):.2f}",
                        str(int(entry.get("search_window_candles", 0) or 0)),
                        str(int(entry.get("optimizer_worker_processes", 0) or 0)),
                        "yes" if bool(entry.get("full_history_optimization")) else "no",
                        "yes" if bool(entry.get("optimizer_force_full_scan")) else "no",
                        str(entry.get("session_top_session", "") or "-"),
                    ]
                )
            )
            best_profile = entry.get("best_profile")
            if isinstance(best_profile, dict) and best_profile:
                lines.append(
                    "  best_profile="
                    + json.dumps(best_profile, ensure_ascii=True, sort_keys=True)
                )
            session_leaderboard = entry.get("session_leaderboard")
            if isinstance(session_leaderboard, list) and session_leaderboard:
                lines.append(
                    "  session_leaderboard="
                    + json.dumps(
                        list(session_leaderboard)[:8],
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )

        if entries:
            strategy_groups: dict[str, list[dict[str, object]]] = {}
            for entry in entries:
                strategy_key = str(entry.get("strategy_name", ""))
                strategy_groups.setdefault(strategy_key, []).append(entry)

            lines.extend(
                [
                    "",
                    "Strategy Aggregates (All Coins)",
                    "strategy | runs | profitable_runs | total_pnl_usd | total_trades | low_sample_runs | median_pf_finite | avg_pf_ex_low_sample_finite | avg_win_rate_pct | avg_gate_pass_rate_pct | median_gate_pass_rate_pct | avg_max_dd_pct | avg_total_signals | avg_approved_signals | avg_blocked_signals | avg_real_leverage",
                ]
            )
            sorted_groups = sorted(
                strategy_groups.items(),
                key=lambda item: sum(
                    float(group_entry.get("total_pnl_usd", 0.0) or 0.0)
                    for group_entry in item[1]
                ),
                reverse=True,
            )
            for strategy_key, group_entries in sorted_groups:
                strategy_label = STRATEGY_LABELS.get(strategy_key, strategy_key)
                group_runs = len(group_entries)
                group_profitable_runs = sum(
                    1
                    for group_entry in group_entries
                    if float(group_entry.get("total_pnl_usd", 0.0) or 0.0) > 0.0
                )
                group_total_pnl = sum(
                    float(group_entry.get("total_pnl_usd", 0.0) or 0.0)
                    for group_entry in group_entries
                )
                group_total_trades = sum(
                    int(group_entry.get("total_trades", 0) or 0)
                    for group_entry in group_entries
                )
                group_low_sample_runs = sum(
                    1 for group_entry in group_entries if bool(group_entry.get("low_sample"))
                )
                group_avg_win_rate = (
                    sum(
                        float(group_entry.get("win_rate_pct", 0.0) or 0.0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                gate_pass_rates = [
                    float(group_entry.get("gate_pass_rate_pct", 0.0) or 0.0)
                    for group_entry in group_entries
                ]
                group_avg_gate_pass_rate = (
                    sum(gate_pass_rates) / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_median_gate_pass_rate = (
                    statistics.median(gate_pass_rates)
                    if gate_pass_rates
                    else 0.0
                )
                group_avg_max_dd = (
                    sum(
                        float(group_entry.get("max_drawdown_pct", 0.0) or 0.0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_avg_total_signals = (
                    sum(
                        int(group_entry.get("total_signals", 0) or 0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_avg_approved_signals = (
                    sum(
                        int(group_entry.get("approved_signals", 0) or 0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_avg_blocked_signals = (
                    sum(
                        int(group_entry.get("blocked_signals", 0) or 0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_avg_real_leverage = (
                    sum(
                        float(group_entry.get("real_leverage_avg", 0.0) or 0.0)
                        for group_entry in group_entries
                    )
                    / group_runs
                    if group_runs > 0
                    else 0.0
                )
                group_pf_finite = [
                    float(group_entry.get("profit_factor", 0.0) or 0.0)
                    for group_entry in group_entries
                    if math.isfinite(float(group_entry.get("profit_factor", 0.0) or 0.0))
                ]
                group_median_pf = (
                    statistics.median(group_pf_finite)
                    if group_pf_finite
                    else None
                )
                group_pf_ex_low_sample = [
                    float(group_entry.get("profit_factor", 0.0) or 0.0)
                    for group_entry in group_entries
                    if (
                        not bool(group_entry.get("low_sample"))
                        and math.isfinite(float(group_entry.get("profit_factor", 0.0) or 0.0))
                    )
                ]
                group_avg_pf_ex_low_sample = (
                    sum(group_pf_ex_low_sample) / len(group_pf_ex_low_sample)
                    if group_pf_ex_low_sample
                    else None
                )
                lines.append(
                    " | ".join(
                        [
                            strategy_label,
                            str(group_runs),
                            str(group_profitable_runs),
                            f"{group_total_pnl:.2f}",
                            str(group_total_trades),
                            str(group_low_sample_runs),
                            (
                                self._format_report_profit_factor(float(group_median_pf))
                                if group_median_pf is not None
                                else "n/a"
                            ),
                            (
                                self._format_report_profit_factor(float(group_avg_pf_ex_low_sample))
                                if group_avg_pf_ex_low_sample is not None
                                else "n/a"
                            ),
                            f"{group_avg_win_rate:.2f}",
                            f"{group_avg_gate_pass_rate:.2f}",
                            f"{group_median_gate_pass_rate:.2f}",
                            f"{group_avg_max_dd:.2f}",
                            f"{group_avg_total_signals:.2f}",
                            f"{group_avg_approved_signals:.2f}",
                            f"{group_avg_blocked_signals:.2f}",
                            f"{group_avg_real_leverage:.2f}",
                        ]
                    )
                )

        if errors:
            lines.extend(["", "Errors"])
            for idx, error_text in enumerate(errors, start=1):
                lines.append(f"{idx}. {error_text}")

        lines.extend(
            [
                "",
                "================================================================================",
                "=== ABSCHLUSS-BEWERTUNG (STATISTIK) ===",
                "================================================================================",
                f"Gesamt-Ergebnis:     {total_pnl:.2f} USD",
                f"Profit-Faktor:       {self._format_report_profit_factor(rating_profit_factor)} (Ziel: >= {BACKTEST_DEPLOY_MIN_ROBUST_PF:.1f})",
                f"Trefferquote:        {overall_hit_rate_pct:.2f}% (HAUPTFOKUS)",
                f"Win-Rate Minimum:    > {BACKTEST_DEPLOY_MIN_WIN_RATE_PCT:.2f}%",
                f"Anzahl Trades:       {total_trades_all} (Wichtig: Aussagekräftig erst ab ~25 Trades)",
                "--------------------------------------------------------------------------------",
                "RISIKO- & GEBÜHREN-CHECK:",
                f"Max. Kontoverlust:   {max_drawdown_batch_pct:.2f}% (Warnung ab 15%)",
                f"Ø Gewinn pro Trade:  {average_gain_per_trade_pct:.4f}% (Wichtig für Gebühren-Puffer)",
                f"Längste Pechsträhne: {longest_losing_streak} Trades in Folge verloren",
                "--------------------------------------------------------------------------------",
                "OPTIMIERER-INFO:",
                f"Abdeckung (Grid):    {grid_coverage_pct:.2f}% (Stichproben-Größe)",
                f"Evaluated Profiles:  {evaluated_profiles_total}",
                f"Validated Profiles:  {validated_profiles_total}",
                f"Force Full Scan:     {force_full_scan_runs}/{len(optimization_entries)} run(s)",
                f"Top Session Votes:   {session_top_distribution}",
                f"Zufalls-Seed:        {seed_display}",
                "--------------------------------------------------------------------------------",
                f"URTEIL: {final_verdict}",
                f"GRUND: {verdict_reason}",
                "================================================================================",
            ]
        )

        reports_dir = Path(__file__).resolve().parent
        filename = f"backtest_compact_summary_{end_at.strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = reports_dir / filename
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self._append_backtest_log(f"Compact backtest summary written: {report_path.name}")
        self.statusBar().showMessage(f"Backtest summary saved: {report_path.name}", 8000)

        self._backtest_report_started_at = None
        self._backtest_report_entries.clear()
        self._backtest_report_errors.clear()
        self._backtest_report_pending = False
        self._backtest_report_mode = "single"

    def _close_selected_position(self) -> None:
        selected_row = self.positions_table.currentRow()
        if selected_row < 0:
            self.statusBar().showMessage("Select an open position first.", 3000)
            return

        symbol_item = self.positions_table.item(selected_row, 1)
        last_price_item = self.positions_table.item(selected_row, 4)
        if symbol_item is None or last_price_item is None:
            self.statusBar().showMessage("Could not resolve the selected position.", 3000)
            return

        symbol = symbol_item.text().strip()
        last_price_text = last_price_item.text().replace(",", "").strip()
        try:
            exit_price = float(last_price_text)
        except ValueError:
            self.statusBar().showMessage("Selected position has no valid last price.", 3000)
            return

        if self._bot_thread is not None and self._bot_thread.isRunning():
            self._bot_thread.request_manual_close(symbol, exit_price)
            self._append_live_log(
                f"Manual close requested: symbol={symbol} at price={exit_price:.4f}"
            )
            self.statusBar().showMessage(f"Manual close sent for {symbol}.", 3000)
            return

        with Database(self._db_path) as db:
            engine = PaperTradingEngine(db, enable_persistence=True)
            closed_trade_id = engine.close_position_at_price(
                symbol,
                exit_price,
                status="MANUAL_CLOSE",
            )
            if closed_trade_id is None:
                self.statusBar().showMessage(f"No open position found for {symbol}.", 3000)
                return
            closed_trade = db.fetch_trade_by_id(closed_trade_id)
            self._positions = {
                int(trade.id): {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "qty": trade.qty,
                    "leverage": trade.leverage,
                    "status": trade.status,
                    "exit_time": None if trade.exit_time is None else trade.exit_time.isoformat(),
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "total_fees": trade.total_fees,
                    "high_water_mark": trade.high_water_mark,
                }
                for trade in db.fetch_open_trades()
            }
            self._realized_pnl_total = db.fetch_realized_pnl()
            self._realized_pnl_today = db.fetch_realized_pnl_since(self._today_start())

        self._sync_positions_table()
        if closed_trade is not None:
            self._append_live_log(
                "Trade closed: "
                f"id={closed_trade.id} price={exit_price:.4f} "
                f"reason={closed_trade.status} net_pnl={closed_trade.pnl:.4f} fees={closed_trade.total_fees:.4f}"
            )
        self.statusBar().showMessage(f"Manually closed {symbol}.", 3000)

    def _copy_table_to_clipboard(self, table: QTableWidget, table_name: str) -> None:
        clipboard_text = self._table_to_tsv(table)
        if not clipboard_text:
            self.statusBar().showMessage(f"{table_name} is empty.", 3000)
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(clipboard_text)
        self.statusBar().showMessage(f"{table_name} copied to clipboard.", 3000)
        if table is self.backtest_table:
            self._append_backtest_log(f"{table_name} copied to clipboard.")
        else:
            self._append_live_log(f"{table_name} copied to clipboard.")

    @staticmethod
    def _table_to_tsv(table: QTableWidget) -> str:
        if table.rowCount() == 0 or table.columnCount() == 0:
            return ""

        header_labels = []
        for column in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(column)
            header_labels.append(header_item.text() if header_item is not None else f"Column {column + 1}")

        lines = ["\t".join(header_labels)]
        for row in range(table.rowCount()):
            cells: list[str] = []
            for column in range(table.columnCount()):
                item = table.item(row, column)
                cells.append("" if item is None else item.text())
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def _load_startup_state(self) -> None:
        with Database(self._db_path) as db:
            self._positions = {
                int(trade.id): {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "qty": trade.qty,
                    "leverage": trade.leverage,
                    "status": trade.status,
                    "exit_time": None if trade.exit_time is None else trade.exit_time.isoformat(),
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "total_fees": trade.total_fees,
                    "high_water_mark": trade.high_water_mark,
                }
                for trade in db.fetch_open_trades()
            }
            self._realized_pnl_total = db.fetch_realized_pnl()
            self._realized_pnl_today = db.fetch_realized_pnl_since(self._today_start())
        if self._positions:
            self._append_live_log(
                f"[RECOVERY] Loaded {len(self._positions)} active trade(s) from persistent state."
            )
        self._refresh_trade_readiness(force=True)

    @staticmethod
    def _calculate_unrealized_net_pnl(trade: dict, last_price: float) -> float:
        entry_price = float(trade["entry_price"])
        qty = float(trade["qty"])
        paid_fees = float(trade.get("total_fees", 0.0) or 0.0)
        exit_fee = qty * last_price * (settings.trading.taker_fee_pct / 100.0)
        if str(trade["side"]).upper() == "LONG":
            gross_pnl = (last_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - last_price) * qty
        return gross_pnl - paid_fees - exit_fee

    @staticmethod
    def _today_start() -> datetime:
        now = datetime.now()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _metric_color(value: float) -> str:
        if value > 0:
            return ACCENT_PROFIT
        if value < 0:
            return ACCENT_DANGER
        return TEXT_PRIMARY

    @staticmethod
    def _make_pnl_item(value: float) -> QTableWidgetItem:
        item = TradingTerminalWindow._make_item(f"{value:,.4f}", True)
        item.setForeground(QColor(TradingTerminalWindow._metric_color(value)))
        return item

    @staticmethod
    def _format_timestamp(timestamp: str | None) -> str:
        if not timestamp:
            return "-"
        try:
            parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if parsed_timestamp.tzinfo is None:
                parsed_timestamp = parsed_timestamp.replace(tzinfo=UTC)
            return parsed_timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(timestamp)

    def _format_log_html(self, message: str) -> tuple[str, bool]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        display_message = message
        if "Trade opened:" in message:
            display_message = display_message.replace("side=LONG", "[BUY]")
            display_message = display_message.replace("side=SHORT", "[SELL]")

        safe_message = html.escape(display_message)
        is_trade_event = "Trade opened:" in message or "Trade closed:" in message
        highlight_tokens = {
            "[HEARTBEAT]": "#00ffff",
            "[BUY]": ACCENT_PRIMARY,
            "[SELL]": ACCENT_WARNING,
            "[WINNER]": ACCENT_PROFIT,
        }
        for token, color in highlight_tokens.items():
            escaped_token = html.escape(token)
            safe_message = safe_message.replace(
                escaped_token,
                f"<span style='color:{color}; font-weight:700'>{escaped_token}</span>",
            )
        return f"<span style='color:{TEXT_MUTED}'>[{timestamp}]</span> {safe_message}", is_trade_event

    def _append_live_log(self, message: str) -> None:
        self._update_realized_metrics_from_log(message)
        formatted_message, is_trade_event = self._format_log_html(message)
        self._append_html_log(self.log_output, formatted_message, auto_scroll=is_trade_event)

    def _append_backtest_log(self, message: str) -> None:
        progress_prefix = ""
        if message.startswith(self._HISTORY_PROGRESS_PREFIX):
            progress_prefix = self._HISTORY_PROGRESS_PREFIX
        elif message.startswith(self._SIGNAL_CACHE_PROGRESS_PREFIX):
            progress_prefix = self._SIGNAL_CACHE_PROGRESS_PREFIX
        if progress_prefix:
            compact_message = message[len(progress_prefix):].strip()
            formatted_message, _is_trade_event = self._format_log_html(compact_message)
            self._append_html_log(
                self.backtest_log_output,
                formatted_message,
                auto_scroll=True,
                replace_last=self._backtest_progress_log_active,
            )
            self._backtest_progress_log_active = True
            return
        self._backtest_progress_log_active = False
        formatted_message, _is_trade_event = self._format_log_html(message)
        self._append_html_log(self.backtest_log_output, formatted_message, auto_scroll=True)

    @staticmethod
    def _append_html_log(
        log_widget: QTextEdit,
        html_message: str,
        *,
        auto_scroll: bool,
        replace_last: bool = False,
    ) -> None:
        scrollbar = log_widget.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= max(0, scrollbar.maximum() - 4)
        if replace_last:
            document = log_widget.document()
            last_block = document.lastBlock()
            if last_block.isValid():
                cursor = QTextCursor(last_block)
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        log_widget.append(html_message)
        document = log_widget.document()
        while document.blockCount() > TradingTerminalWindow._MAX_LOG_LINES:
            first_block = document.firstBlock()
            if not first_block.isValid():
                break
            trim_cursor = QTextCursor(first_block)
            trim_cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            trim_cursor.removeSelectedText()
            trim_cursor.deleteChar()
        if auto_scroll and was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _sync_backtest_symbol_universe(self) -> None:
        configured_symbols = list(_configured_backtest_symbols())
        if not configured_symbols:
            return

        if hasattr(self, "backtest_coin_combo"):
            existing_combo_symbols = [
                self.backtest_coin_combo.itemText(index)
                for index in range(self.backtest_coin_combo.count())
            ]
            if existing_combo_symbols != configured_symbols:
                current_symbol = self.backtest_coin_combo.currentText()
                was_blocked = self.backtest_coin_combo.blockSignals(True)
                self.backtest_coin_combo.clear()
                self.backtest_coin_combo.addItems(configured_symbols)
                target_symbol = (
                    current_symbol
                    if current_symbol in configured_symbols
                    else configured_symbols[0]
                )
                self.backtest_coin_combo.setCurrentText(target_symbol)
                self.backtest_coin_combo.blockSignals(was_blocked)

        stale_symbols = [
            symbol
            for symbol in list(self._backtest_coin_cards.keys())
            if symbol not in configured_symbols
        ]
        for symbol in stale_symbols:
            card = self._backtest_coin_cards.pop(symbol, None)
            if card is None:
                continue
            with suppress(Exception):
                card.setParent(None)
            with suppress(Exception):
                card.deleteLater()

        for symbol in configured_symbols:
            if symbol in self._backtest_coin_cards:
                continue
            card = LiveSymbolCard(symbol, compact=True, show_deploy_button=True)
            card.clicked.connect(self._toggle_backtest_symbol_selection)
            card.deploy_requested.connect(self._deploy_backtest_symbol_to_live)
            self._backtest_coin_cards[symbol] = card

        self._selected_backtest_symbols = {
            symbol for symbol in self._selected_backtest_symbols if symbol in configured_symbols
        }
        if not self._selected_backtest_symbols:
            fallback_symbol = (
                self._current_backtest_symbol()
                if hasattr(self, "backtest_coin_combo")
                else configured_symbols[0]
            )
            if fallback_symbol not in configured_symbols:
                fallback_symbol = configured_symbols[0]
            self._selected_backtest_symbols = {fallback_symbol}

        if hasattr(self, "backtest_symbols_layout"):
            self._rebuild_backtest_symbol_grid()
        self._refresh_backtest_symbol_cards()

    def _get_selected_live_symbols(self) -> list[str]:
        return [
            symbol
            for symbol in settings.live.available_symbols
            if symbol in self._selected_live_symbols
        ]

    def _get_selected_backtest_symbols(self) -> list[str]:
        return [
            symbol
            for symbol in _configured_backtest_symbols()
            if symbol in self._selected_backtest_symbols
        ]

    def _get_selected_backtest_strategy_names(self) -> list[str]:
        return [
            strategy_name
            for strategy_name in BACKTEST_STRATEGY_NAMES
            if strategy_name in self._selected_backtest_strategies
        ]

    def _get_selected_backtest_intervals(self) -> list[str]:
        return [
            interval
            for interval in BACKTEST_INTERVAL_OPTIONS
            if interval in self._selected_backtest_intervals
        ]

    def _select_default_live_symbols(self) -> None:
        if not self._coin_cards:
            return
        self._selected_live_symbols = set(self._coin_cards.keys())
        if not self._selected_live_symbols:
            self._selected_live_symbols.add(settings.live.available_symbols[0])
        self._refresh_live_symbol_cards()

    def _select_default_backtest_symbols(self) -> None:
        if not self._backtest_coin_cards:
            return
        configured_symbols = _configured_backtest_symbols()
        if not configured_symbols:
            return
        default_symbol = self._current_backtest_symbol() or configured_symbols[0]
        self._selected_backtest_symbols = {default_symbol}
        if self.backtest_coin_combo.findText(default_symbol) >= 0:
            was_blocked = self.backtest_coin_combo.blockSignals(True)
            self.backtest_coin_combo.setCurrentText(default_symbol)
            self.backtest_coin_combo.blockSignals(was_blocked)
        self._refresh_backtest_symbol_cards()

    def _select_default_backtest_strategies(self) -> None:
        if not self._backtest_strategy_buttons:
            return
        self._selected_backtest_strategies = set()
        if hasattr(self, "backtest_strategy_combo"):
            was_blocked = self.backtest_strategy_combo.blockSignals(True)
            self.backtest_strategy_combo.setCurrentIndex(-1)
            self.backtest_strategy_combo.blockSignals(was_blocked)
        self._refresh_backtest_strategy_buttons()

    def _select_default_backtest_intervals(self) -> None:
        if not self._backtest_interval_buttons:
            return
        default_interval = self._current_backtest_interval()
        if default_interval not in BACKTEST_INTERVAL_OPTIONS:
            default_interval = BACKTEST_INTERVAL_OPTIONS[0]
        self._selected_backtest_intervals = {default_interval}
        if hasattr(self, "backtest_interval_combo"):
            was_blocked = self.backtest_interval_combo.blockSignals(True)
            self.backtest_interval_combo.setCurrentText(default_interval)
            self.backtest_interval_combo.blockSignals(was_blocked)
        self._refresh_backtest_interval_buttons()

    def _is_optimization_mode_active(self) -> bool:
        return hasattr(self, "optimize_button") and self.optimize_button.isChecked()

    def _toggle_live_symbol_selection(self, symbol: str) -> None:
        if self._bot_thread is not None and self._bot_thread.isRunning():
            self.statusBar().showMessage(
                "Live coin selection is locked while the bot is running (scroll remains enabled).",
                3000,
            )
            return
        if symbol in self._selected_live_symbols:
            self._selected_live_symbols.remove(symbol)
        else:
            self._selected_live_symbols.add(symbol)
        self._refresh_live_symbol_cards()
        self._refresh_trade_readiness(force=True)
        self._update_status_label()

    def _toggle_backtest_symbol_selection(self, symbol: str) -> None:
        deselected_current_symbol = False
        if symbol in self._selected_backtest_symbols:
            if len(self._selected_backtest_symbols) > 1:
                self._selected_backtest_symbols.remove(symbol)
                deselected_current_symbol = self._current_backtest_symbol() == symbol
        else:
            self._selected_backtest_symbols.add(symbol)
            self._set_current_backtest_symbol(symbol)
            return

        if deselected_current_symbol and self._selected_backtest_symbols:
            next_symbol = self._get_selected_backtest_symbols()[0]
            self._set_current_backtest_symbol(next_symbol, ensure_selected=False)
            return
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _refresh_live_symbol_cards(self) -> None:
        self._update_coin_radar_count()
        default_strategy_name = self._current_live_strategy_name()
        for symbol, card in self._coin_cards.items():
            is_selected = symbol in self._selected_live_symbols
            runtime_profile = self._live_runtime_profiles.get(symbol, {})
            is_live_candidate = bool(runtime_profile.get("live_candidate", False))
            resolved_strategy_name = resolve_strategy_for_symbol(symbol, default_strategy_name)
            resolved_interval = resolve_interval_for_symbol(symbol, self._live_interval)
            card.set_selected(is_selected)
            card.set_profit_factor(self._live_profit_factors.get(symbol))
            card.set_win_rate(self._live_win_rates.get(symbol))
            card.set_trade_count(self._live_trade_counts.get(symbol))
            card.set_strategy_badge(
                "LIVE_CANDIDATE" if is_live_candidate else get_strategy_badge(symbol, default_strategy_name),
                resolved_strategy_name,
                resolved_interval,
            )
            status_text, status_color = self._determine_symbol_status(symbol, is_selected)
            card.set_status(status_text, status_color)

    def _is_backtest_symbol_deploy_ready(self, symbol: str) -> bool:
        summary = self._backtest_compact_summaries.get(symbol, {})
        win_rate_pct = float(summary.get("win_rate_pct", self._backtest_win_rates.get(symbol, 0.0)) or 0.0)
        trade_count = int(summary.get("trade_count", self._backtest_trade_counts.get(symbol, 0)) or 0)
        net_pnl_usd = float(summary.get("net_pnl_usd", 0.0) or 0.0)
        robust_pf = float(summary.get("robust_profit_factor", self._backtest_profit_factors.get(symbol, 0.0)) or 0.0)
        return (
            win_rate_pct >= float(BACKTEST_DEPLOY_MIN_WIN_RATE_PCT)
            and trade_count >= int(BACKTEST_DEPLOY_MIN_TRADE_COUNT)
            and net_pnl_usd > 0.0
            and robust_pf >= float(BACKTEST_DEPLOY_MIN_ROBUST_PF)
        )

    def _refresh_backtest_symbol_cards(self) -> None:
        if not hasattr(self, "backtest_strategy_combo"):
            return
        strategy_name = self._current_backtest_strategy_name()
        for symbol, card in self._backtest_coin_cards.items():
            is_selected = symbol in self._selected_backtest_symbols
            card.set_selected(is_selected)
            compact_summary = self._backtest_compact_summaries.get(symbol)
            card.set_compact_summary(compact_summary)
            card.set_profit_factor(self._backtest_profit_factors.get(symbol))
            card.set_win_rate(self._backtest_win_rates.get(symbol))
            card.set_trade_count(self._backtest_trade_counts.get(symbol))
            winner_strategy_name = ""
            if isinstance(compact_summary, dict):
                winner_strategy_name = str(compact_summary.get("strategy_name", "") or "").strip()
                if winner_strategy_name:
                    winner_strategy_name = self._sanitize_backtest_strategy_name(
                        symbol,
                        winner_strategy_name,
                    )
            resolved_strategy = (
                winner_strategy_name
                if winner_strategy_name
                else self._resolve_backtest_strategy_for_symbol(symbol, strategy_name)
            )
            card.set_strategy_badge(
                get_strategy_badge(symbol, resolved_strategy, use_coin_override=False),
                resolved_strategy,
            )
            if symbol in self._backtest_deployed_symbols:
                card.set_deploy_state("deployed")
            elif self._backtest_best_profiles.get(symbol) and self._is_backtest_symbol_deploy_ready(symbol):
                card.set_deploy_state("ready")
            elif self._backtest_best_profiles.get(symbol):
                card.set_deploy_state("blocked")
            else:
                card.set_deploy_state("pending")
            status_text, status_color = self._determine_backtest_symbol_status(symbol, is_selected)
            card.set_status(status_text, status_color)
        self._sync_backtest_symbol_selector_viewport()

    def _refresh_backtest_strategy_buttons(self) -> None:
        for strategy_name, button in self._backtest_strategy_buttons.items():
            is_selected = strategy_name in self._selected_backtest_strategies
            button.blockSignals(True)
            button.setChecked(is_selected)
            button.blockSignals(False)
            button.setStyleSheet(BUTTON_STYLE_BLUE if is_selected else BUTTON_STYLE_DISABLED)

    def _refresh_backtest_interval_buttons(self) -> None:
        for interval, button in self._backtest_interval_buttons.items():
            is_selected = interval in self._selected_backtest_intervals
            button.blockSignals(True)
            button.setChecked(is_selected)
            button.blockSignals(False)
            button.setStyleSheet(BUTTON_STYLE_BLUE if is_selected else BUTTON_STYLE_DISABLED)

    def _toggle_backtest_strategy_selection(self, strategy_name: str, checked: bool) -> None:
        if checked:
            self._selected_backtest_strategies.add(strategy_name)
        else:
            self._selected_backtest_strategies.discard(strategy_name)
        if len(self._selected_backtest_strategies) == 1:
            only_strategy = next(iter(self._selected_backtest_strategies))
            combo_index = self.backtest_strategy_combo.findData(only_strategy)
            if combo_index >= 0:
                was_blocked = self.backtest_strategy_combo.blockSignals(True)
                self.backtest_strategy_combo.setCurrentIndex(combo_index)
                self.backtest_strategy_combo.blockSignals(was_blocked)
        elif len(self._selected_backtest_strategies) > 1:
            was_blocked = self.backtest_strategy_combo.blockSignals(True)
            self.backtest_strategy_combo.setCurrentIndex(-1)
            self.backtest_strategy_combo.blockSignals(was_blocked)
        elif not self._selected_backtest_strategies:
            was_blocked = self.backtest_strategy_combo.blockSignals(True)
            self.backtest_strategy_combo.setCurrentIndex(-1)
            self.backtest_strategy_combo.blockSignals(was_blocked)
        self._refresh_backtest_strategy_buttons()
        self._refresh_backtest_symbol_cards()
        self._update_status_label()

    def _toggle_backtest_interval_selection(self, interval: str, checked: bool) -> None:
        if interval not in BACKTEST_INTERVAL_OPTIONS:
            return
        if checked:
            self._selected_backtest_intervals.add(interval)
        else:
            self._selected_backtest_intervals.discard(interval)

        if not self._selected_backtest_intervals:
            self._selected_backtest_intervals.add(interval)

        selected_intervals = self._get_selected_backtest_intervals()
        if len(selected_intervals) == 1:
            selected_interval = selected_intervals[0]
            if self.backtest_interval_combo.currentText() != selected_interval:
                was_blocked = self.backtest_interval_combo.blockSignals(True)
                self.backtest_interval_combo.setCurrentText(selected_interval)
                self.backtest_interval_combo.blockSignals(was_blocked)
        self._refresh_backtest_interval_buttons()
        self._update_status_label()

    def _select_all_backtest_intervals(self) -> None:
        self._selected_backtest_intervals = set(BACKTEST_INTERVAL_OPTIONS)
        self._refresh_backtest_interval_buttons()
        self._update_status_label()

    def _select_primary_backtest_interval(self) -> None:
        primary_interval = self._current_backtest_interval()
        if primary_interval not in BACKTEST_INTERVAL_OPTIONS:
            primary_interval = BACKTEST_INTERVAL_OPTIONS[0]
        self._selected_backtest_intervals = {primary_interval}
        self._refresh_backtest_interval_buttons()
        self._update_status_label()

    def _rebuild_backtest_symbol_grid(self) -> None:
        if not hasattr(self, "backtest_symbols_layout"):
            return

        while self.backtest_symbols_layout.count():
            self.backtest_symbols_layout.takeAt(0)

        columns = BACKTEST_SYMBOL_GRID_COLUMNS
        ordered_symbols = list(_configured_backtest_symbols())
        for index, symbol in enumerate(ordered_symbols):
            card = self._backtest_coin_cards[symbol]
            row = index // columns
            column = index % columns
            self.backtest_symbols_layout.addWidget(card, row, column)

        for column in range(columns):
            self.backtest_symbols_layout.setColumnStretch(column, 1)
        final_row = (len(ordered_symbols) + columns - 1) // columns
        self.backtest_symbols_layout.setRowStretch(final_row, 1)
        self._sync_backtest_symbol_selector_viewport()

    def _sync_backtest_symbol_selector_viewport(self) -> None:
        if not hasattr(self, "backtest_symbols_scroll") or not hasattr(self, "backtest_symbols_layout"):
            return

        symbols = list(_configured_backtest_symbols())
        if not symbols:
            self.backtest_symbols_scroll.setMinimumHeight(120)
            self.backtest_symbols_scroll.setMaximumHeight(120)
            return

        self.backtest_symbols_layout.activate()
        self.backtest_symbols_container.adjustSize()
        viewport_height = max(
            int(self.backtest_symbols_container.sizeHint().height()),
            int(self.backtest_symbols_container.minimumSizeHint().height()),
        )
        frame_height = max(self.backtest_symbols_scroll.frameWidth() * 2, 0)
        target_height = max(120, viewport_height + frame_height)
        self.backtest_symbols_scroll.setMinimumHeight(target_height)
        self.backtest_symbols_scroll.setMaximumHeight(target_height)

    def _update_live_symbol_card(self, symbol: str) -> None:
        card = self._coin_cards.get(symbol)
        if card is None:
            return
        self._refresh_live_symbol_cards()

    def _update_coin_radar_count(self) -> None:
        if not hasattr(self, "coin_radar_count_label"):
            return
        coin_count = len(self._coin_cards)
        suffix = "" if coin_count == 1 else "s"
        self.coin_radar_count_label.setText(f"({coin_count} coin{suffix})")

    def _determine_symbol_status(self, symbol: str, is_selected: bool) -> tuple[str, str]:
        runtime_profile = self._live_runtime_profiles.get(symbol, {})
        if bool(runtime_profile.get("live_candidate", False)):
            candidate_status = str(
                runtime_profile.get("live_candidate_status", "ACTIVE")
            ).strip().upper() or "ACTIVE"
            if candidate_status == "KILL_SWITCHED":
                return "KILL_SWITCHED", ACCENT_DANGER
            if candidate_status == "PAUSED":
                return "PAUSED", ACCENT_WARNING
            return "ACTIVE", ACCENT_PROFIT
        heartbeat_info = self._heartbeat_snapshot.get(symbol)
        if heartbeat_info is not None and str(heartbeat_info.get("status", "OK")).upper() == "LAGGING":
            return "LAGGING", ACCENT_WARNING
        if any(str(position["symbol"]) == symbol for position in self._positions.values()):
            return "OPEN", ACCENT_PROFIT
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            if symbol == self._current_backtest_symbol():
                return "TESTING", ACCENT_WARNING
        runtime_settings = self._resolve_readiness_runtime_settings()
        resolved_strategy_name = self._resolve_readiness_strategy_for_symbol(
            symbol,
            runtime_settings=runtime_settings,
        )
        resolved_interval = self._resolve_readiness_interval_for_symbol(
            symbol,
            runtime_settings=runtime_settings,
        )
        if self._is_live_profile_loaded_for_symbol(
            symbol,
            strategy_name=resolved_strategy_name,
            interval=resolved_interval,
            runtime_settings=runtime_settings,
        ):
            return "ACTIVE", ACCENT_PROFIT
        if self._bot_thread is not None and self._bot_thread.isRunning() and is_selected:
            return "LIVE", ACCENT_PRIMARY
        if is_selected:
            return "ARMED", ACCENT_PRIMARY
        return "IDLE", TEXT_MUTED

    def _determine_backtest_symbol_status(self, symbol: str, is_selected: bool) -> tuple[str, str]:
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            if symbol == self._current_backtest_symbol():
                return "TESTING", ACCENT_WARNING
            if any(
                queued_symbol == symbol
                for queued_symbol, _strategy_name, _interval_override in self._batch_queue
            ):
                return "QUEUED", ACCENT_PRIMARY
        if is_selected:
            return "SELECTED", ACCENT_PRIMARY
        return "IDLE", TEXT_MUTED

    def _filter_live_symbols(self, query: str) -> None:
        normalized_query = query.strip().upper()
        for symbol, card in self._coin_cards.items():
            matches = normalized_query in symbol.upper()
            card.setVisible(matches)
        self._refresh_live_symbol_cards()
        self._update_status_label()

    def _select_winner_symbols(self) -> None:
        winner_count = 0
        self._selected_live_symbols.clear()
        for symbol, card in self._coin_cards.items():
            profit_factor = self._live_profit_factors.get(symbol)
            should_select = profit_factor is not None and profit_factor > 1.0 and not card.isHidden()
            if should_select:
                self._selected_live_symbols.add(symbol)
                winner_count += 1
        self._refresh_live_symbol_cards()
        self._update_status_label()
        if winner_count == 0:
            self.statusBar().showMessage("No winners with Profit Factor > 1.0 available yet.", 4000)
            return
        self.statusBar().showMessage(f"Selected {winner_count} winner coins.", 3000)

    def _select_all_symbols(self) -> None:
        visible_symbols = [
            symbol
            for symbol, card in self._coin_cards.items()
            if not card.isHidden()
        ]
        if not visible_symbols:
            self.statusBar().showMessage("No visible coins to toggle.", 3000)
            return

        all_visible_selected = all(
            symbol in self._selected_live_symbols for symbol in visible_symbols
        )
        if all_visible_selected:
            for symbol in visible_symbols:
                self._selected_live_symbols.discard(symbol)
            status_message = f"Deselected {len(visible_symbols)} visible coins."
        else:
            for symbol in visible_symbols:
                self._selected_live_symbols.add(symbol)
            status_message = f"Selected {len(visible_symbols)} visible coins."

        self._refresh_live_symbol_cards()
        self._update_status_label()
        self.statusBar().showMessage(status_message, 3000)

    def _select_all_backtest_symbols(self) -> None:
        visible_symbols = list(self._backtest_coin_cards)
        if not visible_symbols:
            self.statusBar().showMessage("No backtest coins available to toggle.", 3000)
            return

        all_selected = all(symbol in self._selected_backtest_symbols for symbol in visible_symbols)
        if all_selected:
            current_symbol = self._current_backtest_symbol()
            self._selected_backtest_symbols = {current_symbol} if current_symbol else set()
            status_message = "Cleared backtest selection to the active coin."
        else:
            self._selected_backtest_symbols = set(visible_symbols)
            status_message = f"Selected {len(visible_symbols)} backtest coins."

        self._refresh_backtest_symbol_cards()
        self._update_status_label()
        self.statusBar().showMessage(status_message, 3000)

    def _clear_backtest_coin_cards(self) -> None:
        backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
        if backtest_running or self._batch_active:
            self.statusBar().showMessage(
                "Cannot clear card results while a backtest is running.",
                3000,
            )
            return

        self._backtest_profit_factors.clear()
        self._backtest_win_rates.clear()
        self._backtest_trade_counts.clear()
        self._backtest_compact_summaries.clear()
        self._backtest_best_profiles.clear()
        self._last_backtest_result = None

        self.backtest_total_pnl_label.setText("Total PnL: -")
        self.backtest_win_rate_label.setText("Win Rate: -")
        self.backtest_total_trades_label.setText("Total Trades: -")

        self._refresh_backtest_symbol_cards()
        self._update_status_label()
        self._append_backtest_log("Backtest card results cleared.")
        self.statusBar().showMessage("Backtest card results cleared.", 3000)

    def _ensure_live_symbol_is_visible(self, symbol: str) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return
        if normalized_symbol not in self._coin_cards:
            card = LiveSymbolCard(normalized_symbol)
            card.clicked.connect(self._toggle_live_symbol_selection)
            self._coin_cards[normalized_symbol] = card
            insert_index = max(self.live_symbols_layout.count() - 1, 0)
            self.live_symbols_layout.insertWidget(insert_index, card)
        self._update_coin_radar_count()
        if self.symbol_combo.findText(normalized_symbol) < 0:
            was_blocked = self.symbol_combo.blockSignals(True)
            self.symbol_combo.addItem(normalized_symbol)
            self.symbol_combo.blockSignals(was_blocked)

    def _deploy_backtest_symbol_to_live(self, symbol: str) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return
        if normalized_symbol in self._backtest_deployed_symbols:
            self.statusBar().showMessage(f"{normalized_symbol} is already deployed to live.", 4000)
            return
        if not self._is_backtest_symbol_deploy_ready(normalized_symbol):
            compact_summary = self._backtest_compact_summaries.get(normalized_symbol, {})
            win_rate_pct = float(compact_summary.get("win_rate_pct", 0.0) or 0.0)
            trade_count = int(compact_summary.get("trade_count", 0) or 0)
            net_pnl_usd = float(compact_summary.get("net_pnl_usd", 0.0) or 0.0)
            robust_pf = float(compact_summary.get("robust_profit_factor", 0.0) or 0.0)
            self.statusBar().showMessage(
                (
                    f"Deploy blocked for {normalized_symbol}: "
                    f"Win-Rate must be >= {BACKTEST_DEPLOY_MIN_WIN_RATE_PCT:.0f}% "
                    f"(current {win_rate_pct:.2f}%), "
                    f"Trades >= {BACKTEST_DEPLOY_MIN_TRADE_COUNT} (current {trade_count}), "
                    f"Robust PF >= {BACKTEST_DEPLOY_MIN_ROBUST_PF:.1f} (current {robust_pf:.2f}), "
                    f"Net-PnL > 0 (current {net_pnl_usd:.2f})."
                ),
                6000,
            )
            return
        best_profile = self._backtest_best_profiles.get(normalized_symbol)
        if not best_profile:
            self.statusBar().showMessage(
                f"No optimization profile available for {normalized_symbol} yet.",
                4000,
            )
            return

        min_confidence_pct = float(self.confidence_spin.value())
        try:
            config_module.migrate_to_live(
                normalized_symbol,
                best_profile,
                min_confidence=min_confidence_pct,
            )
        except Exception as exc:
            logging.getLogger("gui").exception(
                "Live deployment failed for %s",
                normalized_symbol,
            )
            error_text = f"Live deployment failed for {normalized_symbol}: {exc}"
            self._append_backtest_log(error_text)
            self.statusBar().showMessage(error_text, 6000)
            return

        self._backtest_deployed_symbols.add(normalized_symbol)
        self._ensure_live_symbol_is_visible(normalized_symbol)
        self._rebuild_backtest_symbol_grid()
        if self.backtest_coin_combo.findText(normalized_symbol) < 0:
            was_blocked = self.backtest_coin_combo.blockSignals(True)
            self.backtest_coin_combo.addItem(normalized_symbol)
            self.backtest_coin_combo.blockSignals(was_blocked)
        self._refresh_live_symbol_cards()
        self._refresh_backtest_symbol_cards()
        self._update_status_label()
        success_message = (
            f"[SYSTEM] {normalized_symbol} successfully migrated to Live. "
            "Config updated and saved."
        )
        self._append_backtest_log(success_message)
        self.statusBar().showMessage(success_message, 6000)

    def _set_live_symbol_controls_enabled(self, enabled: bool) -> None:
        self.live_search_input.setEnabled(enabled)
        self.select_all_button.setEnabled(enabled)
        self.select_winners_button.setEnabled(enabled)
        for card in self._coin_cards.values():
            card.setEnabled(True)
            card.setCursor(
                Qt.CursorShape.PointingHandCursor
                if enabled
                else Qt.CursorShape.ArrowCursor
            )

    def _set_backtest_symbol_controls_enabled(self, enabled: bool) -> None:
        self.backtest_symbols_scroll.setEnabled(enabled)
        self.backtest_select_all_button.setEnabled(enabled)
        if hasattr(self, "backtest_clear_cards_button"):
            self.backtest_clear_cards_button.setEnabled(enabled)
        self._set_backtest_period_controls_enabled(enabled)
        if hasattr(self, "backtest_select_all_intervals_button"):
            self.backtest_select_all_intervals_button.setEnabled(enabled)
        if hasattr(self, "backtest_single_interval_button"):
            self.backtest_single_interval_button.setEnabled(enabled)
        for card in self._backtest_coin_cards.values():
            card.setEnabled(enabled)
        for button in self._backtest_strategy_buttons.values():
            button.setEnabled(enabled)
        for button in self._backtest_interval_buttons.values():
            button.setEnabled(enabled)

    def _update_performance_header(self, total_unrealized_pnl: float = 0.0) -> None:
        equity = settings.trading.start_capital + self._realized_pnl_total + total_unrealized_pnl
        day_pnl = self._realized_pnl_today + total_unrealized_pnl
        open_trade_count = len(self._positions)

        self.balance_card.set_value(f"{equity:,.2f} USDT", self._metric_color(equity - settings.trading.start_capital))
        self.open_trades_card.set_value(str(open_trade_count), TEXT_PRIMARY)
        self.day_pnl_card.set_value(f"{day_pnl:+,.2f} USDT", self._metric_color(day_pnl))
        self._append_equity_point(equity)

    def _update_realized_metrics_from_log(self, message: str) -> None:
        if not message.startswith("Trade closed:"):
            return
        match = re.search(r"net_pnl=([-+]?\d+(?:\.\d+)?)", message)
        if match is None:
            return
        realized_pnl = float(match.group(1))
        self._realized_pnl_total += realized_pnl
        self._realized_pnl_today += realized_pnl
        self._update_performance_header(self._current_total_unrealized_pnl())

    def _current_total_unrealized_pnl(self) -> float:
        total_unrealized = 0.0
        for trade in self._positions.values():
            last_price = self._latest_prices.get(str(trade["symbol"]), float(trade["entry_price"]))
            total_unrealized += self._calculate_unrealized_net_pnl(trade, last_price)
        return total_unrealized

    def _update_progress(self, value: int, text: str) -> None:
        progress_value = max(0, min(100, int(value)))
        self._progress_generation += 1
        generation = self._progress_generation
        self.statusBar().showMessage(text)
        if progress_value <= 0:
            self.progress_bar.hide()
            self.progress_bar.setValue(0)
            return

        self.progress_bar.show()
        self.progress_bar.setValue(progress_value)
        self.progress_bar.setFormat(f"{progress_value}%")
        if progress_value >= 100:
            QTimer.singleShot(2000, lambda: self._hide_progress_bar(generation))

    def _hide_progress_bar(self, generation: int) -> None:
        if generation != self._progress_generation:
            return
        self.progress_bar.hide()
        self.progress_bar.setValue(0)

    def _reset_progress(self, message: str | None = None) -> None:
        self._progress_generation += 1
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        if message:
            self.statusBar().showMessage(message, 5000)

    def _start_candle_timer(self) -> None:
        self._candle_timer = QTimer(self)
        self._candle_timer.setInterval(1000)
        self._candle_timer.timeout.connect(self._update_candle_timer)
        self._candle_timer.timeout.connect(self._update_clocks)
        self._candle_timer.start()
        self._update_candle_timer()
        self._update_clocks()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_backtest_splitter_sizes)
        QTimer.singleShot(0, self._sync_backtest_symbol_selector_viewport)

    def _apply_backtest_splitter_sizes(self) -> None:
        splitter = self._analytics_splitter
        if splitter is None or splitter.orientation() != Qt.Orientation.Horizontal:
            return
        total_width = max(splitter.width(), 1)
        log_width = max(int(total_width * 0.40), 420)
        main_width = max(total_width - log_width, 640)
        splitter.setSizes([main_width, log_width])

    def _current_live_timer_interval(self) -> str:
        selected_symbols = self._get_selected_live_symbols()
        if not selected_symbols:
            return self._live_interval
        selected_intervals = {
            resolve_interval_for_symbol(symbol, self._live_interval)
            for symbol in selected_symbols
        }
        return min(selected_intervals, key=self._interval_total_seconds)

    def _update_candle_timer(self) -> None:
        timer_interval = self._current_live_timer_interval()
        total_seconds = self._interval_total_seconds(timer_interval)
        if total_seconds <= 0:
            self.candle_timer_progress.setValue(0)
            self.candle_timer_remaining_label.setText("--:--")
            return

        if hasattr(self, "candle_timer_label"):
            self.candle_timer_label.setText(f"Next Candle Check ({timer_interval})")
        elapsed_seconds = int(datetime.now().timestamp()) % total_seconds
        progress_value = int((elapsed_seconds / total_seconds) * 100)
        remaining_seconds = total_seconds - elapsed_seconds

        minutes, seconds = divmod(remaining_seconds, 60)
        self.candle_timer_progress.setValue(progress_value)
        self.candle_timer_remaining_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _update_clocks(self) -> None:
        utc_now = datetime.now(UTC).strftime("%H:%M:%S")
        local_datetime = datetime.now().astimezone()
        local_zone = "CEST" if (local_datetime.dst() or 0) else "CET"
        local_now = f"{local_datetime.strftime('%H:%M:%S')} {local_zone}"
        self.utc_time_label.setText(f"[ UTC: {utc_now} ]")
        self.local_time_label.setText(f"[ Local: {local_now} ]")

    def _flash_market_pulse(self) -> None:
        self._heartbeat_generation += 1
        generation = self._heartbeat_generation
        self.market_pulse_label.setStyleSheet(
            "background: #00e676;"
            "border: 1px solid #7dffb1;"
            "border-radius: 8px;"
        )
        QTimer.singleShot(260, lambda: self._fade_market_pulse(generation))

    def _fade_market_pulse(self, generation: int) -> None:
        if generation != self._heartbeat_generation:
            return
        self.market_pulse_label.setStyleSheet(
            f"background: #1a1f24; border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
        )

    def _append_equity_point(self, equity: float) -> None:
        if self._equity_history and abs(self._equity_history[-1] - equity) < 1e-9:
            self._refresh_trade_readiness()
            return
        self._equity_history.append(equity)
        self._equity_history = self._equity_history[-50:]
        self._refresh_trade_readiness()

    def _refresh_trade_readiness(self, *, force: bool = False) -> None:
        if not hasattr(self, "readiness_bars"):
            return
        now = time.monotonic()
        if (
            not force
            and self._readiness_last_refresh_ts > 0.0
            and (now - self._readiness_last_refresh_ts) < self._readiness_refresh_interval_seconds
        ):
            return
        self._readiness_last_refresh_ts = now

        runtime_settings = self._resolve_readiness_runtime_settings()
        symbols = list(runtime_settings.live.available_symbols)

        try:
            with Database(self._db_path) as db:
                items = self._build_trade_readiness_items(
                    db,
                    symbols,
                    runtime_settings=runtime_settings,
                )
        except Exception:
            items = []
        self.readiness_bars.set_items(items)

    def _build_trade_readiness_items(
        self,
        db: Database,
        symbols: list[str],
        *,
        runtime_settings,
    ) -> list[dict[str, object]]:
        open_symbols = {
            str(position.get("symbol", "")).strip().upper()
            for position in self._positions.values()
        }
        items: list[dict[str, object]] = []
        for symbol in symbols:
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol:
                continue
            if normalized_symbol in open_symbols:
                items.append(
                    {
                        "symbol": normalized_symbol,
                        "score": 100.0,
                        "color": "#546e7a",
                        "eta_text": "LIVE",
                        "eta_color": TEXT_MUTED,
                        "eta_blink": False,
                    }
                )
                continue

            strategy_name = self._resolve_readiness_strategy_for_symbol(
                normalized_symbol,
                runtime_settings=runtime_settings,
            )
            interval = self._resolve_readiness_interval_for_symbol(
                normalized_symbol,
                runtime_settings=runtime_settings,
            )
            profile_loaded = self._is_live_profile_loaded_for_symbol(
                normalized_symbol,
                strategy_name=strategy_name,
                interval=interval,
                runtime_settings=runtime_settings,
            )
            leverage_confirmed = self._is_live_leverage_confirmed_for_symbol(
                normalized_symbol,
                runtime_settings=runtime_settings,
            )
            if not profile_loaded or not leverage_confirmed:
                lock_reason = "CFG" if not profile_loaded else "25x"
                items.append(
                    {
                        "symbol": normalized_symbol,
                        "score": 0.0,
                        "color": "#546e7a" if not profile_loaded else ACCENT_WARNING,
                        "eta_text": f"LOCK {lock_reason}",
                        "eta_color": TEXT_MUTED if not profile_loaded else ACCENT_WARNING,
                        "eta_blink": False,
                    }
                )
                continue
            candles = db.fetch_recent_candles(normalized_symbol, interval, limit=320)
            readiness = self._compute_symbol_trade_readiness(
                strategy_name=strategy_name,
                symbol=normalized_symbol,
                candles=candles,
                runtime_settings=runtime_settings,
            )
            items.append(
                {
                    "symbol": normalized_symbol,
                    "score": float(readiness.get("score", 0.0) or 0.0),
                    "color": str(readiness.get("color", ACCENT_PRIMARY)),
                    "eta_text": str(readiness.get("eta_text", "--")),
                    "eta_color": str(readiness.get("eta_color", TEXT_MUTED)),
                    "eta_blink": bool(readiness.get("eta_blink", False)),
                }
            )
        return items

    def _resolve_readiness_runtime_settings(self):
        try:
            current_mtime_ns = self._readiness_config_file.stat().st_mtime_ns
        except OSError:
            return self._readiness_runtime_settings

        if current_mtime_ns == self._readiness_config_mtime_ns:
            return self._readiness_runtime_settings

        try:
            importlib.invalidate_caches()
            refreshed_config = importlib.reload(config_module)
            refreshed_settings = getattr(refreshed_config, "settings", None)
            if refreshed_settings is not None:
                self._readiness_runtime_settings = refreshed_settings
                self._readiness_config_mtime_ns = current_mtime_ns
                self._readiness_config_reload_error_mtime_ns = None
        except Exception:
            if self._readiness_config_reload_error_mtime_ns != current_mtime_ns:
                logging.getLogger("gui").exception(
                    "Failed to hot-reload config.py for trade readiness."
                )
                self._readiness_config_reload_error_mtime_ns = current_mtime_ns
            self._readiness_config_mtime_ns = current_mtime_ns

        return self._readiness_runtime_settings

    def _resolve_readiness_strategy_for_symbol(
        self,
        symbol: str,
        *,
        runtime_settings,
    ) -> str:
        fallback_strategy = str(
            self.strategy_combo.currentData() or runtime_settings.strategy.default_strategy_name
        )
        allowed_strategies = {
            str(strategy_name)
            for strategy_name in runtime_settings.strategy.available_strategies
        }
        coin_strategy_map = getattr(runtime_settings.strategy, "coin_strategies", {}) or {}

        resolved_strategy = str(coin_strategy_map.get(symbol, fallback_strategy))
        if resolved_strategy in allowed_strategies:
            return resolved_strategy
        if fallback_strategy in allowed_strategies:
            return fallback_strategy
        return str(runtime_settings.strategy.default_strategy_name)

    def _resolve_readiness_interval_for_symbol(
        self,
        symbol: str,
        *,
        runtime_settings,
    ) -> str:
        allowed_timeframes = {str(interval) for interval in runtime_settings.api.timeframes}
        fallback_interval = str(self._live_interval or runtime_settings.trading.interval)
        if fallback_interval not in allowed_timeframes:
            fallback_interval = str(runtime_settings.trading.interval)
        profile = runtime_settings.trading.coin_profiles.get(symbol)
        if profile is not None and profile.interval is not None:
            candidate_interval = str(profile.interval)
            if candidate_interval in allowed_timeframes:
                return candidate_interval
        return fallback_interval

    def _is_live_profile_loaded_for_symbol(
        self,
        symbol: str,
        *,
        strategy_name: str,
        interval: str,
        runtime_settings,
    ) -> bool:
        production_registry = getattr(config_module, "PRODUCTION_PROFILE_REGISTRY", {}) or {}
        production_profile = production_registry.get(symbol)
        if not isinstance(production_profile, dict):
            return False
        expected_strategy = str(production_profile.get("strategy_name", "") or "").strip()
        expected_interval = str(production_profile.get("interval", "") or "").strip()
        if expected_strategy != str(strategy_name):
            return False
        if expected_interval != str(interval):
            return False
        profile_settings = runtime_settings.trading.coin_profiles.get(symbol)
        if profile_settings is None:
            return False
        expected_leverage = 25
        with suppress(Exception):
            expected_leverage = int(production_profile.get("default_leverage", 25) or 25)
        return profile_settings.default_leverage == expected_leverage

    def _is_live_leverage_confirmed_for_symbol(
        self,
        symbol: str,
        *,
        runtime_settings,
    ) -> bool:
        production_registry = getattr(config_module, "PRODUCTION_PROFILE_REGISTRY", {}) or {}
        production_profile = production_registry.get(symbol, {})
        expected_leverage = 25
        with suppress(Exception):
            expected_leverage = int(production_profile.get("default_leverage", 25) or 25)
        runtime_profile = self._live_runtime_profiles.get(symbol)
        if runtime_profile is not None:
            effective_leverage = runtime_profile.get("effective_leverage")
            with suppress(Exception):
                return int(effective_leverage) == int(expected_leverage)
            return False
        _ = runtime_settings
        return False

    def _compute_symbol_trade_readiness(
        self,
        *,
        strategy_name: str,
        symbol: str,
        candles: list,
        runtime_settings,
    ) -> dict[str, object]:
        if not candles:
            return {
                "score": 0.0,
                "color": self._readiness_color(0.0),
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }
        params = runtime_settings.strategy.coin_strategy_params.get(symbol, {})
        if strategy_name == "ema_cross_volume":
            score = self._compute_ema_trade_readiness(
                candles=candles,
                strategy_params=params,
                runtime_settings=runtime_settings,
            )
            return {
                "score": score,
                "color": self._readiness_color(score),
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }
        if strategy_name == "frama_cross":
            score = self._compute_frama_trade_readiness(
                candles=candles,
                strategy_params=params,
                runtime_settings=runtime_settings,
            )
            return {
                "score": score,
                "color": self._readiness_color(score),
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }
        if strategy_name == "supertrend_ema":
            score = self._compute_supertrend_trade_readiness(
                candles=candles,
                strategy_params=params,
                runtime_settings=runtime_settings,
            )
            return {
                "score": score,
                "color": self._readiness_color(score),
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }
        if strategy_name == "dual_thrust":
            return self._compute_dual_thrust_trade_readiness(
                candles=candles,
                strategy_params=params,
                runtime_settings=runtime_settings,
            )
        score = self._compute_fallback_trade_readiness(candles=candles)
        return {
            "score": score,
            "color": self._readiness_color(score),
            "eta_text": "--",
            "eta_color": TEXT_MUTED,
            "eta_blink": False,
        }

    def _compute_ema_trade_readiness(
        self,
        *,
        candles: list,
        strategy_params: dict[str, float],
        runtime_settings,
    ) -> float:
        candles_df = self._candles_to_frame(candles)
        fast_period = int(
            strategy_params.get("ema_fast_period", runtime_settings.strategy.ema_fast_period)
        )
        slow_period = int(
            strategy_params.get("ema_slow_period", runtime_settings.strategy.ema_slow_period)
        )
        volume_period = int(runtime_settings.strategy.volume_sma_period)
        volume_multiplier = float(
            strategy_params.get("volume_multiplier", runtime_settings.strategy.volume_multiplier)
        )
        if not math.isfinite(volume_multiplier) or volume_multiplier <= 0.0:
            volume_multiplier = float(runtime_settings.strategy.volume_multiplier)
        if not math.isfinite(volume_multiplier) or volume_multiplier <= 0.0:
            volume_multiplier = 1.0
        required = max(fast_period, slow_period, volume_period) + 2
        if len(candles_df) < required:
            return 0.0

        close_series = pd.to_numeric(candles_df["close"], errors="coerce").ffill().fillna(0.0)
        volume_series = pd.to_numeric(candles_df["volume"], errors="coerce").ffill().fillna(0.0)
        ema_fast = close_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow_period, adjust=False).mean()
        volume_sma = volume_series.rolling(window=volume_period, min_periods=1).mean()
        volume_threshold = volume_sma * volume_multiplier

        current_close = max(float(close_series.iloc[-1]), 1e-9)
        current_gap = abs(float(ema_fast.iloc[-1]) - float(ema_slow.iloc[-1])) / current_close
        gap_score = max(0.0, 1.0 - min(current_gap / 0.004, 1.0))

        threshold_value = max(float(volume_threshold.iloc[-1]), 1e-9)
        volume_ratio = float(volume_series.iloc[-1]) / threshold_value
        volume_score = max(0.0, min(volume_ratio, 1.0))

        crossed_long = (
            float(ema_fast.iloc[-2]) <= float(ema_slow.iloc[-2])
            and float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
            and volume_ratio >= 1.0
        )
        crossed_short = (
            float(ema_fast.iloc[-2]) >= float(ema_slow.iloc[-2])
            and float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
            and volume_ratio >= 1.0
        )
        if crossed_long or crossed_short:
            return 100.0
        return float(max(0.0, min(gap_score * 72.0 + volume_score * 28.0, 100.0)))

    def _compute_frama_trade_readiness(
        self,
        *,
        candles: list,
        strategy_params: dict[str, float],
        runtime_settings,
    ) -> float:
        candles_df = self._candles_to_frame(candles)
        fast_period = int(
            strategy_params.get("frama_fast_period", runtime_settings.strategy.frama_fast_period)
        )
        slow_period = int(
            strategy_params.get("frama_slow_period", runtime_settings.strategy.frama_slow_period)
        )
        required = max(fast_period, slow_period, 22)
        if len(candles_df) < required:
            return 0.0

        close_values = pd.to_numeric(candles_df["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        high_values = pd.to_numeric(candles_df["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        low_values = pd.to_numeric(candles_df["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        volume_values = pd.to_numeric(candles_df["volume"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        close_values = np.nan_to_num(close_values, nan=0.0, posinf=0.0, neginf=0.0)
        high_values = np.nan_to_num(high_values, nan=0.0, posinf=0.0, neginf=0.0)
        low_values = np.nan_to_num(low_values, nan=0.0, posinf=0.0, neginf=0.0)
        volume_values = np.nan_to_num(volume_values, nan=0.0, posinf=0.0, neginf=0.0)
        close_values = np.where(close_values <= 0.0, np.maximum(np.roll(close_values, 1), 1.0), close_values)
        close_values[0] = max(close_values[0], 1.0)
        high_values = np.where(high_values <= 0.0, close_values, high_values)
        low_values = np.where(low_values <= 0.0, close_values, low_values)
        volume_values = np.where(volume_values <= 0.0, 1.0, volume_values)

        try:
            frama_fast = calculate_frama_series(close_values, high_values, low_values, fast_period)
            frama_slow = calculate_frama_series(close_values, high_values, low_values, slow_period)
        except Exception:
            # Fallback proxy if FRAMA computation fails.
            close_series = pd.Series(close_values)
            frama_fast = close_series.ewm(span=fast_period, adjust=False).mean().to_numpy(dtype=np.float64)
            frama_slow = close_series.ewm(span=slow_period, adjust=False).mean().to_numpy(dtype=np.float64)

        warmup_end = min(len(close_values), max(slow_period, 1))
        if warmup_end > 0:
            frama_fast[:warmup_end] = close_values[:warmup_end]
            frama_slow[:warmup_end] = close_values[:warmup_end]

        current_close = max(float(close_values[-1]), 1e-9)
        current_gap = abs(float(frama_fast[-1]) - float(frama_slow[-1])) / current_close
        crossover_score = max(0.0, 1.0 - min(current_gap / 0.0035, 1.0))

        volume_sma_20 = pd.Series(volume_values).rolling(window=20, min_periods=1).mean().to_numpy(dtype=np.float64)
        volume_multiplier = float(
            strategy_params.get("volume_multiplier", runtime_settings.strategy.volume_multiplier)
        )
        if not math.isfinite(volume_multiplier) or volume_multiplier <= 0.0:
            volume_multiplier = float(runtime_settings.strategy.volume_multiplier)
        if not math.isfinite(volume_multiplier) or volume_multiplier <= 0.0:
            volume_multiplier = 0.8
        volume_confirm_threshold = max(float(volume_sma_20[-1]) * volume_multiplier, 1e-9)
        volume_ratio = float(volume_values[-1]) / volume_confirm_threshold
        volume_score = max(0.0, min(volume_ratio, 1.0))

        crossed_long = float(frama_fast[-1]) > float(frama_slow[-1]) and float(frama_fast[-2]) <= float(frama_slow[-2])
        crossed_short = float(frama_fast[-1]) < float(frama_slow[-1]) and float(frama_fast[-2]) >= float(frama_slow[-2])
        gap_confirmed = abs(float(frama_fast[-1]) - float(frama_slow[-1])) >= current_close * 0.0005
        volume_confirmed = volume_ratio >= 1.0
        if (crossed_long or crossed_short) and gap_confirmed and volume_confirmed:
            return 100.0
        return float(max(0.0, min(crossover_score * 70.0 + volume_score * 30.0, 100.0)))

    def _compute_supertrend_trade_readiness(
        self,
        *,
        candles: list,
        strategy_params: dict[str, float],
        runtime_settings,
    ) -> float:
        candles_df = self._candles_to_frame(candles)
        supertrend_length = max(
            2,
            int(
                strategy_params.get(
                    "supertrend_ema_supertrend_length",
                    strategy_params.get(
                        "supertrend_length",
                        runtime_settings.strategy.supertrend_ema_supertrend_length,
                    ),
                )
            ),
        )
        supertrend_multiplier = float(
            strategy_params.get(
                "supertrend_ema_supertrend_multiplier",
                strategy_params.get(
                    "supertrend_multiplier",
                    runtime_settings.strategy.supertrend_ema_supertrend_multiplier,
                ),
            )
        )
        ema_length = max(
            2,
            int(
                strategy_params.get(
                    "supertrend_ema_ema_length",
                    strategy_params.get(
                        "ema_length",
                        runtime_settings.strategy.supertrend_ema_ema_length,
                    ),
                )
            ),
        )
        required = max(supertrend_length + 3, ema_length + 2, 12)
        if len(candles_df) < required or supertrend_multiplier <= 0.0:
            return 0.0

        close_series = pd.to_numeric(candles_df["close"], errors="coerce").ffill().fillna(0.0)
        high_series = pd.to_numeric(candles_df["high"], errors="coerce").ffill().fillna(0.0)
        low_series = pd.to_numeric(candles_df["low"], errors="coerce").ffill().fillna(0.0)
        ema_series = close_series.ewm(span=ema_length, adjust=False).mean()

        supertrend_line, direction_series = self._calculate_supertrend_for_readiness(
            high_series=high_series,
            low_series=low_series,
            close_series=close_series,
            length=supertrend_length,
            multiplier=supertrend_multiplier,
        )
        if len(direction_series) < 2:
            return 0.0

        previous_direction = direction_series.shift(1)
        flip_up = previous_direction.lt(0) & direction_series.gt(0)
        flip_down = previous_direction.gt(0) & direction_series.lt(0)
        signal_long = bool(flip_up.iloc[-1] and close_series.iloc[-1] > ema_series.iloc[-1])
        signal_short = bool(flip_down.iloc[-1] and close_series.iloc[-1] < ema_series.iloc[-1])
        if signal_long or signal_short:
            return 100.0

        current_close = max(float(close_series.iloc[-1]), 1e-9)
        current_line = float(supertrend_line.iloc[-1])
        if not math.isfinite(current_line):
            return 0.0
        current_ema = float(ema_series.iloc[-1])
        current_direction = int(direction_series.iloc[-1])

        line_gap_pct = abs(current_close - current_line) / current_close
        ema_gap_pct = abs(current_close - current_ema) / current_close
        line_score = max(0.0, 1.0 - min(line_gap_pct / 0.012, 1.0))
        ema_score = max(0.0, 1.0 - min(ema_gap_pct / 0.008, 1.0))
        trend_aligned = (
            (current_direction > 0 and current_close > current_ema)
            or (current_direction < 0 and current_close < current_ema)
        )
        trend_score = 1.0 if trend_aligned else 0.35
        return float(max(0.0, min((line_score * 55.0) + (ema_score * 30.0) + (trend_score * 15.0), 100.0)))

    @staticmethod
    def _calculate_supertrend_for_readiness(
        *,
        high_series: pd.Series,
        low_series: pd.Series,
        close_series: pd.Series,
        length: int,
        multiplier: float,
    ) -> tuple[pd.Series, pd.Series]:
        previous_close = close_series.shift(1)
        true_range = pd.concat(
            [
                (high_series - low_series).abs(),
                (high_series - previous_close).abs(),
                (low_series - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(
            alpha=1.0 / float(max(length, 1)),
            adjust=False,
            min_periods=max(length, 1),
        ).mean()
        hl2 = (high_series + low_series) * 0.5
        basic_upper = hl2 + (float(multiplier) * atr)
        basic_lower = hl2 - (float(multiplier) * atr)

        row_count = len(close_series)
        final_upper = np.full(row_count, np.nan, dtype="float64")
        final_lower = np.full(row_count, np.nan, dtype="float64")
        supertrend = np.full(row_count, np.nan, dtype="float64")
        direction = np.zeros(row_count, dtype="int8")
        close_values = close_series.to_numpy(dtype="float64", copy=False)
        upper_values = basic_upper.to_numpy(dtype="float64", copy=False)
        lower_values = basic_lower.to_numpy(dtype="float64", copy=False)

        for index in range(row_count):
            current_close = close_values[index]
            current_upper = upper_values[index]
            current_lower = lower_values[index]

            if index == 0:
                final_upper[index] = current_upper
                final_lower[index] = current_lower
                if np.isnan(current_upper) and np.isnan(current_lower):
                    direction[index] = 0
                    supertrend[index] = np.nan
                elif np.isnan(current_upper):
                    direction[index] = 1
                    supertrend[index] = current_lower
                else:
                    direction[index] = -1
                    supertrend[index] = current_upper
                continue

            previous_upper = final_upper[index - 1]
            previous_lower = final_lower[index - 1]
            previous_close_value = close_values[index - 1]

            if np.isnan(current_upper):
                final_upper[index] = previous_upper
            elif (
                np.isnan(previous_upper)
                or current_upper < previous_upper
                or (not np.isnan(previous_close_value) and previous_close_value > previous_upper)
            ):
                final_upper[index] = current_upper
            else:
                final_upper[index] = previous_upper

            if np.isnan(current_lower):
                final_lower[index] = previous_lower
            elif (
                np.isnan(previous_lower)
                or current_lower > previous_lower
                or (not np.isnan(previous_close_value) and previous_close_value < previous_lower)
            ):
                final_lower[index] = current_lower
            else:
                final_lower[index] = previous_lower

            previous_direction = int(direction[index - 1])
            if np.isnan(final_upper[index]) and np.isnan(final_lower[index]):
                direction[index] = previous_direction
                supertrend[index] = supertrend[index - 1]
                continue
            if np.isnan(current_close):
                direction[index] = previous_direction
                supertrend[index] = supertrend[index - 1]
                continue

            if previous_direction <= 0:
                if np.isnan(final_upper[index]):
                    direction[index] = 1
                    supertrend[index] = final_lower[index]
                elif current_close <= final_upper[index]:
                    direction[index] = -1
                    supertrend[index] = final_upper[index]
                else:
                    direction[index] = 1
                    supertrend[index] = final_lower[index]
            else:
                if np.isnan(final_lower[index]):
                    direction[index] = -1
                    supertrend[index] = final_upper[index]
                elif current_close >= final_lower[index]:
                    direction[index] = 1
                    supertrend[index] = final_lower[index]
                else:
                    direction[index] = -1
                    supertrend[index] = final_upper[index]

        return (
            pd.Series(supertrend, index=close_series.index, dtype="float64"),
            pd.Series(direction, index=close_series.index, dtype="int8"),
        )

    def _compute_dual_thrust_trade_readiness(
        self,
        *,
        candles: list,
        strategy_params: dict[str, float],
        runtime_settings,
    ) -> dict[str, object]:
        candles_df = self._candles_to_frame(candles)
        period = max(
            1,
            int(
                strategy_params.get(
                    "dual_thrust_period",
                    runtime_settings.strategy.dual_thrust_period,
                )
            ),
        )
        k1 = float(strategy_params.get("dual_thrust_k1", runtime_settings.strategy.dual_thrust_k1))
        k2 = float(strategy_params.get("dual_thrust_k2", runtime_settings.strategy.dual_thrust_k2))
        required = max(period + 2, 6)
        if len(candles_df) < required or k1 <= 0.0 or k2 <= 0.0:
            return {
                "score": 0.0,
                "color": ACCENT_PRIMARY,
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }

        open_series = pd.to_numeric(candles_df["open"], errors="coerce").ffill().fillna(0.0)
        high_series = pd.to_numeric(candles_df["high"], errors="coerce").ffill().fillna(0.0)
        low_series = pd.to_numeric(candles_df["low"], errors="coerce").ffill().fillna(0.0)
        close_series = pd.to_numeric(candles_df["close"], errors="coerce").ffill().fillna(0.0)

        highest_high = high_series.rolling(window=period, min_periods=period).max().shift(1)
        highest_close = close_series.rolling(window=period, min_periods=period).max().shift(1)
        lowest_close = close_series.rolling(window=period, min_periods=period).min().shift(1)
        lowest_low = low_series.rolling(window=period, min_periods=period).min().shift(1)
        dual_range = pd.concat(
            [
                highest_high - lowest_close,
                highest_close - lowest_low,
            ],
            axis=1,
        ).max(axis=1)
        upper_trigger = open_series + (dual_range * k1)
        lower_trigger = open_series - (dual_range * k2)

        current_price = float(close_series.iloc[-1])
        upper_level = float(upper_trigger.iloc[-1])
        lower_level = float(lower_trigger.iloc[-1])
        if not math.isfinite(current_price) or not math.isfinite(upper_level) or not math.isfinite(lower_level):
            return {
                "score": 0.0,
                "color": ACCENT_PRIMARY,
                "eta_text": "--",
                "eta_color": TEXT_MUTED,
                "eta_blink": False,
            }

        half_span_series = ((upper_trigger - lower_trigger).abs() / 2.0).replace(0.0, np.nan)
        midpoint_series = (upper_trigger + lower_trigger) / 2.0
        trigger_proximity_series = (
            (close_series - midpoint_series).abs() / half_span_series
        ).clip(lower=0.0, upper=1.0)
        current_proximity = float(trigger_proximity_series.iloc[-1])
        if not math.isfinite(current_proximity):
            current_proximity = 0.0
        score = float(max(0.0, min(current_proximity * 100.0, 100.0)))

        eta_candles = self._estimate_candles_to_threshold(
            trigger_proximity_series.to_numpy(dtype=np.float64, copy=False),
            target_value=1.0,
            direction="up",
        )
        eta_text, eta_color, eta_blink = self._format_eta_display(eta_candles)

        return {
            "score": score,
            "color": (
                ACCENT_PROFIT
                if score >= 100.0
                else "#fbc02d"
                if score >= 70.0
                else ACCENT_PRIMARY
            ),
            "eta_text": eta_text,
            "eta_color": eta_color,
            "eta_blink": eta_blink,
        }

    def _compute_fallback_trade_readiness(self, *, candles: list) -> float:
        candles_df = self._candles_to_frame(candles)
        if len(candles_df) < 6:
            return 0.0
        close_series = pd.to_numeric(candles_df["close"], errors="coerce").ffill().fillna(0.0)
        current_close = max(float(close_series.iloc[-1]), 1e-9)
        recent_change = abs(float(close_series.iloc[-1]) - float(close_series.iloc[-6])) / current_close
        return float(max(0.0, (1.0 - min(recent_change / 0.02, 1.0)) * 100.0))

    @staticmethod
    def _candles_to_frame(candles: list) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": candle.symbol,
                    "interval": candle.interval,
                    "open_time": candle.open_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in candles
            ]
        )

    @staticmethod
    def _estimate_candles_to_threshold(
        values: np.ndarray,
        *,
        target_value: float,
        direction: str,
        lookback_candles: int = 5,
    ) -> float | None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size < (lookback_candles + 1):
            return None

        recent = finite_values[-(lookback_candles + 1):]
        current_value = float(recent[-1])
        if direction == "down":
            if current_value <= target_value:
                return 0.0
            slope_per_candle = (float(recent[-1]) - float(recent[0])) / max(lookback_candles, 1)
            if slope_per_candle >= 0.0:
                return None
            return max((current_value - target_value) / abs(slope_per_candle), 0.0)

        if direction == "up":
            if current_value >= target_value:
                return 0.0
            slope_per_candle = (float(recent[-1]) - float(recent[0])) / max(lookback_candles, 1)
            if slope_per_candle <= 0.0:
                return None
            return max((target_value - current_value) / slope_per_candle, 0.0)

        return None

    @staticmethod
    def _format_eta_display(eta_candles: float | None) -> tuple[str, str, bool]:
        if eta_candles is None or not math.isfinite(eta_candles):
            return "--", TEXT_MUTED, False
        estimated_candles = max(0, int(math.ceil(eta_candles)))
        eta_text = f"~{estimated_candles}c"
        if estimated_candles <= 2:
            return eta_text, "#39ff88", True
        if estimated_candles <= 5:
            return eta_text, "#fbc02d", False
        return eta_text, TEXT_MUTED, False

    @staticmethod
    def _readiness_color(score: float) -> str:
        bounded_score = max(0.0, min(100.0, float(score)))
        if bounded_score >= 80.0:
            return "#2e7d32"
        if bounded_score >= 55.0:
            return "#ef6c00"
        if bounded_score >= 30.0:
            return "#d84315"
        return "#c62828"

    def _build_trade_health_bar(self, trade: dict, last_price: float) -> QProgressBar:
        stop_loss_pct, trailing_activation_pct = self._resolve_trade_visual_thresholds(str(trade["symbol"]))
        entry_price = float(trade["entry_price"])
        side = str(trade["side"]).upper()

        if side == "LONG":
            stop_price = entry_price * (1.0 - (stop_loss_pct / 100.0))
            activation_price = entry_price * (1.0 + (trailing_activation_pct / 100.0))
            denominator = max(activation_price - stop_price, 1e-9)
            progress_ratio = (last_price - stop_price) / denominator
            trailing_active = last_price >= activation_price
        else:
            stop_price = entry_price * (1.0 + (stop_loss_pct / 100.0))
            activation_price = entry_price * (1.0 - (trailing_activation_pct / 100.0))
            denominator = max(stop_price - activation_price, 1e-9)
            progress_ratio = (stop_price - last_price) / denominator
            trailing_active = last_price <= activation_price

        progress_value = max(0, min(100, int(progress_ratio * 100)))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(progress_value)
        bar.setTextVisible(True)
        bar.setFormat("TRAIL" if trailing_active else f"{progress_value}%")
        bar.setFixedHeight(18)
        chunk_color = "#fbc02d" if trailing_active else ACCENT_PRIMARY
        bar.setToolTip(
            f"Stop: {stop_price:,.4f}\n"
            f"Trail Trigger: {activation_price:,.4f}\n"
            f"Last: {last_price:,.4f}"
        )
        bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: {SURFACE_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                color: {TEXT_PRIMARY};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {chunk_color};
                border-radius: 6px;
            }}
            """
        )
        return bar

    def _resolve_trade_visual_thresholds(self, symbol: str) -> tuple[float, float]:
        profile = settings.trading.coin_profiles.get(symbol)
        stop_loss_pct = settings.trading.stop_loss_pct
        trailing_activation_pct = settings.trading.trailing_activation_pct
        if profile is not None:
            if profile.stop_loss_pct is not None:
                stop_loss_pct = profile.stop_loss_pct
            if profile.trailing_activation_pct is not None:
                trailing_activation_pct = profile.trailing_activation_pct
        return stop_loss_pct, trailing_activation_pct

    @staticmethod
    def _interval_total_seconds(interval: str) -> int:
        normalized = interval.strip().lower()
        if normalized.endswith("m"):
            return int(normalized[:-1]) * 60
        if normalized.endswith("h"):
            return int(normalized[:-1]) * 3600
        return 15 * 60

    def _apply_button_states(self) -> None:
        bot_running = self._bot_thread is not None and self._bot_thread.isRunning()
        backtest_running = self._backtest_thread is not None and self._backtest_thread.isRunning()
        optimization_active = self._is_optimization_mode_active()
        batch_running_text = "Sweep Running..." if self._batch_mode == "dual_interval" else "Batch Running..."

        if bot_running and not self._bot_stop_requested:
            self._set_button_state(self.start_button, "Running...", False, BUTTON_STYLE_DISABLED)
            self._set_button_state(self.stop_button, "Stop", True, BUTTON_STYLE_RED)
        elif bot_running and self._bot_stop_requested:
            self._set_button_state(self.start_button, "Running...", False, BUTTON_STYLE_DISABLED)
            self._set_button_state(self.stop_button, "Stopping...", False, BUTTON_STYLE_DISABLED)
        else:
            self._set_button_state(self.start_button, "Start", True, BUTTON_STYLE_GREEN)
            self._set_button_state(self.stop_button, "Stop", False, BUTTON_STYLE_DISABLED)

        if self._batch_active and not backtest_running:
            self._set_button_state(self.backtest_button, "Run Backtest", False, BUTTON_STYLE_DISABLED)
            self._set_button_state(self.optimize_button, "Optimize Profile", False, BUTTON_STYLE_DISABLED)
            self._set_button_state(self.run_selected_button, batch_running_text, False, BUTTON_STYLE_DISABLED)
            self._set_button_state(self.run_all_button, batch_running_text, False, BUTTON_STYLE_DISABLED)
            if self._backtest_stop_requested:
                self._set_button_state(
                    self.stop_backtest_button, "Stopping...", False, BUTTON_STYLE_DISABLED
                )
            else:
                self._set_button_state(
                    self.stop_backtest_button, "Stop Backtest", True, BUTTON_STYLE_RED
                )
            return
        if backtest_running:
            self._set_button_state(self.backtest_button, "Simulating...", False, BUTTON_STYLE_DISABLED)
            optimize_text = "Optimizing..." if optimization_active else "Optimize Profile"
            self._set_button_state(self.optimize_button, optimize_text, False, BUTTON_STYLE_DISABLED)
            run_selected_text = batch_running_text if self._batch_active else "Run Selected Coins"
            self._set_button_state(self.run_selected_button, run_selected_text, False, BUTTON_STYLE_DISABLED)
            run_all_text = batch_running_text if self._batch_active else "Run All Coins"
            self._set_button_state(self.run_all_button, run_all_text, False, BUTTON_STYLE_DISABLED)
            if self._backtest_stop_requested:
                self._set_button_state(
                    self.stop_backtest_button, "Stopping...", False, BUTTON_STYLE_DISABLED
                )
            else:
                self._set_button_state(
                    self.stop_backtest_button, "Stop Backtest", True, BUTTON_STYLE_RED
                )
            return
        self._set_button_state(self.backtest_button, "Run Backtest", True, BUTTON_STYLE_BLUE)
        optimize_style = BUTTON_STYLE_GREEN if optimization_active else BUTTON_STYLE_BLUE
        self._set_button_state(self.optimize_button, "Optimize Profile", True, optimize_style)
        self._set_button_state(self.run_selected_button, "Run Selected Coins", True, BUTTON_STYLE_BLUE)
        self._set_button_state(self.run_all_button, "Run All Coins", True, BUTTON_STYLE_BLUE)
        self._set_button_state(self.stop_backtest_button, "Stop Backtest", False, BUTTON_STYLE_DISABLED)

    @staticmethod
    def _set_button_state(button: QPushButton, text: str, enabled: bool, style_sheet: str) -> None:
        button.setText(text)
        button.setEnabled(enabled)
        button.setStyleSheet(style_sheet)

    def _update_status_label(self) -> None:
        backtest_symbol = self._current_backtest_symbol()
        live_symbols = self._get_selected_live_symbols()
        backtest_strategy_names = self._get_selected_backtest_strategy_names()
        live_label = ", ".join(live_symbols[:3])
        if len(live_symbols) > 3:
            live_label += f" +{len(live_symbols) - 3}"
        if not live_label:
            live_label = "-"
        live_leverage = self.leverage_spin.value()
        live_interval_label = self._summarize_live_intervals()
        backtest_leverage = self._current_backtest_leverage()
        backtest_interval = self._summarize_backtest_intervals()
        live_strategy_label = self._summarize_live_resolved_strategies()
        if len(backtest_strategy_names) <= 1:
            backtest_strategy_label = self.backtest_strategy_combo.currentText()
        else:
            backtest_strategy_label = f"{len(backtest_strategy_names)} strategies"
        backtest_mode = "Optimize" if self._is_optimization_mode_active() else "Backtest"
        if self._bot_thread is not None and self._bot_thread.isRunning():
            self.status_label.setText(
                f"Running: {live_label} / {live_interval_label} / {live_leverage}x / {live_strategy_label}"
            )
            return
        self.status_label.setText(
            f"Idle / {backtest_mode} {backtest_symbol} / {backtest_interval} / {backtest_strategy_label} / {backtest_leverage}x / "
            f"Live {live_label} / {live_interval_label} / {live_leverage}x / {live_strategy_label}"
        )

    @staticmethod
    def _make_item(text: str, align_right: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter if align_right else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        item.setTextAlignment(int(alignment))
        return item


def main() -> int:
    _setup_gui_file_logging()
    logger = logging.getLogger("gui")
    logger.info("Starting GUI application.")
    app = QApplication(sys.argv)
    window = TradingTerminalWindow()
    window.show()
    exit_code = app.exec()
    logger.info("GUI application exited with code %s.", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
