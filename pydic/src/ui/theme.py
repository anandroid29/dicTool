"""
theme.py — PyDIC professional dark theme
Color palette: deep navy background, electric blue accent, slate text.
"""

# Raw color tokens
C_BG       = "#08111d"
C_SURFACE  = "#0e1c2e"
C_CARD     = "#132035"
C_RAISED   = "#1a2d47"
C_BORDER   = "#1e3a5a"
C_ACCENT   = "#3b82f6"
C_ACCENT_D = "#1d4ed8"
C_ACCENT_G = "#0ea5e9"
C_SUCCESS  = "#10b981"
C_WARNING  = "#f59e0b"
C_DANGER   = "#ef4444"
C_TEXT     = "#e2e8f0"
C_TEXT2    = "#94a3b8"
C_TEXT3    = "#475569"
C_RUN      = "#10b981"
C_RUN_H    = "#059669"

STYLESHEET = f"""
/* ── Base ──────────────────────────────────────────────────────────── */
QWidget {{
    background: {C_BG};
    color: {C_TEXT};
    font-family: "Inter", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial;
    font-size: 12px;
    border: none;
    selection-background-color: {C_ACCENT};
    selection-color: #ffffff;
}}

QMainWindow, QDialog {{
    background: {C_BG};
}}

/* ── Scrollbars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {C_SURFACE};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: {C_SURFACE};
    height: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {C_BORDER};
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {C_RAISED};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {C_BORDER};
    border-color: {C_ACCENT};
    color: #ffffff;
}}
QPushButton:pressed  {{ background: {C_SURFACE}; }}
QPushButton:disabled {{ background: {C_SURFACE}; color: {C_TEXT3}; border-color: {C_SURFACE}; }}

QPushButton[class="accent"] {{
    background: {C_ACCENT};
    color: #ffffff;
    border: none;
    font-weight: 700;
}}
QPushButton[class="accent"]:hover   {{ background: {C_ACCENT_G}; }}
QPushButton[class="accent"]:pressed {{ background: {C_ACCENT_D}; }}

QPushButton[class="run"] {{
    background: {C_RUN};
    color: #ffffff;
    border: none;
    font-weight: 700;
    font-size: 13px;
    padding: 10px 28px;
    border-radius: 8px;
}}
QPushButton[class="run"]:hover   {{ background: {C_RUN_H}; }}
QPushButton[class="run"]:pressed {{ background: #047857; }}
QPushButton[class="run"]:disabled {{ background: {C_RAISED}; color: {C_TEXT3}; }}

QPushButton[class="danger"] {{
    background: {C_DANGER};
    color: #ffffff;
    border: none;
    font-weight: 700;
}}
QPushButton[class="danger"]:hover {{ background: #dc2626; }}

/* ── Tool Buttons ───────────────────────────────────────────────────── */
QToolButton {{
    background: {C_RAISED};
    color: {C_TEXT2};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 5px;
}}
QToolButton:hover   {{ background: {C_BORDER}; color: {C_TEXT}; }}
QToolButton:checked {{
    background: {C_ACCENT};
    color: #ffffff;
    border-color: {C_ACCENT};
}}

/* ── Inputs ─────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {C_ACCENT};
    background: {C_CARD};
}}
QLineEdit:read-only {{ color: {C_TEXT2}; }}

QSpinBox, QDoubleSpinBox {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {C_ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    border: none;
    background: transparent;
    width: 18px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ color: {C_TEXT2}; }}

QComboBox {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
}}
QComboBox:focus {{ border-color: {C_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    selection-background-color: {C_ACCENT};
    outline: none;
}}

/* ── Sliders ────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {C_RAISED};
    height: 5px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {C_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {C_ACCENT_G}; }}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: 3px;
    opacity: 0.6;
}}

/* ── Check / Radio ──────────────────────────────────────────────────── */
QCheckBox {{
    color: {C_TEXT2};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {C_BORDER};
    border-radius: 3px;
    background: {C_SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT};
}}
QCheckBox:hover {{ color: {C_TEXT}; }}

/* ── Progress bar ───────────────────────────────────────────────────── */
QProgressBar {{
    background: {C_RAISED};
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT_G});
    border-radius: 5px;
}}

/* ── Group boxes ────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    font-size: 10px;
    font-weight: 700;
    color: {C_TEXT3};
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background: {C_BG};
}}

/* ── Tables ─────────────────────────────────────────────────────────── */
QTableWidget {{
    background: {C_SURFACE};
    alternate-background-color: {C_CARD};
    gridline-color: {C_BORDER};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    font-family: "Fira Code", "JetBrains Mono", "Cascadia Code", monospace;
    font-size: 11px;
}}
QHeaderView::section {{
    background: {C_CARD};
    color: {C_TEXT2};
    border: none;
    border-bottom: 1px solid {C_BORDER};
    padding: 5px 10px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QTableWidget::item:selected {{ background: #1e3a70; color: {C_TEXT}; }}

/* ── Menu bar ───────────────────────────────────────────────────────── */
QMenuBar {{
    background: {C_SURFACE};
    color: {C_TEXT2};
    border-bottom: 1px solid {C_BORDER};
    padding: 2px;
}}
QMenuBar::item:selected {{ background: {C_ACCENT}; color: #fff; border-radius: 4px; }}
QMenu {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: 4px;
    border-radius: 6px;
}}
QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {C_ACCENT}; color: #fff; }}
QMenu::separator {{ background: {C_BORDER}; height: 1px; margin: 4px 8px; }}

/* ── Status bar ─────────────────────────────────────────────────────── */
QStatusBar {{
    background: {C_SURFACE};
    color: {C_TEXT3};
    border-top: 1px solid {C_BORDER};
    font-size: 11px;
    padding: 2px 8px;
}}

/* ── Tooltip ────────────────────────────────────────────────────────── */
QToolTip {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: 5px 8px;
    border-radius: 5px;
    font-size: 11px;
}}

/* ── Frames ─────────────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {C_BORDER};
    background: {C_BORDER};
}}

/* ── List widget ────────────────────────────────────────────────────── */
QListWidget {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    font-size: 11px;
}}
QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {C_ACCENT}; color: #fff; }}
QListWidget::item:hover {{ background: {C_RAISED}; }}

/* ── Splitter ───────────────────────────────────────────────────────── */
QSplitter::handle {{ background: {C_BORDER}; }}
"""
