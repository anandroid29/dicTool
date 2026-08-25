"""
theme.py — PyDIC interface theme.

A measurement instrument, not a consumer application. Three rules follow from
that, and every token below is chosen to serve them.

**Neutral greys.** The background is deliberately free of any colour cast. This
is not a stylistic preference: the whole point of the results view is to judge a
field rendered in turbo, viridis or a diverging map, and simultaneous contrast
makes a surrounding hue shift the apparent colour of everything inside it. A
blue-tinted surround biases the reading of a blue-to-red scale, which is exactly
the judgement the operator is there to make. Grey surrounds are what image
analysis and photometry software use, for this reason.

**A restrained accent.** Chrome should not compete with data for attention. The
accent is a desaturated steel blue used to mark focus and selection only; it
never appears at high saturation next to the canvas, where it would draw the eye
away from the field.

**Density over decoration.** Small radii, tight padding, a monospace face for
numbers so digits align down a column and can be compared at a glance. Rounded,
airy, high-contrast interfaces read as consumer software and waste the screen
area that field data should occupy.
"""
from pathlib import Path

# ── Surfaces ───────────────────────────────────────────────────────────
# A neutral ramp. Steps are close together: hierarchy comes from borders and
# type, not from large jumps in luminance that fight the image for attention.
C_BG       = "#1a1c1e"
C_SURFACE  = "#212426"
C_CARD     = "#282b2e"
C_RAISED   = "#31353a"
C_BORDER   = "#3c4247"

# ── Accent ─────────────────────────────────────────────────────────────
# Steel blue, desaturated. Legible against every surface above and distinct
# from the ends of the scientific colormaps it sits beside.
C_ACCENT   = "#5a8cb0"
C_ACCENT_D = "#41697f"
C_ACCENT_G = "#74a6c9"

# ── Status ─────────────────────────────────────────────────────────────
# Muted so that a warning reads as information rather than an alarm, and so no
# status colour is mistaken for part of a field being displayed.
C_SUCCESS  = "#6a9c74"
C_WARNING  = "#c2954f"
C_DANGER   = "#bf6259"

# ── Type ───────────────────────────────────────────────────────────────
C_TEXT     = "#e6e8ea"
C_TEXT2    = "#a2a8ad"
C_TEXT3    = "#6b7378"

C_RUN      = "#4f7d5a"
C_RUN_H    = "#5e9169"

# ── Geometry ───────────────────────────────────────────────────────────
# One radius for the whole interface. Small: a 2 px corner reads as a machined
# edge, an 8 px one reads as a phone app.
RADIUS     = "3px"

# Numbers are set in a monospace face wherever they are read rather than merely
# displayed, so that magnitudes line up and a changed digit is visible.
FONT_UI    = '"Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial'
FONT_MONO  = '"JetBrains Mono", "Fira Code", "Cascadia Mono", "SF Mono", "Consolas", monospace'

_CHECKMARK = (Path(__file__).resolve().parent / "assets" / "checkmark.svg").as_posix()

STYLESHEET = f"""
/* ── Base ──────────────────────────────────────────────────────────── */
QWidget {{
    background: {C_BG};
    color: {C_TEXT};
    font-family: {FONT_UI};
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
    border-radius: {RADIUS};
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: {RADIUS};
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: {C_SURFACE};
    height: 6px;
    border-radius: {RADIUS};
}}
QScrollBar::handle:horizontal {{
    background: {C_BORDER};
    border-radius: {RADIUS};
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {C_RAISED};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
}}
QPushButton[class="run"]:hover   {{ background: {C_RUN_H}; }}
QPushButton[class="run"]:pressed {{ background: #3f6849; }}
QPushButton[class="run"]:disabled {{ background: {C_RAISED}; color: {C_TEXT3}; }}

QPushButton[class="danger"] {{
    background: {C_DANGER};
    color: #ffffff;
    border: none;
    font-weight: 700;
}}
QPushButton[class="danger"]:hover {{ background: #cf7168; }}

/* ── Tool Buttons ───────────────────────────────────────────────────── */
QToolButton {{
    background: {C_RAISED};
    color: {C_TEXT2};
    border: 1px solid {C_BORDER};
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
}}
QSlider::handle:horizontal {{
    background: {C_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: {RADIUS};
}}
QSlider::handle:horizontal:hover {{ background: {C_ACCENT_G}; }}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
    background: {C_SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT};
    image: url("{_CHECKMARK}");
}}
QCheckBox:hover {{ color: {C_TEXT}; }}
QCheckBox:disabled {{ color: {C_TEXT3}; }}
QCheckBox::indicator:disabled {{ border-color: {C_RAISED}; background: {C_BG}; }}

/* QRadioButton had no rules at all, so radios fell back to the native widget
   style and rendered as a pale slab against the dark theme. Same geometry as
   the checkbox, round indicator, accent dot when selected. */
QRadioButton {{
    color: {C_TEXT2};
    spacing: 6px;
    padding: 1px 4px;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {C_BORDER};
    border-radius: {RADIUS};
    background: {C_SURFACE};
}}
QRadioButton::indicator:hover {{ border-color: {C_ACCENT}; }}
QRadioButton::indicator:checked {{
    border-color: {C_ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.62, fx:0.5, fy:0.5,
                stop:0 {C_ACCENT}, stop:0.55 {C_ACCENT},
                stop:0.6 {C_SURFACE}, stop:1 {C_SURFACE});
}}
QRadioButton:checked {{ color: {C_TEXT}; font-weight: 600; }}
QRadioButton:hover {{ color: {C_TEXT}; }}
QRadioButton:disabled {{ color: {C_TEXT3}; }}
QRadioButton::indicator:disabled {{ border-color: {C_RAISED}; background: {C_BG}; }}

/* ── Progress bar ───────────────────────────────────────────────────── */
QProgressBar {{
    background: {C_RAISED};
    border: none;
    border-radius: {RADIUS};
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT_G});
    border-radius: {RADIUS};
}}

/* ── Group boxes ────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {C_BORDER};
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
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
QMenuBar::item:selected {{ background: {C_ACCENT}; color: #fff; border-radius: {RADIUS}; }}
QMenu {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: 4px;
    border-radius: {RADIUS};
}}
QMenu::item {{ padding: 6px 20px; border-radius: {RADIUS}; }}
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
    border-radius: {RADIUS};
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
    border-radius: {RADIUS};
    font-size: 11px;
}}
QListWidget::item {{ padding: 4px 8px; border-radius: {RADIUS}; }}
QListWidget::item:selected {{ background: {C_ACCENT}; color: #fff; }}
QListWidget::item:hover {{ background: {C_RAISED}; }}

/* ── Splitter ───────────────────────────────────────────────────────── */
QSplitter::handle {{ background: {C_BORDER}; }}
"""
