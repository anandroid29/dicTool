"""Shared field-selection metadata and controls for result viewers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QToolButton, QWidget

from strainx.ui.theme import C_ACCENT, C_BORDER, C_RAISED, C_TEXT, C_TEXT2


RESULT_FIELDS = {
    "u": ("Instantaneous displacement u", "px"),
    "v": ("Instantaneous displacement v", "px"),
    "mag_inc": ("Instantaneous displacement magnitude", "px"),
    "Vx": ("Velocity Vx", "px/s"),
    "Vy": ("Velocity Vy", "px/s"),
    "Veff": ("Effective velocity", "px/s"),
    "Exx_rate": ("Strain rate Ėxx", "s⁻¹"),
    "Exy_rate": ("Tensor shear strain rate Ėxy", "s⁻¹"),
    "Eyy_rate": ("Strain rate Ėyy", "s⁻¹"),
    "Eeff_rate": ("Effective strain rate", "s⁻¹"),
    "Exx_gl": ("Accumulated strain Exx", "dimensionless"),
    "Eyy_gl": ("Accumulated strain Eyy", "dimensionless"),
    "Exy_gl": ("Accumulated tensor shear strain Exy", "dimensionless"),
    "Eeff_gl": ("Accumulated equivalent strain magnitude", "dimensionless"),
}

RESULT_FIELD_GROUPS = {
    "Displacement": ["u", "v", "mag_inc"],
    "Velocity": ["Vx", "Vy", "Veff"],
    "Strain rate": ["Exx_rate", "Eyy_rate", "Exy_rate", "Eeff_rate"],
    "Strain": ["Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"],
}

FIELD_SHORT = {
    "u": "u", "v": "v", "magnitude": "|d|", "mag_inc": "|d|",
    "corr": "ZNSSD", "valid": "valid",
    "Vx": "Vx", "Vy": "Vy", "Veff": "|V|",
    "Exx_rate": "Ėxx", "Exy_rate": "Ėxy", "Eyy_rate": "Ėyy",
    "Eeff_rate": "Ėeff",
    "Exx_gl": "Exx", "Eyy_gl": "Eyy", "Exy_gl": "Exy",
    "Eeff_gl": "Eeq",
}

CMAPS = [
    "turbo", "jet", "rainbow", "nipy_spectral",
    "RdBu_r", "seismic", "bwr", "coolwarm",
    "viridis", "inferno", "magma", "plasma", "cividis",
    "hot", "afmhot", "gist_heat", "copper", "gray",
]
DEFAULT_CMAP = "turbo"
DEFAULT_COVERAGE_TEXT = "99%"


def field_short(key: str) -> str:
    return FIELD_SHORT.get(key, key)


def field_group(field: str, groups: Mapping[str, Sequence[str]]) -> str:
    return next((name for name, keys in groups.items() if field in keys),
                next(iter(groups), ""))


class FieldFamilySelector(QWidget):
    """A readable family dropdown plus short, fully-tooltipped field buttons."""

    field_changed = pyqtSignal(str)

    def __init__(self, fields: Mapping[str, tuple[str, str]],
                 groups: Mapping[str, Sequence[str]], current: str,
                 parent=None) -> None:
        super().__init__(parent)
        self.fields = dict(fields)
        self.groups = {name: list(keys) for name, keys in groups.items()}
        self._field = current if current in self.fields else next(iter(self.fields))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.groups)
        self.category_combo.setFixedWidth(132)
        self.category_combo.setToolTip("Which family of results to display.")
        layout.addWidget(self.category_combo)

        self.buttons: dict[str, QToolButton] = {}
        for key, (label, unit) in self.fields.items():
            button = QToolButton()
            button.setText(field_short(key))
            button.setToolTip(f"{label} ({unit})" if unit else label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, name=key: self.set_field(name))
            layout.addWidget(button)
            self.buttons[key] = button

        self.category_combo.currentTextChanged.connect(self.select_category)
        self.available_fields = set(self.fields)
        self.set_field(self._field, emit=False)

    @property
    def field(self) -> str:
        return self._field

    def select_category(self, name: str, emit: bool = True) -> None:
        keys = [key for key in self.groups.get(name, [])
                if key in self.available_fields]
        if keys:
            self.set_field(self._field if self._field in keys else keys[0],
                           emit=emit)

    def set_field(self, field: str, emit: bool = True) -> None:
        if field not in self.fields or field not in self.available_fields:
            return
        changed = field != self._field
        self._field = field
        group = field_group(field, self.groups)
        if self.category_combo.currentText() != group:
            self.category_combo.blockSignals(True)
            self.category_combo.setCurrentText(group)
            self.category_combo.blockSignals(False)
        visible = set(self.groups.get(group, []))
        active = (
            f"QToolButton{{background:{C_ACCENT};color:#fff;border:none;"
            "border-radius:3px;font-size:10px;font-weight:700;padding:3px 8px;}"
        )
        inactive = (
            f"QToolButton{{background:{C_RAISED};color:{C_TEXT2};"
            f"border:1px solid {C_BORDER};border-radius:3px;font-size:10px;"
            f"padding:3px 8px;}}QToolButton:hover{{background:{C_BORDER};"
            f"color:{C_TEXT};}}"
        )
        for key, button in self.buttons.items():
            button.setVisible(key in visible)
            button.setChecked(key == field)
            button.setStyleSheet(active if key == field else inactive)
        if emit and changed:
            self.field_changed.emit(field)

    def set_available_fields(self, fields) -> None:
        """Disable result families absent from a loaded cache."""
        self.available_fields = set(fields) & set(self.fields)
        model = self.category_combo.model()
        for row, name in enumerate(self.groups):
            enabled = any(key in self.available_fields
                          for key in self.groups[name])
            item = model.item(row)
            if item is not None:
                item.setEnabled(enabled)
        for key, button in self.buttons.items():
            button.setEnabled(key in self.available_fields)
        if self._field not in self.available_fields and self.available_fields:
            replacement = next(
                key for key in self.fields if key in self.available_fields)
            self.set_field(replacement)
