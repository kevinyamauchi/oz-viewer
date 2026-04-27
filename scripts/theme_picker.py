# /// script
# requires-python = ">=3.11"
# dependencies = ["cmap", "pydantic", "PySide6"]
# ///
"""Interactive theme designer for oz-viewer.

Usage
-----
    # Start with the built-in dark theme
    uv run --script scripts/theme_picker.py

    # Load an existing theme JSON
    uv run --script scripts/theme_picker.py --load path/to/theme.json

    # Pre-set the save path
    uv run --script scripts/theme_picker.py --output my_theme.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing oz_viewer.theme from the local source tree when run as a
# uv script (oz_viewer is not a listed script dependency).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oz_viewer.theme._model import PaletteColorGroup, Theme, ThemePalette
from oz_viewer.theme._registry import get_theme, list_themes

# ---------------------------------------------------------------------------
# Role metadata: display order, section grouping, human-readable labels
# ---------------------------------------------------------------------------

_ROLES: list[tuple[str, str]] = [
    # (field_name, display_label)
    ("window", "Window"),
    ("base", "Base (inputs / lists)"),
    ("alternate_base", "Alternate base"),
    ("button", "Button"),
    ("window_text", "Window text"),
    ("text", "Text"),
    ("bright_text", "Bright text"),
    ("button_text", "Button text"),
    ("link", "Link"),
    ("link_visited", "Link visited"),
    ("highlight", "Highlight (selection)"),
    ("highlighted_text", "Highlighted text"),
    ("tool_tip_base", "Tooltip base"),
    ("tool_tip_text", "Tooltip text"),
]

_SECTIONS: list[tuple[str, list[str]]] = [
    ("Backgrounds", ["window", "base", "alternate_base", "button"]),
    (
        "Text",
        ["window_text", "text", "bright_text", "button_text", "link", "link_visited"],
    ),
    ("Selection", ["highlight", "highlighted_text"]),
    ("Tooltips", ["tool_tip_base", "tool_tip_text"]),
]

_ROLE_LABEL: dict[str, str] = dict(_ROLES)

# ---------------------------------------------------------------------------
# Helper: contrasting text colour for a background hex
# ---------------------------------------------------------------------------


def _contrasting_text(hex_color: str) -> str:
    """Return '#000000' or '#ffffff' depending on luminance of *hex_color*."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


# ---------------------------------------------------------------------------
# ColorButton
# ---------------------------------------------------------------------------


class ColorButton:
    """A QPushButton that shows a solid color and opens QColorDialog on click."""

    def __init__(self, field_name: str, initial_hex: str, parent=None) -> None:
        from PySide6.QtWidgets import QPushButton

        self.field_name = field_name
        self._hex = initial_hex
        self._btn = QPushButton(parent)
        self._btn.setFixedHeight(28)
        self._btn.clicked.connect(self._pick_color)
        self._refresh()

    # callback set by the editor after construction
    on_changed: object = None

    def _refresh(self) -> None:
        fg = _contrasting_text(self._hex)
        self._btn.setStyleSheet(
            f"background-color: {self._hex}; color: {fg};"
            " border: 1px solid #555; border-radius: 3px;"
        )
        self._btn.setText(self._hex.upper())

    def _pick_color(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        initial = QColor(self._hex)
        color = QColorDialog.getColor(initial, self._btn, f"Pick {self.field_name}")
        if color.isValid():
            self._hex = color.name().upper()
            self._refresh()
            if callable(self.on_changed):
                self.on_changed(self.field_name, self._hex)

    @property
    def widget(self):
        """Return the underlying ``QPushButton`` widget.

        Returns
        -------
        PySide6.QtWidgets.QPushButton
            The push-button that displays the color swatch.
        """
        return self._btn

    @property
    def hex_value(self) -> str:
        """Return the current color as an uppercase hex string.

        Returns
        -------
        str
            Six-digit hex color string, e.g. ``"#FF8800"``.
        """
        return self._hex

    def set_hex(self, hex_color: str) -> None:
        """Set the displayed color without opening the color dialog.

        Parameters
        ----------
        hex_color : str
            Six-digit hex color string accepted by ``QColor``.
        """
        self._hex = hex_color.upper()
        self._refresh()


# ---------------------------------------------------------------------------
# Widget gallery (live preview panel)
# ---------------------------------------------------------------------------


def _make_gallery(parent=None):
    from PySide6.QtWidgets import (
        QCheckBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QListWidget,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    container = QWidget(parent)
    layout = QVBoxLayout(container)
    layout.setSpacing(8)

    layout.addWidget(QLabel("Widget preview"))

    btn_normal = QPushButton("Normal button")
    layout.addWidget(btn_normal)

    btn_disabled = QPushButton("Disabled button")
    btn_disabled.setEnabled(False)
    layout.addWidget(btn_disabled)

    line_edit = QLineEdit()
    line_edit.setPlaceholderText("Text input…")
    layout.addWidget(line_edit)

    list_widget = QListWidget()
    for item in ("Item A", "Item B (selected)", "Item C"):
        list_widget.addItem(item)
    list_widget.setCurrentRow(1)
    list_widget.setFixedHeight(90)
    layout.addWidget(list_widget)

    check = QCheckBox("Checkbox")
    check.setChecked(True)
    layout.addWidget(check)

    progress = QProgressBar()
    progress.setValue(60)
    layout.addWidget(progress)

    group = QGroupBox("Group box")
    group_layout = QVBoxLayout(group)
    group_layout.addWidget(QPushButton("Button inside group"))
    group_layout.addWidget(QLineEdit("Text inside group"))
    layout.addWidget(group)

    layout.addStretch()
    return container


# ---------------------------------------------------------------------------
# Main editor window
# ---------------------------------------------------------------------------


class ThemePickerWindow:
    """Interactive theme editor window.

    Displays a scrollable panel of colour pickers (grouped by role) on the
    left and a live widget gallery on the right.  A toolbar exposes controls
    for naming the theme, loading built-in themes, loading/saving JSON files,
    and printing the theme dict to the terminal.

    Parameters
    ----------
    initial_theme : Theme
        The theme to display on startup.
    output_path : str or None
        Pre-filled save path offered in the Save dialog.  ``None`` defaults
        to ``"<theme-name>.json"``.
    """

    def __init__(self, initial_theme: Theme, output_path: str | None = None) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QFormLayout,
            QGroupBox,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPushButton,
            QScrollArea,
            QSplitter,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )

        self._app = QApplication.instance()
        self._output_path = output_path

        # --- state ---
        self._colors: dict[str, str] = self._theme_to_hex_dict(initial_theme)
        self._name = initial_theme.name

        # --- main window ---
        self._win = QMainWindow()
        self._win.setWindowTitle("oz-viewer theme picker")
        self._win.resize(960, 700)

        # --- toolbar ---
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self._win.addToolBar(toolbar)

        toolbar.addWidget(QLabel("  Theme name: "))
        self._name_edit = QLineEdit(self._name)
        self._name_edit.setFixedWidth(160)
        self._name_edit.textChanged.connect(self._on_name_changed)
        toolbar.addWidget(self._name_edit)

        toolbar.addSeparator()

        # theme selector dropdown
        toolbar.addWidget(QLabel("  Load built-in: "))
        self._builtin_combo = QComboBox()
        self._builtin_combo.addItems(list_themes())
        self._builtin_combo.setCurrentText(initial_theme.name)
        self._builtin_combo.currentTextChanged.connect(self._load_builtin)
        toolbar.addWidget(self._builtin_combo)

        toolbar.addSeparator()

        load_btn = QPushButton("Load JSON…")
        load_btn.clicked.connect(self._load_file)
        toolbar.addWidget(load_btn)

        save_btn = QPushButton("Save JSON…")
        save_btn.clicked.connect(self._save_file)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        print_btn = QPushButton("Print dict to terminal")
        print_btn.clicked.connect(self._print_dict)
        toolbar.addWidget(print_btn)

        # --- central splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._win.setCentralWidget(splitter)

        # --- left: scrollable color buttons ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(310)

        scroll_contents = QWidget()
        scroll_layout = QVBoxLayout(scroll_contents)
        scroll_layout.setSpacing(4)
        scroll_layout.setContentsMargins(8, 8, 8, 8)

        self._buttons: dict[str, ColorButton] = {}

        for section_label, field_names in _SECTIONS:
            group = QGroupBox(section_label)
            form = QFormLayout(group)
            form.setSpacing(4)
            for field_name in field_names:
                hex_val = self._colors.get(field_name, "#888888")
                btn = ColorButton(field_name, hex_val)
                btn.on_changed = self._on_color_changed
                self._buttons[field_name] = btn
                form.addRow(_ROLE_LABEL[field_name], btn.widget)
            scroll_layout.addWidget(group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_contents)
        splitter.addWidget(scroll_area)

        # --- right: gallery ---
        self._gallery = _make_gallery()
        splitter.addWidget(self._gallery)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._apply_to_app()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _theme_to_hex_dict(theme: Theme) -> dict[str, str]:
        result: dict[str, str] = {}
        group = theme.palette.active
        for field_name, _ in _ROLES:
            val = getattr(group, field_name, None)
            if val is not None:
                r8 = val.rgba8
                result[field_name] = f"#{r8.r:02X}{r8.g:02X}{r8.b:02X}"
            else:
                result[field_name] = "#888888"
        return result

    def _hex_dict_to_theme(self) -> Theme:
        from cmap import Color

        color_kwargs: dict[str, Color | None] = {}
        for field_name, _ in _ROLES:
            hex_val = self._colors.get(field_name)
            color_kwargs[field_name] = Color(hex_val) if hex_val else None

        group = PaletteColorGroup(**color_kwargs)
        return Theme(name=self._name, palette=ThemePalette(active=group))

    def _apply_to_app(self) -> None:
        theme = self._hex_dict_to_theme()
        self._app.setStyle("Fusion")
        self._app.setPalette(theme.palette.to_qpalette())

    def _populate_buttons(self) -> None:
        for field_name, btn in self._buttons.items():
            btn.set_hex(self._colors.get(field_name, "#888888"))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_color_changed(self, field_name: str, hex_val: str) -> None:
        self._colors[field_name] = hex_val
        self._apply_to_app()

    def _on_name_changed(self, text: str) -> None:
        self._name = text

    def _load_builtin(self, name: str) -> None:
        theme = get_theme(name)
        self._colors = self._theme_to_hex_dict(theme)
        self._name = theme.name
        self._name_edit.setText(self._name)
        self._populate_buttons()
        self._apply_to_app()

    def _load_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self._win, "Load theme JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        from oz_viewer.theme import load_theme_file

        theme = load_theme_file(path)
        self._colors = self._theme_to_hex_dict(theme)
        self._name = theme.name
        self._name_edit.setText(self._name)
        self._populate_buttons()
        self._apply_to_app()

    def _save_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        default = self._output_path or f"{self._name}.json"
        path, _ = QFileDialog.getSaveFileName(
            self._win, "Save theme JSON", default, "JSON files (*.json)"
        )
        if not path:
            return
        from oz_viewer.theme import save_theme_file

        save_theme_file(self._hex_dict_to_theme(), path)
        print(f"Saved theme to {path}")

    def _print_dict(self) -> None:
        theme = self._hex_dict_to_theme()
        # model_dump_json → back to plain dict → pretty-print JSON
        d = json.loads(theme.model_dump_json())
        print("\n# --- theme dict (paste into _defaults.py) ---")
        print(json.dumps(d, indent=4))
        print("# ---\n")

    @property
    def window(self):
        """Return the top-level ``QMainWindow`` instance.

        Returns
        -------
        PySide6.QtWidgets.QMainWindow
            The editor's main window.
        """
        return self._win


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch the theme picker window."""
    parser = argparse.ArgumentParser(description="oz-viewer theme picker")
    parser.add_argument(
        "--load", metavar="PATH", help="Load a theme JSON file on startup"
    )
    parser.add_argument("--output", metavar="PATH", help="Pre-set the save path")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    if args.load:
        from oz_viewer.theme import load_theme_file

        initial = load_theme_file(args.load)
    else:
        initial = get_theme("dark")

    picker = ThemePickerWindow(initial, output_path=args.output)
    picker.window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
