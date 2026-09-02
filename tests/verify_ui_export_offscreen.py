"""Offscreen visual/integration checks for Results and video export.

This is a manual verification script because it writes screenshots and an
exported mosaic for human inspection.  It does not require a display server.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QScrollArea, QCheckBox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydic.core.analysis import DICAnalysis, PairResult
from pydic.core.rg_dic import DICParams
from pydic.core.units import Calibration
from pydic.ui.pages.results_page import CMAPS, FIELDS, ResultsPage
from pydic.ui.pages.params_page import ParamsPage
from pydic.ui.pages.dynamic_roi_page import DynamicROIPage
from pydic.ui.image_canvas import ROITool
from pydic.ui.pages.video_export_dialog import VideoExportDialog
from pydic.ui.render import PanelSpec, RangeSpec
from pydic.ui.video_export import ExportSpec, export_video
from pydic.ui.video_importer import VideoImporterDialog
from pydic.ui.theme import STYLESHEET


OUT = ROOT / "output" / "verification" / "ui"
OUT.mkdir(parents=True, exist_ok=True)


def make_analysis() -> DICAnalysis:
    frame_paths = [ROOT / "sample video_frames" / f"frame_{i:06d}.png"
                   for i in range(4)]
    ref = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Cannot load {frame_paths[0]}")
    h, w = ref.shape
    yy, xx = np.mgrid[:h, :w]
    roi = ((xx > w * 0.18) & (xx < w * 0.82) &
           (yy > h * 0.12) & (yy < h * 0.88))

    # Construct a lightweight real DICAnalysis so the UI and exporter exercise
    # production accessors without rerunning the solver.
    analysis = object.__new__(DICAnalysis)
    analysis.ref_path = str(frame_paths[0])
    analysis.def_paths = [str(p) for p in frame_paths]
    analysis._ref_image = ref.astype(np.float64) / 255.0
    analysis._roi_mask = roi
    analysis.params = DICParams(subset_spacing=10)
    analysis.calibration = Calibration.from_pixel_size(0.025, "mm")
    analysis.fps = 2412.0
    analysis.results = []
    analysis.dynamic_include_mask = None
    analysis.dynamic_exclude_mask = None

    base = 0.018 * np.sin(xx / 95.0) * np.cos(yy / 72.0)
    for i, path in enumerate(frame_paths):
        k = i + 1.0
        u = np.where(roi, 0.72 + 0.04 * np.sin(yy / 60.0), np.nan)
        v = np.where(roi, -0.18 + 0.03 * np.cos(xx / 80.0), np.nan)
        exx = np.where(roi, k * base, np.nan)
        eyy = np.where(roi, -0.55 * k * base, np.nan)
        exy = np.where(roi, 0.35 * k * base, np.nan)
        eeq = np.where(roi, np.sqrt(exx * exx + eyy * eyy + 2 * exy * exy), np.nan)
        rate = eeq * analysis.fps
        zeros = np.where(roi, 0.0, np.nan)
        analysis.results.append(PairResult(
            image_path=str(path), u=u, v=v,
            Exx=exx, Exy=exy, Eyy=eyy, Eeff=eeq,
            du_dx=zeros, du_dy=zeros, dv_dx=zeros, dv_dy=zeros,
            corr=np.where(roi, 0.97, np.nan), valid=roi.copy(),
            u_inc=u, v_inc=v, mag_inc=np.hypot(u, v),
            Vx=u * analysis.fps, Vy=v * analysis.fps,
            Veff=np.hypot(u, v) * analysis.fps,
            Exx_rate=exx * analysis.fps, Exy_rate=exy * analysis.fps,
            Gxy_rate=2 * exy * analysis.fps, Eyy_rate=eyy * analysis.fps,
            Eeff_rate=rate,
            Exx_inf=exx, Eyy_inf=eyy, Exy_inf=exy, Gxy_inf=2 * exy,
            Eeff_inf=eeq,
            Exx_gl=exx + 0.5 * exx * exx,
            Eyy_gl=eyy + 0.5 * eyy * eyy,
            Exy_gl=exy + 0.25 * exx * exy,
            Gxy_gl=2 * exy + 0.5 * exx * exy,
            Eeff_gl=eeq * 1.03,
        ))
    return analysis


class FakeWizard:
    def __init__(self, analysis):
        self.analysis = analysis
        self.seed_xy = None
        self.use_gpu = False

    def new_session(self):
        pass

    def go_before_params(self): pass
    def go_dynamic_roi(self): pass
    def go_analysis(self): pass
    def go_roi(self): pass
    def go_params(self): pass


def visible_overflow(page: ResultsPage) -> dict:
    top = page.layout().itemAt(0).widget()
    children = [c for c in top.findChildren(QWidget)
                if c.parentWidget() is top and c.isVisible()]
    rightmost = max((c.geometry().right() for c in children), default=0)
    return {
        "toolbar_width": top.width(),
        "rightmost_visible_child": rightmost,
        "overflow_px": max(0, rightmost - (top.width() - 1)),
        "visible_direct_children": len(children),
    }


def save_widget(widget: QWidget, name: str) -> None:
    QApplication.processEvents()
    if not widget.grab().save(str(OUT / name)):
        raise RuntimeError(f"Could not save screenshot {name}")


def main() -> None:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLESHEET)
    analysis = make_analysis()
    wizard = FakeWizard(analysis)
    page = ResultsPage(wizard)
    page.resize(1366, 768)
    page.show()
    page.on_enter()
    QApplication.processEvents()
    save_widget(page, "results_1366.png")
    wide = visible_overflow(page)

    page._cat_combo.setCurrentText("Strain")
    page.resize(1024, 720)
    QApplication.processEvents()
    save_widget(page, "results_1024_gl.png")
    narrow = visible_overflow(page)

    source_h, source_w = analysis.reference_image.shape[:2]
    dialog = VideoExportDialog(
        FIELDS, CMAPS, len(analysis.results), "Eeff_gl", "turbo",
        analysis.fps, source_size=(source_w, source_h))
    dialog.rows.setValue(2)
    dialog.cols.setValue(2)
    dialog.resize(900, 760)
    dialog.show()
    QApplication.processEvents()

    raw = dialog._editors[1]
    controls = ("colour_hdr", "lbl_field", "field", "lbl_cmap", "cmap",
                "lbl_range", "range_mode", "rng_row", "symmetric", "colorbar")
    raw_hidden = {name: getattr(raw, name).isHidden() for name in controls}
    content_choices = [raw.content.itemData(i)
                       for i in range(raw.content.count())]
    all_fields_present = all(e.field.count() == len(FIELDS) for e in dialog._editors)
    save_widget(dialog, "video_export_2x2.png")

    settings_wizard = FakeWizard(make_analysis())
    params_page = ParamsPage(settings_wizard)
    params_page.resize(1200, 720)
    params_page.show()
    params_page.on_enter()
    params_page._sp_spacing.setValue(2)
    params_page._sp_strain.setValue(3)
    QApplication.processEvents()
    valid_strain_warning_hidden = params_page._strain_warn.isHidden()
    params_page._sp_spacing.setValue(4)
    QApplication.processEvents()
    save_widget(params_page, "parameters_1200x720.png")

    dynamic_page = DynamicROIPage(settings_wizard)
    dynamic_page.resize(1200, 720)
    dynamic_page.show()
    dynamic_page.on_enter()
    QApplication.processEvents()
    save_widget(dynamic_page, "dynamic_roi_reset_zoom.png")

    importer = VideoImporterDialog(
        initial_calibration=Calibration.from_pixel_size(0.025, "mm"))
    importer.resize(760, 900)
    importer.show()
    QApplication.processEvents()
    save_widget(importer, "video_import_calibration.png")

    panels = [
        PanelSpec(content="field", field="Eeff_gl", cmap="turbo",
                  background="Deformed frame", show_colorbar=True),
        PanelSpec(content="image", background="Deformed frame"),
        PanelSpec(content="streaklines", background="Transparent"),
        PanelSpec(content="field", field="Exx_gl", cmap="RdBu_r",
                  background="Reference frame", show_colorbar=True,
                  range_spec=RangeSpec(mode="manual", vmin=0.0, vmax=0.06)),
    ]
    export_spec = ExportSpec(rows=2, cols=2, panels=panels,
                             cell_w=320, cell_h=180, codec="PNG image sequence",
                             first=0, last=0, fps=analysis.fps)
    seq_dir = Path(export_video(analysis, export_spec, str(OUT / "mosaic.png"),
                                markers=[(460.0, 250.0), (620.0, 330.0)]))
    exported = seq_dir / "frame_00000.png"
    image = cv2.imread(str(exported), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Exported mosaic could not be read")

    evidence = {
        "results_toolbar_1366": wide,
        "results_toolbar_1024": narrow,
        "raw_irrelevant_controls_all_hidden": all(raw_hidden.values()),
        "raw_control_visibility": raw_hidden,
        "streaklines_only_choice_removed": "streaklines" not in content_choices,
        "video_cell_size_matches_source":
            (dialog.cell_w.value(), dialog.cell_h.value()) == (source_w, source_h),
        "all_result_fields_available_in_each_panel": all_fields_present,
        "field_count": len(FIELDS),
        "video_import_pixel_size_mm":
            importer.get_calibration().pixel_size_in("mm"),
        "parameters_has_vertical_scroll_panel": bool(
            params_page.findChildren(QScrollArea)),
        "valid_3px_window_2px_spacing_warning_hidden":
            valid_strain_warning_hidden,
        "strain_warning_visible": params_page._strain_warn.isVisible(),
        "strain_warning_height": params_page._strain_warn.height(),
        "dynamic_roi_reset_zoom_visible":
            dynamic_page._reset_view_btn.isVisible(),
        "dynamic_roi_defaults_include_rect": (
            dynamic_page._btn_inc.isChecked()
            and dynamic_page._tool_buttons[ROITool.RECTANGLE].isChecked()),
        "fill_holes_is_checkbox": isinstance(dynamic_page._fill_chk, QCheckBox),
        "exported_mosaic": str(exported),
        "export_shape": list(image.shape),
        "png_has_alpha": image.ndim == 3 and image.shape[2] == 4,
    }
    (OUT / "ui_export_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    page.close()
    dialog.close()
    importer.close()
    params_page.close()
    dynamic_page.close()
    app.processEvents()


if __name__ == "__main__":
    main()
