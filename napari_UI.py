import numpy as np
import json
import csv
import sys
import os
import time
import napari
import torch
from datetime import datetime
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QSlider, QGroupBox, QFileDialog, QApplication, QMessageBox, QProgressBar,
    QSpinBox, QDialog, QColorDialog, QSplitter, QListWidget, QAbstractItemView,
    QListWidgetItem, QDialogButtonBox, QDoubleSpinBox, QInputDialog, QLineEdit, QComboBox, QScrollArea
)
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk, dilation, ball, binary_opening
from aicsimageio import AICSImage
import tifffile
import imageio
from skimage.transform import rotate
from skimage.measure import label, regionprops
from skimage.segmentation import watershed  
import matplotlib.pyplot as plt
from qtpy.QtGui import QIntValidator
from skimage.draw import polygon as sk_polygon  
from scipy import ndimage as ndi
from skimage.draw import polygon2mask
import re
import xml.etree.ElementTree as ET
from skimage.morphology import disk, dilation, remove_small_objects
from skimage.feature import peak_local_max
from cellpose import models, train
from cellpose import io as cellpose_io







###############################################################################
# ----------------------------- FILE LOADING ---------------------------------
###############################################################################

def load_lif_memmap(path, scene_index=0, max_ch=4):
    """Load LIF using memory mapping for instant slice access without RAM overhead.
    Returns (list_of_memmaps, contrast_limits_dict, voxel_size(Z,Y,X), paths).
    """
    print("Creating memory-mapped arrays from LIF (memmap)...")
    img = AICSImage(str(path))

    # Determine number of channels
    try:
        c_total = int(img.dims.C)
    except Exception:
        c_total = 1

    n_ch = min(max_ch, max(1, c_total))

    # Get the underlying dask array in CZYX order if possible
    try:
        arr = img.get_image_dask_data("CZYX", S=scene_index)
    except Exception:
        arr = img.get_image_dask_data("CYX", S=scene_index)

    print(f"Image array shape (expected C,Z,Y,X or C,Y,X): {arr.shape}")

    voxel_size = (1.0, 1.0, 1.0)
    try:
        p = getattr(img, "physical_pixel_sizes", None)
        if p is not None:
            # NamedTuple: PhysicalPixelSizes(Z=..., Y=..., X=...) 
            if hasattr(p, "Z") and hasattr(p, "Y") and hasattr(p, "X"):
                z = p.Z
                y = p.Y
                x = p.X
            else:
                # tuple-like; docs indicate order Z,Y,X 
                z, y, x = tuple(p)[:3]

            if z is not None and y is not None and x is not None:
                voxel_size = (float(z), float(y), float(x))
        else:
            # Older AICSImageIO API: returns X,Y,Z 
            if hasattr(img, "get_physical_pixel_size"):
                pxyz = img.get_physical_pixel_size(scene_index)
                if isinstance(pxyz, (list, tuple)) and len(pxyz) >= 3:
                    x, y, z = pxyz[:3]
                    voxel_size = (float(z), float(y), float(x))
    except Exception:
        pass

    # Create temporary memory-mapped files
    temp_dir = Path.cwd() / "napari_mmap_files"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Normalize arr to shape (C,Z,Y,X) where possible
    if arr.ndim == 4:
        c, z, y, x = arr.shape
    elif arr.ndim == 3:
        # could be (C,Y,X) -> treat Z=1
        c = arr.shape[0]
        z = 1
        y = arr.shape[1]
        x = arr.shape[2]
    else:
        raise ValueError(f"Unsupported LIF array shape: {arr.shape}")

    n_z = z

    # create memmaps for up to n_ch channels
    memmaps = []
    paths = []
    for ch_idx in range(min(n_ch, c)):
        pth = temp_dir / f"ch{ch_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"
        mm = np.memmap(pth, dtype=np.float32, mode='w+', shape=(n_z, y, x))
        memmaps.append(mm)
        paths.append(pth)

    print("Writing data to memory-mapped files (this may take a few minutes)...")
    chunk_size = 30
    start_time = time.time()

    for idx, start_z in enumerate(range(0, n_z, chunk_size)):
        end_z = min(start_z + chunk_size, n_z)
        for ch_i in range(len(memmaps)):
            if arr.ndim == 4:
                chunk = arr[ch_i, start_z:end_z].compute().astype(np.float32)
            else:
                chunk = arr[ch_i][np.newaxis, ...].astype(np.float32)

            memmaps[ch_i][start_z:end_z] = chunk
            memmaps[ch_i].flush()

        elapsed = time.time() - start_time
        chunks_done = idx + 1
        total_chunks = (n_z + chunk_size - 1) // chunk_size
        remaining_chunks = total_chunks - chunks_done
        time_per_chunk = elapsed / chunks_done
        est_remaining = remaining_chunks * time_per_chunk
        print(f"  Processed slices {start_z}-{end_z-1} of {n_z-1} ({elapsed:.1f}s elapsed, ~{est_remaining:.1f}s remaining)")

    print("Memory-mapped files created successfully!")

    # Calculate contrast limits per channel
    contrast_limits = {}
    mid_z = max(0, n_z // 2)
    for i, mm in enumerate(memmaps, start=1):
        try:
            sample = mm[mid_z]
            lo, hi = np.percentile(sample, [1, 99.5])
        except Exception:
            lo, hi = 0, 65535
        contrast_limits[f'ch{i}'] = (float(lo), float(hi))

    print(f"Voxel size (Z, Y, X): {voxel_size}")
    return memmaps, contrast_limits, voxel_size, paths


def load_tiff_memmap(path, max_ch=4):
    """Load TIFF using memory-mapped arrays via tifffile, write float32 memmaps.
    Returns (list_of_memmaps, contrast_limits_dict, voxel_size(Z,Y,X), paths).
    Tries to extract voxel size from:
      1) OME-TIFF OME-XML
      2) ImageJ metadata
      3) TIFF resolution tags
    """

    print("Creating memory-mapped arrays from TIFF (memmap) using tifffile...")

    # ---------- voxel size extraction helpers ----------
    def _safe_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _parse_ome_voxel_size(tif):
        """Return (Z,Y,X) in microns if possible, else None."""
        ome = getattr(tif, "ome_metadata", None)
        if not ome:
            return None
        try:
            root = ET.fromstring(ome)
            # Find first Pixels element (ignore namespaces by matching suffix)
            pixels = None
            for el in root.iter():
                if el.tag.endswith("Pixels"):
                    pixels = el
                    break
            if pixels is None:
                return None

            psx = _safe_float(pixels.attrib.get("PhysicalSizeX"))
            psy = _safe_float(pixels.attrib.get("PhysicalSizeY"))
            psz = _safe_float(pixels.attrib.get("PhysicalSizeZ"))

            # Units exist too, but we assume µm if present/typical; you can extend conversion later.
            # If any are missing, keep None.
            if psx is None and psy is None and psz is None:
                return None

            # OME uses X,Y,Z naming; return (Z,Y,X)
            z = 1.0 if psz is None else float(psz)
            y = 1.0 if psy is None else float(psy)
            x = 1.0 if psx is None else float(psx)
            return (z, y, x)
        except Exception:
            return None

    def _resolution_unit_to_microns(unit_code_or_str):
        """
        TIFF ResolutionUnit: 2=inches, 3=centimeters (common).
        Return microns per unit.
        """
        try:
            # tifffile may return an int or an enum-like
            u = unit_code_or_str
            if hasattr(u, "value"):
                u = u.value
            if isinstance(u, str):
                s = u.lower()
                if "inch" in s:
                    return 25400.0
                if "centimeter" in s or "cm" == s:
                    return 10000.0
                return None
            # numeric
            if int(u) == 2:   # inch
                return 25400.0
            if int(u) == 3:   # centimeter
                return 10000.0
            return None
        except Exception:
            return None

    def _parse_tiff_resolution_xy(tif):
        """Return (Y,X) microns-per-pixel from TIFF tags if present, else None."""
        try:
            page0 = tif.pages[0]
            tags = page0.tags

            # XResolution/YResolution are rational pixels per ResolutionUnit
            if "XResolution" not in tags or "YResolution" not in tags or "ResolutionUnit" not in tags:
                return None

            xres = tags["XResolution"].value
            yres = tags["YResolution"].value
            unit = tags["ResolutionUnit"].value

            # Convert rationals to float (pixels per unit)
            xres_f = float(xres[0]) / float(xres[1]) if isinstance(xres, tuple) else float(xres)
            yres_f = float(yres[0]) / float(yres[1]) if isinstance(yres, tuple) else float(yres)

            microns_per_unit = _resolution_unit_to_microns(unit)
            if microns_per_unit is None:
                return None

            # pixel size (units/pixel) = (microns/unit) / (pixels/unit)
            px = microns_per_unit / xres_f if xres_f > 0 else None
            py = microns_per_unit / yres_f if yres_f > 0 else None
            if px is None or py is None:
                return None
            return (float(py), float(px))
        except Exception:
            return None

    def _parse_imagej_voxel_size(tif):
        """
        ImageJ TIFF:
          - Z spacing often in imagej_metadata['spacing'] (in units of 'unit') 
          - XY sometimes via TIFF resolution tags; ImageJ may also embed things in ImageDescription.
        """
        try:
            ij = getattr(tif, "imagej_metadata", None)
            if not ij:
                return None

            unit = ij.get("unit", None)  # often 'um' 
            spacing = _safe_float(ij.get("spacing", None))  # z spacing

            # XY: best effort via TIFF tags
            yx = _parse_tiff_resolution_xy(tif)

            # If unit is not microns, we don't convert here (extend if you need)
            # We'll still return values as-is; but prefer microns.
            if yx is None and spacing is None:
                return None

            # default XY if missing
            y = 1.0
            x = 1.0
            if yx is not None:
                y, x = yx

            z = 1.0 if spacing is None else float(spacing)
            return (z, float(y), float(x))
        except Exception:
            return None

    def _parse_generic_voxel_size(tif):
        """Generic fallback: XY from resolution tags, Z unknown."""
        yx = _parse_tiff_resolution_xy(tif)
        if yx is None:
            return None
        y, x = yx
        return (1.0, float(y), float(x))

    # ---------- read pixels (prefer memmap) ----------
    try:
        data = tifffile.memmap(str(path))
        arr = np.asarray(data)
    except Exception:
        try:
            arr = tifffile.imread(str(path))
        except Exception as e:
            # last resort: AICSImage -> delegate
            img = AICSImage(str(path))
            try:
                arr = img.get_image_dask_data("CZYX")
            except Exception:
                arr = img.get_image_dask_data("CYX")
            try:
                c_total = int(img.dims.C)
            except Exception:
                c_total = arr.shape[0] if arr.ndim == 4 else 1
            return load_lif_memmap(path, scene_index=0, max_ch=min(max_ch, max(1, c_total)))

    # ---------- extract voxel size ----------
    voxel_size = (1.0, 1.0, 1.0)
    try:
        with tifffile.TiffFile(str(path)) as tif:
            vs = _parse_ome_voxel_size(tif)
            if vs is None:
                vs = _parse_imagej_voxel_size(tif)
            if vs is None:
                vs = _parse_generic_voxel_size(tif)
            if vs is not None:
                voxel_size = tuple(map(float, vs))
    except Exception:
        pass

    # ---------- normalize array into (C, Z, Y, X) ----------
    arr = np.asarray(arr)
    if arr.ndim == 4:
        a0, a1, a2, a3 = arr.shape
        if a0 <= 4 and a1 > 4:
            arr_czyx = arr  # (C,Z,Y,X)
        elif a1 <= 4 and a0 > 4:
            arr_czyx = np.moveaxis(arr, [0, 1], [1, 0])  # (Z,C,Y,X)->(C,Z,Y,X)
        else:
            arr_czyx = arr
    elif arr.ndim == 3:
        arr_czyx = np.expand_dims(arr, axis=0)  # (1,Z,Y,X)
    elif arr.ndim == 2:
        arr_czyx = arr.reshape((1, 1) + arr.shape)  # (1,1,Y,X)
    else:
        arr_czyx = np.reshape(arr, (1, 1) + arr.shape[-2:])

    c_total = arr_czyx.shape[0]
    n_ch = min(max_ch, max(1, c_total))

    # ---------- write float32 memmaps ----------
    temp_dir = Path.cwd() / "napari_mmap_files"
    temp_dir.mkdir(parents=True, exist_ok=True)

    n_z, y, x = arr_czyx.shape[1], arr_czyx.shape[2], arr_czyx.shape[3]

    memmaps = []
    paths = []
    for ch_idx in range(n_ch):
        p = temp_dir / f"ch{ch_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"
        mm = np.memmap(p, dtype=np.float32, mode='w+', shape=(n_z, y, x))
        memmaps.append(mm)
        paths.append(p)

    print("Writing data to memory-mapped files (this may take a few minutes)...")
    chunk_size = 30
    start_time = time.time()
    for idx, start_z in enumerate(range(0, n_z, chunk_size)):
        end_z = min(start_z + chunk_size, n_z)
        for ch_i in range(n_ch):
            chunk = arr_czyx[ch_i, start_z:end_z].astype(np.float32, copy=False)
            memmaps[ch_i][start_z:end_z] = chunk
            memmaps[ch_i].flush()

        elapsed = time.time() - start_time
        chunks_done = idx + 1
        total_chunks = (n_z + chunk_size - 1) // chunk_size
        remaining_chunks = total_chunks - chunks_done
        time_per_chunk = elapsed / chunks_done
        est_remaining = remaining_chunks * time_per_chunk
        print(f"  Processed slices {start_z}-{end_z-1} of {n_z-1} ({elapsed:.1f}s elapsed, ~{est_remaining:.1f}s remaining)")

    # ---------- contrast limits ----------
    contrast_limits = {}
    mid_z = max(0, n_z // 2)
    for i, mm in enumerate(memmaps, start=1):
        try:
            sample = mm[mid_z]
            lo, hi = np.percentile(sample, [1, 99.5])
        except Exception:
            lo, hi = 0, 65535
        contrast_limits[f'ch{i}'] = (float(lo), float(hi))

    print(f"Voxel size (Z, Y, X): {voxel_size}")
    print("Memory-mapped TIFF files created successfully!")
    return memmaps, contrast_limits, voxel_size, paths


###############################################################################
# ----------------------------- MASK COMPUTATION-----------------------------
###############################################################################

def compute_mask(img2d, threshold, blur_enabled=False, blur_sigma=1.0,
                 min_size=0, do_close=False, close_radius=3):
    data = np.asarray(img2d).astype(np.float32)
    if blur_enabled:
        data = gaussian(data, sigma=blur_sigma, preserve_range=True)

    mask = data > float(threshold)

    if min_size and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)  # can delete everything 
    if do_close:
        mask = closing(mask, footprint=disk(int(close_radius)))

    return mask.astype(np.uint8)
###############################################################################
# ----------------------- ANALYSIS POPUP DIALOG ----------------------------
###############################################################################

class AnalysisPopupDialog(QDialog):
    """Separate popup for analysis buttons."""
    
    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget
        self.setWindowTitle("Analysis Tools")
        self.setGeometry(100, 100, 400, 300)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()

        # Keep only the pared-down buttons requested in implementation.txt
        save_btn = QPushButton("Save Metadata")
        save_btn.clicked.connect(self.parent_widget.save_output)
        layout.addWidget(save_btn)

        load_meta_btn = QPushButton("Load Metadata")
        load_meta_btn.clicked.connect(self.parent_widget.load_metadata)
        layout.addWidget(load_meta_btn)

        load_channels_btn = QPushButton("Load Channels...")
        load_channels_btn.clicked.connect(self.parent_widget.load_channels)
        layout.addWidget(load_channels_btn)

        export_btn = QPushButton("Export Selected Channels")
        export_btn.clicked.connect(self.parent_widget.export_selected_layers)
        layout.addWidget(export_btn)

        coloc_btn = QPushButton("Colocalize Z Stack")
        coloc_btn.clicked.connect(self.parent_widget.generate_coloc_entire_stack)
        layout.addWidget(coloc_btn)

        maxproj_btn = QPushButton("Generate Max Projection")
        maxproj_btn.clicked.connect(self.parent_widget.generate_max_projection)
        layout.addWidget(maxproj_btn)

        # Keep screenshot/video buttons associated with projections
        self.video_btn = QPushButton("Save Video of Current View")
        self.video_btn.clicked.connect(self.parent_widget.save_video)
        layout.addWidget(self.video_btn)

        self.screenshot_btn = QPushButton("Save Screenshot of Current View")
        self.screenshot_btn.clicked.connect(
            lambda _=False: self.parent_widget.save_screenshot(
                viewer=self.parent_widget.viewer,
                parent=self,
                default_name="main_viewer.png",
                canvas_only=True,
            )
        )
        layout.addWidget(self.screenshot_btn)

        # 3D surfaces from selected layers
        self.surfaces3d_btn = QPushButton("Generate 3D Surfaces (Marching Cubes)")
        self.surfaces3d_btn.clicked.connect(lambda _=False: self.parent_widget.generate_3d_surfaces())
        layout.addWidget(self.surfaces3d_btn)

        self.cellcount_btn = QPushButton("Cell Counter (ROI examples → detect)")
        self.cellcount_btn.clicked.connect(lambda _=False: self.parent_widget.open_cell_counter())
        layout.addWidget(self.cellcount_btn)

        layout.addStretch()
        self.setLayout(layout)

###############################################################################
#Mask tuner for individual channel mask genearation and preview before writing to main viewer
class MaskTunerDialog(QDialog):
    """
    Interactive per-channel threshold/mask tuner.

    - Embedded napari viewer (2D slice view driven by this dialog's Z slider)
    - Right-side per-layer controls for channel + mask (visible/opacity/color/contrast)
    - Save/Load metadata that includes ALL widget + layer control settings
    """
    def __init__(self, parent_widget, channel_index, channel_data, file_path=None):
        super().__init__(parent_widget)

        # ---- basic state ----
        self.parent_widget = parent_widget
        self.channel_index = int(channel_index)  # 0-based
        self.file_path = str(file_path) if file_path is not None else str(
            getattr(parent_widget, "file_path", "unknown")
        )

        self.arr = np.asarray(channel_data)
        self.arr3 = self.arr[np.newaxis, ...] if self.arr.ndim == 2 else self.arr
        self.nz = int(self.arr3.shape[0]) if self.arr3.ndim == 3 else 1

        self.z = int(getattr(parent_widget, "current_z", 0))
        self.z = max(0, min(self.z, max(self.nz - 1, 0)))

        # thresholding state (drives mask computation)
        self.blur_enabled = False
        self.blur_sigma = 1.0
        self.threshold = 500

        # full 3D mask storage; displayed mask is a 2D slice
        self.preview_masks = np.zeros(self.arr3.shape, dtype=np.uint8)

        self.setWindowTitle(f"Mask Tuner — Channel {self.channel_index + 1}")
        self.setMinimumSize(950, 700)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        outer = QVBoxLayout(self)

        # ---- embedded viewer + right panel ----
        self.viewer = napari.Viewer()

        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        # Left: napari viewer widget
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(self.viewer.window._qt_viewer)
        split.addWidget(left)

        # Create layers BEFORE creating controls that reference them
        self.img_layer = self.viewer.add_image(
            self.arr3[self.z],
            name=f"Channel {self.channel_index + 1}",
            blending="additive",
            opacity=1.0,
        )
        self.mask_layer = self.viewer.add_image(
            self.preview_masks[self.z],
            name="Mask (preview)",
            colormap="yellow",
            blending="additive",
            opacity=0.6,
            contrast_limits=(0, 1),
        )

        # Right: controls panel (we will store handles to all widgets)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 6, 6, 6)

        self.layer_ctrl_widgets = {}  # keys: "channel", "mask" -> dict of widgets

        gb1, ctrl1 = self._layer_controls_group(
            title=f"Channel {self.channel_index + 1} Controls",
            layer=self.img_layer,
            is_mask=False,
        )
        self.layer_ctrl_widgets["channel"] = ctrl1
        right_lay.addWidget(gb1)

        gb2, ctrl2 = self._layer_controls_group(
            title="Mask Controls",
            layer=self.mask_layer,
            is_mask=True,
        )
        self.layer_ctrl_widgets["mask"] = ctrl2
        right_lay.addWidget(gb2)

        right_lay.addStretch(1)
        split.addWidget(right)
        split.setSizes([700, 250])

        # ---- bottom control row ----
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Z"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, max(self.nz - 1, 0))
        self.z_slider.setValue(self.z)
        ctrl.addWidget(self.z_slider)

        self.z_label = QLabel(f"{self.z}/{max(self.nz - 1, 0)}")
        self.z_label.setFixedWidth(90)
        ctrl.addWidget(self.z_label)

        self.otsu_btn = QPushButton("Otsu (this slice)")
        ctrl.addWidget(self.otsu_btn)

        ctrl.addWidget(QLabel("Threshold"))
        self.th_slider = QSlider(Qt.Horizontal)
        th_max = int(np.max(self.arr3)) if self.arr3.size else 1
        self.th_slider.setRange(0, max(th_max, 1))
        self.th_slider.setValue(int(self.threshold))
        ctrl.addWidget(self.th_slider)

        self.th_val = QLabel(str(int(self.threshold)))
        self.th_val.setFixedWidth(70)
        ctrl.addWidget(self.th_val)

        self.blur_cb = QCheckBox("Blur")
        ctrl.addWidget(self.blur_cb)

        ctrl.addWidget(QLabel("σ"))
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(0, 50)  # 0..5.0
        self.sigma_slider.setValue(int(round(self.blur_sigma * 10)))
        ctrl.addWidget(self.sigma_slider)

        self.sigma_val = QLabel(f"{self.blur_sigma:.1f}")
        self.sigma_val.setFixedWidth(60)
        ctrl.addWidget(self.sigma_val)

        self.preview_all_btn = QPushButton("Preview full stack")
        ctrl.addWidget(self.preview_all_btn)

        self.accept_btn = QPushButton("Accept (write to main)")
        ctrl.addWidget(self.accept_btn)

        outer.addLayout(ctrl)

        # ---- meta row ----
        meta_row = QHBoxLayout()
        self.save_meta_btn = QPushButton("Save Channel Mask Meta")
        self.load_meta_btn = QPushButton("Load Channel Mask Meta")
        meta_row.addWidget(self.save_meta_btn)
        meta_row.addWidget(self.load_meta_btn)
        meta_row.addStretch(1)
        outer.addLayout(meta_row)

        # ---- signals ----
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self.th_slider.valueChanged.connect(self._on_threshold_changed)
        self.blur_cb.stateChanged.connect(self._on_blur_toggle)
        self.sigma_slider.valueChanged.connect(self._on_sigma_changed)

        self.otsu_btn.clicked.connect(self._otsu_current_slice)
        self.preview_all_btn.clicked.connect(self._preview_full_stack)
        self.accept_btn.clicked.connect(self._accept_mask)

        self.save_meta_btn.clicked.connect(self._save_channel_meta)
        self.load_meta_btn.clicked.connect(self._load_channel_meta)

        # ---- initialize preview ----
        self._preview_current_slice()

    # ---------------- helpers ----------------
    def _set_slider_from_lineedit(self, lineedit, slider, lo, hi):
        try:
            val = int(lineedit.text())
        except Exception:
            return
        val = max(int(lo), min(int(hi), val))
        slider.setValue(val)

    def _safe_get_colormap_str(self, layer):
        """Return something JSON-serializable for colormap."""
        try:
            cm = getattr(layer, "colormap", None)
            if cm is None:
                return None
            # napari colormap objects have .name; hex strings also fine
            name = getattr(cm, "name", None)
            if isinstance(name, str) and name:
                return name
            if isinstance(cm, str) and cm:
                return cm
            return str(cm)
        except Exception:
            return None

    def _apply_layer_settings(self, layer, settings):
        """Apply layer settings dict safely."""
        if not isinstance(settings, dict):
            return
        try:
            if "visible" in settings:
                layer.visible = bool(settings["visible"])
        except Exception:
            pass
        try:
            if "opacity" in settings:
                layer.opacity = float(settings["opacity"])
        except Exception:
            pass
        try:
            if "contrast_limits" in settings and settings["contrast_limits"] is not None:
                a, b = settings["contrast_limits"]
                layer.contrast_limits = (float(a), float(b))
        except Exception:
            pass
        try:
            if "colormap" in settings and settings["colormap"]:
                layer.colormap = settings["colormap"]
        except Exception:
            pass

    # ---------------- UI handlers ----------------
    def _on_z_changed(self, v):
        self.z = int(v)
        if self.nz <= 0:
            return
        self.z = max(0, min(self.z, self.nz - 1))

        self.z_label.setText(f"{self.z}/{max(self.nz - 1, 0)}")

        # Update displayed slices (2D)
        self.img_layer.data = self.arr3[self.z]
        self.mask_layer.data = self.preview_masks[self.z]
        try:
            self.mask_layer.contrast_limits = (0, 1)
        except Exception:
            pass

        self._preview_current_slice()

    def _on_threshold_changed(self, v):
        self.threshold = int(v)
        self.th_val.setText(str(int(self.threshold)))
        self._preview_current_slice()

    def _on_blur_toggle(self, s):
        self.blur_enabled = bool(s)
        self._preview_current_slice()

    def _on_sigma_changed(self, v):
        self.blur_sigma = float(v) / 10.0
        self.sigma_val.setText(f"{self.blur_sigma:.1f}")
        if self.blur_enabled:
            self._preview_current_slice()

    def _otsu_current_slice(self):
        sl = self.arr3[self.z]
        try:
            th = float(threshold_otsu(sl))
        except Exception:
            th = float(np.mean(sl))
        self.threshold = int(th)

        self.th_slider.blockSignals(True)
        try:
            self.th_slider.setValue(self.threshold)
        finally:
            self.th_slider.blockSignals(False)

        self.th_val.setText(str(int(self.threshold)))
        self._preview_current_slice()

    # ---------------- mask preview ----------------
    def _preview_current_slice(self):
        m = compute_mask(
            self.arr3[self.z],
            threshold=self.threshold,
            blur_enabled=self.blur_enabled,
            blur_sigma=self.blur_sigma,
            min_size=0,
            do_close=False,
        )
        self.preview_masks[self.z] = m

        self.mask_layer.data = self.preview_masks[self.z]
        try:
            self.mask_layer.contrast_limits = (0, 1)
        except Exception:
            pass

    def _preview_full_stack(self):
        for z in range(self.nz):
            self.preview_masks[z] = compute_mask(
                self.arr3[z],
                threshold=self.threshold,
                blur_enabled=self.blur_enabled,
                blur_sigma=self.blur_sigma,
                min_size=0,
                do_close=False,
            )
        self.mask_layer.data = self.preview_masks[self.z]

    # ---------------- accept into main ----------------
    def _accept_mask(self):
        try:
            # Always build a complete 3D mask using the *current* parameters
            self._preview_full_stack()

            masks_out = (self.preview_masks > 0).astype(np.uint8)

            self.parent_widget.create_mask_layer(self.channel_index, masks_out)
            try:
                self.parent_widget.z_changed(int(getattr(self.parent_widget, "current_z", 0)))
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write mask to main UI: {e}")
            return

        self.accept()

    # ---------------- save/load per-channel meta ----------------
    def _meta_path_default(self):
        meta_dir = Path("meta_data")
        meta_dir.mkdir(exist_ok=True)
        stem = Path(self.file_path).stem
        return meta_dir / f"{stem}.channel{self.channel_index + 1}.maskmeta.json"

    def _collect_ui_state(self):
        """Collect ALL UI widget values + relevant layer settings."""
        md = {
            "type": "mask_tuner_channel_meta_v2",
            "source_file": str(self.file_path),
            "timestamp": datetime.now().isoformat(),
            "channel_index_0based": int(self.channel_index),

            # thresholding controls
            "z": int(self.z_slider.value()),
            "threshold": int(self.th_slider.value()),
            "blur_enabled": bool(self.blur_cb.isChecked()),
            "blur_sigma": float(self.sigma_slider.value()) / 10.0,

            # layer settings and right-panel widget states
            "layers": {
                "channel": {
                    "visible": bool(getattr(self.img_layer, "visible", True)),
                    "opacity": float(getattr(self.img_layer, "opacity", 1.0)),
                    "contrast_limits": list(getattr(self.img_layer, "contrast_limits", (0, 1))),
                    "colormap": self._safe_get_colormap_str(self.img_layer),
                    "ui": self.layer_ctrl_widgets.get("channel", {}).get("ui_state", {}),
                },
                "mask": {
                    "visible": bool(getattr(self.mask_layer, "visible", True)),
                    "opacity": float(getattr(self.mask_layer, "opacity", 0.6)),
                    "contrast_limits": list(getattr(self.mask_layer, "contrast_limits", (0, 1))),
                    "colormap": self._safe_get_colormap_str(self.mask_layer),
                    "ui": self.layer_ctrl_widgets.get("mask", {}).get("ui_state", {}),
                }
            }
        }
        return md

    def _save_channel_meta(self):
        try:
            path = self._meta_path_default()

            # Update stored per-layer UI values right before save
            for key in ("channel", "mask"):
                ctrl = self.layer_ctrl_widgets.get(key, {})
                if not ctrl:
                    continue
                ctrl["ui_state"] = {
                    "visible": bool(ctrl["visible_cb"].isChecked()),
                    "opacity_slider": int(ctrl["opacity_slider"].value()),
                    "min_slider": int(ctrl["min_slider"].value()),
                    "max_slider": int(ctrl["max_slider"].value()),
                }

            md = self._collect_ui_state()

            with open(path, "w") as f:
                json.dump(md, f, indent=4)
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save channel meta: {e}")

    def _load_channel_meta(self):
        try:
            path = self._meta_path_default()
            if not path.exists():
                QMessageBox.warning(self, "Not found", f"No meta file found:\n{path}")
                return

            with open(path, "r") as f:
                md = json.load(f)

            # ---- apply thresholding sliders ----
            self.z_slider.blockSignals(True)
            self.th_slider.blockSignals(True)
            self.blur_cb.blockSignals(True)
            self.sigma_slider.blockSignals(True)
            try:
                z = int(md.get("z", self.z))
                z = max(0, min(z, self.nz - 1))
                th = int(md.get("threshold", self.threshold))
                be = bool(md.get("blur_enabled", self.blur_enabled))
                bs = float(md.get("blur_sigma", self.blur_sigma))

                self.z_slider.setValue(z)
                self.th_slider.setValue(th)
                self.blur_cb.setChecked(be)
                self.sigma_slider.setValue(int(round(bs * 10)))
            finally:
                self.z_slider.blockSignals(False)
                self.th_slider.blockSignals(False)
                self.blur_cb.blockSignals(False)
                self.sigma_slider.blockSignals(False)

            # update internal state + labels
            self.z = int(self.z_slider.value())
            self.threshold = int(self.th_slider.value())
            self.blur_enabled = bool(self.blur_cb.isChecked())
            self.blur_sigma = float(self.sigma_slider.value()) / 10.0
            self.z_label.setText(f"{self.z}/{max(self.nz - 1, 0)}")
            self.th_val.setText(str(int(self.threshold)))
            self.sigma_val.setText(f"{self.blur_sigma:.1f}")

            # ---- apply layer settings (visible/opacity/contrast/colormap) ----
            layers_md = md.get("layers", {})
            self._apply_layer_settings(self.img_layer, layers_md.get("channel", {}))
            self._apply_layer_settings(self.mask_layer, layers_md.get("mask", {}))

            # ---- apply right-panel UI sliders to match ----
            def _apply_ctrl(key):
                ctrl = self.layer_ctrl_widgets.get(key, {})
                ui = (layers_md.get(key, {}) or {}).get("ui", {}) or (layers_md.get(key, {}) or {}).get("ui_state", {})
                if not ctrl or not ui:
                    return

                for wname, setter in [
                    ("visible_cb", "setChecked"),
                    ("opacity_slider", "setValue"),
                    ("min_slider", "setValue"),
                    ("max_slider", "setValue"),
                ]:
                    if wname not in ctrl:
                        continue
                    widget = ctrl[wname]
                    widget.blockSignals(True)
                    try:
                        if wname == "visible_cb" and "visible" in ui:
                            getattr(widget, setter)(bool(ui["visible"]))
                        elif wname == "opacity_slider" and "opacity_slider" in ui:
                            getattr(widget, setter)(int(ui["opacity_slider"]))
                        elif wname == "min_slider" and "min_slider" in ui:
                            getattr(widget, setter)(int(ui["min_slider"]))
                        elif wname == "max_slider" and "max_slider" in ui:
                            getattr(widget, setter)(int(ui["max_slider"]))
                    finally:
                        widget.blockSignals(False)

                # trigger layer update from sliders
                try:
                    ctrl["apply_fn"]()
                except Exception:
                    pass

            _apply_ctrl("channel")
            _apply_ctrl("mask")

            # update displayed slices + recompute mask
            self.img_layer.data = self.arr3[self.z]
            self.mask_layer.data = self.preview_masks[self.z]
            self._preview_current_slice()

            QMessageBox.information(self, "Loaded", f"Loaded:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load channel meta: {e}")

    # ---------------- right-side layer controls ----------------
    def _layer_controls_group(self, title, layer, is_mask=False):
        """
        Returns (groupbox, ctrl_dict) where ctrl_dict stores widget handles
        and an apply_fn() that writes widget values to the napari layer.
        """
        gb = QGroupBox(title)
        vlay = QVBoxLayout(gb)

        # Visible + opacity + color
        top = QHBoxLayout()
        vis_cb = QCheckBox("Visible")
        vis_cb.setChecked(True)
        top.addWidget(vis_cb)

        top.addWidget(QLabel("Opacity"))
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(int(getattr(layer, "opacity", 1.0) * 100))
        top.addWidget(opacity_slider)

        color_btn = QPushButton("Color")
        top.addWidget(color_btn)
        vlay.addLayout(top)

        # Contrast limits sliders
        if is_mask:
            lo, hi = 0, 1
            default_min, default_max = 0, 1
        else:
            try:
                sl = np.asarray(layer.data)
                default_min = int(np.nanmin(sl))
                default_max = int(np.nanmax(sl))
                if default_max <= default_min:
                    default_max = default_min + 1
                lo, hi = 0, max(default_max, 1)
            except Exception:
                lo, hi = 0, 65535
                default_min, default_max = 0, 65535

        def _minmax_row(label, init):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sld = QSlider(Qt.Horizontal)
            sld.setRange(int(lo), int(hi))
            sld.setValue(int(init))
            row.addWidget(sld)
            le = QLineEdit(str(int(init)))
            le.setFixedWidth(80)
            le.setValidator(QIntValidator(int(lo), int(hi)))
            le.editingFinished.connect(
                lambda le=le, sld=sld: self._set_slider_from_lineedit(le, sld, int(lo), int(hi))
            )
            sld.valueChanged.connect(lambda v, le=le: le.setText(str(int(v))))
            row.addWidget(le)
            vlay.addLayout(row)
            return sld

        min_slider = _minmax_row("Min:", default_min)
        max_slider = _minmax_row("Max:", default_max)

        # Apply function: write UI -> layer
        def apply_fn():
            try:
                layer.visible = bool(vis_cb.isChecked())
            except Exception:
                pass
            try:
                layer.opacity = float(opacity_slider.value()) / 100.0
            except Exception:
                pass
            try:
                mn = int(min_slider.value())
                mx = int(max_slider.value())
                if mx <= mn:
                    mx = mn + 1
                layer.contrast_limits = (mn, mx)
            except Exception:
                pass
            if is_mask:
                # keep mask visible even if user moves contrast around
                try:
                    layer.contrast_limits = (0, 1)
                except Exception:
                    pass

        # Connect signals
        vis_cb.stateChanged.connect(lambda _=None: apply_fn())
        opacity_slider.valueChanged.connect(lambda _=None: apply_fn())
        min_slider.valueChanged.connect(lambda _=None: apply_fn())
        max_slider.valueChanged.connect(lambda _=None: apply_fn())

        def pick_color():
            col = QColorDialog.getColor(parent=self)
            if not col.isValid():
                return
            try:
                layer.colormap = col.name()
            except Exception:
                pass
        color_btn.clicked.connect(pick_color)

        # initialize
        apply_fn()

        ctrl = {
            "visible_cb": vis_cb,
            "opacity_slider": opacity_slider,
            "min_slider": min_slider,
            "max_slider": max_slider,
            "apply_fn": apply_fn,
            "ui_state": {},   # filled on save
        }
        return gb, ctrl

###############################################################################

# Building of the main UI
class UIWidget(QWidget):
    def __init__(self, viewer, channel_list, channel_layers, contrast_limits, file_path):
        super().__init__()
        self.viewer = viewer
        self.channel_list = channel_list
        self.channel_layers = channel_layers
        self.contrast_limits = contrast_limits
        self.file_path = file_path

        # determine Z size from first channel
        first = np.asarray(channel_list[0])
        self.n_z = first.shape[0] if first.ndim == 3 else 1
        self.current_z = 0

        self.setLayout(QVBoxLayout())
        self.build_ui()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.coloc_controls = getattr(self, "coloc_controls", [])
        self.coloc_control_boxes = getattr(self, "coloc_control_boxes", [])
        arr = np.asarray(channel_list[0])
        print("dtype:", arr.dtype, "shape:", arr.shape, "min/max:", float(np.min(arr)), float(np.max(arr)))

    def build_ui(self):
        layout = self.layout()

        # Z slider
        z_box = QGroupBox('Z Slice Navigation')
        z_layout = QVBoxLayout()
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, max(self.n_z - 1, 0))
        self.z_slider.valueChanged.connect(self.z_changed)
        z_layout.addWidget(self.z_slider)
        z_box.setLayout(z_layout)
        layout.addWidget(z_box)

        def infer_max_range_from_channels(channel_list, fallback=65535):
            try:
                best_dtype = None
                best_p99 = 0.0
                best_max = 0.0

                for ch in channel_list:
                    arr = np.asarray(ch)
                    if best_dtype is None:
                        best_dtype = arr.dtype

                    # sample a few z planes
                    if arr.ndim == 3 and arr.shape[0] > 1:
                        z_idx = [0, arr.shape[0] // 2, arr.shape[0] - 1]
                        sample = np.asarray(arr[z_idx, ...]).ravel()
                    else:
                        sample = arr.ravel()

                    if np.issubdtype(sample.dtype, np.floating):
                        sample = sample[np.isfinite(sample)]

                    if sample.size == 0:
                        continue

                    best_max = max(best_max, float(np.max(sample)))
                    best_p99 = max(best_p99, float(np.percentile(sample, 99)))

                # integer range
                if best_dtype is not None and np.issubdtype(best_dtype, np.integer):
                    return int(min(np.iinfo(best_dtype).max, fallback))

                # float range
                v = best_p99 if best_p99 > 0 else best_max
                if v <= 0:
                    return fallback

                if v <= 255: return 255
                if v <= 4095: return 4095
                if v <= 65535: return 65535
                return int(min(v, fallback))

            except Exception:
                return fallback

        max_range = infer_max_range_from_channels(self.channel_list, fallback=65535)

        # Per-channel controls
        for idx, layer in enumerate(self.channel_layers, start=1):
            gb = QGroupBox(f'Channel {idx} Controls')
            vlay = QVBoxLayout()

            # Visibility + opacity
            top_h = QHBoxLayout()
            vis_cb = QCheckBox('Visible')
            vis_cb.setChecked(True)
            vis_cb.stateChanged.connect(lambda s, lyr=layer: setattr(lyr, 'visible', bool(s)))
            top_h.addWidget(vis_cb)

            top_h.addWidget(QLabel('Opacity'))
            op = QSlider(Qt.Horizontal)
            op.setRange(0, 100)
            op.setValue(int(layer.opacity * 100))
            op.valueChanged.connect(lambda v, lyr=layer: setattr(lyr, 'opacity', v / 100.0))
            top_h.addWidget(op)

            # Color picker
            color_btn = QPushButton('Color')
            def make_color_picker(lyr):
                def pick():
                    col = QColorDialog.getColor(parent=self)
                    if col.isValid():
                        try:
                            lyr.colormap = col.name()
                        except Exception:
                            pass
                return pick
            color_btn.clicked.connect(make_color_picker(layer))
            top_h.addWidget(color_btn)

            vlay.addLayout(top_h)

            # ---------------------------------------------------
            # MIN / MAX CONTRAST SLIDERS (kept)
            # ---------------------------------------------------
            ch_min_layout = QHBoxLayout()
            ch_min_layout.addWidget(QLabel('Min:'))
            ch_min = QSlider(Qt.Horizontal)
            ch_min.setRange(0, max_range)

            try:
                lim = self.contrast_limits.get(f'ch{idx}', None)
            except Exception:
                lim = None
            if lim is None:
                try:
                    lim = tuple(layer.contrast_limits)
                except Exception:
                    lim = (0, max_range)

            ch_min.setValue(int(lim[0]))
            ch_min_layout.addWidget(ch_min)

            ch_min_val = QLineEdit(str(int(lim[0])))
            ch_min_val.setFixedWidth(80)
            ch_min_val.setValidator(QIntValidator(0, max_range))
            ch_min_val.editingFinished.connect(
                lambda le=ch_min_val, sl=ch_min: self._set_slider_from_lineedit(le, sl, 0, max_range)
            )
            ch_min_layout.addWidget(ch_min_val)
            vlay.addLayout(ch_min_layout)

            ch_max_layout = QHBoxLayout()
            ch_max_layout.addWidget(QLabel('Max:'))
            ch_max = QSlider(Qt.Horizontal)
            ch_max.setRange(0, max_range)
            ch_max.setValue(int(lim[1]))
            ch_max_layout.addWidget(ch_max)

            ch_max_val = QLineEdit(str(int(lim[1])))
            ch_max_val.setFixedWidth(80)
            ch_max_val.setValidator(QIntValidator(0, max_range))
            ch_max_val.editingFinished.connect(
                lambda le=ch_max_val, sl=ch_max: self._set_slider_from_lineedit(le, sl, 0, max_range)
            )
            ch_max_layout.addWidget(ch_max_val)
            vlay.addLayout(ch_max_layout)

            # Apply min/max updates to napari
            def _apply_contrast(_=None, lyr=layer, smin=ch_min, smax=ch_max, le_min=ch_min_val, le_max=ch_max_val):
                lo = int(smin.value())
                hi = int(smax.value())
                if hi < lo:
                    hi = lo
                    smax.blockSignals(True)
                    smax.setValue(hi)
                    smax.blockSignals(False)

                le_min.setText(str(lo))
                le_max.setText(str(hi))

                try:
                    lyr.contrast_limits = (lo, hi)
                except Exception:
                    pass

            ch_min.valueChanged.connect(_apply_contrast)
            ch_max.valueChanged.connect(_apply_contrast)

            # Initialize text fields
            ch_min_val.setText(str(int(ch_min.value())))
            ch_max_val.setText(str(int(ch_max.value())))

            gb.setLayout(vlay)
            layout.addWidget(gb)

        # Container for dynamic mask controls
        self.mask_container = QVBoxLayout()
        layout.addLayout(self.mask_container)

        # Analysis buttons
        btns = QHBoxLayout()
        analysis_btn = QPushButton('Data Analysis')
        analysis_btn.clicked.connect(self.open_analysis)
        btns.addWidget(analysis_btn)

        preprocess_btn = QPushButton('Pre-process Analysis')
        preprocess_btn.clicked.connect(self.open_preprocess)
        btns.addWidget(preprocess_btn)

        layout.addLayout(btns)
    
    def _set_slider_from_lineedit(self, lineedit, slider, lo, hi):
        try:
            val = int(lineedit.text())
        except Exception:
            return
        val = max(int(lo), min(int(hi), val))
        try:
            lineedit.setText(str(int(val)))
        except Exception:
            pass
        slider.setValue(int(val))

    def _get_layer_by_name(self, name):
        for lyr in self.viewer.layers:
            if getattr(lyr, "name", None) == name:
                return lyr
        return None

    def _add_coloc_controls(self, layer, title: str, max_range: int = 65535, is_binary: bool = False):
        gb = QGroupBox(title)
        v = QVBoxLayout()

        # Opacity
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        op = QSlider(Qt.Horizontal)
        op.setRange(0, 100)
        op.setValue(int(layer.opacity * 100) if layer.opacity is not None else 70)
        op_val = QLineEdit(str(op.value()))
        op_val.setFixedWidth(60)
        op_val.setValidator(QIntValidator(0, 100))
        def _set_opacity(val):
            layer.opacity = int(val) / 100.0
            op_val.setText(str(int(val)))
        op.valueChanged.connect(_set_opacity)
        op_row.addWidget(op)
        op_row.addWidget(op_val)
        v.addLayout(op_row)

        # Contrast limits (min/max)
        cl_row1 = QHBoxLayout()
        cl_row1.addWidget(QLabel("Min"))
        smin = QSlider(Qt.Horizontal)
        smin.setRange(0, 1 if is_binary else max_range)
        cl_row1.addWidget(smin)
        v.addLayout(cl_row1)

        cl_row2 = QHBoxLayout()
        cl_row2.addWidget(QLabel("Max"))
        smax = QSlider(Qt.Horizontal)
        smax.setRange(0, 1 if is_binary else max_range)
        cl_row2.addWidget(smax)
        v.addLayout(cl_row2)

        # initialize from layer if present
        try:
            c0, c1 = layer.contrast_limits
        except Exception:
            c0, c1 = (0, 1) if is_binary else (0, max_range)

        # clamp + set
        c0 = int(max(0, min(c0, smin.maximum())))
        c1 = int(max(0, min(c1, smax.maximum())))
        if c1 < c0:
            c1 = c0

        smin.setValue(c0)
        smax.setValue(c1)

        def _apply_clims(_=None):
            lo = int(smin.value())
            hi = int(smax.value())
            if hi < lo:
                hi = lo
                smax.setValue(hi)
            try:
                layer.contrast_limits = (lo, hi)
            except Exception:
                pass

        smin.valueChanged.connect(_apply_clims)
        smax.valueChanged.connect(_apply_clims)

        gb.setLayout(v)
        ctrl = {"opacity": op, "cmin": smin, "cmax": smax, "layer": layer}
        return gb, ctrl

    def ensure_layer_controls(self, layer, key, title=None, max_range=65535):
        """
        Create controls once per 'key'. key can be ('mask', ch_idx) or ('derived', name).
        """
        if not hasattr(self, "layer_control_widgets"):
            self.layer_control_widgets = {}

        if key in self.layer_control_widgets:
            return  # already have controls

        gb = QGroupBox(title or layer.name)
        vlay = QVBoxLayout(gb)

        # Visibility + Opacity
        row = QHBoxLayout()
        vis = QCheckBox("Visible")
        vis.setChecked(True)
        vis.stateChanged.connect(lambda s, lyr=layer: setattr(lyr, "visible", bool(s)))
        row.addWidget(vis)

        row.addWidget(QLabel("Opacity"))
        op = QSlider(Qt.Horizontal)
        op.setRange(0, 100)
        op.setValue(int(getattr(layer, "opacity", 1.0) * 100))
        op.valueChanged.connect(lambda v, lyr=layer: setattr(lyr, "opacity", v / 100.0))
        row.addWidget(op)
        vlay.addLayout(row)

        # Contrast limits (simple version)
        try:
            mn, mx = layer.contrast_limits
        except Exception:
            mn, mx = 0, max_range

        min_sl = QSlider(Qt.Horizontal); min_sl.setRange(0, max_range); min_sl.setValue(int(mn))
        max_sl = QSlider(Qt.Horizontal); max_sl.setRange(0, max_range); max_sl.setValue(int(mx))

        def upd():
            try:
                a = int(min_sl.value())
                b = int(max_sl.value())
                if b <= a: b = a + 1
                layer.contrast_limits = (a, b)
            except Exception:
                pass

        min_sl.valueChanged.connect(lambda _=None: upd())
        max_sl.valueChanged.connect(lambda _=None: upd())

        vlay.addWidget(QLabel("Min"))
        vlay.addWidget(min_sl)
        vlay.addWidget(QLabel("Max"))
        vlay.addWidget(max_sl)

        self.layer_control_widgets[key] = gb

        # Put it somewhere: e.g. self.mask_container or a right dock widget layout you already have
        self.mask_container.addWidget(gb)

    def z_changed(self, v):
        v = int(v)
        self.current_z = v

        # ---- channels ----
        for idx, ch in enumerate(self.channel_list):
            arr = np.asarray(ch)
            lyr = self.channel_layers[idx]
            if arr.ndim == 3:
                z = max(0, min(v, arr.shape[0] - 1))
                lyr.data = arr[z]
            else:
                lyr.data = arr

        # ---- masks ----
        if hasattr(self, "masks_by_channel"):
            for ch_idx, m3 in list(self.masks_by_channel.items()):
                if m3 is None:
                    continue
                m3 = np.asarray(m3)
                if m3.ndim == 2:
                    m2 = (m3 > 0).astype(np.uint8)
                else:
                    z = max(0, min(v, m3.shape[0] - 1))
                    m2 = (m3[z] > 0).astype(np.uint8)

                lay = None
                if hasattr(self, "mask_layers"):
                    lay = self.mask_layers.get(ch_idx, None)
                if lay is None:
                    name = f"Channel {ch_idx+1} mask"
                    try:
                        lay = self.viewer.layers[name]
                    except Exception:
                        lay = None

                if lay is not None:
                    lay.data = m2
                    try:
                        lay.contrast_limits = (0, 1)
                    except Exception:
                        pass

        # ---- derived (coloc/multiply) ----
        if hasattr(self, "derived_stacks"):
            self.derived_layers = getattr(self, "derived_layers", {})
            for k, arr3 in list(self.derived_stacks.items()):
                lay = self.derived_layers.get(k, None)
                if lay is None:
                    continue
                arr3 = np.asarray(arr3)
                if arr3.ndim == 3:
                    z = max(0, min(v, arr3.shape[0] - 1))
                    lay.data = arr3[z]
                else:
                    lay.data = arr3

                    def _set_slider_from_lineedit(self, lineedit, slider, lo, hi):
                        try:
                            val = int(lineedit.text())
                        except Exception:
                            return
                        val = max(lo, min(hi, val))
                        slider.setValue(val)
        # prune deleted coloc controls
        if hasattr(self, "coloc_controls"):
            alive = []
            alive_boxes = []
            for ctrl, box in zip(self.coloc_controls, getattr(self, "coloc_control_boxes", [])):
                lay = ctrl.get("layer", None)
                if lay is not None and lay in self.viewer.layers:
                    alive.append(ctrl); alive_boxes.append(box)
                else:
                    try: box.setParent(None)
                    except Exception: pass
            self.coloc_controls = alive
            self.coloc_control_boxes = alive_boxes

    def _add_channel_controls(self, layer, idx, max_range=65535):
        """Create UI controls for one channel and return (groupbox, ctrl_dict)."""
        gb = QGroupBox(f"Channel {idx} Controls")
        vlay = QVBoxLayout(gb)

        # Visibility + Opacity row
        top_h = QHBoxLayout()
        vis_cb = QCheckBox("Visible")
        vis_cb.setChecked(True)
        vis_cb.stateChanged.connect(lambda s, lyr=layer: setattr(lyr, "visible", bool(s)))
        top_h.addWidget(vis_cb)

        top_h.addWidget(QLabel("Opacity"))
        op = QSlider(Qt.Horizontal)
        op.setRange(0, 100)
        op.setValue(int(getattr(layer, "opacity", 1.0) * 100))
        op.valueChanged.connect(lambda v, lyr=layer: setattr(lyr, "opacity", v / 100.0))
        top_h.addWidget(op)

        # Color
        color_btn = QPushButton("Color")
        def make_color_picker(lyr):
            def pick():
                col = QColorDialog.getColor(parent=self)
                if col.isValid():
                    try:
                        lyr.colormap = col.name()
                    except Exception:
                        pass
            return pick
        color_btn.clicked.connect(make_color_picker(layer))
        top_h.addWidget(color_btn)

        vlay.addLayout(top_h)

        # ----------------------------------------------------------
        # Simple MIN/MAX CONTRAST LIMIT SLIDERS (napari-native)
        # ----------------------------------------------------------
        lim = self.contrast_limits.get(f"ch{idx}", (0, max_range))

        # Min
        ch_min_layout = QHBoxLayout()
        ch_min_layout.addWidget(QLabel("Min:"))
        ch_min = QSlider(Qt.Horizontal)
        ch_min.setRange(0, max_range)
        ch_min.setValue(int(lim[0]))
        ch_min_layout.addWidget(ch_min)

        ch_min_val = QLineEdit(str(int(lim[0])))
        ch_min_val.setFixedWidth(80)
        ch_min_val.setValidator(QIntValidator(0, max_range))
        ch_min_val.editingFinished.connect(
            lambda le=ch_min_val, sl=ch_min: self._set_slider_from_lineedit(le, sl, 0, max_range)
        )
        ch_min_layout.addWidget(ch_min_val)
        vlay.addLayout(ch_min_layout)

        # Max
        ch_max_layout = QHBoxLayout()
        ch_max_layout.addWidget(QLabel("Max:"))
        ch_max = QSlider(Qt.Horizontal)
        ch_max.setRange(0, max_range)
        ch_max.setValue(int(lim[1]))
        ch_max_layout.addWidget(ch_max)

        ch_max_val = QLineEdit(str(int(lim[1])))
        ch_max_val.setFixedWidth(80)
        ch_max_val.setValidator(QIntValidator(0, max_range))
        ch_max_val.editingFinished.connect(
            lambda le=ch_max_val, sl=ch_max: self._set_slider_from_lineedit(le, sl, 0, max_range)
        )
        ch_max_layout.addWidget(ch_max_val)
        vlay.addLayout(ch_max_layout)

        # --- update napari contrast limits ---
        def _apply_contrast_from_sliders(_=None, lyr=layer,
                                        smin=ch_min, smax=ch_max,
                                        le_min=ch_min_val, le_max=ch_max_val):
            lo = int(smin.value())
            hi = int(smax.value())

            if hi < lo:
                hi = lo
                smax.blockSignals(True)
                smax.setValue(hi)
                smax.blockSignals(False)

            # keep text boxes synced
            le_min.setText(str(lo))
            le_max.setText(str(hi))

            try:
                lyr.contrast_limits = (lo, hi)
            except Exception:
                pass

        ch_min.valueChanged.connect(_apply_contrast_from_sliders)
        ch_max.valueChanged.connect(_apply_contrast_from_sliders)

        # normalize line edits
        ch_min_val.setText(str(int(ch_min.value())))
        ch_max_val.setText(str(int(ch_max.value())))

        ctrl = {
            "visible_cb": vis_cb,
            "opacity_slider": op,
            "min_slider": ch_min,
            "max_slider": ch_max,
            "min_val": ch_min_val,
            "max_val": ch_max_val,
            "layer": layer,
            "groupbox": gb,
        }
        return gb, ctrl

    def open_analysis(self):
        # Keep a reference so dialog doesn't get garbage-collected
        self.analysis_dialog = AnalysisPopupDialog(self)
        self.analysis_dialog.show()

    def save_output(self):
        """Save metadata (UI + per-channel display settings) to JSON."""
        try:
            metadata_dir = Path("meta_data")
            metadata_dir.mkdir(exist_ok=True)

            original_filename = Path(self.file_path)  # <-- use file_path
            json_filename = original_filename.stem + ".json"
            metadata_path = metadata_dir / json_filename

            channels_meta = []
            for i, layer in enumerate(getattr(self, "channel_layers", []), start=1):
                if layer is None:
                    continue

                ch = {
                    "channel_index": i,
                    "layer_name": getattr(layer, "name", f"ch{i}"),
                    "visible": bool(getattr(layer, "visible", True)),
                    "opacity": float(getattr(layer, "opacity", 1.0)),
                }

                # napari Image layer properties (safe if present) 
                try:
                    cl = getattr(layer, "contrast_limits", None)
                    if cl is not None:
                        ch["contrast_limits"] = [float(cl[0]), float(cl[1])]
                except Exception:
                    pass

                try:
                    cm = getattr(layer, "colormap", None)
                    # colormap can be an object; str() is at least JSON-serializable
                    if cm is not None:
                        ch["colormap"] = str(cm)
                except Exception:
                    pass

                # If you stored widgets per channel, save the actual slider values too
                if hasattr(self, "channel_controls") and i - 1 < len(self.channel_controls):
                    ctrl = self.channel_controls[i - 1]
                    try:
                        ch["ui"] = {
                            "opacity_slider": ctrl["opacity_slider"].value() / 100.0,
                            "min": ctrl["min_slider"].value(),
                            "max": ctrl["max_slider"].value(),
                            "brightness": ctrl["brightness_slider"].value(),
                            "contrast_percent": ctrl["contrast_slider"].value(),
                            "visible_checkbox": bool(ctrl["visible_cb"].isChecked()),
                        }
                    except Exception:
                        pass

                channels_meta.append(ch)

            metadata = {
                "original_file": str(original_filename),
                "timestamp": datetime.now().isoformat(),  # JSON-friendly 
                "current_z_slice": int(getattr(self, "current_z", 0)),
                "total_z_slices": int(getattr(self, "n_z", 1)),
                "channels": channels_meta,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            QMessageBox.information(self, "Success", f"Metadata saved to:\n{metadata_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save metadata: {str(e)}")
    
    def load_metadata(self):
        """Load metadata JSON from meta_data/<thisfile>.json and apply to UI/layers."""
        try:
            # Match save_output(): metadata filename is based on self.file_path
            srcp = Path(self.file_path)
            meta_dir = Path("meta_data")
            metadata_path = meta_dir / (srcp.stem + ".json")  # same stem logic as save_output 

            if not metadata_path.exists():
                QMessageBox.critical(self, "Not Found", f"No metadata found.\nExpected: {metadata_path}")
                return

            with open(metadata_path, "r") as f:
                md = json.load(f)

            # ---------- Restore per-channel ----------
            channels = md.get("channels", [])
            for ch in channels:
                try:
                    idx1 = int(ch.get("channel_index", -1))   # 1-based in your JSON
                    i = idx1 - 1                              # 0-based
                    if i < 0:
                        continue

                    lyr = None
                    if hasattr(self, "channel_layers") and i < len(self.channel_layers):
                        lyr = self.channel_layers[i]

                    # Layer properties
                    if lyr is not None:
                        if "visible" in ch:
                            try:
                                lyr.visible = bool(ch["visible"])
                            except Exception:
                                pass

                        if "opacity" in ch:
                            try:
                                lyr.opacity = float(ch["opacity"])
                            except Exception:
                                pass

                        if "contrast_limits" in ch:
                            try:
                                cl0, cl1 = ch["contrast_limits"]
                                lyr.contrast_limits = (float(cl0), float(cl1))
                            except Exception:
                                pass

                        # Only set if it's a simple usable string (your saver uses str(cm))
                        if "colormap" in ch:
                            cm = ch.get("colormap", None)
                            if isinstance(cm, str) and cm:
                                try:
                                    lyr.colormap = cm
                                except Exception:
                                    pass

                    # UI controls (only if you created/stored self.channel_controls)
                    ui = ch.get("ui", None)
                    if ui and hasattr(self, "channel_controls") and i < len(self.channel_controls):
                        ctrl = self.channel_controls[i]

                        def _set(widget, value):
                            widget.blockSignals(True)
                            try:
                                # checkbox vs slider
                                if hasattr(widget, "setChecked"):
                                    widget.setChecked(bool(value))
                                else:
                                    widget.setValue(int(value))
                            finally:
                                widget.blockSignals(False)

                        if "visible_cb" in ctrl and "visible_checkbox" in ui:
                            _set(ctrl["visible_cb"], ui["visible_checkbox"])

                        if "opacity_slider" in ctrl and "opacity_slider" in ui:
                            # saved as 0..1, slider expects 0..100
                            _set(ctrl["opacity_slider"], int(round(float(ui["opacity_slider"]) * 100.0)))

                        if "min_slider" in ctrl and "min" in ui:
                            _set(ctrl["min_slider"], ui["min"])
                        if "max_slider" in ctrl and "max" in ui:
                            _set(ctrl["max_slider"], ui["max"])
                        if "brightness_slider" in ctrl and "brightness" in ui:
                            _set(ctrl["brightness_slider"], ui["brightness"])
                        if "contrast_slider" in ctrl and "contrast_percent" in ui:
                            _set(ctrl["contrast_slider"], ui["contrast_percent"])

                except Exception:
                    pass

            # ---------- Restore Z ----------
            if "current_z_slice" in md and hasattr(self, "z_slider"):
                try:
                    z = int(md.get("current_z_slice", 0))
                    z = max(0, min(z, max(self.n_z - 1, 0)))
                    self.z_slider.setValue(z)  # should call z_changed
                except Exception:
                    pass

            # Force refresh (helpful if some signals were blocked)
            try:
                self.z_changed(self.z_slider.value())
            except Exception:
                pass

            QMessageBox.information(self, "Loaded", f"Metadata loaded successfully from:\n{metadata_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load metadata:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def open_cell_counter(self):
        """
        Single-image-layer Cell Counter with two detection backends:
        - Classic (threshold/cleanup/watershed) for mask generation
        - ML (Cellpose train/load/infer), optional

        Intended workflow:
        A) Manual detection + mask generation (Classic + label editing)
        B) ML (train/infer) using the curated labels mask
        C) Analysis (measure intensities, screenshots)
        """
        try:
            layer_names = [lyr.name for lyr in self.viewer.layers]
            if not layer_names:
                QMessageBox.warning(self, "No layers", "No layers found in viewer.")
                return

            # --- dialog: select exactly ONE layer ---
            dlg = QDialog(self)
            dlg.setWindowTitle("Cell counter: select layer")
            dlg.setMinimumWidth(420)
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("Select ONE layer to use for cell detection + intensity measurement:"))

            lst = QListWidget()
            lst.setSelectionMode(QAbstractItemView.SingleSelection)
            for nm in layer_names:
                it = QListWidgetItem(nm)
                lst.addItem(it)
                if lst.currentItem() is None and nm.startswith("Channel ") and nm[8:].isdigit():
                    lst.setCurrentItem(it)
            v.addWidget(lst)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            v.addWidget(buttons)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() != QDialog.Accepted:
                return

            item = lst.currentItem()
            if item is None:
                QMessageBox.information(self, "Cell Counter", "No layer selected.")
                return

            selected_name = item.text()

            # Grab the actual layer object
            try:
                src_layer = self.viewer.layers[selected_name]
            except Exception:
                src_layer = None

            # Resolve 3D stack
            s3 = self._resolve_stack3d(selected_name)
            if s3 is None:
                QMessageBox.warning(self, "Missing data", f"Could not resolve stack for: {selected_name}")
                return

            vol = np.asarray(s3, dtype=np.float32)
            if vol.ndim != 3:
                QMessageBox.warning(self, "Unsupported", f"{selected_name} is not 3D (shape {vol.shape}).")
                return

            # Pull current colormap from main layer
            cmap_arg = "gray"
            if src_layer is not None:
                cm = getattr(src_layer, "colormap", None)
                if isinstance(cm, str) and cm:
                    cmap_arg = cm
                elif isinstance(cm, tuple) and len(cm) >= 2 and isinstance(cm[0], str):
                    cmap_arg = (cm[0], cm[1])

            # --- create cell-counter viewer ---
            cc = napari.Viewer()
            cc.title = "Cell Counter"

            detection_layer_name = f"Detection input - {selected_name}"
            det_layer = cc.add_image(
                vol,
                name=detection_layer_name,
                colormap=cmap_arg,
                blending="additive",
                opacity=1.0,
            )

            # Scale bar
            cc.scale_bar.visible = True
            cc.scale_bar.unit = "um"
            cc.scale_bar.font_size = 12
            cc.scale_bar.color = "white"
            cc.scale_bar.box = True
            cc.scale_bar.position = "bottom_right"

            # Shapes (3D)
            pos_shapes = cc.add_shapes(
                name="Positive ROIs (seed somas)",
                opacity=0.7,
                ndim=3,
                edge_color="lime",
                face_color="transparent",
            )
            neg_shapes = cc.add_shapes(
                name="Negative ROIs (exclude junk)",
                opacity=0.7,
                ndim=3,
                edge_color="red",
                face_color="transparent",
            )

            # Labels
            labels_layer = cc.add_labels(np.zeros(vol.shape, dtype=np.int32), name="Detected cells (labels mask)")
            labels_layer.visible = True

            # Store refs (add ML fields)
            self._cell_counter_state = dict(
                viewer=cc,
                vol=vol,
                detection_layer=det_layer,
                source_main_layer=src_layer,
                pos_shapes=pos_shapes,
                neg_shapes=neg_shapes,
                labels=labels_layer,
                # ML state:
                ml_backend="none",          # "none" | "cellpose" | "stardist" | "custom"
                ml_model_path="",           # where weights/config live
                ml_device=self._torch_best_device(),
                status_label=None,
            )

            # Promote accidentally-2D shapes to 3D (slice-locked)
            def _promote_shapes_to_3d(layer, event=None):
                try:
                    z_now = int(cc.dims.point[0]) if hasattr(cc, "dims") and len(cc.dims.point) > 0 else 0
                except Exception:
                    z_now = 0

                changed = False
                new_data = []
                for poly in layer.data:
                    pts = np.asarray(poly) if poly is not None else None
                    if pts is None or pts.ndim != 2 or pts.shape[0] < 3:
                        new_data.append(poly)
                        continue
                    if pts.shape[1] == 2:
                        pts3 = np.c_[np.full(len(pts), z_now), pts]
                        new_data.append(pts3)
                        changed = True
                    else:
                        new_data.append(pts)

                if changed:
                    layer.data = new_data

            try:
                pos_shapes.events.data.connect(lambda e: _promote_shapes_to_3d(pos_shapes, e))
            except Exception:
                pass
            try:
                neg_shapes.events.data.connect(lambda e: _promote_shapes_to_3d(neg_shapes, e))
            except Exception:
                pass

            # ---------------- Dock controls (reorganized) ----------------
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(6)


            # Status line (top)
            status = QLabel(f"Device: {self._cell_counter_state['ml_device']} | Backend: (none)")
            layout.addWidget(status)
            self._cell_counter_state["status_label"] = status

            # ===== Section A: Manual detection / mask generation =====
            layout.addWidget(QLabel("Manual detection + mask generation (Classic)"))

            layout.addWidget(QLabel("Step 1: Draw GREEN ROIs on somas (seeds) and RED ROIs on junk to exclude."))
            layout.addWidget(QLabel("Step 2: Click 'Generate/Update labels mask (Classic)'."))
            layout.addWidget(QLabel("Step 3: Manually edit 'Detected cells (labels mask)' if needed (this mask trains ML)."))

            # Threshold
            th_row = QHBoxLayout()
            th_row.addWidget(QLabel("Threshold (0–1)"))
            th = QSlider(Qt.Horizontal)
            th.setRange(0, 100)
            th.setValue(50)
            th_row.addWidget(th)
            th_val = QLineEdit("0.50")
            th_val.setFixedWidth(70)
            th_row.addWidget(th_val)
            layout.addLayout(th_row)

            def _sync_th(v=None):
                if v is None:
                    v = th.value()
                th_val.setText(f"{float(v) / 100.0:.2f}")

            th.valueChanged.connect(_sync_th)

            # Min voxels
            minvox_row = QHBoxLayout()
            minvox_row.addWidget(QLabel("Min voxels"))
            minvox = QSpinBox()
            minvox.setRange(1, 10_000_000)
            minvox.setValue(200)
            minvox_row.addWidget(minvox)
            layout.addLayout(minvox_row)

            # Max voxels
            maxvox_row = QHBoxLayout()
            maxvox_row.addWidget(QLabel("Max voxels (0=off)"))
            maxvox = QSpinBox()
            maxvox.setRange(0, 10_000_000)
            maxvox.setValue(0)
            maxvox_row.addWidget(maxvox)
            layout.addLayout(maxvox_row)

            # Compactness
            comp_row = QHBoxLayout()
            comp_row.addWidget(QLabel("Watershed compactness"))
            comp = QDoubleSpinBox()
            comp.setDecimals(2)
            comp.setRange(0.0, 50.0)
            comp.setValue(0.0)
            comp_row.addWidget(comp)
            layout.addLayout(comp_row)

            # Neg ROI grow
            neggrow_row = QHBoxLayout()
            neggrow_row.addWidget(QLabel("Neg ROI grow (px)"))
            neg_grow = QSpinBox()
            neg_grow.setRange(0, 50)
            neg_grow.setValue(2)
            neggrow_row.addWidget(neg_grow)
            layout.addLayout(neggrow_row)

            # Smoothing
            gs_row = QHBoxLayout()
            gs_row.addWidget(QLabel("Gaussian sigma (0=off)"))
            gs = QDoubleSpinBox()
            gs.setDecimals(2)
            gs.setRange(0.0, 10.0)
            gs.setValue(1.0)
            gs_row.addWidget(gs)
            layout.addLayout(gs_row)

            # Soma opening
            open_row = QHBoxLayout()
            open_row.addWidget(QLabel("Soma opening radius (px)"))
            open_r = QSpinBox()
            open_r.setRange(0, 10)
            open_r.setValue(1)
            open_row.addWidget(open_r)
            layout.addLayout(open_row)

            # Auto-split
            auto_split = QCheckBox("Auto-split touching cells (distance peaks)")
            auto_split.setChecked(True)
            layout.addWidget(auto_split)

            seed_row = QHBoxLayout()
            seed_row.addWidget(QLabel("Min seed distance (px)"))
            seed_dist = QSpinBox()
            seed_dist.setRange(1, 100)
            seed_dist.setValue(8)
            seed_row.addWidget(seed_dist)
            layout.addLayout(seed_row)

            # Manual/classic run button (renamed to clarify purpose)
            classic_btn = QPushButton("Generate/Update labels mask (Classic)")
            layout.addWidget(classic_btn)

            def _run_classic():
                self._cell_counter_set_status("Backend: classic | Generating labels...")
                self._cell_counter_run_detection(
                    threshold=float(th.value()) / 100.0,
                    min_voxels=int(minvox.value()),
                    max_voxels=int(maxvox.value()),
                    compactness=float(comp.value()),
                    neg_grow=int(neg_grow.value()),
                    gaussian_sigma=float(gs.value()),
                    open_radius=int(open_r.value()),
                    auto_split=bool(auto_split.isChecked()),
                    min_seed_distance=int(seed_dist.value()),
                )
                # _cell_counter_run_detection already shows a QMessageBox; keep status updated anyway
                self._cell_counter_set_status("Backend: classic | Labels updated")

            classic_btn.clicked.connect(lambda _=False: _run_classic())

            # ===== Section B: ML (Cellpose) =====
            layout.addWidget(QLabel("ML model (Cellpose): train / load / infer (uses labels mask)"))
            layout.addWidget(QLabel("Train uses the current 'Detected cells (labels mask)' (0=bg, 1..N instances)."))

            ml_backend_row = QHBoxLayout()
            ml_backend_row.addWidget(QLabel("ML backend"))
            ml_backend = QComboBox()
            ml_backend.addItems(["Cellpose (recommended)"])  # keep focused until other backends implemented
            ml_backend_row.addWidget(ml_backend)
            layout.addLayout(ml_backend_row)

            device_row = QHBoxLayout()
            device_row.addWidget(QLabel("Compute device"))
            device = QComboBox()
            device.addItems(["auto", "cuda", "mps", "cpu"])
            device.setCurrentIndex(0)
            device_row.addWidget(device)
            layout.addLayout(device_row)

            model_path_row = QHBoxLayout()
            model_path_row.addWidget(QLabel("Model path"))
            model_path = QLineEdit("")
            model_path_row.addWidget(model_path)
            browse_btn = QPushButton("Browse")
            model_path_row.addWidget(browse_btn)
            layout.addLayout(model_path_row)

            def _browse_model():
                # NOTE: using qt_viewer here matches your existing code; you can swap to cc.window._qt_window later.
                pth = QFileDialog.getExistingDirectory(cc.window.qt_viewer, "Select model folder")
                if pth:
                    model_path.setText(pth)

            browse_btn.clicked.connect(_browse_model)

            # ---- NEW: Model mode (pretrained cpsam vs custom weights) ----
            ml_mode_row = QHBoxLayout()
            ml_mode_row.addWidget(QLabel("Model mode"))

            ml_mode = QComboBox()
            ml_mode.addItems([
                "Pretrained (cpsam) – no training data",
                "Custom weights (from folder/file) – train or load",
            ])
            ml_mode.setCurrentIndex(0)

            ml_mode_row.addWidget(ml_mode)
            layout.addLayout(ml_mode_row)

            self._cell_counter_state["ml_mode_combo"] = ml_mode
            # -------------------------------------------------------------

            load_btn = QPushButton("Load model / weights")
            train_btn = QPushButton("Train Cellpose from labels mask (manual-first)")
            infer_btn = QPushButton("Run Cellpose inference → write labels mask")
            save_btn = QPushButton("Save model / weights")

            layout.addWidget(load_btn)
            layout.addWidget(train_btn)
            layout.addWidget(infer_btn)
            layout.addWidget(save_btn)

            # --- ML progress UI (dock-local, in addition to napari's global progress) ---
            ml_prog_label = QLabel("ML progress: idle")
            ml_prog_bar = QProgressBar()
            ml_prog_bar.setRange(0, 100)
            ml_prog_bar.setValue(0)
            ml_prog_bar.setTextVisible(True)

            layout.addWidget(ml_prog_label)
            layout.addWidget(ml_prog_bar)

            cancel_btn = QPushButton("Cancel training")
            layout.addWidget(cancel_btn)
            cancel_btn.clicked.connect(lambda _=False: self._cell_counter_ml_cancel())

            self._cell_counter_state["ml_progress_label"] = ml_prog_label
            self._cell_counter_state["ml_progress_bar"] = ml_prog_bar
            self._cell_counter_state["ml_worker"] = None
            self._cell_counter_state["ml_cancel_flag"] = False

            # --- UPDATED: Load should support both modes ---
            # If mode=Pretrained -> just set cpsam and clear any custom file
            # If mode=Custom -> try to resolve weights file now (prompt if needed) and store it
            def _ml_load_clicked():
                st = getattr(self, "_cell_counter_state", None)
                if not st:
                    return
                mode_txt = ""
                try:
                    mode_txt = st["ml_mode_combo"].currentText()
                except Exception:
                    mode_txt = ""

                mp = str(model_path.text()).strip()

                if "Pretrained (cpsam)" in (mode_txt or ""):
                    st["ml_last_trained_model_file"] = ""
                    st["ml_model_path"] = mp
                    self._cell_counter_set_status(f"Device: {self._torch_best_device()} | Backend: cellpose | Weights: cpsam")
                    self._ml_ui_set_progress("ML progress: using pretrained cpsam", 0)
                    return

                # Custom mode: prompt/resolve a weights file now
                w = self._cell_counter_resolve_infer_weights(model_path=mp)
                st["ml_model_path"] = mp
                st["ml_last_trained_model_file"] = w if (w and w != "cpsam") else st.get("ml_last_trained_model_file", "")
                self._cell_counter_set_status(f"Device: {self._torch_best_device()} | Backend: cellpose | Weights: {os.path.basename(w) if os.path.exists(w) else w}")
                self._ml_ui_set_progress("ML progress: custom weights loaded", 0)

            load_btn.clicked.connect(lambda _=False: _ml_load_clicked())

            train_btn.clicked.connect(lambda _=False: self._cell_counter_ml_train_threaded(
                backend=str(ml_backend.currentText()),
                device=str(device.currentText()),
                model_path=str(model_path.text()).strip(),
            ))

            infer_btn.clicked.connect(lambda _=False: self._cell_counter_ml_infer(
                backend=str(ml_backend.currentText()),
                device=str(device.currentText()),
                model_path=str(model_path.text()).strip(),
            ))

            save_btn.clicked.connect(lambda _=False: self._cell_counter_ml_save(
                backend=str(ml_backend.currentText()),
                device=str(device.currentText()),
                model_path=str(model_path.text()).strip(),
            ))


            ml_mode_row = QHBoxLayout()
            ml_mode_row.addWidget(QLabel("Model mode"))

            ml_mode = QComboBox()
            ml_mode.addItems([
                "Pretrained (cpsam) – no training data",
                "Custom weights (from folder/file) – train or load",
            ])
            ml_mode.setCurrentIndex(0)

            ml_mode_row.addWidget(ml_mode)
            layout.addLayout(ml_mode_row)

            self._cell_counter_state["ml_mode_combo"] = ml_mode

            # ===== Section C: Analysis / export =====
            layout.addWidget(QLabel("Analysis / export"))

            measure_btn = QPushButton("Measure intensities per cell (min/max/mean; exclude zeros)")
            layout.addWidget(measure_btn)

            ss_btn = QPushButton("Save Screenshot")
            ss_btn.clicked.connect(
                lambda _=False: self.save_screenshot(
                    viewer=cc, parent=self, default_name="cell_counter.png", canvas_only=True
                )
            )
            layout.addWidget(ss_btn)

            measure_btn.clicked.connect(lambda _=False: self.calculate_intensities_per_cell())

            layout.addStretch(1)
            container.setLayout(layout)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)

            cc.window.add_dock_widget(scroll, area="right")
            return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open cell counter:\n{e}")
            
    def _cell_counter_resolve_infer_weights(self, model_path: str = "") -> str:
        """
        Returns a string to pass to CellposeModel(pretrained_model=...).
        Either "cpsam" or a path to a trained weights file.
        """
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return "cpsam"

        mode_combo = st.get("ml_mode_combo", None)
        mode_txt = mode_combo.currentText() if mode_combo is not None else ""

        # Mode A: always use cpsam
        if "Pretrained (cpsam)" in (mode_txt or ""):
            return "cpsam"

        # Mode B: custom weights: try last trained file, then meta.json, else ask user
        model_path = (model_path or st.get("ml_model_path", "") or "").strip()
        model_file = (st.get("ml_last_trained_model_file", "") or "").strip()

        meta_file = os.path.join(model_path, "cellpose_model_meta.json") if model_path else ""
        if (not model_file) and meta_file and os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                model_file = (meta.get("model_out_path", "") or "").strip()
            except Exception:
                model_file = ""

        if model_file and os.path.exists(model_file):
            return model_file

        # Fall back: prompt user to pick weights file
        cc = st.get("viewer", None)
        parent = cc.window._qt_window if (cc is not None and hasattr(cc.window, "_qt_window")) else self
        picked, _ = QFileDialog.getOpenFileName(
            parent,
            "Select Cellpose model weights file",
            model_path or "",
            "All files (*)",
        )
        if picked:
            st["ml_last_trained_model_file"] = picked
            return picked

        # If user cancels, default back to cpsam (safe)
        return "cpsam"

    def _fmt_hms(self, sec: float) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _cell_counter_ml_cancel(self):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return
        st["ml_cancel_flag"] = True
        # request abort from napari worker too (works only if worker yields/checks abort_requested)
        wk = st.get("ml_worker", None)
        if wk is not None:
            try:
                wk.quit()
            except Exception:
                pass
        self._ml_ui_set_progress("ML progress: cancel requested...", None)

    def _ml_ui_set_progress(self, text: str = None, pct: int = None):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return
        lab = st.get("ml_progress_label", None)
        bar = st.get("ml_progress_bar", None)
        if text is not None and lab is not None:
            try:
                lab.setText(text)
            except Exception:
                pass
        if pct is not None and bar is not None:
            try:
                bar.setValue(int(np.clip(pct, 0, 100)))
            except Exception:
                pass
    
    def _cell_counter_ml_train_threaded(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            QMessageBox.warning(self, "ML", "No cell-counter session found.")
            return

        if st.get("ml_worker", None) is not None:
            QMessageBox.information(self, "ML", "Training is already running.")
            return

        model_path = (model_path or "").strip()
        if not model_path:
            cc = st.get("viewer", None)
            parent = cc.window._qt_window if hasattr(cc.window, "_qt_window") else None  # avoids qt_viewer deprecation
            QMessageBox.warning(parent or self, "ML", "Please choose a Model path before training.")
            return

        # Collect hyperparameters on UI thread (mac-safe)
        cc = st["viewer"]
        model_name, ok = QInputDialog.getText(
            cc.window._qt_window if hasattr(cc.window, "_qt_window") else None,
            "Cellpose model name",
            "Enter a name for the trained model:",
            text=f"cellpose_custom_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        if not ok:
            return
        model_name = (model_name or "").strip() or f"cellpose_custom_{time.strftime('%Y%m%d_%H%M%S')}"

        n_epochs, ok = QInputDialog.getInt(
            cc.window._qt_window if hasattr(cc.window, "_qt_window") else None,
            "Epochs",
            "Number of training epochs:",
            value=50, min=1, max=5000
        )
        if not ok:
            return

        lr, ok = QInputDialog.getDouble(
            cc.window._qt_window if hasattr(cc.window, "_qt_window") else None,
            "Learning rate",
            "Learning rate:",
            value=1e-5, min=1e-7, max=1e-2, decimals=8
        )
        if not ok:
            return

        wd, ok = QInputDialog.getDouble(
            cc.window._qt_window if hasattr(cc.window, "_qt_window") else None,
            "Weight decay",
            "Weight decay:",
            value=0.1, min=0.0, max=10.0, decimals=6
        )
        if not ok:
            return

        st["ml_cancel_flag"] = False
        self._ml_ui_set_progress("ML progress: starting...", 0)
        self._cell_counter_set_status("Backend: cellpose | Starting training...")

        worker = self._cellpose_train_worker(
            backend=backend,
            device=device,
            model_path=model_path,
            model_name=model_name,
            n_epochs=int(n_epochs),
            lr=float(lr),
            wd=float(wd),
            start_model="cpsam",
        )

        worker.yielded.connect(self._on_cellpose_train_yielded)
        worker.returned.connect(self._on_cellpose_train_returned)
        worker.errored.connect(self._on_cellpose_train_errored)
        worker.finished.connect(self._on_cellpose_train_finished)

        st["ml_worker"] = worker
        worker.start()


    @thread_worker(progress={"desc": "Cellpose training"})  # omit total => indeterminate is fine
    def _cellpose_train_worker(
        self,
        backend="Cellpose (recommended)",
        device="auto",
        model_path="",
        model_name="cellpose_custom",
        n_epochs=50,
        lr=1e-5,
        wd=0.1,
        start_model="cpsam",
    ):
        t0 = time.time()
        yield {"pct": 0, "msg": "Validating session..."}

        st = getattr(self, "_cell_counter_state", None)
        if not st:
            raise RuntimeError("No cell-counter session found.")
        if "Cellpose" not in (backend or ""):
            raise RuntimeError("Only Cellpose backend is implemented right now.")

        model_path = (model_path or "").strip()
        if not model_path:
            raise RuntimeError("Model path is empty. Choose a model folder first.")
        os.makedirs(model_path, exist_ok=True)

        # Device
        dev = self._torch_best_device() if (device == "auto") else self._cellpose_device_from_ui(device)

        # Pull data
        vol = np.asarray(st["vol"], dtype=np.float32)  # (Z,Y,X)
        labels_layer = st["labels"]
        lab3d = np.asarray(labels_layer.data, dtype=np.int32)

        # Manual-first: require labels to exist (recommended)
        if int(lab3d.max()) <= 0:
            raise RuntimeError("No labels available. Generate/Edit 'Detected cells (labels mask)' first, then train.")

        yield {"pct": 10, "msg": "Building 2D training set..."}

        train_imgs, train_masks = self._cell_counter_build_cellpose_training_set(
            vol3d=vol,
            labels3d=lab3d,
            max_slices=64,
            min_masks_per_slice=1,
        )
        if not train_imgs:
            raise RuntimeError("No labeled Z-slices found to train on.")

        # Early cancel point (works)
        if getattr(self, "_cell_counter_state", {}).get("ml_cancel_flag", False):
            raise RuntimeError("Training cancelled before model init.")

        yield {"pct": 20, "msg": f"Init Cellpose model ({start_model}) on {dev}..."}

        try:
            cellpose_io.logger_setup()
        except Exception:
            pass

        try:
            import torch
            device_obj = torch.device(dev)
        except Exception:
            device_obj = None

        gpu_flag = (dev == "cuda" or dev == "mps")
        model = models.CellposeModel(gpu=gpu_flag, device=device_obj, pretrained_model=str(start_model))

        yield {"pct": 30, "msg": f"Training... elapsed {self._fmt_hms(time.time()-t0)} (epochs={int(n_epochs)})"}

        # After this point we cannot cancel cleanly until train_seg returns.
        # IMPORTANT CHANGE: force saving into model_path via save_path (supported by Cellpose train API).
        model_out_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=train_imgs,
            train_labels=train_masks,
            test_data=None,
            test_labels=None,
            weight_decay=float(wd),
            learning_rate=float(lr),
            n_epochs=int(n_epochs),
            model_name=str(model_name),
            save_path=str(model_path),
        )

        meta = dict(
            backend="cellpose",
            pretrained_start=str(start_model),
            device=str(dev),
            model_name=str(model_name),
            model_out_path=str(model_out_path),
            n_train_images=len(train_imgs),
            n_epochs=int(n_epochs),
            learning_rate=float(lr),
            weight_decay=float(wd),
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        with open(os.path.join(model_path, "cellpose_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        yield {"pct": 100, "msg": f"Done in {self._fmt_hms(time.time()-t0)} | saved: {model_out_path}"}
        return dict(
            model_out_path=str(model_out_path),
            model_name=str(model_name),
            meta_path=os.path.join(model_path, "cellpose_model_meta.json"),
        )


    def _on_cellpose_train_yielded(self, payload):
        # payload is dict: {"pct": int, "msg": str}
        try:
            msg = str(payload.get("msg", ""))
            pct = payload.get("pct", None)
        except Exception:
            msg, pct = str(payload), None

        if msg:
            print(f"[Cellpose] {msg}", flush=True)
            self._cell_counter_set_status(msg)
            self._ml_ui_set_progress("ML progress: ", None)

        if pct is not None:
            self._ml_ui_set_progress(None, int(pct))

    def _on_cellpose_train_returned(self, result):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return
        try:
            st["ml_last_trained_model_file"] = result.get("model_out_path", "")
            st["ml_model_path"] = os.path.dirname(result.get("meta_path", "")) if result.get("meta_path", "") else st.get("ml_model_path", "")
            st["ml_backend"] = "cellpose"
        except Exception:
            pass

        self._cell_counter_set_status("Backend: cellpose | Training complete")
        self._ml_ui_set_progress("ML progress: complete", 100)

        try:
            cc = st["viewer"]
            QMessageBox.information(
                cc.window.qt_viewer,
                "ML",
                f"Training complete.\nModel: {st.get('ml_last_trained_model_file','')}",
            )
        except Exception:
            pass

    def _on_cellpose_train_errored(self, err):
        self._cell_counter_set_status("Backend: cellpose | Training failed")
        self._ml_ui_set_progress(f"ML progress: error: {err}", 0)

        st = getattr(self, "_cell_counter_state", None)
        try:
            cc = st["viewer"] if st else None
            parent = cc.window.qt_viewer if cc is not None else self
            QMessageBox.critical(parent, "ML", f"Cellpose training failed:\n{err}")
        except Exception:
            pass

    def _on_cellpose_train_finished(self):
        st = getattr(self, "_cell_counter_state", None)
        if st:
            st["ml_worker"] = None


    def _cell_counter_ml_load(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return
        if device == "auto":
            device = self._torch_best_device()
        st["ml_device"] = device
        st["ml_model_path"] = model_path
        st["ml_backend"] = "cellpose" if "Cellpose" in backend else "stardist" if "StarDist" in backend else "custom"
        self._cell_counter_set_status(f"Device: {device} | Backend: {st['ml_backend']} | Model: {model_path or '(none)'}")

    def _cell_counter_ml_save(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        QMessageBox.information(self, "ML", "Save is wired, but training/inference implementation is not added yet.")

    def _cell_counter_ml_train(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        QMessageBox.information(self, "ML", "Train is wired, but training implementation is not added yet.")

    def _cell_counter_ml_infer(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                QMessageBox.warning(self, "ML", "No cell-counter session found.")
                return

            if "Cellpose" not in (backend or ""):
                QMessageBox.warning(self, "ML", "Only Cellpose backend is implemented right now.")
                return

            cc = st["viewer"]
            vol = np.asarray(st["vol"], dtype=np.float32)  # (Z,Y,X)
            labels_layer = st["labels"]

            dev = self._cellpose_device_from_ui(device)
            st["ml_device"] = dev
            st["ml_backend"] = "cellpose"

            if not model_path:
                model_path = st.get("ml_model_path", "") or ""
            if model_path:
                st["ml_model_path"] = model_path

            # IMPORTANT CHANGE: resolve weights based on Model mode
            weights = self._cell_counter_resolve_infer_weights(model_path=model_path)

            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Infer ({'cpsam' if weights=='cpsam' else 'custom'})")

            # Build model
            try:
                import torch
                device_obj = torch.device(dev)
            except Exception:
                device_obj = None
            gpu_flag = (dev == "cuda" or dev == "mps")

            model = models.CellposeModel(gpu=gpu_flag, device=device_obj, pretrained_model=weights)

            # Ask for key eval params
            diameter, ok = QInputDialog.getDouble(
                cc.window.qt_viewer,
                "Cellpose diameter",
                "Approx cell diameter in pixels (0 = auto):",
                value=0.0,
                min=0.0,
                max=10000.0,
                decimals=2,
            )
            if not ok:
                return

            # 2D-per-slice inference
            Z = vol.shape[0]
            out = np.zeros_like(vol, dtype=np.int32)
            next_id = 0

            channels = [0, 0]

            for z in range(Z):
                img2 = np.asarray(vol[z], dtype=np.float32)

                masks, flows, styles, diams = model.eval(
                    img2,
                    channels=channels,
                    diameter=(None if float(diameter) <= 0 else float(diameter)),
                    do_3D=False,
                )

                masks = np.asarray(masks, dtype=np.int32)
                if masks.size == 0 or int(masks.max()) == 0:
                    continue

                unique_ids = np.unique(masks)
                unique_ids = unique_ids[unique_ids != 0]
                if unique_ids.size:
                    remap = {int(uid): (next_id + i + 1) for i, uid in enumerate(unique_ids)}
                    next_id += int(unique_ids.size)
                    m2 = np.zeros_like(masks, dtype=np.int32)
                    for uid, nid in remap.items():
                        m2[masks == uid] = nid
                    out[z] = m2

            labels_layer.data = out
            n_cells = int(out.max())

            # Track last-used weights (for custom mode, this is your selected file; for cpsam it's just "cpsam")
            st["ml_last_trained_model_file"] = weights

            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Done ({n_cells} labels)")
            QMessageBox.information(cc.window.qt_viewer, "ML", f"Cellpose inference complete.\nDetected (slice-stacked) objects: {n_cells}")

        except Exception as e:
            try:
                self._cell_counter_set_status("ML inference failed")
            except Exception:
                pass
            QMessageBox.critical(self, "Error", f"Cellpose inference failed:\n{e}")

    def _torch_best_device(self):
        """Return a torch.device string: 'cuda' | 'mps' | 'cpu'."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if mps_ok:
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"
    
    def _cellpose_device_from_ui(self, device_choice: str):
        """
        Map UI 'auto|cuda|mps|cpu' to a torch.device-like string for Cellpose.
        Cellpose accepts either gpu=True or an explicit device in newer APIs;
        we keep it simple and pass a device string into CellposeModel where possible.
        """
        dc = (device_choice or "auto").strip().lower()
        if dc == "auto":
            return self._torch_best_device()
        if dc in ("cuda", "mps", "cpu"):
            return dc
        return self._torch_best_device()
    
    def _cell_counter_build_cellpose_training_set(self, vol3d: np.ndarray, labels3d: np.ndarray,
                                             max_slices: int = 64,
                                             min_masks_per_slice: int = 1):
        """
        Build 2D training data for Cellpose from a 3D stack.
        Returns (train_images, train_masks) where each is a list of 2D arrays.

        Strategy:
        - Use Z slices where labels3d has at least `min_masks_per_slice` instances.
        - Subsample up to `max_slices` slices to keep training time reasonable.
        """
        assert vol3d.ndim == 3 and labels3d.ndim == 3
        Z = vol3d.shape[0]

        slice_ids = []
        for z in range(Z):
            lab2 = labels3d[z]
            n_inst = int(lab2.max())
            if n_inst >= int(min_masks_per_slice):
                slice_ids.append(z)

        if not slice_ids:
            return [], []

        # Subsample evenly if too many
        if len(slice_ids) > int(max_slices):
            idx = np.linspace(0, len(slice_ids) - 1, int(max_slices)).round().astype(int)
            slice_ids = [slice_ids[i] for i in idx]

        imgs = []
        masks = []
        for z in slice_ids:
            img2 = np.asarray(vol3d[z], dtype=np.float32)
            m2 = np.asarray(labels3d[z], dtype=np.int32)
            imgs.append(img2)
            masks.append(m2)

        return imgs, masks

    def _cell_counter_ml_train(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                QMessageBox.warning(self, "ML", "No cell-counter session found.")
                return

            if "Cellpose" not in (backend or ""):
                QMessageBox.warning(self, "ML", "Only Cellpose backend is implemented right now.")
                return

            cc = st["viewer"]
            vol = np.asarray(st["vol"], dtype=np.float32)  # (Z,Y,X)
            labels_layer = st["labels"]

            # Choose / create model output directory
            if not model_path:
                model_path = QFileDialog.getExistingDirectory(cc.window.qt_viewer, "Select folder to save model")
                if not model_path:
                    return
            os.makedirs(model_path, exist_ok=True)

            # Determine device
            dev = self._cellpose_device_from_ui(device)
            st["ml_device"] = dev
            st["ml_backend"] = "cellpose"
            st["ml_model_path"] = model_path
            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Training...")

            # Ensure we have labels to train on: use existing labels if present,
            # otherwise generate pseudo-labels via classic detection first.
            lab3d = np.asarray(labels_layer.data, dtype=np.int32)
            if int(lab3d.max()) <= 0:
                # Run classic detection with conservative defaults (you can tune these)
                self._cell_counter_run_detection(
                    threshold=0.50,
                    min_voxels=200,
                    max_voxels=0,
                    compactness=0.0,
                    neg_grow=2,
                    gaussian_sigma=1.0,
                    open_radius=1,
                    auto_split=True,
                    min_seed_distance=8,
                )
                lab3d = np.asarray(labels_layer.data, dtype=np.int32)

            if int(lab3d.max()) <= 0:
                QMessageBox.warning(cc.window.qt_viewer, "ML", "No labels available for training (Detected cells is empty).")
                self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Training failed")
                return

            # Build 2D training set from labeled slices
            train_imgs, train_masks = self._cell_counter_build_cellpose_training_set(
                vol3d=vol,
                labels3d=lab3d,
                max_slices=64,
                min_masks_per_slice=1,
            )
            if not train_imgs:
                QMessageBox.warning(cc.window.qt_viewer, "ML", "No labeled Z-slices found to train on.")
                self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Training failed")
                return

            # Ask for training hyperparameters
            model_name, ok = QInputDialog.getText(
                cc.window.qt_viewer,
                "Cellpose model name",
                "Enter a name for the trained model:",
                text=f"cellpose_custom_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            if not ok:
                return
            model_name = (model_name or "").strip() or f"cellpose_custom_{time.strftime('%Y%m%d_%H%M%S')}"

            n_epochs, ok = QInputDialog.getInt(
                cc.window.qt_viewer, "Epochs", "Number of training epochs:", value=50, min=1, max=5000
            )
            if not ok:
                return

            lr, ok = QInputDialog.getDouble(
                cc.window.qt_viewer, "Learning rate", "Learning rate:", value=1e-5, min=1e-7, max=1e-2, decimals=8
            )
            if not ok:
                return

            wd, ok = QInputDialog.getDouble(
                cc.window.qt_viewer, "Weight decay", "Weight decay:", value=0.1, min=0.0, max=10.0, decimals=6
            )
            if not ok:
                return

            # Cellpose logger setup (optional)
            try:
                cellpose_io.logger_setup()
            except Exception:
                pass

            # Create a Cellpose model starting from built-in pretrained (recommended by Cellpose docs). [page:1]
            # We try to pass an explicit device if available; otherwise rely on gpu flag.
            # Cellpose's internal device assignment supports CUDA and checks MPS availability. [web:479]
            try:
                import torch
                device_obj = torch.device(dev)
            except Exception:
                device_obj = None

            gpu_flag = (dev == "cuda" or dev == "mps")
            model = models.CellposeModel(gpu=gpu_flag, device=device_obj, pretrained_model="cpsam")

            # Train and save model
            # train.train_seg returns model_path and losses per docs. [page:1]
            model_out_path, train_losses, test_losses = train.train_seg(
                model.net,
                train_data=train_imgs,
                train_labels=train_masks,
                test_data=None,
                test_labels=None,
                weight_decay=float(wd),
                learning_rate=float(lr),
                n_epochs=int(n_epochs),
                model_name=str(model_name),
            )

            # Record where it saved + metadata
            meta = dict(
                backend="cellpose",
                pretrained_start="cpsam",
                device=dev,
                model_name=model_name,
                model_out_path=str(model_out_path),
                n_train_images=len(train_imgs),
                n_epochs=int(n_epochs),
                learning_rate=float(lr),
                weight_decay=float(wd),
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(os.path.join(model_path, "cellpose_model_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            st["ml_model_path"] = model_path
            st["ml_backend"] = "cellpose"
            st["ml_last_trained_model_file"] = str(model_out_path)

            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Trained: {model_name}")
            QMessageBox.information(
                cc.window.qt_viewer,
                "ML",
                f"Training complete.\nModel name: {model_name}\nSaved model file:\n{model_out_path}\nMeta in:\n{os.path.join(model_path, 'cellpose_model_meta.json')}",
            )

        except Exception as e:
            try:
                self._cell_counter_set_status("ML training failed")
            except Exception:
                pass
            QMessageBox.critical(self, "Error", f"Cellpose training failed:\n{e}")

    def _cell_counter_ml_infer(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                QMessageBox.warning(self, "ML", "No cell-counter session found.")
                return

            if "Cellpose" not in (backend or ""):
                QMessageBox.warning(self, "ML", "Only Cellpose backend is implemented right now.")
                return

            cc = st["viewer"]
            vol = np.asarray(st["vol"], dtype=np.float32)  # (Z,Y,X)
            labels_layer = st["labels"]

            dev = self._cellpose_device_from_ui(device)
            st["ml_device"] = dev
            st["ml_backend"] = "cellpose"

            if not model_path:
                model_path = st.get("ml_model_path", "") or ""
            if model_path:
                st["ml_model_path"] = model_path

            # Resolve model file
            model_file = st.get("ml_last_trained_model_file", "")

            meta_file = os.path.join(model_path, "cellpose_model_meta.json") if model_path else ""
            if (not model_file) and meta_file and os.path.exists(meta_file):
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    model_file = meta.get("model_out_path", "") or ""
                except Exception:
                    model_file = ""

            if not model_file or not os.path.exists(model_file):
                # Let user pick the trained model file directly
                model_file, _ = QFileDialog.getOpenFileName(
                    cc.window.qt_viewer,
                    "Select Cellpose model file",
                    model_path or "",
                    "All files (*)",
                )
                if not model_file:
                    return

            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Infer...")

            # Build model
            try:
                import torch
                device_obj = torch.device(dev)
            except Exception:
                device_obj = None
            gpu_flag = (dev == "cuda" or dev == "mps")

            model = models.CellposeModel(gpu=gpu_flag, device=device_obj, pretrained_model=model_file)

            # Ask for key eval params
            diameter, ok = QInputDialog.getDouble(
                cc.window.qt_viewer,
                "Cellpose diameter",
                "Approx cell diameter in pixels (0 = auto):",
                value=0.0,
                min=0.0,
                max=10000.0,
                decimals=2,
            )
            if not ok:
                return

            # 2D-per-slice inference
            Z = vol.shape[0]
            out = np.zeros_like(vol, dtype=np.int32)
            next_id = 0

            # Basic channel handling: single-channel grayscale => channels=[0,0]
            # (Cellpose uses channel indices; for grayscale you typically pass [0,0].)
            channels = [0, 0]

            for z in range(Z):
                img2 = np.asarray(vol[z], dtype=np.float32)

                masks, flows, styles, diams = model.eval(
                    img2,
                    channels=channels,
                    diameter=(None if float(diameter) <= 0 else float(diameter)),
                    do_3D=False,
                )

                masks = np.asarray(masks, dtype=np.int32)
                if masks.size == 0 or int(masks.max()) == 0:
                    continue

                # Relabel slice so IDs are unique across Z
                m = masks
                unique_ids = np.unique(m)
                unique_ids = unique_ids[unique_ids != 0]
                if unique_ids.size:
                    remap = {int(uid): (next_id + i + 1) for i, uid in enumerate(unique_ids)}
                    next_id += int(unique_ids.size)
                    m2 = np.zeros_like(m, dtype=np.int32)
                    for uid, nid in remap.items():
                        m2[m == uid] = nid
                    out[z] = m2

            labels_layer.data = out
            n_cells = int(out.max())
            st["ml_last_trained_model_file"] = model_file

            self._cell_counter_set_status(f"Device: {dev} | Backend: cellpose | Done ({n_cells} labels)")
            QMessageBox.information(cc.window.qt_viewer, "ML", f"Cellpose inference complete.\nDetected (slice-stacked) objects: {n_cells}")

        except Exception as e:
            try:
                self._cell_counter_set_status("ML inference failed")
            except Exception:
                pass
            QMessageBox.critical(self, "Error", f"Cellpose inference failed:\n{e}")

    def _cell_counter_ml_save(self, backend="Cellpose (recommended)", device="auto", model_path=""):
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                return

            cc = st["viewer"]
            if "Cellpose" not in (backend or ""):
                QMessageBox.warning(self, "ML", "Only Cellpose backend is implemented right now.")
                return

            model_file = st.get("ml_last_trained_model_file", "")
            if not model_file or not os.path.exists(model_file):
                QMessageBox.information(cc.window.qt_viewer, "ML", "No trained model file found to save. Train first.")
                return

            if not model_path:
                model_path = QFileDialog.getExistingDirectory(cc.window.qt_viewer, "Select folder to save/copy model")
                if not model_path:
                    return
            os.makedirs(model_path, exist_ok=True)

            # Copy model file into folder
            dst = os.path.join(model_path, os.path.basename(model_file))
            if os.path.abspath(dst) != os.path.abspath(model_file):
                shutil.copy2(model_file, dst)

            # Also copy meta if present from current ml_model_path
            src_meta = ""
            if st.get("ml_model_path", ""):
                cand = os.path.join(st["ml_model_path"], "cellpose_model_meta.json")
                if os.path.exists(cand):
                    src_meta = cand
            if src_meta:
                shutil.copy2(src_meta, os.path.join(model_path, "cellpose_model_meta.json"))

            st["ml_model_path"] = model_path
            self._cell_counter_set_status(f"Device: {st.get('ml_device','?')} | Backend: cellpose | Model saved")
            QMessageBox.information(cc.window.qt_viewer, "ML", f"Saved/copied model to:\n{model_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save model failed:\n{e}")

    def _cell_counter_set_status(self, text: str):
        st = getattr(self, "_cell_counter_state", None)
        if not st:
            return
        lab = st.get("status_label", None)
        if lab is not None:
            try:
                lab.setText(text)
            except Exception:
                pass

    def _cell_counter_run_detection(
        self,
        threshold=0.5,
        min_voxels=200,
        max_voxels=0,
        compactness=0.0,
        neg_grow=2,
        gaussian_sigma=1.0,
        open_radius=1,
        auto_split=True,
        min_seed_distance=8,
    ):
        """
        Improved seeded segmentation:

        - Foreground from threshold on normalized volume
        - Optional smoothing
        - Optional soma-only cleanup (3D opening + remove_small_objects)
        - Positive ROIs -> markers (seeds)
        - Negative ROIs -> exclusion mask
        - Optional auto-split markers from distance peaks
        - Watershed on distance transform
        - Size filtering
        """
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                QMessageBox.warning(self, "Cell Counter", "No cell-counter session found.")
                return

            cc = st["viewer"]
            vol = np.asarray(st["vol"], dtype=np.float32)  # (Z,Y,X)
            pos_shapes = st["pos_shapes"]
            neg_shapes = st["neg_shapes"]
            labels_layer = st["labels"]

            # Normalize volume to 0..1 for thresholding
            vmin, vmax = np.percentile(vol, (1, 99))
            if vmax - vmin <= 1e-6:
                norm = np.clip(vol, 0, 1)
            else:
                norm = np.clip((vol - vmin) / (vmax - vmin), 0, 1)

            Z, Y, X = norm.shape

            def _rasterize_shapes_to_masks(shapes_layer):
                out = []
                if shapes_layer is None or len(shapes_layer.data) == 0:
                    return out

                for poly in shapes_layer.data:
                    if poly is None:
                        continue
                    pts = np.asarray(poly)
                    if pts.ndim != 2 or pts.shape[0] < 3:
                        continue

                    if pts.shape[1] == 3:
                        zz = pts[:, 0]
                        yy = pts[:, 1]
                        xx = pts[:, 2]
                        z = int(np.clip(np.round(np.median(zz)), 0, Z - 1))
                    elif pts.shape[1] == 2:
                        try:
                            z = int(cc.dims.point[0]) if hasattr(cc, "dims") and len(cc.dims.point) > 0 else 0
                        except Exception:
                            z = 0
                        z = int(np.clip(z, 0, Z - 1))
                        yy = pts[:, 0]
                        xx = pts[:, 1]
                    else:
                        continue

                    rr, cc_ = sk_polygon(yy, xx, shape=(Y, X))
                    m = np.zeros((Y, X), dtype=bool)
                    m[rr, cc_] = True
                    out.append((z, m))
                return out

            # Seeds from positive ROIs
            seeds = np.zeros((Z, Y, X), dtype=np.int32)
            pos = _rasterize_shapes_to_masks(pos_shapes)
            if len(pos) == 0:
                QMessageBox.information(cc.window.qt_viewer, "Cell Counter", "Draw at least one GREEN cell-body ROI first.")
                return

            seed_id = 0
            for z, m2d in pos:
                seed_id += 1
                seeds[z, m2d] = seed_id

            # Negative mask (with grow)
            neg_mask = np.zeros((Z, Y, X), dtype=bool)
            neg = _rasterize_shapes_to_masks(neg_shapes)
            if len(neg) > 0:
                se2 = disk(int(neg_grow)) if int(neg_grow) > 0 else None
                for z, m2d in neg:
                    if se2 is not None:
                        m2d = dilation(m2d, se2)
                    neg_mask[z, m2d] = True

            # Foreground mask from threshold (optionally after smoothing)
            img = norm
            if float(gaussian_sigma) > 0:
                img = gaussian(img, sigma=float(gaussian_sigma), preserve_range=True)

            fg = (img >= float(threshold))
            if neg_mask.any():
                fg = fg & (~neg_mask)

            # Soma-only cleanup: opening removes thin processes [web:293]
            r = int(open_radius)
            if r > 0:
                fg = binary_opening(fg, footprint=ball(r))

            # Remove small junk objects (3D) [web:461]
            fg = remove_small_objects(fg, min_size=int(min_voxels), connectivity=1)

            if not fg.any():
                labels_layer.data = np.zeros_like(seeds, dtype=np.int32)
                QMessageBox.information(cc.window.qt_viewer, "Cell Counter", "No foreground after threshold/cleanup.")
                return

            # Watershed on distance transform with markers [web:292]
            dist = ndi.distance_transform_edt(fg)
            markers = seeds.copy()

            # Optional auto-split: add distance peaks as extra markers [web:8]
            if bool(auto_split):
                coords = peak_local_max(
                    dist,
                    min_distance=int(max(1, min_seed_distance)),
                    labels=fg,
                    exclude_border=False,
                )
                for (zz, yy, xx) in coords:
                    if markers[zz, yy, xx] == 0:
                        seed_id += 1
                        markers[zz, yy, xx] = seed_id

            seg = watershed(-dist, markers=markers, mask=fg, compactness=float(compactness))

            # Size filtering
            counts = np.bincount(seg.ravel())
            out = np.zeros_like(seg, dtype=np.int32)
            new_id = 0
            for old_id in range(1, int(seg.max()) + 1):
                n = int(counts[old_id]) if old_id < len(counts) else 0
                if n < int(min_voxels):
                    continue
                if int(max_voxels) > 0 and n > int(max_voxels):
                    continue
                new_id += 1
                out[seg == old_id] = new_id

            labels_layer.data = out
            n_cells = int(new_id)
            print(f"Cell counter: {n_cells} cells detected")
            QMessageBox.information(cc.window.qt_viewer, "Cell Counter", f"Detected {n_cells} cell bodies.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cell detection failed:\n{e}")

    def calculate_intensities_per_cell(self):
        """
        Export a CSV with per-cell fluorescence intensity stats from a chosen 3D Image layer
        in the Cell Counter viewer (one channel at a time):
        - Cell # (1..N)
        - min (excluding zeros)
        - max (excluding zeros)
        - average (mean, excluding zeros)
        """
        try:
            st = getattr(self, "_cell_counter_state", None)
            if not st:
                QMessageBox.warning(self, "Cell Counter", "No cell-counter session found.")
                return

            cc = st["viewer"]
            labels_layer = st["labels"]
            lab = np.asarray(labels_layer.data)  # 0=background

            if lab.ndim != 3:
                QMessageBox.warning(cc.window.qt_viewer, "Cell Counter", "Detected cells layer must be 3D.")
                return

            n_cells = int(lab.max())
            if n_cells <= 0:
                QMessageBox.information(cc.window.qt_viewer, "Cell Counter", "No detected cells to export.")
                return

            # ---------------------------------------------------------
            # 1) Collect candidate intensity source layers (3D Images)
            # ---------------------------------------------------------
            candidates = []  # list of (name, layer)
            for lyr in cc.layers:
                # Must have data
                if not hasattr(lyr, "data"):
                    continue

                # Never allow measuring intensities from the Labels layer itself
                # (Labels are integer IDs per pixel/voxel) 
                if lyr is labels_layer or lyr.__class__.__name__.lower() == "labels":
                    continue

                # Must be array-like and shape-match labels
                try:
                    arr = np.asarray(lyr.data)
                except Exception:
                    continue

                # We only support 3D scalar volumes here
                if arr.ndim != 3:
                    continue
                if arr.shape != lab.shape:
                    continue

                candidates.append((getattr(lyr, "name", "image"), lyr))

            if not candidates:
                QMessageBox.warning(
                    cc.window.qt_viewer,
                    "Cell Counter",
                    "No valid 3D Image layers found that match the labels shape.\n"
                    f"Labels shape: {lab.shape}\n\n"
                    "Tip: In your Cell Counter viewer, ensure the fluorescence volume you want to measure\n"
                    "is present as a 3D Image layer with the same (Z,Y,X) shape."
                )
                return

            # Default to 'Detection input' if present
            names = [nm for nm, _ in candidates]
            default_index = 0
            if "Detection input" in names:
                default_index = names.index("Detection input")

            chosen_name, ok = QInputDialog.getItem(
                cc.window.qt_viewer,
                "Choose intensity layer",
                "Select the 3D Image layer to measure intensities from:",
                names,
                default_index,
                False
            )
            if not ok or not chosen_name:
                return

            chosen_layer = dict(candidates)[chosen_name]
            vol = np.asarray(chosen_layer.data, dtype=np.float32)

            # Final guard
            if vol.ndim != 3 or vol.shape != lab.shape:
                QMessageBox.warning(
                    cc.window.qt_viewer,
                    "Cell Counter",
                    f"Selected layer does not match labels shape.\n"
                    f"Image shape: {vol.shape}, Labels shape: {lab.shape}"
                )
                return

            # ---------------------------------------------------------
            # 2) Choose output file
            # ---------------------------------------------------------
            base, ok = QInputDialog.getText(
                cc.window.qt_viewer,
                "CSV name",
                "Enter output CSV filename (without extension):",
                text="cell_intensities"
            )
            if not ok:
                return
            base = (base or "").strip() or "cell_intensities"
            default_name = base if base.lower().endswith(".csv") else (base + ".csv")

            save_path, _ = QFileDialog.getSaveFileName(
                cc.window.qt_viewer,
                "Save cell intensities CSV",
                default_name,
                "CSV files (*.csv)"
            )
            if not save_path:
                return
            if not save_path.lower().endswith(".csv"):
                save_path += ".csv"

            # ---------------------------------------------------------
            # 3) Compute per-cell stats (exclude zeros)
            # ---------------------------------------------------------
            rows = []
            for cell_id in range(1, n_cells + 1):
                m = (lab == cell_id)
                if not np.any(m):
                    continue

                vals = vol[m]
                vals = vals[vals > 0]  # exclude 0 background

                if vals.size == 0:
                    min_v = float("nan")
                    max_v = float("nan")
                    mean_v = float("nan")
                else:
                    min_v = float(vals.min())
                    max_v = float(vals.max())
                    mean_v = float(vals.mean())

                rows.append((cell_id, min_v, max_v, mean_v))

            if not rows:
                QMessageBox.information(cc.window.qt_viewer, "Cell Counter", "No cells had non-zero intensity pixels.")
                return

            with open(save_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Cell #", "min", "max", "average"])
                w.writerows(rows)

            print(f"Exported intensities for {len(rows)} cells from layer '{chosen_name}' to: {save_path}")
            QMessageBox.information(
                cc.window.qt_viewer,
                "Cell Counter",
                f"Exported {len(rows)} cells.\nLayer: {chosen_name}\nSaved to:\n{save_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Intensity export failed:\n{e}")

    def open_preprocess(self):
        self_parent = self
        dlg = QDialog(self)
        dlg.setWindowTitle("Pre-process Analysis")

        v = QVBoxLayout(dlg)

        def _update_z_label():
            if not hasattr(self_parent, "z_range_label"):
                return
            end = max(self_parent.n_z - 1, 0)

            # Optional: show crop provenance (original indices)
            if hasattr(self_parent, "last_crop_range") and self_parent.last_crop_range is not None:
                a, b = self_parent.last_crop_range
                self_parent.z_range_label.setText(
                    f"Z: {self_parent.current_z}   (range 0–{end})   |   cropped from {a}–{b}"
                )
            else:
                self_parent.z_range_label.setText(f"Z: {self_parent.current_z}   (range 0–{end})")

        def _refresh_z_controls():
            """Update n_z-dependent UI + refresh viewer to current slice."""
            end = max(self_parent.n_z - 1, 0)

            self_parent.z_slider.blockSignals(True)
            self_parent.z_slider.setRange(0, end)
            self_parent.current_z = max(0, min(self_parent.current_z, end))
            self_parent.z_slider.setValue(self_parent.current_z)
            self_parent.z_slider.blockSignals(False)

            _update_z_label()
            self_parent.z_changed(self_parent.current_z)

        delete_z = QPushButton("Delete z slices")

        def do_delete():
            z0, ok1 = QInputDialog.getInt(
                self_parent, "Delete Start", "Start Z slice to delete:",
                0, 0, max(0, self_parent.n_z - 1)
            )
            if not ok1:
                return

            z1, ok2 = QInputDialog.getInt(
                self_parent, "Delete End", "End Z slice to delete:",
                z0, 0, max(0, self_parent.n_z - 1)
            )
            if not ok2:
                return

            if z0 > z1:
                QMessageBox.warning(self_parent, "Invalid Range", "Start Z must be <= End Z")
                return

            try:
                for i, ch in enumerate(self_parent.channel_list):
                    arr = np.asarray(ch)
                    if arr.ndim == 3:
                        new = np.concatenate([arr[:z0], arr[z1+1:]], axis=0) if (z1 + 1) < arr.shape[0] else arr[:z0]
                        self_parent.channel_list[i] = new

                first = np.asarray(self_parent.channel_list[0])
                self_parent.n_z = first.shape[0] if first.ndim == 3 else 1

                # After deleting, crop provenance is no longer meaningful
                self_parent.last_crop_range = None

                _refresh_z_controls()
                QMessageBox.information(self_parent, "Deleted", f"Deleted Z {z0}-{z1}")
            except Exception as e:
                QMessageBox.critical(self_parent, "Error", f"Failed to delete slices: {e}")

        delete_z.clicked.connect(do_delete)
        v.addWidget(delete_z)

        crop_z = QPushButton("Crop z slices")

        def do_crop():
            if self_parent.n_z <= 0:
                QMessageBox.warning(self_parent, "No Data", "No Z slices available")
                return

            z0, ok1 = QInputDialog.getInt(
                self_parent, "Crop Start", "Start Z slice to keep:",
                0, 0, max(0, self_parent.n_z - 1)
            )
            if not ok1:
                return

            z1, ok2 = QInputDialog.getInt(
                self_parent, "Crop End", "End Z slice to keep:",
                z0, 0, max(0, self_parent.n_z - 1)
            )
            if not ok2:
                return

            if z0 > z1:
                QMessageBox.warning(self_parent, "Invalid Range", "Start Z must be <= End Z")
                return

            try:
                for i, ch in enumerate(self_parent.channel_list):
                    arr = np.asarray(ch)
                    if arr.ndim == 3:
                        self_parent.channel_list[i] = arr[z0:z1+1].copy()

                first = np.asarray(self_parent.channel_list[0])
                self_parent.n_z = first.shape[0] if first.ndim == 3 else 1

                # Store the original crop selection for display only
                self_parent.last_crop_range = (z0, z1)

                _refresh_z_controls()
                QMessageBox.information(self_parent, "Cropped", f"Kept Z {z0}-{z1} (now {self_parent.n_z} slices)")
            except Exception as e:
                QMessageBox.critical(self_parent, "Error", f"Failed to crop slices: {e}")

        crop_z.clicked.connect(do_crop)
        v.addWidget(crop_z)

        dlg.setLayout(v)
        dlg.show()

        rot90 = QPushButton('Rotate 90')
        rot90.clicked.connect(lambda: self.rotate_stack(90))
        v.addWidget(rot90)

        rot180 = QPushButton('Rotate 180')
        rot180.clicked.connect(lambda: self.rotate_stack(180))
        v.addWidget(rot180)

        gen_mask_btn = QPushButton("Generate Mask Channel(s) (Otsu)")

        def do_generate_masks():
            txt, ok = QInputDialog.getText(
                self_parent,
                "Channels",
                f'Enter channel numbers (comma-separated) or "all":',
                text="1",
            )
            if not ok or not txt:
                return

            txt = txt.strip()
            n = len(self_parent.channel_list)

            if txt.lower() == "all":
                choices = list(range(1, n + 1))
            else:
                choices = []
                for p in [p.strip() for p in txt.split(",") if p.strip()]:
                    try:
                        v = int(p)
                        if 1 <= v <= n:
                            choices.append(v)
                    except Exception:
                        pass

            if not choices:
                QMessageBox.warning(self_parent, "No Channels", "No valid channels selected")
                return

            # Keep references so dialogs don't get GC'd
            self_parent._mask_tuners = getattr(self_parent, "_mask_tuners", [])

            for chnum in choices:
                idx = chnum - 1
                arr = self_parent.channel_list[idx]

                dlg = MaskTunerDialog(self_parent, idx, arr, file_path=self_parent.file_path)
                dlg.setModal(False)

                self_parent._mask_tuners.append(dlg)

                # remove reference when actually destroyed
                dlg.destroyed.connect(
                    lambda _=None, d=dlg: self_parent._mask_tuners.remove(d)
                    if d in self_parent._mask_tuners else None
                )

                dlg.show()

        gen_mask_btn.clicked.connect(do_generate_masks)
        v.addWidget(gen_mask_btn)

    def threshold_channel_dialog(self, ch_idx):
        """Simple threshold dialog with preview and confirm flow.
        Uses the currently displayed slice for defaults and computes masks
        across the full Z-stack when applying.
        """
        arr = np.asarray(self.channel_list[ch_idx])
        if arr.ndim != 3:
            QMessageBox.information(self, 'Single-slice', 'Channel has no Z dimension; thresholding will operate on full image')

        dialog = QDialog(self)
        dialog.setWindowTitle(f'Threshold Channel {ch_idx+1}')
        layout = QVBoxLayout()

        # Slider for threshold
        th_slider = QSlider(Qt.Horizontal)
        th_slider.setRange(0, 65535)
        # Default based on current Z slice (if available)
        cur_z = int(getattr(self, 'current_z', 0))
        if arr.ndim == 3:
            cur_z = max(0, min(cur_z, arr.shape[0]-1))
        default = int(np.percentile(arr[cur_z], 50)) if arr.size > 0 else 100
        th_slider.setValue(default)
        th_label = QLabel(str(default))
        th_slider.valueChanged.connect(lambda v: th_label.setText(str(v)))

        layout.addWidget(QLabel('Manual threshold'))
        layout.addWidget(th_slider)
        layout.addWidget(th_label)

        # Preview button: shows mask preview across Z-stack
        preview_btn = QPushButton('Preview Mask (full stack)')
        def do_preview():
            th = th_slider.value()
            if arr.ndim == 3:
                masks = np.zeros_like(arr, dtype=np.uint8)
                for z in range(arr.shape[0]):
                    masks[z] = compute_mask(arr[z], threshold=th, blur_enabled=False)
                v = napari.Viewer()
                v.add_image(arr, name=f'Ch{ch_idx+1} stack')
                v.add_image(masks, name='Mask Preview', colormap='gray', opacity=0.6)
            else:
                mask = compute_mask(arr, threshold=th, blur_enabled=False)
                v = napari.Viewer()
                v.add_image(arr, name=f'Ch{ch_idx+1} image')
                v.add_image(mask, name='Mask Preview', colormap='gray', opacity=0.6)
        preview_btn.clicked.connect(do_preview)
        layout.addWidget(preview_btn)

        # Otsu auto-threshold button (uses current slice)
        otsu_btn = QPushButton('Use Otsu (current slice)')
        def do_otsu():
            try:
                if arr.ndim == 3:
                    zidx = int(getattr(self, 'current_z', 0))
                    zidx = max(0, min(zidx, arr.shape[0]-1))
                    t = int(threshold_otsu(arr[zidx]))
                else:
                    t = int(threshold_otsu(arr))
                th_slider.setValue(t)
                th_label.setText(str(t))
            except Exception as e:
                QMessageBox.critical(self, 'Otsu Error', f'Otsu failed: {e}')
        otsu_btn.clicked.connect(do_otsu)
        layout.addWidget(otsu_btn)

        # Confirm/cancel buttons
        btns = QHBoxLayout()
        ok_btn = QPushButton('Apply and Close')
        cancel_btn = QPushButton('Cancel')
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        def do_apply():
            th = th_slider.value()
            # compute full 3D mask or single mask
            if arr.ndim == 3:
                masks = np.zeros_like(arr, dtype=np.uint8)
                for z in range(arr.shape[0]):
                    masks[z] = compute_mask(arr[z], threshold=th)
            else:
                masks = compute_mask(arr, threshold=th)
            # create mask layer and add controls in main UI
            self.create_mask_layer(ch_idx, masks)
            dialog.accept()

        def do_cancel():
            reply = QMessageBox.question(self, 'Still Thresholding?', 'Are you done thresholding this channel?', QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                dialog.reject()

        ok_btn.clicked.connect(do_apply)
        cancel_btn.clicked.connect(do_cancel)

        dialog.setLayout(layout)
        dialog.exec_()

    def create_mask_layer(self, ch_idx, mask_array):
        name = f"Channel {ch_idx+1} mask"

        mask3 = np.asarray(mask_array)
        if mask3.ndim == 2:
            mask3 = mask3[np.newaxis, ...]
        mask3 = (mask3 > 0).astype(np.uint8)

        self.masks_by_channel = getattr(self, "masks_by_channel", {})
        self.masks_by_channel[ch_idx] = mask3

        z = int(getattr(self, "current_z", 0))
        z = max(0, min(z, mask3.shape[0] - 1))
        mask2d = mask3[z]

        try:
            lay = self.viewer.layers.get(name)
        except Exception:
            lay = None

        if lay is None:
            lay = self.viewer.add_image(
                mask2d,
                name=name,
                colormap="yellow",
                opacity=0.6,
                blending="additive",
                contrast_limits=(0, 1),
            )
        else:
            lay.data = mask2d
            try:
                lay.contrast_limits = (0, 1)
            except Exception:
                pass

        self.mask_layers = getattr(self, "mask_layers", {})
        self.mask_layers[ch_idx] = lay

    def rotate_stack(self, deg):
        for i, ch in enumerate(self.channel_list):
            arr = np.asarray(ch)
            if arr.ndim == 3:
                for z in range(arr.shape[0]):
                    arr[z] = rotate(arr[z], deg, preserve_range=True)
            else:
                arr = rotate(arr, deg, preserve_range=True)
            self.channel_list[i] = arr
        # refresh current slice
        self.z_changed(self.current_z)

    def load_channels(self):
        """Load more channels and append them to the existing viewer/UI (no overwrite)."""
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load LIF/TIFF",
                "",
                "LIF (*.lif);;TIFF (*.tif *.tiff);;All files (*)",
            )
            if not path:
                return

            p = Path(path)
            if p.suffix.lower() == ".lif":
                mmaps, clims, vox, temp = load_lif_memmap(p, max_ch=4)
            else:
                mmaps, clims, vox, temp = load_tiff_memmap(p, max_ch=4)

            # Ensure lists exist
            if not hasattr(self, "channel_list") or self.channel_list is None:
                self.channel_list = []
            if not hasattr(self, "channel_layers") or self.channel_layers is None:
                self.channel_layers = []
            if not hasattr(self, "channel_controls") or self.channel_controls is None:
                self.channel_controls = []

            # Update contrast_limits dict with any new ones we got
            if isinstance(clims, dict):
                # NOTE: clims keys might be "ch1/ch2" relative to the loaded file.
                # We'll set defaults for new global indices below anyway.
                pass

            # determine slider max range from existing + new clims
            max_range = 65535
            try:
                vals = []
                if isinstance(getattr(self, "contrast_limits", None), dict):
                    vals += [int(v[1]) for v in self.contrast_limits.values() if isinstance(v, (tuple, list)) and len(v) == 2]
                if isinstance(clims, dict):
                    vals += [int(v[1]) for v in clims.values() if isinstance(v, (tuple, list)) and len(v) == 2]
                if vals:
                    mr = max(vals)
                    max_range = max(1000, min(mr * 2, 65535))
            except Exception:
                pass

            # Append new channels
            start_idx = len(self.channel_list)  # 0-based count before adding
            for j, ch in enumerate(mmaps):
                new_idx = start_idx + j + 1  # 1-based channel number for naming

                self.channel_list.append(ch)

                # Add layer to viewer (do NOT clear existing)
                lay = self.viewer.add_image(ch, name=f"Channel {new_idx}", opacity=1.0)
                self.channel_layers.append(lay)

                # Set default contrast_limits entry for this new channel if missing
                if not hasattr(self, "contrast_limits") or self.contrast_limits is None:
                    self.contrast_limits = {}
                if f"ch{new_idx}" not in self.contrast_limits:
                    # try to use clims from loaded file (best-effort), else fallback
                    guess = None
                    if isinstance(clims, dict):
                        guess = clims.get(f"ch{j+1}", None)
                    self.contrast_limits[f"ch{new_idx}"] = guess if guess else (0, max_range)

                # Add UI controls for the new channel
                gb, ctrl = self._add_channel_controls(lay, new_idx, max_range=max_range)
                self.channel_controls.append(ctrl)

                # Put the groupbox somewhere sensible:
                # simplest: add at end of main layout
                self.layout().insertWidget(self.layout().count() - 1, gb)

            # Update n_z and z slider range based on first channel (your app’s convention)
            first = np.asarray(self.channel_list[0])
            self.n_z = first.shape[0] if first.ndim == 3 else 1
            self.z_slider.setRange(0, max(self.n_z - 1, 0))

            # Track temp paths
            self._temp_paths = getattr(self, "_temp_paths", []) + (temp or [])

            QMessageBox.information(self, "Loaded", f"Added {len(mmaps)} channels (now {len(self.channel_list)} total).")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load channels: {e}")

    def export_selected_layers(self):
        try:
            # list image-like layers
            img_layers = [lyr for lyr in self.viewer.layers if hasattr(lyr, "data")]
            if not img_layers:
                QMessageBox.warning(self, "No layers", "No exportable layers found.")
                return

            names = [lyr.name for lyr in img_layers]
            txt, ok = QInputDialog.getText(self, "Export", f"Enter comma-separated layer numbers to export (1-{len(names)}):\n" +
                                        "\n".join([f"{i+1}: {n}" for i, n in enumerate(names)]))
            if not ok or not txt:
                return

            nums = [int(x.strip()) for x in txt.split(",") if x.strip().isdigit()]
            out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
            if not out_dir:
                return

            for n in nums:
                i = n - 1
                if i < 0 or i >= len(img_layers):
                    continue
                lyr = img_layers[i]
                arr = np.asarray(lyr.data)
                out_path = Path(out_dir) / f"{Path(self.file_path).stem}_{lyr.name}.tif"
                tifffile.imwrite(str(out_path), arr)
            QMessageBox.information(self, "Exported", "Selected layers exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def _pick_layer_name(self, title, prompt, layer_names):
        name, ok = QInputDialog.getItem(self, title, prompt, layer_names, 0, False)
        if not ok or not name:
            return None
        return str(name)

    def _is_mask_like(self, name, arr):
        if "mask" in (name or "").lower():
            return True
        a = np.asarray(arr)
        try:
            u = np.unique(a)
            if u.size <= 3 and set(u.tolist()).issubset({0, 1, 255}):
                return True
        except Exception:
            pass
        return False
    
    def _resolve_stack3d(self, layer_name: str):
        """
        Return a full 3D stack (Z,Y,X) for any selectable layer name:
        - "Channel N"          -> self.channel_list[N-1]
        - "Channel N mask"     -> self.masks_by_channel[N-1]
        - derived layer name   -> self.derived_stacks[...] (supports name-keyed or id-keyed)
        - else fallback        -> viewer layer.data if it is already 3D (or 2D->Z=1)
        """
        # ---- Channels ----
        if layer_name.startswith("Channel ") and layer_name[8:].isdigit():
            idx = int(layer_name.split()[1]) - 1
            if 0 <= idx < len(getattr(self, "channel_list", [])):
                arr = np.asarray(self.channel_list[idx])
                return arr if arr.ndim == 3 else arr[np.newaxis, ...]

        # ---- Masks ----
        if layer_name.startswith("Channel ") and layer_name.endswith(" mask"):
            try:
                idx = int(layer_name.split()[1]) - 1
            except Exception:
                idx = None
            if idx is not None:
                m3 = getattr(self, "masks_by_channel", {}).get(idx, None)
                if m3 is not None:
                    m3 = np.asarray(m3)
                    return m3 if m3.ndim == 3 else m3[np.newaxis, ...]

        # ---- Derived stacks (coloc/multiply outputs) ----
        if hasattr(self, "derived_stacks"):
            ds = getattr(self, "derived_stacks", {})

            # name-keyed
            if layer_name in ds:
                arr = np.asarray(ds[layer_name])
                return arr if arr.ndim == 3 else arr[np.newaxis, ...]

            # id(lay)-keyed: find layer object by name
            dl = getattr(self, "derived_layers", {})
            for k, lay in dl.items():
                if getattr(lay, "name", None) == layer_name and k in ds:
                    arr = np.asarray(ds[k])
                    return arr if arr.ndim == 3 else arr[np.newaxis, ...]

        # ---- Fallback to napari layer.data ----
        lay = self._get_layer_by_name(layer_name)
        if lay is None:
            return None
        arr = np.asarray(lay.data)
        return arr if arr.ndim == 3 else arr[np.newaxis, ...]

    def _compute_reasonable_maxrange(self, arr3, fallback=65535):
        """Pick a slider max that makes sense for integer images; clamp to 65535 for UI."""
        try:
            a = np.asarray(arr3)
            if np.issubdtype(a.dtype, np.integer):
                mx = int(np.iinfo(a.dtype).max)
                return int(max(1000, min(mx, 65535)))
        except Exception:
            pass
        return int(fallback)

    def _add_coloc_controls(self, lay, out3, title: str, out_is_mask: bool):
        """
        Create and insert per-coloc UI controls (opacity + contrast min/max + brightness/contrast-mult).
        Stores bookkeeping so you don't create duplicates.
        """
        # init containers
        self.coloc_controls = getattr(self, "coloc_controls", [])
        self.coloc_control_boxes = getattr(self, "coloc_control_boxes", [])

        # don’t duplicate controls for the same layer object
        for c in self.coloc_controls:
            if c.get("layer", None) is lay:
                return

        gb = QGroupBox(title)
        v = QVBoxLayout()

        # ---- Opacity ----
        v.addWidget(QLabel("Opacity"))
        op_row = QHBoxLayout()
        op = QSlider(Qt.Horizontal)
        op.setRange(0, 100)
        op.setValue(int((lay.opacity if lay.opacity is not None else 0.7) * 100))
        op_val = QLineEdit(str(op.value()))
        op_val.setFixedWidth(60)
        op_val.setValidator(QIntValidator(0, 100))
        op_row.addWidget(op)
        op_row.addWidget(op_val)
        v.addLayout(op_row)

        def _apply_op(val=None):
            val = op.value() if val is None else int(val)
            lay.opacity = val / 100.0
            op_val.setText(str(val))

        op.valueChanged.connect(_apply_op)
        op_val.editingFinished.connect(lambda: self.setsliderfromlineedit(op_val, op, 0, 100))

        # ---- Contrast limits ----
        maxrange = 1 if out_is_mask else self._compute_reasonable_maxrange(out3, fallback=65535)

        v.addWidget(QLabel("Contrast"))
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min"))
        smin = QSlider(Qt.Horizontal)
        smin.setRange(0, maxrange)
        min_row.addWidget(smin)
        min_val = QLineEdit("0")
        min_val.setFixedWidth(80)
        min_val.setValidator(QIntValidator(0, maxrange))
        min_row.addWidget(min_val)
        v.addLayout(min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max"))
        smax = QSlider(Qt.Horizontal)
        smax.setRange(0, maxrange)
        max_row.addWidget(smax)
        max_val = QLineEdit(str(maxrange))
        max_val.setFixedWidth(80)
        max_val.setValidator(QIntValidator(0, maxrange))
        max_row.addWidget(max_val)
        v.addLayout(max_row)

        # initialize from layer if present; else default
        try:
            c0, c1 = lay.contrast_limits
            c0 = int(np.clip(int(c0), 0, maxrange))
            c1 = int(np.clip(int(c1), 0, maxrange))
        except Exception:
            c0, c1 = (0, 1) if out_is_mask else (0, maxrange)

        if c1 < c0:
            c1 = c0

        smin.setValue(c0); smax.setValue(c1)
        min_val.setText(str(c0)); max_val.setText(str(c1))

        def _apply_clims(_=None):
            lo = int(smin.value())
            hi = int(smax.value())
            if hi < lo:
                hi = lo
                smax.setValue(hi)
                max_val.setText(str(hi))
            min_val.setText(str(lo))
            max_val.setText(str(hi))
            try:
                lay.contrast_limits = (lo, hi)
            except Exception:
                pass

        smin.valueChanged.connect(_apply_clims)
        smax.valueChanged.connect(_apply_clims)
        min_val.editingFinished.connect(lambda: self.setsliderfromlineedit(min_val, smin, 0, maxrange))
        max_val.editingFinished.connect(lambda: self.setsliderfromlineedit(max_val, smax, 0, maxrange))

        # ---- Optional display tweaks (brightness + contrast multiplier) ----
        # This mirrors your existing pattern (brightness shift, contrast multiplier). [file:375]
        v.addWidget(QLabel("Display (optional)"))
        disp_row = QHBoxLayout()

        bright = QSlider(Qt.Horizontal)
        bright.setRange(-2000, 2000)
        bright.setValue(0)
        bright_val = QLineEdit("0")
        bright_val.setFixedWidth(60)
        bright_val.setValidator(QIntValidator(-2000, 2000))

        mult = QSlider(Qt.Horizontal)
        mult.setRange(10, 300)
        mult.setValue(100)
        mult_val = QLineEdit(str(mult.value()))
        mult_val.setFixedWidth(60)
        mult_val.setValidator(QIntValidator(10, 300))

        # layout
        disp_row.addWidget(QLabel("Brightness"))
        disp_row.addWidget(bright)
        disp_row.addWidget(bright_val)
        disp_row.addWidget(QLabel("Contrast"))
        disp_row.addWidget(mult)
        disp_row.addWidget(mult_val)
        v.addLayout(disp_row)

        def _apply_display(_=None):
            # derive from current min/max sliders (like your other update*display funcs) [file:375]
            try:
                base_lo = float(smin.value())
                base_hi = float(smax.value())
                center = (base_lo + base_hi) / 2.0
                width = max(base_hi - base_lo, 1.0)
                b = float(bright.value())
                m = float(mult.value()) / 100.0
                new_lo = center - (width / 2.0) * m + b
                new_hi = center + (width / 2.0) * m + b
                lay.contrast_limits = (new_lo, new_hi)
            except Exception:
                pass

            bright_val.setText(str(int(bright.value())))
            mult_val.setText(str(int(mult.value())))

        bright.valueChanged.connect(_apply_display)
        mult.valueChanged.connect(_apply_display)
        bright_val.editingFinished.connect(lambda: self.setsliderfromlineedit(bright_val, bright, -2000, 2000))
        mult_val.editingFinished.connect(lambda: self.setsliderfromlineedit(mult_val, mult, 10, 300))

        gb.setLayout(v)

        # insert into UI (just before final stretch)
        self.layout().insertWidget(self.layout().count() - 1, gb)

        # bookkeeping
        self.coloc_controls.append(
            dict(layer=lay, opacity=op, cmin=smin, cmax=smax, bright=bright, mult=mult, groupbox=gb)
        )
        self.coloc_control_boxes.append(gb)

    def generate_coloc_entire_stack(self):
        """
        Multiply (mask x target) where target can be: mask, channel, or an existing derived layer.
        Always operates on full 3D stacks (Z,Y,X) resolved from canonical storage, not the 2D view.
        """
        try:
            layer_names = [lyr.name for lyr in self.viewer.layers]
            if not layer_names:
                QMessageBox.warning(self, "No layers", "No layers found in viewer.")
                return

            mask_name = self._pick_layer_name(
                "Select Mask", "Select MASK layer (binarized as >0):", layer_names
            )
            if mask_name is None:
                return

            target_name = self._pick_layer_name(
                "Select Target", "Select TARGET layer (mask/channel/derived):", layer_names
            )
            if target_name is None:
                return

            mask3 = self._resolve_stack3d(mask_name)
            target3 = self._resolve_stack3d(target_name)

            if mask3 is None or target3 is None:
                QMessageBox.warning(self, "Missing data", "Could not resolve full stack for one of the selections.")
                return

            mask3 = np.asarray(mask3)
            target3 = np.asarray(target3)

            if mask3.ndim != 3 or target3.ndim != 3:
                QMessageBox.warning(self, "Bad dims", f"Expected 3D stacks; got {mask3.ndim}D and {target3.ndim}D.")
                return

            if mask3.shape != target3.shape:
                QMessageBox.warning(self, "Shape mismatch", f"Mask {mask3.shape} != Target {target3.shape}")
                return

            mask_bin = (mask3 > 0)

            target_is_mask = self._is_mask_like(target_name, target3)

            if target_is_mask:
                out3 = (mask_bin & (target3 > 0)).astype(np.uint8)
                out_is_mask = True
            else:
                out3 = target3 * mask_bin.astype(target3.dtype)
                out_is_mask = False

            requested_name = f"Mul({target_name})_xMask({mask_name})"

            z0 = int(getattr(self, "current_z", 0))
            z0 = max(0, min(z0, out3.shape[0] - 1))
            out2 = out3[z0]

            lay = self._get_layer_by_name(requested_name)

            if out_is_mask:
                cmap = "yellow"
            else:
                tlay = self._get_layer_by_name(target_name)
                cmap = getattr(tlay, "colormap", None) if tlay is not None else None
                cmap = cmap or "yellow"

            created_new = False
            if lay is None:
                lay = self.viewer.add_image(
                    out2,
                    name=requested_name,
                    colormap=cmap,
                    blending="additive",
                    opacity=0.6 if out_is_mask else 0.7,
                )
                created_new = True
            else:
                lay.data = out2
                try:
                    lay.colormap = cmap
                except Exception:
                    pass

            if out_is_mask:
                try:
                    lay.contrast_limits = (0, 1)
                except Exception:
                    pass

            # store derived using stable object key
            self.derived_stacks = getattr(self, "derived_stacks", {})
            self.derived_layers = getattr(self, "derived_layers", {})
            k = id(lay)
            self.derived_stacks[k] = out3
            self.derived_layers[k] = lay

            # NEW: create per-coloc sliders/controls (only once per layer)
            # (If you re-run and update same named layer, controls won't duplicate.)
            self._add_coloc_controls(
                lay,
                out3=out3,
                title=f"Coloc: {lay.name}",
                out_is_mask=bool(out_is_mask),
            )

            # force sync right now
            self.z_changed(int(getattr(self, "current_z", 0)))

            QMessageBox.information(self, "Done", f"Created/updated: {lay.name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Multiply failed: {e}")
            
    def generate_3d_surfaces(self):
        """
        3D surface generator (marching cubes) where surface color matches the
        *current* channel color in the main viewer UI, including user-picked hex colors.

        IMPORTANT: napari Surface layers are colored via vertex_values+colormap or vertex_colors,
        not face_color/edge_color. We set vertex_colors for a uniform solid surface color.
        """
        try:
            # ----------------------------
            # 0) Collect candidate layers
            # ----------------------------
            layer_names = [lyr.name for lyr in self.viewer.layers]
            if not layer_names:
                QMessageBox.warning(self, "No layers", "No layers found in viewer.")
                return None

            # ----------------------------
            # 1) Voxel size: always ask user
            #    (use metadata only as suggested defaults)
            # ----------------------------
            suggested = None

            # (A) If you already stored voxel_size somewhere, use it as suggestion
            try:
                vs = getattr(self, "voxel_size", None)
                if isinstance(vs, (list, tuple)) and len(vs) == 3:
                    suggested = (float(vs[0]), float(vs[1]), float(vs[2]))
            except Exception:
                pass

            # (B) Try TIFF metadata as suggestion (still only a suggestion)
            if suggested is None:
                path = getattr(self, "last_loaded_path", None)
                if path:
                    try:
                        from tifffile import TiffFile
                        with TiffFile(str(path)) as tif:
                            ij = getattr(tif, "imagej_metadata", None) or {}
                            z = ij.get("spacing", None)

                            tags = tif.pages[0].tags
                            xr = tags["XResolution"].value if "XResolution" in tags else None
                            yr = tags["YResolution"].value if "YResolution" in tags else None
                            ru = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None

                            def _res_to_size(res, unit_tag):
                                if res is None:
                                    return None
                                num, den = res
                                if num == 0:
                                    return None
                                pixels_per_unit = num / den
                                if unit_tag == 2:      # inch
                                    microns_per_unit = 25400.0
                                elif unit_tag == 3:    # cm
                                    microns_per_unit = 10000.0
                                else:
                                    microns_per_unit = 1.0
                                return microns_per_unit / pixels_per_unit

                            x_um = _res_to_size(xr, ru)
                            y_um = _res_to_size(yr, ru)

                            if z is not None and x_um is not None and y_um is not None:
                                suggested = (float(z), float(y_um), float(x_um))
                    except Exception:
                        pass

            if suggested is None:
                suggested = (1.0, 1.0, 1.0)

            # Always prompt user (defaulting to suggested values) 
            z_um, ok1 = QInputDialog.getDouble(
                self, "Voxel size", "Z voxel size (µm):", float(suggested[0]), 1e-6, 1e9, 6
            )
            if not ok1:
                return None
            y_um, ok2 = QInputDialog.getDouble(
                self, "Voxel size", "Y voxel size (µm):", float(suggested[1]), 1e-6, 1e9, 6
            )
            if not ok2:
                return None
            x_um, ok3 = QInputDialog.getDouble(
                self, "Voxel size", "X voxel size (µm):", float(suggested[2]), 1e-6, 1e9, 6
            )
            if not ok3:
                return None

            spacing_z, spacing_y, spacing_x = (float(z_um), float(y_um), float(x_um))

            # ----------------------------
            # 2) Checklist dialog
            # ----------------------------
            dlg = QDialog(self)
            dlg.setWindowTitle("Select layers for 3D surfaces")
            dlg.setMinimumWidth(420)
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("Check layers to convert to 3D surfaces (requires 3D stack):"))

            lst = QListWidget()
            lst.setSelectionMode(QAbstractItemView.NoSelection)
            for nm in layer_names:
                it = QListWidgetItem(nm)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                lst.addItem(it)
            v.addWidget(lst)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            v.addWidget(buttons)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() != QDialog.Accepted:
                return None

            selected = [lst.item(i).text() for i in range(lst.count())
                        if lst.item(i).checkState() == Qt.Checked]
            if not selected:
                QMessageBox.information(self, "3D Surfaces", "No layers selected.")
                return None

            # ----------------------------
            # 3) Create 3D viewer
            # ----------------------------
            v3 = napari.Viewer(ndisplay=3)
            v3.title = "3D Surfaces"
            try:
                v3.scale_bar.visible = True
                v3.scale_bar.unit = "um"
                v3.scale_bar.font_size = 12
                v3.scale_bar.color = "white"
                v3.scale_bar.box = True
                v3.scale_bar.position = "bottom_right"
            except Exception:
                pass
            try:
                v3.dims.axis_labels = ["Z", "Y", "X"]
            except Exception:
                pass

            dock = QWidget()
            dock_lay = QVBoxLayout(dock)
            ss_btn = QPushButton("Save Screenshot")
            ss_btn.clicked.connect(
                lambda _=False: self.save_screenshot(
                    viewer=v3, parent=self, default_name="3d_surfaces.png", canvas_only=True
                )
            )
            dock_lay.addWidget(ss_btn)
            dock_lay.addStretch(1)
            v3.window.add_dock_widget(dock, area="right")

            # ----------------------------
            # Helpers: color extraction
            # ----------------------------
            from skimage.measure import marching_cubes

            def _fallback_color(name):
                palette = [
                    (1.0, 0.2, 0.2, 0.85),
                    (0.2, 1.0, 0.2, 0.85),
                    (0.2, 0.5, 1.0, 0.85),
                    (1.0, 1.0, 0.2, 0.85),
                    (1.0, 0.2, 1.0, 0.85),
                    (0.2, 1.0, 1.0, 0.85),
                ]
                return palette[abs(hash(name)) % len(palette)]

            def _hex_to_rgba(hex_str, alpha=0.85):
                s = str(hex_str).strip()
                if not s.startswith("#"):
                    return None
                h = s[1:]
                if len(h) == 6:
                    r = int(h[0:2], 16) / 255.0
                    g = int(h[2:4], 16) / 255.0
                    b = int(h[4:6], 16) / 255.0
                    return (r, g, b, float(alpha))
                if len(h) == 8:
                    a = int(h[0:2], 16) / 255.0
                    r = int(h[2:4], 16) / 255.0
                    g = int(h[4:6], 16) / 255.0
                    b = int(h[6:8], 16) / 255.0
                    return (r, g, b, a)
                return None

            NAMED_RGBA = {
                "red":     (1.0, 0.0, 0.0, 0.85),
                "green":   (0.0, 1.0, 0.0, 0.85),
                "blue":    (0.0, 0.4, 1.0, 0.85),
                "magenta": (1.0, 0.0, 1.0, 0.85),
                "cyan":    (0.0, 1.0, 1.0, 0.85),
                "yellow":  (1.0, 1.0, 0.0, 0.85),
                "gray":    (0.85, 0.85, 0.85, 0.85),
                "grey":    (0.85, 0.85, 0.85, 0.85),
            }

            def _rgba_from_layer_current(layer_obj, fallback_name):
                if layer_obj is None:
                    return _fallback_color(fallback_name)

                cm = getattr(layer_obj, "colormap", None)
                if cm is None:
                    return _fallback_color(fallback_name)

                if isinstance(cm, str):
                    rgba = _hex_to_rgba(cm)
                    if rgba is not None:
                        return rgba
                    if cm.lower() in NAMED_RGBA:
                        return NAMED_RGBA[cm.lower()]
                    return _fallback_color(fallback_name)

                if isinstance(cm, tuple) and len(cm) >= 2:
                    name = cm[0]
                    cmap_obj = cm[1]

                    if isinstance(name, str):
                        rgba = _hex_to_rgba(name)
                        if rgba is not None:
                            return rgba
                        if name.lower() in NAMED_RGBA:
                            return NAMED_RGBA[name.lower()]

                    try:
                        if hasattr(cmap_obj, "map"):
                            rgba = cmap_obj.map([0.95])[0]
                            rgba = tuple(float(x) for x in rgba)
                            if len(rgba) == 4:
                                return (rgba[0], rgba[1], rgba[2], 0.85)
                    except Exception:
                        pass

                    return _fallback_color(fallback_name)

                try:
                    if hasattr(cm, "map"):
                        rgba = cm.map([0.95])[0]
                        rgba = tuple(float(x) for x in rgba)
                        if len(rgba) == 4:
                            return (rgba[0], rgba[1], rgba[2], 0.85)
                except Exception:
                    pass

                return _fallback_color(fallback_name)

            # ----------------------------
            # 4) Marching cubes per layer
            # ----------------------------
            made_any = False
            warnings = []

            for name in selected:
                try:
                    src_layer = self.viewer.layers[name]
                except Exception:
                    src_layer = None

                stack3 = self._resolve_stack3d(name)
                if stack3 is None:
                    warnings.append(f"{name}: could not resolve stack")
                    continue

                vol = np.asarray(stack3)
                if vol.ndim != 3:
                    warnings.append(f"{name}: not 3D (shape {vol.shape})")
                    continue

                volf = vol.astype(np.float32)

                vmin, vmax = np.percentile(volf, (1, 99))
                if vmax - vmin <= 1e-6:
                    norm = np.clip(volf, 0, 1)
                else:
                    norm = np.clip((volf - vmin) / (vmax - vmin), 0.0, 1.0)

                iso_level = 0.5
                if np.nanmax(norm) < iso_level:
                    warnings.append(f"{name}: no voxels above iso-level")
                    continue

                try:
                    verts, faces, normals, values = marching_cubes(
                        norm,
                        level=iso_level,
                        spacing=(spacing_z, spacing_y, spacing_x),
                        step_size=2,
                    )
                except Exception as e:
                    warnings.append(f"{name}: marching cubes failed ({e})")
                    continue

                rgba = _rgba_from_layer_current(src_layer, name)
                vertex_colors = np.tile(np.array(rgba, dtype=np.float32), (verts.shape[0], 1))

                v3.add_surface(
                    (verts, faces, values),
                    name=f"{name} surface",
                    shading="smooth",
                    vertex_colors=vertex_colors,
                )

                made_any = True

            if not made_any:
                QMessageBox.information(self, "3D Surfaces", "No surfaces were created.\n" + "\n".join(warnings[:8]))
                try:
                    v3.close()
                except Exception:
                    pass
                return None

            msg = f"3D viewer opened.\nVoxel size (Z,Y,X) = ({spacing_z}, {spacing_y}, {spacing_x}) µm."
            if warnings:
                msg += "\n\nNotes:\n" + "\n".join(warnings[:10])
            QMessageBox.information(self, "3D Surfaces", msg)

            return v3

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate 3D surfaces:\n{e}")
            return None

    def extract_roi_intensities_for_viewer(self, proj_viewer, projections):
        """
        Extract ROI intensities from a MAX-PROJECTION viewer.

        - Uses Shapes drawn on proj_viewer.
        - Prompts user to choose which *_max image layer to measure.
        - Measures on that chosen 2D max-projection image only (no coloc stack).
        - Writes results to an Excel file (safe-save temp then atomic replace).
        """
        try:
            # ----------------------------
            # 0) Find Shapes layers
            # ----------------------------
            try:
                from napari.layers import Shapes
            except Exception:
                Shapes = None

            shapes_layers = []
            for layer in proj_viewer.layers:
                if Shapes is not None and isinstance(layer, Shapes):
                    shapes_layers.append(layer)
                else:
                    if getattr(layer, "shape_type", None) is not None or layer.__class__.__name__ == "Shapes":
                        shapes_layers.append(layer)

            if len(shapes_layers) == 0:
                QMessageBox.critical(
                    proj_viewer.window.qt_viewer,
                    "No shape layers found to extract pixel intensities.",
                    "No shape layers found to extract pixel intensities.",
                )
                return

            # ----------------------------
            # 1) Ask which max-projection channel to quantify
            # ----------------------------
            # Prefer *_max layers, since generate_max_projection names them that way.
            max_img_layer_names = []
            for lyr in proj_viewer.layers:
                # shapes layers shouldn't be candidates
                if lyr in shapes_layers:
                    continue
                if getattr(lyr, "data", None) is None:
                    continue
                nm = getattr(lyr, "name", "") or ""
                if nm.endswith("_max"):
                    max_img_layer_names.append(nm)

            # Fallback: any non-shapes layer with 2D image data
            if not max_img_layer_names:
                for lyr in proj_viewer.layers:
                    if lyr in shapes_layers:
                        continue
                    try:
                        arr = np.asarray(lyr.data)
                    except Exception:
                        continue
                    if arr.ndim == 2:
                        max_img_layer_names.append(getattr(lyr, "name", "image"))

            if not max_img_layer_names:
                QMessageBox.critical(self, "No projection images", "No max-projection image layers found to quantify.")
                return

            chosen_layer_name, ok = QInputDialog.getItem(
                self,
                "Choose channel",
                "Which max-projection layer should be used for ROI intensity extraction?",
                max_img_layer_names,
                0,
                False,
            )
            if not ok or not chosen_layer_name:
                return

            try:
                img_layer = proj_viewer.layers[chosen_layer_name]  # napari supports indexing by unique name 
            except Exception:
                img_layer = None

            if img_layer is None or getattr(img_layer, "data", None) is None:
                QMessageBox.critical(self, "Missing data", f"Could not access layer data for: {chosen_layer_name}")
                return

            img2d = np.asarray(img_layer.data)
            if img2d.ndim != 2:
                QMessageBox.critical(self, "Unsupported", f"Selected layer is not 2D: {chosen_layer_name} has shape {img2d.shape}")
                return

            ydim, xdim = img2d.shape

            # ----------------------------
            # 2) Ask metadata
            # ----------------------------
            protein, ok = QInputDialog.getText(self, "Protein/labeling", "Enter the name or protein or what has been labeled :")
            if not ok or not protein:
                return

            brain_num, ok = QInputDialog.getText(self, "Image #, or ID", "Enter ID number:")
            if not ok or not brain_num:
                return

            slide_date, ok = QInputDialog.getText(self, "Image Date", "Enter image date:")
            if not ok or not slide_date:
                return

            file_name, ok = QInputDialog.getText(self, "File name", "File will be saved as ROI_intensity_<file name>.xlsx :")
            if not ok or not file_name:
                return

            out_path = Path(f"ROI_intensity_{file_name}.xlsx")

            bg_sub_answer, ok = QInputDialog.getItem(
                self,
                "Background Subtracted?",
                "Is this projection already background subtracted?",
                ["Yes", "No"],
                0,
                False,
            )
            if not ok:
                return

            # ----------------------------
            # 3) Excel setup
            # ----------------------------
            try:
                from openpyxl import Workbook, load_workbook
            except Exception:
                QMessageBox.critical(self, "Missing Dependency", "openpyxl is required to write Excel files. Please install it.")
                return

            region_names = [layer.name for layer in shapes_layers]
            region_bg_cols = [f"region_background_value_{name}" for name in region_names]

            headers = [
                "file_name",
                "protein_name",
                "brain_number",
                "slide_date",
                "background_subtracted",
                "quant_layer",
            ] + region_bg_cols + region_names

            if not out_path.exists():
                wb = Workbook()
                ws = wb.active
                ws.append(headers)
                wb.save(out_path)

            wb = load_workbook(out_path)
            ws = wb.active

            # Map header -> column index
            header_to_col = {str(cell.value): i + 1 for i, cell in enumerate(ws[1]) if cell.value is not None}

            # Append row index (we will append at the end)
            append_row_idx = ws.max_row + 1

            tmp_path = out_path.with_name(out_path.stem + "_temp.xlsx")

            def safe_save():
                wb.save(tmp_path)

            # ----------------------------
            # 4) ROI processing
            # ----------------------------  

            results = {}
            bg_values_for_row = {}

            for layer in shapes_layers:
                name = layer.name or f"Region_{len(results) + 1}"
                bg_val = None

                # Ask per ROI if not already bg-subtracted
                if bg_sub_answer == "No":
                    bg_each, ok_each = QInputDialog.getItem(
                        self,
                        "Background Subtraction",
                        f"Background subtract region '{name}'?",
                        ["Yes", "No"],
                        1,
                        False,
                    )
                    if ok_each and bg_each == "Yes":
                        val, okv = QInputDialog.getDouble(
                            self,
                            "Background Value",
                            f"Enter background value for region '{name}':",
                            0.0,
                            0.0,
                            float(np.nanmax(img2d) if img2d.size else 65535.0),
                            3,
                        )
                        if okv:
                            bg_val = float(val)

                bg_values_for_row[name] = "" if bg_val is None else float(bg_val)

                # Build combined mask from all polygons in this shapes layer
                mask2d = np.zeros((ydim, xdim), dtype=bool)
                for shape in getattr(layer, "data", []):
                    try:
                        poly = np.asarray(shape)
                        if poly.ndim != 2 or poly.shape[0] < 3:
                            continue
                        # polygon2mask expects (row, col) coords 
                        poly_rc = np.column_stack([poly[:, 0], poly[:, 1]])
                        m = polygon2mask((ydim, xdim), poly_rc)
                        mask2d |= m.astype(bool)
                    except Exception:
                        continue

                if not mask2d.any():
                    results[name] = 0.0
                    continue

                roi_pixels = img2d[mask2d].astype(np.float32)

                if bg_val is not None:
                    roi_pixels = roi_pixels - float(bg_val)
                    roi_pixels[roi_pixels < 0] = 0.0

                nz = roi_pixels[roi_pixels > 0]
                avg_val = float(nz.mean()) if nz.size > 0 else 0.0
                results[name] = avg_val

                # Optional verification overlay (shows ROI pixels from chosen layer)
                try:
                    verification2d = np.zeros((ydim, xdim), dtype=np.float32)
                    verification2d[mask2d] = img2d[mask2d].astype(np.float32)
                    proj_viewer.add_image(verification2d, name=f"ROI_{name}", blending="additive", opacity=0.7)
                except Exception:
                    pass

            # ----------------------------
            # 5) Append final row + save atomically
            # ----------------------------
            file_name_val = str(getattr(self, "file_path", ""))  # fallback if you don't have app_state

            row = [
                file_name_val,
                protein,
                brain_num,
                slide_date,
                "Yes" if bg_sub_answer == "Yes" else "No",
                str(chosen_layer_name),
            ]

            for nm in region_names:
                row.append(bg_values_for_row.get(nm, ""))

            for nm in region_names:
                row.append(results.get(nm, 0.0))

            ws.append(row)
            safe_save()

            # Atomic replace
            tmp_path.replace(out_path)

            QMessageBox.information(self, "ROI Extraction Complete", f"ROI intensities written to: {out_path}")

        except Exception as e:
            # Delete failed temp file
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            QMessageBox.critical(self, "Error", f"Failed ROI extraction (viewer):\n{e}")
            import traceback
            traceback.print_exc()

    def generate_max_projection(self):
        try:
            # -------- Build selectable items: use viewer layer names, but resolve stacks via _resolve_stack3d --------
            layer_names = [lyr.name for lyr in self.viewer.layers]
            if not layer_names:
                QMessageBox.warning(self, "No layers", "No layers found in viewer.")
                return

            # -------- Popup dialog with checklist --------
            dlg = QDialog(self)
            dlg.setWindowTitle("Max projection: select layers")
            dlg.setMinimumWidth(420)

            main = QVBoxLayout(dlg)
            main.addWidget(QLabel("Check layers to max-project (channels, masks, derived/coloc):"))

            lst = QListWidget()
            lst.setSelectionMode(QAbstractItemView.NoSelection)

            # Pre-check channels by default (optional)
            for nm in layer_names:
                it = QListWidgetItem(nm)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                default_checked = nm.startswith("Channel ") and nm[8:].isdigit()
                it.setCheckState(Qt.Checked if default_checked else Qt.Unchecked)
                lst.addItem(it)

            main.addWidget(lst)

            # Convenience buttons
            btn_row = QHBoxLayout()
            btn_all = QPushButton("Select all")
            btn_none = QPushButton("Select none")
            btn_row.addWidget(btn_all)
            btn_row.addWidget(btn_none)
            btn_row.addStretch(1)
            main.addLayout(btn_row)

            def _set_all(state):
                for i in range(lst.count()):
                    lst.item(i).setCheckState(state)

            btn_all.clicked.connect(lambda: _set_all(Qt.Checked))
            btn_none.clicked.connect(lambda: _set_all(Qt.Unchecked))

            # OK/Cancel
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            main.addWidget(buttons)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() != QDialog.Accepted:
                return

            selected = []
            for i in range(lst.count()):
                it = lst.item(i)
                if it.checkState() == Qt.Checked:
                    selected.append(it.text())

            if not selected:
                QMessageBox.information(self, "MaxProj", "No layers selected.")
                return

            # -------- Create projection viewer --------
            proj = napari.Viewer()
            proj.scale_bar.visible = True
            proj.scale_bar.unit = 'um'
            proj.scale_bar.font_size = 12
            proj.scale_bar.color = 'white'
            projections = {}
            proj_layers = []

            for name in selected:
                stack3 = self._resolve_stack3d(name)
                if stack3 is None:
                    # skip silently or warn; here we warn once per missing
                    QMessageBox.warning(self, "Missing data", f"Could not resolve full stack for: {name}")
                    continue

                arr = np.asarray(stack3)
                if arr.ndim == 3:
                    mp = arr.max(axis=0)
                elif arr.ndim == 2:
                    mp = arr
                else:
                    QMessageBox.warning(self, "Unsupported", f"{name} has unsupported dims: {arr.shape}")
                    continue

                # choose a colormap: reuse existing layer colormap if possible
                cmap = None
                try:
                    src_layer = self._get_layer_by_name(name)
                    cmap = getattr(src_layer, "colormap", None) or getattr(src_layer, "colormap_name", None)
                except Exception:
                    cmap = None

                # reasonable defaults
                if cmap is None:
                    cmap = "gray"

                try:
                    lay = proj.add_image(
                        mp,
                        name=f"{name}_max",
                        colormap=cmap,
                        blending="additive",
                        opacity=0.7,
                    )
                except Exception:
                    lay = proj.add_image(mp, name=f"{name}_max")

                proj_layers.append(lay)
                projections[name] = mp

            if not proj_layers:
                QMessageBox.information(self, "MaxProj", "No projections were created.")
                try:
                    proj.close()
                except Exception:
                    pass
                return

            # -------- Controls dock (screenshot + per-layer opacity + ROI) --------
            widget = QWidget()
            vlay = QVBoxLayout()

            ss_btn = QPushButton("Save Screenshot")
            ss_btn.clicked.connect(
                lambda _=False: self.save_screenshot(
                    viewer=proj,
                    parent=self,
                    default_name="max_projection.png",
                    canvas_only=True,
                )
            )
            vlay.addWidget(ss_btn)

            try:
                for lay in proj_layers:
                    vlay.addWidget(QLabel(lay.name))
                    sl = QSlider(Qt.Horizontal)
                    sl.setRange(0, 100)
                    sl.setValue(int(getattr(lay, "opacity", 0.7) * 100))
                    sl.valueChanged.connect(lambda val, L=lay: setattr(L, "opacity", val / 100.0))
                    vlay.addWidget(sl)
            except Exception:
                pass

            roi_btn = QPushButton("Extract ROI Intensities")
            roi_btn.clicked.connect(lambda: self.extract_roi_intensities_for_viewer(proj, projections))
            vlay.addWidget(roi_btn)

            widget.setLayout(vlay)
            proj.window.add_dock_widget(widget, area="right")

            QMessageBox.information(self, "MaxProj", f"Max projection viewer opened ({len(proj_layers)} layers).")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Max projection failed: {e}")

    def save_video(self):
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'MP4 Video (*.mp4)')
            if not save_path:
                return
            if not save_path.lower().endswith('.mp4'):
                save_path += '.mp4'
            frames = []
            for z in range(self.n_z):
                self.z_slider.setValue(z)
                QApplication.processEvents()
                frames.append(self.viewer.screenshot(canvas_only=True))
            writer = imageio.get_writer(save_path, format='ffmpeg', mode='I', fps=10, codec='libx264')
            for f in frames:
                writer.append_data(f)
            writer.close()
            QMessageBox.information(self, 'Saved', f'Video saved to {save_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save video: {e}')

    def save_screenshot(self, viewer=None, parent=None, default_name="screenshot.png", canvas_only=True):
        """
        One screenshot function for any napari Viewer.
        - viewer: napari.Viewer (defaults to self.viewer)
        - parent: QWidget for dialogs/message boxes (defaults to self)
        """
        try:
            viewer = viewer if viewer is not None else self.viewer
            parent = parent if parent is not None else self

            save_path, _ = QFileDialog.getSaveFileName(
                parent,
                "Save Screenshot",
                default_name,
                "PNG (*.png);;JPEG (*.jpg *.jpeg)"
            )
            if not save_path:
                return

            if not any(save_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                save_path += ".png"

            img = viewer.screenshot(canvas_only=canvas_only)  # returns RGBA ndarray [page:1]
            imageio.imwrite(save_path, img)

            QMessageBox.information(parent, "Saved", f"Screenshot saved to {save_path}")

        except Exception as e:
            QMessageBox.critical(parent if parent is not None else self, "Error", f"Failed to save screenshot: {e}")


###############################################################################
# ------------------------------ MAIN APP ------------------------------------
###############################################################################

def main():

    # Ensure QApplication exists (only one per process).
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # File selection dialog
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle("Select LIF or TIFF file")
    file_dialog.setNameFilters(["LIF files (*.lif)", "TIFF files (*.tif *.tiff)", "All files (*)"])
    file_dialog.setFileMode(QFileDialog.ExistingFile)

    if file_dialog.exec_():
        file_path = Path(file_dialog.selectedFiles()[0])
    else:
        print("No file selected.")
        return

    print(f"Loading: {file_path}")

    try:
        # Load (memmap)
        if file_path.suffix.lower() == ".lif":
            channel_mmaps, contrast_limits, voxel_size, temp_paths = load_lif_memmap(file_path, max_ch=4)
        else:
            channel_mmaps, contrast_limits, voxel_size, temp_paths = load_tiff_memmap(file_path, max_ch=4)

        print(f"Loaded {len(channel_mmaps)} channels (memmaps)")

        # Create napari viewer
        viewer = napari.Viewer()
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
        viewer.scale_bar.font_size = 12
        viewer.scale_bar.color = "white"

        # Add channel layers
        channel_layers = []
        default_colormaps = ["blue", "green", "red", "magenta"]
        for idx, ch in enumerate(channel_mmaps, start=1):
            name = f"Channel {idx}"
            cmap = default_colormaps[idx - 1] if idx - 1 < len(default_colormaps) else "gray"
            layer = viewer.add_image(ch, name=name, opacity=0.5, colormap=cmap, blending="additive")
            channel_layers.append(layer)

        # Dock widget
        widget = UIWidget(viewer, channel_mmaps, channel_layers, contrast_limits, file_path)
        widget._temp_paths = list(temp_paths) if temp_paths is not None else []
        try:
            widget.voxel_size = voxel_size
        except Exception:
            pass
        viewer.window.add_dock_widget(widget, area="right")

        # ---- QUIT / CLEANUP (no widget.cleanup) ----
        def _cleanup_and_stop_workers():
            # 1) remove temp memmap files
            for p in getattr(widget, "_temp_paths", []) or []:
                try:
                    os.remove(str(p))
                except Exception:
                    pass

            # 2) ask napari workers to quit; prevents hanging exit 
            try:
                from napari.qt.threading import WorkerBase
                WorkerBase.await_workers(msecs=2000)
            except Exception:
                pass

        try:
            app.aboutToQuit.connect(_cleanup_and_stop_workers)
        except Exception:
            pass

        # Run napari event loop; returns after last viewer closed.
        napari.run()

        # One more short wait for workers right before exiting 
        try:
            from napari.qt.threading import WorkerBase
            WorkerBase.await_workers(msecs=500)
        except Exception:
            pass

        sys.exit(0)
        import threading
        print([t.name for t in threading.enumerate()])

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading file: {e}")
        QMessageBox.critical(None, "Error", f"Failed to load file:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()