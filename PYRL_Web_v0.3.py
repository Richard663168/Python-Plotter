"""PYRL Web v0.3

Streamlit conversion of PYRL V9.2.

Supported presets in v0.3:
- UV-VIS
- UV-VIS (eV)
- XRD
- CD
- gCD
- PL
- CP-PL
- gCPPL
- CV
- EL
- EL Time
- EIS
- CIE
- EL Spectrum
- FTIR

The original Tkinter PYRL V9.2 file is not modified.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import EngFormatter, FuncFormatter, MultipleLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import streamlit as st


# =============================================================================
# Constants copied from / aligned with PYRL V9.2
# =============================================================================

PRESET_COLORS = [
    "#000000", "#C42238", "#066190", "#7DA494", "#f79059", "#9F8DB8",
    "#A8DDE1", "#75B5DC", "#478ECC", "#326DB6", "#2C4CA0", "#313772",
    "#FEE3CE", "#EABAA1", "#DC917B", "#D16D5B", "#C44438", "#B7282E",
    "#CFEADF", "#A4CBB7", "#81B095", "#669877", "#4D7E54", "#376439",
    "#EBCCE2", "#DDA1BE", "#C87D98", "#B25F79", "#9B3F5C", "#832440",
]

# Human-readable names for the exact V9.2 colors.  These labels are only UI
# helpers; the underlying hex values remain unchanged.
PALETTE_CHOICES = [
    ("Core — Black", "#000000"),
    ("Core — Red", "#C42238"),
    ("Core — Blue", "#066190"),
    ("Core — Sage", "#7DA494"),
    ("Core — Orange", "#f79059"),
    ("Core — Purple", "#9F8DB8"),
    ("Blue 1 — lightest", "#A8DDE1"),
    ("Blue 2", "#75B5DC"),
    ("Blue 3", "#478ECC"),
    ("Blue 4", "#326DB6"),
    ("Blue 5", "#2C4CA0"),
    ("Blue 6 — darkest", "#313772"),
    ("Warm 1 — lightest", "#FEE3CE"),
    ("Warm 2", "#EABAA1"),
    ("Warm 3", "#DC917B"),
    ("Warm 4", "#D16D5B"),
    ("Warm 5", "#C44438"),
    ("Warm 6 — darkest", "#B7282E"),
    ("Green 1 — lightest", "#CFEADF"),
    ("Green 2", "#A4CBB7"),
    ("Green 3", "#81B095"),
    ("Green 4", "#669877"),
    ("Green 5", "#4D7E54"),
    ("Green 6 — darkest", "#376439"),
    ("Magenta 1 — lightest", "#EBCCE2"),
    ("Magenta 2", "#DDA1BE"),
    ("Magenta 3", "#C87D98"),
    ("Magenta 4", "#B25F79"),
    ("Magenta 5", "#9B3F5C"),
    ("Magenta 6 — darkest", "#832440"),
]

PALETTE_NAME_TO_HEX = dict(PALETTE_CHOICES)
PALETTE_HEX_TO_NAME = {hex_value.lower(): name for name, hex_value in PALETTE_CHOICES}

EL_LABELS = [
    "Voltage (V)", "Current (mA)", "J (mA/cm²)", "cd", "L (cd/m²)",
    "Current Efficiency (cd/A)", "Luminous Flux (lm)", "Pₑ𝒻𝒻 (lm/W)",
    "x", "y", "u′", "v′", "Color Temp",
    "Dominant Wavelength (nm)", "Purity", "Peak Wavelength (nm)", "FWHM",
    "QE (%)", "Output Power (µW)", "Input Power (µW)", "Efficiency (%)",
    "Peak Counts", "Radiance (W/sr/m²)",
]

EL_TIME_LABELS = [
    "Time (s)", "Voltage (V)", "Current (mA)", "J (mA/cm²)", "cd", "L (cd/m²)",
    "Current Efficiency (cd/A)", "Luminous Flux (lm)", "Pₑ𝒻𝒻 (lm/W)",
    "x", "y", "u′", "v′", "Color Temp",
    "Dominant Wavelength (nm)", "Purity", "Peak Wavelength (nm)", "FWHM",
    "QE (%)", "Output Power (µW)", "Input Power (µW)", "Efficiency (%)",
    "Peak Counts",
]

EIS_LABELS = [
    "Index", "Frequency (Hz)", "Z' (Ω)", "-Z'' (Ω)",
    "Z (Ω)", "-Phase (°)", "Time (s)",
]

SIMPLE_PRESETS = {"UV-VIS", "UV-VIS (eV)", "XRD", "PL", "CV", "FTIR"}
TABLE_PRESETS = {"EL", "EL Time", "EIS"}
ALL_PRESETS = [
    "UV-VIS", "UV-VIS (eV)", "XRD", "CD", "gCD",
    "PL", "CP-PL", "gCPPL", "CV",
    "EL", "EL Time", "EIS", "EL Spectrum", "CIE", "FTIR",
]


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="PYRL Web",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PYRL Web")
st.caption("Browser-based PYRL plotting — v0.3")


# =============================================================================
# General data helpers
# =============================================================================

def _safe_name(value, fallback: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    text = str(value).strip()
    return text if text and text.lower() != "nan" else fallback


def clean_dataset_filename(filename: str) -> str:
    """Return the human-readable sample/device name from an instrument export.

    Examples
    --------
    B16LED1 CH1_251121134425_Table.txt -> B16LED1 CH1
    B16LED1 CH1_Table.txt              -> B16LED1 CH1

    The timestamp pattern is intentionally conservative so meaningful
    underscores elsewhere in a sample name are preserved.
    """
    stem = Path(filename).stem.strip()

    # Typical EL-system export: <name>_<long numeric timestamp>_Table
    cleaned = re.sub(r"_\d{8,20}_Table$", "", stem, flags=re.IGNORECASE)

    # Also tolerate exports that contain _Table without a timestamp.
    cleaned = re.sub(r"_Table$", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip() or stem


def _unique_name(name: str, existing: dict) -> str:
    if name not in existing:
        return name
    original = name
    suffix = 2
    while f"{original} ({suffix})" in existing:
        suffix += 1
    return f"{original} ({suffix})"


def optional_float(text: str):
    text = str(text).strip()
    return None if text == "" else float(text)


def to_numeric_pair(df: pd.DataFrame, x_label: str, y_label: str) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[x_label], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_label], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def to_decay_ratio_percent(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return arr
    y0 = arr[mask][0]
    if y0 == 0:
        return arr
    return (arr / y0) * 100.0


def decay_axis_label(label: str) -> str:
    label_map = {
        "L (cd/m²)": "Luminance Decay Ratio (%)",
        "cd": "Luminance Decay Ratio (%)",
        "Current (mA)": "Current Decay Ratio (%)",
        "J (mA/cm²)": "Current Density Decay Ratio (%)",
    }
    return label_map.get(label, f"{label} Decay Ratio (%)")


# =============================================================================
# Parsers — preserve V9.2 file conventions
# =============================================================================

def parse_uvvis(file_bytes: bytes, energy: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    raw = pd.read_csv(BytesIO(file_bytes), header=None, dtype=str)
    if raw.empty or raw.shape[1] < 2:
        raise ValueError("UV-Vis file must contain at least two columns.")

    sample_names = list(raw.iloc[0, ::2].values)
    data = raw.iloc[2:].reset_index(drop=True)
    x = pd.to_numeric(data.iloc[:, 0].astype(str).str.strip(), errors="coerce")

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, raw_name in enumerate(sample_names):
        y_col = i * 2 + 1
        if y_col >= data.shape[1]:
            continue
        y = pd.to_numeric(data.iloc[:, y_col], errors="coerce")
        mask = x.notna() & y.notna() & (x > 200)
        if not mask.any():
            continue
        x_vals = x[mask].to_numpy(dtype=float)
        if energy:
            x_vals = 1240.0 / x_vals
        y_vals = y[mask].to_numpy(dtype=float)
        name = _unique_name(_safe_name(raw_name, f"Sample {i + 1}"), result)
        result[name] = (x_vals, y_vals)

    if not result:
        raise ValueError("No valid UV-Vis datasets were found in the uploaded file.")
    return result


def parse_xrd(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(
        BytesIO(file_bytes), sep=r"\s+", skiprows=3, header=None, engine="python"
    )
    if df.shape[1] < 2:
        raise ValueError(f"{filename}: expected at least two columns after the 3-line header.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError(f"{filename}: no valid XRD data found.")
    return {
        clean_dataset_filename(filename): (
            x[mask].to_numpy(dtype=float),
            y[mask].to_numpy(dtype=float),
        )
    }


def parse_pl(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(BytesIO(file_bytes), header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{filename}: PL file must contain at least two columns.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    x_vals = x[mask].to_numpy(dtype=float)
    y_vals = y[mask].to_numpy(dtype=float)
    integer_mask = np.isclose(np.mod(x_vals, 1.0), 0.0)
    x_vals = x_vals[integer_mask]
    y_vals = y_vals[integer_mask]
    if x_vals.size == 0:
        raise ValueError(f"{filename}: no valid integer-wavelength PL points found.")
    return {clean_dataset_filename(filename): (x_vals, y_vals)}


def parse_cv(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Parse the CV convention used by V9.2: col 1 = x, col 3 = current ×1000."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(BytesIO(file_bytes), header=None)
    elif suffix == ".csv":
        df = pd.read_csv(BytesIO(file_bytes), header=None)
    else:
        raise ValueError(f"{filename}: unsupported CV format '{suffix}'.")
    if df.shape[1] < 3:
        raise ValueError(f"{filename}: CV file must contain at least three columns.")
    x = pd.to_numeric(df.iloc[1:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[1:, 2], errors="coerce") * 1000.0
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError(f"{filename}: no valid CV data found.")
    return {
        clean_dataset_filename(filename): (
            x[mask].to_numpy(dtype=float),
            y[mask].to_numpy(dtype=float),
        )
    }


def parse_ftir(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read V9.2 two-column FTIR .dpt files (wavenumber, absorbance)."""
    try:
        df = pd.read_csv(
            BytesIO(file_bytes), header=None, comment="#", sep=r"[,\s]+", engine="python"
        )
    except Exception:
        df = pd.read_csv(BytesIO(file_bytes), header=None, comment="#", sep=r"\s+", engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{filename}: FTIR file must contain at least two columns.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError(f"{filename}: no valid FTIR numeric data found.")
    stem = Path(filename).stem
    name = stem.split("_", 1)[1] if "_" in stem else stem
    return {
        name: (
            x[mask].to_numpy(dtype=float),
            y[mask].to_numpy(dtype=float),
        )
    }


def read_cd_xydata(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the XYDATA section of the CD CSV export used by V9.2."""
    df = pd.read_csv(BytesIO(file_bytes), header=None, usecols=[0, 1], on_bad_lines="skip")
    starts = df.index[df.iloc[:, 0].astype(str).str.strip().eq("XYDATA")]
    if len(starts) == 0:
        raise ValueError(f"{filename}: XYDATA row not found.")
    start = int(starts[0]) + 1
    end_candidates = df.index[(df.index > start) & df.iloc[:, 0].isna()]
    end = int(end_candidates.min()) if len(end_candidates) else len(df)
    data = df.iloc[start:end].copy()
    data.columns = ["X", "Y"]
    data["X"] = pd.to_numeric(data["X"], errors="coerce")
    data["Y"] = pd.to_numeric(data["Y"], errors="coerce")
    data = data.dropna(subset=["X", "Y"])
    if data.empty:
        raise ValueError(f"{filename}: no valid numeric CD data found after XYDATA.")
    return data["X"].to_numpy(dtype=float), data["Y"].to_numpy(dtype=float)


def is_cd_background(filename: str) -> bool:
    stem = Path(filename).stem.replace("-1", "")
    return stem.lower() == "bkg"


def parse_cd_raw_files(files) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Expose individual raw CD traces before V9.2 pair/background processing."""
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in files:
        if Path(f.name).suffix.lower() != ".csv" or is_cd_background(f.name):
            continue
        try:
            x, y = read_cd_xydata(f.getvalue(), f.name)
        except Exception:
            # gCD uploads may also contain a UV-VIS CSV; ignore non-CD CSVs here.
            continue
        raw_name = re.sub(r"-1$", "", clean_dataset_filename(f.name))
        name = _unique_name(raw_name, result)
        result[name] = (x, y)
    return result


def process_cd_files(files, identifier_mode: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Replicate V9.2 CD processing: mean of a two-file pair minus bkg.csv."""
    bkg_file = next((f for f in files if is_cd_background(f.name)), None)
    if bkg_file is None:
        raise ValueError("bkg.csv not found in uploaded files.")
    x_bkg, y_bkg = read_cd_xydata(bkg_file.getvalue(), bkg_file.name)

    sample_pairs: Dict[str, list] = {}
    for f in files:
        if is_cd_background(f.name):
            continue
        stem = Path(f.name).stem
        if stem.endswith("-1"):
            stem = stem[:-2]
        if not stem:
            continue
        sample_name = stem[:-1]  # exact V9.2 pairing convention
        sample_pairs.setdefault(sample_name, []).append(f)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []
    for sample_name, pair in sample_pairs.items():
        if len(pair) != 2:
            continue
        try:
            x1, y1 = read_cd_xydata(pair[0].getvalue(), pair[0].name)
            x2, y2 = read_cd_xydata(pair[1].getvalue(), pair[1].name)
        except Exception:
            continue
        if not np.array_equal(x1, x2):
            skipped.append(f"{pair[0].name} / {pair[1].name}: X-values mismatch")
            continue
        if not np.array_equal(x_bkg, x1):
            skipped.append(f"{sample_name}: background X-values do not match sample")
            continue
        y_processed = ((y1 + y2) / 2.0) - y_bkg
        out_name = sample_name
        if identifier_mode:
            match = re.search(r"B\d+F\d+", sample_name)
            if not match:
                continue
            out_name = match.group(0)
        result[out_name] = (x_bkg.copy(), y_processed)

    if not result:
        detail = f" ({'; '.join(skipped[:3])})" if skipped else ""
        raise ValueError("No valid two-file CD pairs could be processed." + detail)
    return result


def process_gcd_files(files, abs_threshold: float = 0.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Replicate V9.2 gCD = processed CD / (32980 × absorbance)."""
    cd_data = process_cd_files(files, identifier_mode=True)
    uvvis_file = next((f for f in files if "uv-vis" in f.name.lower()), None)
    if uvvis_file is None:
        raise ValueError("No UV-VIS file found; its filename must include 'UV-VIS'.")

    df = pd.read_csv(BytesIO(uvvis_file.getvalue()), header=None, dtype=str)
    sample_names = [df.iloc[0, i] for i in range(0, len(df.columns), 2)]
    data = df.iloc[2:].reset_index(drop=True)
    uv_x_series = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    valid_x = uv_x_series.notna() & (uv_x_series > 200)
    uv_x = uv_x_series[valid_x].to_numpy(dtype=float)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, raw_name in enumerate(sample_names):
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        identifier = raw_name.strip().split()[-1]
        if identifier not in cd_data:
            continue
        y_index = idx * 2 + 1
        if y_index >= data.shape[1]:
            continue
        y_uv_series = pd.to_numeric(data.iloc[:, y_index], errors="coerce")
        y_uv = y_uv_series[valid_x].to_numpy(dtype=float)
        if len(uv_x) != len(y_uv):
            continue

        cd_x, y_cd = cd_data[identifier]
        rounded_cd_x = np.round(cd_x, 1)
        rounded_uv_x = np.round(uv_x, 1)
        mask_cd = np.isin(rounded_cd_x, rounded_uv_x)
        mask_uv = np.isin(rounded_uv_x, rounded_cd_x)
        x_common = cd_x[mask_cd]
        aligned_cd_y = y_cd[mask_cd]
        aligned_uv_y = y_uv[mask_uv]
        if len(aligned_cd_y) != len(aligned_uv_y) or len(aligned_cd_y) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gcd_y = np.where(
                aligned_uv_y >= float(abs_threshold),
                aligned_cd_y / (32980.0 * aligned_uv_y),
                0.0,
            )
        result[raw_name] = (x_common, gcd_y)
    if not result:
        raise ValueError("No UV-VIS samples could be matched to processed CD identifiers for gCD.")
    return result


def parse_cppl_file(
    file_bytes: bytes, filename: str, x_column: int = 1, y_column: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Read user-selected 1-based X/Y columns exactly like V9.2 CP-PL."""
    if x_column < 1 or y_column < 1:
        raise ValueError(f"{filename}: CP-PL column numbers must be ≥ 1.")
    df = pd.read_csv(BytesIO(file_bytes), header=None)
    xi, yi = x_column - 1, y_column - 1
    if xi >= df.shape[1] or yi >= df.shape[1]:
        raise ValueError(
            f"{filename}: selected columns exceed file width ({df.shape[1]} columns)."
        )
    x = pd.to_numeric(df.iloc[:, xi], errors="coerce")
    y = pd.to_numeric(df.iloc[:, yi], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError(f"{filename}: no valid numeric CP-PL data found in selected columns.")
    return x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)


def cppl_group_pairs(files) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, Dict[str, object]] = {}
    for f in files:
        stem = Path(f.name).stem
        pre = stem.split("_")[0]
        hand = None
        if re.search(r"RCP", pre, re.IGNORECASE):
            hand = "RCP"
        if re.search(r"LCP", pre, re.IGNORECASE) and hand is None:
            hand = "LCP"
        if hand is None:
            continue
        root = re.sub(r"(RCP|LCP)$", "", pre, flags=re.IGNORECASE).strip() or pre
        groups.setdefault(root, {})[hand] = f
    return groups


def process_gcppl_files(files, column_settings: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pair LCP/RCP CP-PL scans and calculate 2(IL-IR)/(IL+IR), as in V9.2."""
    groups = cppl_group_pairs(files)
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for root, pair in groups.items():
        if "RCP" not in pair or "LCP" not in pair:
            continue
        fr_file = pair["RCP"]
        fl_file = pair["LCP"]
        xr_col, yr_col = column_settings.get(fr_file.name, (1, 2))
        xl_col, yl_col = column_settings.get(fl_file.name, (1, 2))
        try:
            x_r, y_r = parse_cppl_file(fr_file.getvalue(), fr_file.name, xr_col, yr_col)
            x_l, y_l = parse_cppl_file(fl_file.getvalue(), fl_file.name, xl_col, yl_col)
        except Exception:
            continue
        if len(x_r) < 2 or len(x_l) < 2:
            continue
        order_r = np.argsort(x_r)
        order_l = np.argsort(x_l)
        x_r, y_r = x_r[order_r], y_r[order_r]
        x_l, y_l = x_l[order_l], y_l[order_l]
        xmin = max(np.min(x_r), np.min(x_l))
        xmax = min(np.max(x_r), np.max(x_l))
        if xmax <= xmin:
            continue
        npts = max(len(x_r), len(x_l))
        x_common = np.linspace(xmin, xmax, npts)
        fr = interp1d(x_r, y_r, kind="linear", bounds_error=False, fill_value=np.nan)
        fl = interp1d(x_l, y_l, kind="linear", bounds_error=False, fill_value=np.nan)
        y_rcp = fr(x_common)
        y_lcp = fl(x_common)
        denom = y_lcp + y_rcp
        valid = np.isfinite(y_lcp) & np.isfinite(y_rcp) & (denom != 0)
        if not np.any(valid):
            continue
        result[root] = (
            x_common[valid],
            2.0 * (y_lcp[valid] - y_rcp[valid]) / denom[valid],
        )
    if not result:
        raise ValueError(
            "No valid LCP/RCP CP-PL pairs found. Filenames must contain LCP/RCP "
            "in the first underscore-separated block."
        )
    return result


def smooth_gcppl_curve(
    x: np.ndarray, y: np.ndarray, enabled: bool, window: int = 11, poly: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) < 5:
        return x, y
    window = max(int(window), int(poly) + 2)
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window <= poly or window < 3:
        return x, y
    try:
        return x, savgol_filter(y, window_length=window, polyorder=int(poly))
    except Exception:
        return x, y


def read_tabular_el(file_bytes: bytes, labels: list[str]) -> pd.DataFrame:
    """Read the EL/EL-Time table format used by V9.2.

    V9.2 ignores the first column and applies the fixed label list to the
    following columns.
    """
    df = pd.read_csv(BytesIO(file_bytes), delimiter="\t", skiprows=0)
    expected_cols = len(labels)
    if df.shape[1] < expected_cols + 1:
        raise ValueError(
            f"Expected at least {expected_cols + 1} columns including the leading index column; "
            f"found {df.shape[1]}."
        )
    out = df.iloc[:, 1:1 + expected_cols].copy()
    out.columns = labels
    return out


def parse_el_tables(files, time_mode: bool = False) -> Dict[str, pd.DataFrame]:
    labels = EL_TIME_LABELS if time_mode else EL_LABELS
    result: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = _unique_name(clean_dataset_filename(f.name), result)
        result[name] = read_tabular_el(f.getvalue(), labels)
    return result


def parse_eis_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
    if df.shape[1] < len(EIS_LABELS):
        raise ValueError(
            f"{filename}: expected at least {len(EIS_LABELS)} EIS columns; found {df.shape[1]}."
        )
    out = df.iloc[:, :len(EIS_LABELS)].copy()
    out.columns = EIS_LABELS
    return out


def parse_eis_files(files) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = _unique_name(clean_dataset_filename(f.name), result)
        result[name] = parse_eis_file(f.getvalue(), f.name)
    return result


def parse_el_spectrum(file_bytes: bytes) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], list[str]]:
    raw = pd.read_csv(BytesIO(file_bytes), header=None, delimiter="\t", dtype=str)
    if raw.empty or raw.shape[1] < 2:
        raise ValueError("EL Spectrum file must contain wavelength plus at least one spectrum column.")

    labels = [str(v).strip() for v in raw.iloc[0, 1:] if pd.notna(v)]
    data = raw.iloc[1:].reset_index(drop=True)
    x = pd.to_numeric(data.iloc[:, 0], errors="coerce")

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(labels, start=1):
        if i >= data.shape[1]:
            break
        y = pd.to_numeric(data.iloc[:, i], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            continue
        unique = _unique_name(name, result)
        result[unique] = (
            x[mask].to_numpy(dtype=float),
            y[mask].to_numpy(dtype=float),
        )

    if not result:
        raise ValueError("No valid EL spectra were found.")
    return result, list(result.keys())


def build_el_spectrum_metadata(table_bytes: bytes, spectrum_labels: list[str]) -> Dict[str, dict]:
    df = read_tabular_el(table_bytes, EL_LABELS)
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Voltage (V)"])
    if df.empty:
        raise ValueError("No valid voltage rows found in the EL table file.")

    voltages = df["Voltage (V)"].to_numpy(dtype=float)
    meta: Dict[str, dict] = {}
    for name in spectrum_labels:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(name))
        if not match:
            continue
        v_label = float(match.group(0))
        idx = int(np.argmin(np.abs(voltages - v_label)))
        row = df.iloc[idx]
        meta[name] = {col: row[col] for col in df.columns}
    return meta


def format_el_spectrum_legend(dataset_label: str, metric_name: str, metadata: Dict[str, dict]) -> str:
    if not metadata:
        return dataset_label

    key = dataset_label if dataset_label in metadata else None
    if key is None:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(dataset_label))
        if m:
            v_label = float(m.group(0))
            best_key = None
            best_diff = float("inf")
            for candidate in metadata:
                mc = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(candidate))
                if not mc:
                    continue
                diff = abs(float(mc.group(0)) - v_label)
                if diff < best_diff:
                    best_diff = diff
                    best_key = candidate
            key = best_key

    if key is None or metric_name not in metadata.get(key, {}):
        return dataset_label

    value = metadata[key][metric_name]
    try:
        value_float = float(value)
        text_val = f"{value_float:.3f}".rstrip("0").rstrip(".")
    except Exception:
        text_val = str(value)

    unit_match = re.search(r"\(([^)]+)\)", metric_name)
    if unit_match:
        return f"{text_val} {unit_match.group(1)}"
    return f"{metric_name} = {text_val}"


# =============================================================================
# CIE 1931 background / plotting
# =============================================================================

@st.cache_data(show_spinner=False)
def generate_cie_background(extent=(0.0, 0.8, 0.0, 0.9), res=650):
    """Generate the same true-CIE style background used by V9.2.

    Requires the `colour-science` package. If unavailable, the caller simply
    plots the CIE points without the background.
    """
    import colour

    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, int(res * (ymax - ymin) / (xmax - xmin)))
    xg, yg = np.meshgrid(xs, ys)

    eps = 1e-8
    valid = (yg > eps) & (xg >= 0) & (yg >= 0) & ((xg + yg) <= 1.0)

    X = np.zeros_like(xg)
    Y = np.ones_like(yg)
    Z = np.zeros_like(xg)
    X[valid] = xg[valid] / yg[valid]
    Z[valid] = (1.0 - xg[valid] - yg[valid]) / yg[valid]
    XYZ = np.stack([X, Y, Z], axis=-1)

    cs = colour.models.RGB_COLOURSPACES["sRGB"]
    RGB_lin = XYZ @ cs.matrix_XYZ_to_RGB.T
    RGB_lin = np.maximum(RGB_lin, 0.0)
    mx = np.max(RGB_lin, axis=-1, keepdims=True)
    scale = np.where(mx > 1.0, 1.0 / mx, 1.0)
    RGB_lin = RGB_lin * scale
    RGB = colour.models.eotf_inverse_sRGB(RGB_lin)
    RGB = np.clip(RGB, 0, 1)
    RGB[~valid] = 1.0

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(380, 780, 1)
    )
    xy_locus = colour.XYZ_to_xy(cmfs.values)
    poly_x = np.concatenate([xy_locus[:, 0], [xy_locus[0, 0]]])
    poly_y = np.concatenate([xy_locus[:, 1], [xy_locus[0, 1]]])

    p = MplPath(np.column_stack([poly_x, poly_y]))
    inside = p.contains_points(np.column_stack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    RGB[~inside] = 1.0
    return RGB, extent


def draw_cie_background(ax):
    try:
        bg, extent = generate_cie_background()
        ax.imshow(bg, extent=extent, origin="lower", aspect="auto", zorder=0)
    except Exception:
        # Keep the CIE plot usable even if colour-science failed to import.
        pass
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.001, 0.9)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, alpha=0.25)


def plot_cie_dataset(ax, df: pd.DataFrame, name: str, color: str):
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError(f"{name}: no valid CIE x/y values found.")

    ax.scatter(
        x, y, s=40, c=color, edgecolors="black", linewidths=0.5,
        label=name, zorder=10,
    )

    if "L (cd/m²)" in df.columns:
        L = pd.to_numeric(df["L (cd/m²)"], errors="coerce").to_numpy(dtype=float)
        L = L[mask]
        if np.isfinite(L).any():
            idx = int(np.nanargmax(L))
            ax.scatter([x[idx]], [y[idx]], marker="s", s=60, color=color, zorder=11)


# =============================================================================
# Plot utilities
# =============================================================================

def apply_formatter(axis, fmt_name: str, scale_name: str):
    if fmt_name == "scientific":
        if scale_name != "log":
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-3, 3))
            axis.set_major_formatter(fmt)
    elif fmt_name == "engineering":
        axis.set_major_formatter(EngFormatter())
    elif fmt_name == "percent":
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))


def default_axis_labels(preset: str):
    if preset == "UV-VIS":
        return "Wavelength (nm)", "Absorbance"
    if preset == "UV-VIS (eV)":
        return "Energy (eV)", "Absorbance"
    if preset == "XRD":
        return r"2θ (Degrees)", "Intensity (Counts)"
    if preset == "CD":
        return "Wavelength (nm)", "CD (mdeg)"
    if preset == "gCD":
        return "Wavelength (nm)", r"$g_{\mathrm{CD}}$"
    if preset == "PL":
        return "Wavelength (nm)", "Counts"
    if preset == "CP-PL":
        return "Wavelength (nm)", "Intensity (Counts)"
    if preset == "gCPPL":
        return "Wavelength (nm)", r"$g_{\mathrm{CPPL}}$"
    if preset == "CV":
        return "Potential (V)", "Current (mA)"
    if preset == "FTIR":
        return r"Wavenumber (cm$^{-1}$)", "Absorbance"
    if preset == "CIE":
        return "CIE x", "CIE y"
    if preset == "EL Spectrum":
        return "Wavelength (nm)", "Light Intensity (Counts)"
    return "X", "Y"


def minor_tick_geometry(size_index: int):
    if size_index == 0:
        return 0, 0
    if size_index == 1:
        return 2.0, 0.6
    if size_index == 2:
        return 3.5, 0.9
    return 5.0, 1.5


def build_figure(
    *,
    preset: str,
    selected: list[str],
    primary_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    secondary_datasets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cie_frames: Optional[Dict[str, pd.DataFrame]],
    styles: dict,
    plot_mode: str,
    title: str,
    fig_width: float,
    fig_height: float,
    x_min,
    x_max,
    y_min,
    y_max,
    y2_min,
    y2_max,
    x_scale: str,
    y_scale: str,
    y2_scale: str,
    x_format: str,
    y_format: str,
    y2_format: str,
    border_thickness: float,
    line_thickness: float,
    axis_fontsize: int,
    title_fontsize: int,
    legend_fontsize: int,
    legend_position: str,
    custom_text: str,
    text_fontsize: int,
    text_x: float,
    text_y: float,
    backgrounds: list[dict],
    x_label_override: Optional[str] = None,
    y_label_override: Optional[str] = None,
    y2_label_override: Optional[str] = None,
    el_spectrum_metadata: Optional[Dict[str, dict]] = None,
    el_spectrum_legend_metric: str = "Voltage (V)",
    el_spectrum_normalized: bool = False,
    el_spectrum_x_mode: str = "Wavelength (nm)",
    x_minor_size: int = 0,
    y_minor_size: int = 0,
    hide_y_tick_labels: bool = False,
):
    if plot_mode == "Stacked":
        fig, axs = plt.subplots(
            len(selected), 1, sharex=True,
            figsize=(fig_width, fig_height), squeeze=False,
        )
        axes = list(axs[:, 0])
        targets = list(zip(axes, selected))
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]
        targets = [(ax, name) for name in selected]

    xlabel, ylabel = default_axis_labels(preset)
    if x_label_override:
        xlabel = x_label_override
    if y_label_override:
        ylabel = y_label_override

    if preset == "EL Spectrum":
        xlabel = "Energy (eV)" if el_spectrum_x_mode == "Energy (eV)" else "Wavelength (nm)"
        if el_spectrum_normalized:
            ylabel = "Normalized EL Intensity (a.u.)"

    x_minor_len, x_minor_width = minor_tick_geometry(x_minor_size)
    y_minor_len, y_minor_width = minor_tick_geometry(y_minor_size)

    secondary_axes = []

    for ax, name in targets:
        style = styles[name]
        color = style["line_color"]
        marker = None if style["marker"] == "None" else style["marker"]

        if preset == "CIE":
            draw_cie_background(ax)
            plot_cie_dataset(ax, cie_frames[name], name, color)
        else:
            x, y = primary_datasets[name]
            if preset == "EL Spectrum" and el_spectrum_x_mode == "Energy (eV)":
                x = 1240.0 / np.asarray(x, dtype=float)
            if preset == "EL Spectrum" and el_spectrum_normalized:
                y = np.asarray(y, dtype=float)
                m = np.nanmax(y) if y.size else np.nan
                if np.isfinite(m) and m != 0:
                    y = y / m

            display_label = name
            if preset == "EL Spectrum":
                display_label = format_el_spectrum_legend(
                    name, el_spectrum_legend_metric, el_spectrum_metadata or {}
                )

            ax.plot(
                x, y,
                color=color,
                linestyle=style["line_style"],
                linewidth=line_thickness,
                marker=marker,
                markerfacecolor=style["marker_color"],
                markeredgecolor=style["marker_color"],
                label=display_label,
            )

            if secondary_datasets and name in secondary_datasets:
                x2, y2 = secondary_datasets[name]
                ax2 = ax.twinx()
                ax2.plot(
                    x2, y2,
                    color=color,
                    linestyle="--",
                    linewidth=line_thickness,
                    marker=marker,
                    markerfacecolor=style["marker_color"],
                    markeredgecolor=style["marker_color"],
                    label=f"{name} ({y2_label_override or 'Y2'})",
                )
                ax2.set_ylabel(y2_label_override or "Y2", fontsize=axis_fontsize)
                ax2.set_yscale(y2_scale)
                if y2_min is not None or y2_max is not None:
                    ax2.set_ylim(bottom=y2_min, top=y2_max)
                apply_formatter(ax2.yaxis, y2_format, y2_scale)
                ax2.minorticks_on()
                ax2.tick_params(
                    axis="y", which="major", direction="in", length=5, width=1.5,
                    labelsize=axis_fontsize, right=True,
                )
                ax2.tick_params(
                    axis="y", which="minor", direction="in",
                    length=y_minor_len, width=y_minor_width, right=True,
                )
                for spine in ax2.spines.values():
                    spine.set_linewidth(border_thickness)
                secondary_axes.append(ax2)

        for spine in ax.spines.values():
            spine.set_linewidth(border_thickness)
            spine.set_visible(True)

        if preset != "CIE":
            for bg in backgrounds:
                if bg["enabled"]:
                    ax.axvspan(
                        bg["xmin"], bg["xmax"], color=bg["color"],
                        alpha=bg["alpha"], zorder=0,
                    )

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        apply_formatter(ax.xaxis, x_format, x_scale)
        apply_formatter(ax.yaxis, y_format, y_scale)

        ax.minorticks_on()
        ax.tick_params(
            axis="both", which="major", direction="in",
            length=5, width=1.5, labelsize=axis_fontsize,
            top=False, right=False,
        )
        ax.tick_params(
            axis="x", which="minor", direction="in",
            length=x_minor_len, width=x_minor_width, top=False,
        )
        ax.tick_params(
            axis="y", which="minor", direction="in",
            length=y_minor_len, width=y_minor_width, right=False,
        )

        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        if hide_y_tick_labels:
            ax.tick_params(axis="y", which="both", labelleft=False)
        if plot_mode == "Stacked":
            ax.legend(loc=legend_position, frameon=False, fontsize=legend_fontsize)

    axes[-1].set_xlabel(xlabel, fontsize=axis_fontsize)

    if plot_mode == "Single":
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(
                handles, labels, loc=legend_position,
                frameon=False, fontsize=legend_fontsize,
            )
    else:
        for ax in axes:
            ax.set_ylabel("")
        fig.text(0.015, 0.5, ylabel, va="center", rotation="vertical", fontsize=axis_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.985)

    if custom_text.strip():
        fig.text(
            text_x, 1.0 - text_y, custom_text.strip(), fontsize=text_fontsize,
            ha="center", va="center",
        )

    fig.tight_layout(rect=[0.03 if plot_mode == "Stacked" else 0, 0, 1, 0.97])
    return fig


def figure_to_png(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", transparent=False)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# Sidebar — preset + upload
# =============================================================================

with st.sidebar:
    st.header("Data")
    preset = st.selectbox("Preset", ALL_PRESETS, index=0)

    spectrum_file = None
    spectrum_table_file = None

    if preset in ("UV-VIS", "UV-VIS (eV)"):
        uploaded_files = st.file_uploader(
            "Upload UV-Vis CSV", type=["csv"], accept_multiple_files=False,
            help="Uses the paired-column format from PYRL V9.2.",
        )
    elif preset == "XRD":
        uploaded_files = st.file_uploader(
            "Upload XRD files", type=["xy"], accept_multiple_files=True,
        )
    elif preset == "CD":
        uploaded_files = st.file_uploader(
            "Upload CD CSV files", type=["csv"], accept_multiple_files=True,
            help="Upload raw CD scans; include bkg.csv if you want pair/background processing.",
        )
    elif preset == "gCD":
        uploaded_files = st.file_uploader(
            "Upload CD + UV-VIS CSV files", type=["csv"], accept_multiple_files=True,
            help="Requires bkg.csv, paired CD scans, and a UV-VIS file whose filename contains 'UV-VIS'.",
        )
    elif preset == "PL":
        uploaded_files = st.file_uploader(
            "Upload PL CSV files", type=["csv"], accept_multiple_files=True,
        )
    elif preset in {"CP-PL", "gCPPL"}:
        uploaded_files = st.file_uploader(
            "Upload CP-PL CSV files", type=["csv"], accept_multiple_files=True,
            help="For gCPPL, LCP/RCP must appear in the first underscore-separated filename block.",
        )
    elif preset == "CV":
        uploaded_files = st.file_uploader(
            "Upload CV files", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
        )
    elif preset in {"EL", "EL Time", "CIE"}:
        uploaded_files = st.file_uploader(
            "Upload EL table files", type=["txt", "tsv"], accept_multiple_files=True,
            help="Tab-delimited EL table format used by the desktop PYRL.",
        )
    elif preset == "EIS":
        uploaded_files = st.file_uploader(
            "Upload EIS files", type=["csv", "txt", "xlsx", "xls"], accept_multiple_files=True,
        )
    elif preset == "EL Spectrum":
        spectrum_file = st.file_uploader(
            "EL spectrum file", type=["txt", "tsv"], accept_multiple_files=False,
            help="First column = wavelength; first row = spectrum labels/voltages.",
        )
        spectrum_table_file = st.file_uploader(
            "EL table file (recommended)", type=["txt", "tsv"], accept_multiple_files=False,
            help="Used to convert spectrum legend labels to Current, J, or L.",
        )
        uploaded_files = None
    elif preset == "FTIR":
        uploaded_files = st.file_uploader(
            "Upload FTIR .dpt files", type=["dpt"], accept_multiple_files=True,
            help="Two-column V9.2 format: wavenumber and absorbance.",
        )
    else:
        uploaded_files = None

    st.divider()
    plot_mode = st.radio("Plot mode", ["Single", "Stacked"], horizontal=True)


if uploaded_files is None:
    file_list = []
elif isinstance(uploaded_files, list):
    file_list = uploaded_files
else:
    file_list = [uploaded_files]


# =============================================================================
# Parse uploads
# =============================================================================

simple_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
table_frames: Dict[str, pd.DataFrame] = {}
el_spectrum_metadata: Dict[str, dict] = {}
parse_errors: list[str] = []

try:
    if preset in ("UV-VIS", "UV-VIS (eV)") and file_list:
        simple_datasets.update(
            parse_uvvis(file_list[0].getvalue(), energy=(preset == "UV-VIS (eV)"))
        )
    elif preset == "XRD":
        for f in file_list:
            try:
                simple_datasets.update(parse_xrd(f.getvalue(), f.name))
            except Exception as exc:
                parse_errors.append(str(exc))
    elif preset == "PL":
        for f in file_list:
            try:
                simple_datasets.update(parse_pl(f.getvalue(), f.name))
            except Exception as exc:
                parse_errors.append(str(exc))
    elif preset == "CD" and file_list:
        simple_datasets = parse_cd_raw_files(file_list)
    elif preset == "CV":
        for f in file_list:
            try:
                simple_datasets.update(parse_cv(f.getvalue(), f.name))
            except Exception as exc:
                parse_errors.append(str(exc))
    elif preset == "FTIR":
        for f in file_list:
            try:
                parsed = parse_ftir(f.getvalue(), f.name)
                for name, values in parsed.items():
                    simple_datasets[_unique_name(name, simple_datasets)] = values
            except Exception as exc:
                parse_errors.append(str(exc))
    elif preset == "EL" and file_list:
        table_frames = parse_el_tables(file_list, time_mode=False)
    elif preset == "EL Time" and file_list:
        table_frames = parse_el_tables(file_list, time_mode=True)
    elif preset == "CIE" and file_list:
        table_frames = parse_el_tables(file_list, time_mode=False)
    elif preset == "EIS" and file_list:
        table_frames = parse_eis_files(file_list)
    elif preset == "EL Spectrum" and spectrum_file is not None:
        simple_datasets, spectrum_labels = parse_el_spectrum(spectrum_file.getvalue())
        if spectrum_table_file is not None:
            try:
                el_spectrum_metadata = build_el_spectrum_metadata(
                    spectrum_table_file.getvalue(), spectrum_labels
                )
            except Exception as exc:
                parse_errors.append(f"EL Spectrum table metadata: {exc}")
except Exception as exc:
    parse_errors.append(str(exc))

for error in parse_errors:
    st.error(error)

if preset in TABLE_PRESETS or preset == "CIE":
    available_names = list(table_frames.keys())
else:
    available_names = list(simple_datasets.keys())


# =============================================================================
# Tabs
# =============================================================================

general_tab, text_tab, limit_tab, misc_tab, background_tab, giwaxs_tab = st.tabs(
    ["General", "Text", "Limit", "Misc", "Background", "GIWAXS"]
)

# Defaults shared across tabs.
selected: list[str] = []
styles: dict = {}
graph_title = ""
axis_x_choice = None
axis_y_choice = None
axis_y2_choice = "None"
el_time_decay = False
el_spectrum_x_mode = "Wavelength (nm)"
el_spectrum_normalize = False
el_spectrum_legend_metric = "Voltage (V)"
cd_use_processed = False
gcd_abs_threshold = 0.0
cppl_column_settings: dict[str, tuple[int, int]] = {}
gcppl_smooth = False
gcppl_window = 11
gcppl_poly = 2
ftir_to_trans = False
ftir_arb_units = False
hide_y_tick_labels = False

with general_tab:
    # Preset-specific processing controls that determine which datasets exist.
    if preset == "CD" and file_list:
        cd_use_processed = st.checkbox(
            "Process CD: average each two-scan pair and subtract bkg.csv",
            value=False,
            help="Matches the desktop Process behavior. Raw individual traces remain available when unchecked.",
        )
        if cd_use_processed:
            try:
                simple_datasets = process_cd_files(file_list, identifier_mode=False)
            except Exception as exc:
                simple_datasets = {}
                st.error(f"CD processing: {exc}")
        else:
            simple_datasets = parse_cd_raw_files(file_list)
        available_names = list(simple_datasets.keys())

    elif preset == "gCD" and file_list:
        gcd_abs_threshold = st.number_input(
            "Absorbance threshold",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f",
            help="V9.2 sets gCD to 0 wherever absorbance is below this threshold.",
        )
        try:
            simple_datasets = process_gcd_files(file_list, gcd_abs_threshold)
        except Exception as exc:
            simple_datasets = {}
            st.error(f"gCD processing: {exc}")
        available_names = list(simple_datasets.keys())

    elif preset in {"CP-PL", "gCPPL"} and file_list:
        st.subheader("CP-PL column mapping")
        st.caption("Column numbers are 1-based, matching V9.2. Defaults: X = 1, Y = 2.")
        map_cols = st.columns(2)
        for i, f in enumerate(file_list):
            with map_cols[i % 2].expander(clean_dataset_filename(f.name), expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    xcol = int(st.number_input(
                        "X column", min_value=1, value=1, step=1,
                        key=f"cppl_xcol_{preset}_{f.name}",
                    ))
                with c2:
                    ycol = int(st.number_input(
                        "Y column", min_value=1, value=2, step=1,
                        key=f"cppl_ycol_{preset}_{f.name}",
                    ))
                cppl_column_settings[f.name] = (xcol, ycol)

        if preset == "CP-PL":
            simple_datasets = {}
            for f in file_list:
                try:
                    xcol, ycol = cppl_column_settings.get(f.name, (1, 2))
                    x, y = parse_cppl_file(f.getvalue(), f.name, xcol, ycol)
                    name = _unique_name(clean_dataset_filename(f.name), simple_datasets)
                    simple_datasets[name] = (x, y)
                except Exception as exc:
                    st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                gcppl_smooth = st.checkbox("Smooth gCPPL", value=False)
            with c2:
                gcppl_window = int(st.number_input(
                    "Window length", min_value=3, value=11, step=2,
                    help="Savitzky–Golay window; adjusted to a valid odd value when needed.",
                ))
            with c3:
                gcppl_poly = int(st.number_input(
                    "Polynomial order", min_value=1, value=2, step=1,
                ))
            try:
                simple_datasets = process_gcppl_files(file_list, cppl_column_settings)
            except Exception as exc:
                simple_datasets = {}
                st.error(f"gCPPL processing: {exc}")
        available_names = list(simple_datasets.keys())

    elif preset == "FTIR" and file_list:
        c1, c2 = st.columns(2)
        with c1:
            ftir_to_trans = st.checkbox(
                "Convert Absorbance → Transmission (T = 10⁻ᴬ)", value=False
            )
        with c2:
            ftir_arb_units = st.checkbox(
                "Arbitrary units (normalize + hide y numbers)", value=False
            )
        available_names = list(simple_datasets.keys())

    # Re-evaluate names after any processing above.
    if preset in TABLE_PRESETS or preset == "CIE":
        available_names = list(table_frames.keys())
    elif preset not in {"CD", "gCD", "CP-PL", "gCPPL"}:
        available_names = list(simple_datasets.keys())

    if not available_names:
        st.info("Choose a preset and upload compatible files to begin.")
    else:
        dataset_key_suffix = "processed" if preset == "CD" and cd_use_processed else "default"
        selected = st.multiselect(
            "Datasets", available_names, default=available_names,
            key=f"datasets_{preset}_{dataset_key_suffix}",
        )

        if preset == "EL":
            st.subheader("EL Axis Selector")
            c1, c2, c3 = st.columns(3)
            with c1:
                axis_x_choice = st.selectbox("X Axis", EL_LABELS, index=0, key="el_x_axis")
            with c2:
                axis_y_choice = st.selectbox(
                    "Y Axis", EL_LABELS, index=EL_LABELS.index("L (cd/m²)"), key="el_y_axis"
                )
            with c3:
                axis_y2_choice = st.selectbox(
                    "Y2 Axis", ["None"] + EL_LABELS, index=0, key="el_y2_axis"
                )

        elif preset == "EL Time":
            st.subheader("EL Time Axis Selector")
            c1, c2, c3 = st.columns(3)
            with c1:
                axis_x_choice = st.selectbox(
                    "X Axis", EL_TIME_LABELS, index=EL_TIME_LABELS.index("Time (s)"), key="elt_x_axis"
                )
            with c2:
                axis_y_choice = st.selectbox(
                    "Y Axis", EL_TIME_LABELS, index=EL_TIME_LABELS.index("L (cd/m²)"), key="elt_y_axis"
                )
            with c3:
                axis_y2_choice = st.selectbox(
                    "Y2 Axis", ["None"] + EL_TIME_LABELS, index=0, key="elt_y2_axis"
                )
            el_time_decay = st.checkbox(
                "Decay ratio (%)", value=False,
                help="Normalizes each selected Y trace to its first finite value × 100, matching V9.2.",
            )

        elif preset == "EIS":
            st.subheader("EIS Axis Selector")
            c1, c2, c3 = st.columns(3)
            with c1:
                axis_x_choice = st.selectbox(
                    "X Axis", EIS_LABELS, index=EIS_LABELS.index("Z' (Ω)"), key="eis_x_axis"
                )
            with c2:
                axis_y_choice = st.selectbox(
                    "Y Axis", EIS_LABELS, index=EIS_LABELS.index("-Z'' (Ω)"), key="eis_y_axis"
                )
            with c3:
                axis_y2_choice = st.selectbox(
                    "Y2 Axis", ["None"] + EIS_LABELS, index=0, key="eis_y2_axis"
                )

        elif preset == "EL Spectrum":
            st.subheader("EL Spectrum Options")
            c1, c2, c3 = st.columns(3)
            with c1:
                el_spectrum_legend_metric = st.selectbox(
                    "Legend value",
                    ["Voltage (V)", "Current (mA)", "J (mA/cm²)", "L (cd/m²)"],
                    index=0,
                )
            with c2:
                el_spectrum_x_mode = st.selectbox(
                    "X-axis", ["Wavelength (nm)", "Energy (eV)"], index=0
                )
            with c3:
                el_spectrum_normalize = st.checkbox("Normalize EL intensity", value=False)
            if el_spectrum_legend_metric != "Voltage (V)" and not el_spectrum_metadata:
                st.warning("Upload the matching EL table file to convert legend values to Current, J, or L.")

        # Title default mirrors V9.2's last selected filename behavior where possible.
        if preset == "EL Spectrum" and spectrum_file is not None:
            default_title = clean_dataset_filename(spectrum_file.name)
        elif file_list:
            default_title = clean_dataset_filename(file_list[-1].name)
        else:
            default_title = ""
        graph_title = st.text_input("Graph title", value=default_title)

        # Exact V9.2 palette, labeled so no hex memorization is needed.
        st.subheader("Line style & PYRL color palette")
        palette_groups = [
            ("Core", PALETTE_CHOICES[0:6]),
            ("Blues", PALETTE_CHOICES[6:12]),
            ("Warm", PALETTE_CHOICES[12:18]),
            ("Greens", PALETTE_CHOICES[18:24]),
            ("Magentas", PALETTE_CHOICES[24:30]),
        ]
        guide_parts = [
            '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;'
            'margin:2px 0 6px 0;width:100%">'
        ]
        for group_name, group in palette_groups:
            guide_parts.append(
                '<div style="flex:1 1 160px;min-width:150px">'
                f'<div style="font-size:0.82rem;font-weight:600;margin:0 0 4px 0">{group_name}</div>'
                '<div style="display:grid;grid-template-columns:repeat(6,minmax(20px,1fr));gap:4px">'
            )
            for idx, (label, hex_value) in enumerate(group, start=1):
                short_label = label.replace("Core — ", "") if group_name == "Core" else str(idx)
                guide_parts.append(
                    f'<div title="{label}" style="text-align:center;min-width:0">'
                    f'<div style="height:22px;background:{hex_value};border:1px solid #999;border-radius:3px"></div>'
                    f'<div style="font-size:0.60rem;color:#666;white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis;margin-top:1px">{short_label}</div></div>'
                )
            guide_parts.append('</div></div>')
        guide_parts.append('</div>')
        st.markdown("".join(guide_parts), unsafe_allow_html=True)
        st.caption("Datasets are automatically assigned sequential colors. Choose colors by name; hex codes stay hidden unless you use Custom.")

        if selected:
            cols = st.columns(2)
            palette_labels = [name for name, _ in PALETTE_CHOICES]
            for i, name in enumerate(selected):
                default_color = PRESET_COLORS[i % len(PRESET_COLORS)]
                default_palette_name = PALETTE_HEX_TO_NAME[default_color.lower()]
                with cols[i % 2].expander(name, expanded=False):
                    palette_choice = st.selectbox(
                        "Preset color", palette_labels + ["Custom…"],
                        index=palette_labels.index(default_palette_name),
                        key=f"palette_named_{preset}_{name}",
                    )
                    if palette_choice == "Custom…":
                        line_color = st.color_picker(
                            "Custom line color", default_color,
                            key=f"line_color_{preset}_{name}",
                        )
                    else:
                        line_color = PALETTE_NAME_TO_HEX[palette_choice]
                        st.markdown(
                            f'<div style="height:20px;width:100%;background:{line_color};border:1px solid #aaa;'
                            'border-radius:3px"></div>', unsafe_allow_html=True,
                        )
                    line_style = st.selectbox(
                        "Line style", ["-", "--", ":", "-."], index=0,
                        key=f"line_style_{preset}_{name}",
                    )
                    marker = st.selectbox(
                        "Marker", ["None", "o", "s", "^", "v", "D", "x", "+"], index=0,
                        key=f"marker_{preset}_{name}",
                    )
                    marker_follow = st.checkbox(
                        "Marker color follows line", value=True,
                        key=f"marker_follow_{preset}_{name}",
                    )
                    marker_color = line_color if marker_follow else st.color_picker(
                        "Marker color", line_color, key=f"marker_color_{preset}_{name}"
                    )
                    styles[name] = {
                        "line_color": line_color,
                        "line_style": line_style,
                        "marker": marker,
                        "marker_color": marker_color,
                    }

with text_tab:
    custom_text = st.text_input("Custom text", value="")
    text_fontsize = st.number_input("Font size", min_value=8, max_value=36, value=12, step=1)
    text_x = st.slider("Horizontal position", 0.0, 1.0, 0.5, 0.01)
    text_y = st.slider(
        "Vertical position", 0.0, 1.0, 0.5, 0.01,
        help="Uses the same top-to-bottom convention as your Tkinter text canvas.",
    )

with limit_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_min_text = st.text_input("X Min", value="")
        x_max_text = st.text_input("X Max", value="")
    with c2:
        y_min_text = st.text_input("Y Min", value="")
        y_max_text = st.text_input("Y Max", value="")
    with c3:
        y2_min_text = st.text_input("Y2 Min", value="")
        y2_max_text = st.text_input("Y2 Max", value="")

with misc_tab:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_scale = st.selectbox("X-axis scale", ["linear", "log"], index=0)
        x_format = st.selectbox(
            "X-axis format", ["number", "scientific", "engineering", "percent"], index=0
        )
        x_minor_size = st.selectbox("Minor Tick Size (X)", [0, 1, 2, 3], index=0)
        border_thickness = st.number_input("Border thickness", 0.5, 5.0, 1.0, 0.1)
    with c2:
        y_scale = st.selectbox("Y1-axis scale", ["linear", "log"], index=0)
        y_format = st.selectbox(
            "Y1-axis format", ["number", "scientific", "engineering", "percent"], index=0
        )
        y_minor_size = st.selectbox("Minor Tick Size (Y)", [0, 1, 2, 3], index=0)
        line_thickness = st.number_input("Line thickness", 0.1, 10.0, 2.0, 0.1)
    with c3:
        y2_scale = st.selectbox("Y2-axis scale", ["linear", "log"], index=0)
        y2_format = st.selectbox(
            "Y2-axis format", ["number", "scientific", "engineering", "percent"], index=0
        )
        axis_fontsize = st.number_input("Axis/tick font size", 6, 36, 16, 1)
        title_fontsize = st.number_input("Title font size", 6, 36, 18, 1)
    with c4:
        fig_width = st.number_input("Figure width (in)", 2.0, 20.0, 6.7, 0.5)
        fig_height = st.number_input("Figure height (in)", 2.0, 20.0, 5.5, 0.5)
        legend_fontsize = st.number_input("Legend font size", 8, 30, 16, 1)
        legend_position = st.selectbox(
            "Legend position",
            ["upper right", "lower right", "upper left", "lower left", "best"],
            index=0,
        )

with background_tab:
    st.caption("Up to three optional x-range highlight blocks.")
    backgrounds = []
    for i in range(3):
        with st.expander(f"Background block {i + 1}", expanded=False):
            enabled = st.checkbox("Enable", value=False, key=f"bg_enable_{i}")
            c1, c2, c3 = st.columns(3)
            with c1:
                bg_xmin = st.number_input("X min", value=0.0, key=f"bg_xmin_{i}")
            with c2:
                bg_xmax = st.number_input("X max", value=1.0, key=f"bg_xmax_{i}")
            with c3:
                bg_color = st.color_picker("Color", "#D9D9D9", key=f"bg_color_{i}")
            bg_alpha = st.slider(
                "Transparency", 0.0, 1.0, 0.2, 0.05, key=f"bg_alpha_{i}"
            )
            backgrounds.append(
                {
                    "enabled": enabled,
                    "xmin": bg_xmin,
                    "xmax": bg_xmax,
                    "color": bg_color,
                    "alpha": bg_alpha,
                }
            )

with giwaxs_tab:
    st.info(
        "GIWAXS remains reserved for the later migration stage. "
        "v0.3 adds spectroscopy/processing presets while GIWAXS remains reserved for its dedicated migration stage."
    )


# =============================================================================
# Build selected x/y data from parsed tables
# =============================================================================

primary_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
secondary_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
cie_frames: Optional[Dict[str, pd.DataFrame]] = None
x_label_override = None
y_label_override = None
y2_label_override = None

if selected:
    if preset in SIMPLE_PRESETS or preset in {"CD", "gCD", "CP-PL", "gCPPL", "EL Spectrum"}:
        primary_datasets = {name: simple_datasets[name] for name in selected}

        if preset == "gCPPL" and gcppl_smooth:
            primary_datasets = {
                name: smooth_gcppl_curve(x, y, True, gcppl_window, gcppl_poly)
                for name, (x, y) in primary_datasets.items()
            }

        if preset == "FTIR":
            transformed = {}
            for name, (x, y) in primary_datasets.items():
                arr = np.asarray(y, dtype=float)
                if ftir_to_trans:
                    arr = np.power(10.0, -arr)
                if ftir_arb_units:
                    m = np.nanmax(np.abs(arr)) if arr.size else np.nan
                    if np.isfinite(m) and m != 0:
                        arr = arr / m
                transformed[name] = (np.asarray(x, dtype=float), arr)
            primary_datasets = transformed
            y_label_override = "Transmission" if ftir_to_trans else "Absorbance"
            if ftir_arb_units:
                y_label_override += " (arb. unit.)"
                hide_y_tick_labels = True

    elif preset in TABLE_PRESETS:
        if axis_x_choice and axis_y_choice:
            for name in selected:
                df = table_frames[name]
                x, y = to_numeric_pair(df, axis_x_choice, axis_y_choice)
                if preset == "EL Time" and el_time_decay:
                    y = to_decay_ratio_percent(y)
                primary_datasets[name] = (x, y)

                if axis_y2_choice not in (None, "None", ""):
                    x2, y2 = to_numeric_pair(df, axis_x_choice, axis_y2_choice)
                    if preset == "EL Time" and el_time_decay:
                        y2 = to_decay_ratio_percent(y2)
                    secondary_datasets[name] = (x2, y2)

            x_label_override = axis_x_choice
            y_label_override = (
                decay_axis_label(axis_y_choice)
                if preset == "EL Time" and el_time_decay
                else axis_y_choice
            )
            if axis_y2_choice not in (None, "None", ""):
                y2_label_override = (
                    decay_axis_label(axis_y2_choice)
                    if preset == "EL Time" and el_time_decay
                    else axis_y2_choice
                )

    elif preset == "CIE":
        cie_frames = {name: table_frames[name] for name in selected}


# =============================================================================
# Plot preview + download
# =============================================================================

st.divider()
st.subheader("Plot preview")

if available_names and selected:
    try:
        x_min = optional_float(x_min_text)
        x_max = optional_float(x_max_text)
        y_min = optional_float(y_min_text)
        y_max = optional_float(y_max_text)
        y2_min = optional_float(y2_min_text)
        y2_max = optional_float(y2_max_text)

        fig = build_figure(
            preset=preset,
            selected=selected,
            primary_datasets=primary_datasets,
            secondary_datasets=secondary_datasets or None,
            cie_frames=cie_frames,
            styles=styles,
            plot_mode=plot_mode,
            title=graph_title,
            fig_width=float(fig_width),
            fig_height=float(fig_height),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            y2_min=y2_min,
            y2_max=y2_max,
            x_scale=x_scale,
            y_scale=y_scale,
            y2_scale=y2_scale,
            x_format=x_format,
            y_format=y_format,
            y2_format=y2_format,
            border_thickness=float(border_thickness),
            line_thickness=float(line_thickness),
            axis_fontsize=int(axis_fontsize),
            title_fontsize=int(title_fontsize),
            legend_fontsize=int(legend_fontsize),
            legend_position=legend_position,
            custom_text=custom_text,
            text_fontsize=int(text_fontsize),
            text_x=float(text_x),
            text_y=float(text_y),
            backgrounds=backgrounds,
            x_label_override=x_label_override,
            y_label_override=y_label_override,
            y2_label_override=y2_label_override,
            el_spectrum_metadata=el_spectrum_metadata,
            el_spectrum_legend_metric=el_spectrum_legend_metric,
            el_spectrum_normalized=el_spectrum_normalize,
            el_spectrum_x_mode=el_spectrum_x_mode,
            x_minor_size=int(x_minor_size),
            y_minor_size=int(y_minor_size),
            hide_y_tick_labels=hide_y_tick_labels,
        )
        st.pyplot(fig, clear_figure=False, use_container_width=False)
        png = figure_to_png(fig)
        download_name = f"{graph_title.strip() or 'exported_plot'}.png"
        st.download_button(
            "Download PNG (300 dpi)",
            data=png,
            file_name=download_name,
            mime="image/png",
            type="primary",
        )
        plt.close(fig)
    except Exception as exc:
        st.error(f"Could not build plot: {exc}")
elif available_names:
    st.warning("Select at least one dataset in the General tab.")
else:
    st.caption("Your plot will appear here after you upload data.")


with st.expander("v0.3 compatibility notes"):
    st.markdown(
        """
- **UV-VIS / UV-VIS (eV):** paired-column V9.2 layout; filters wavelength ≤ 200 nm.
- **XRD:** `.xy`, first 3 rows skipped, first 2 numeric columns plotted.
- **CD:** reads the `XYDATA` section. Optional processed mode averages each two-scan pair and subtracts `bkg.csv`, following V9.2's filename pairing convention.
- **gCD:** runs the V9.2 CD pair/background processing, matches `B#F#` identifiers to a `UV-VIS` file, then calculates CD/(32980×Abs); absorbance threshold is adjustable.
- **PL:** first 2 CSV columns; integer x/wavelength points retained.
- **CP-PL:** per-file 1-based X/Y column selectors, matching the desktop behavior.
- **gCPPL:** pairs LCP/RCP from the first underscore-separated filename block and calculates `2(IL−IR)/(IL+IR)` on an interpolated common x-grid; optional Savitzky–Golay smoothing is available.
- **CV:** CSV/XLS/XLSX; V9.2 uses column 1 for x and column 3 ×1000 for current.
- **EL:** fixed V9.2 EL column map, arbitrary X/Y/Y2 selection, dual Y-axis plotting.
- **EL Time:** same table convention plus optional first-point decay-ratio normalization.
- **EIS:** first 7 columns mapped to the V9.2 EIS labels; CSV/TXT/XLS/XLSX supported.
- **CIE:** EL-table x/y chromaticity points, CIE 1931 background, and square marker at max luminance.
- **EL Spectrum:** wavelength/eV x-axis, normalization, and Voltage/Current/J/L legend conversion using the matching EL table.
- **FTIR:** two-column `.dpt`; optional absorbance→transmission conversion and arbitrary-unit normalization/hide-y-number behavior.
- **Colors:** exact 30-swatch V9.2 palette remains integrated in the compact horizontal guide.
- **Still to migrate:** CP-EL/gEL, TR CP-EL, TRPL, and GIWAXS.
        """
    )
