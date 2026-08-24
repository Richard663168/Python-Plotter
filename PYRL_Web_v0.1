"""PYRL Web v0.1

Initial Streamlit conversion of PYRL V9.2.
Supported presets in this first web build:
- UV-VIS
- UV-VIS (eV)
- XRD
- PL

The original Tkinter application is not modified.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PYRL Web",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PYRL Web")
st.caption("Browser-based plotting for your spectroscopy / diffraction data — v0.1")


# -----------------------------------------------------------------------------
# Data parsing — extracted from the behavior of PYRL V9.2
# -----------------------------------------------------------------------------
def _safe_name(value, fallback: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    text = str(value).strip()
    return text if text and text.lower() != "nan" else fallback


def parse_uvvis(file_bytes: bytes, energy: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Parse the paired-column UV-Vis format used by PYRL V9.2.

    V9.2 behavior retained:
    - first row contains sample names in columns A, C, E, ...
    - first two rows are metadata/header rows
    - wavelength is taken from column A
    - rows at wavelength <= 200 nm are excluded
    - each sample's y column is B, D, F, ...
    - optional wavelength -> energy conversion uses E = 1240 / wavelength
    """
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

        name = _safe_name(raw_name, f"Sample {i + 1}")
        # Avoid accidental duplicate labels in the web UI.
        original = name
        suffix = 2
        while name in result:
            name = f"{original} ({suffix})"
            suffix += 1
        result[name] = (x_vals, y_vals)

    if not result:
        raise ValueError("No valid UV-Vis datasets were found in the uploaded file.")
    return result


def parse_xrd(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Parse PYRL V9.2 XRD .xy files: whitespace-delimited, skip first 3 rows."""
    df = pd.read_csv(
        BytesIO(file_bytes),
        sep=r"\s+",
        skiprows=3,
        header=None,
        engine="python",
    )
    if df.shape[1] < 2:
        raise ValueError(f"{filename}: expected at least two numeric columns after the 3-line header.")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError(f"{filename}: no valid XRD data found.")

    name = Path(filename).stem
    return {name: (x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))}


def parse_pl(file_bytes: bytes, filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Parse PYRL V9.2 PL CSV files.

    V9.2 behavior retained:
    - use first two columns
    - coerce non-numeric values to NaN and drop them
    - keep only rows whose x/wavelength value is an integer
    """
    df = pd.read_csv(BytesIO(file_bytes), header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{filename}: PL file must contain at least two columns.")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    integer_mask = np.isclose(np.mod(x.to_numpy(dtype=float), 1.0), 0.0)
    x_vals = x.to_numpy(dtype=float)[integer_mask]
    y_vals = y.to_numpy(dtype=float)[integer_mask]

    if x_vals.size == 0:
        raise ValueError(f"{filename}: no valid integer-wavelength PL points found.")

    name = Path(filename).stem
    return {name: (x_vals, y_vals)}


# -----------------------------------------------------------------------------
# Plot utilities
# -----------------------------------------------------------------------------
def optional_float(text: str):
    text = str(text).strip()
    if text == "":
        return None
    return float(text)


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
    if preset == "PL":
        return "Wavelength (nm)", "Counts"
    return "X", "Y"


def build_figure(
    datasets,
    selected,
    preset,
    styles,
    plot_mode,
    title,
    fig_width,
    fig_height,
    x_min,
    x_max,
    y_min,
    y_max,
    x_scale,
    y_scale,
    x_format,
    y_format,
    border_thickness,
    line_thickness,
    axis_fontsize,
    title_fontsize,
    legend_fontsize,
    legend_position,
    custom_text,
    text_fontsize,
    text_x,
    text_y,
    backgrounds,
):
    if plot_mode == "Stacked":
        fig, axs = plt.subplots(
            len(selected),
            1,
            sharex=True,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        axes = list(axs[:, 0])
        targets = list(zip(axes, selected))
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]
        targets = [(ax, name) for name in selected]

    xlabel, ylabel = default_axis_labels(preset)

    for ax, name in targets:
        x, y = datasets[name]
        style = styles[name]
        marker = None if style["marker"] == "None" else style["marker"]

        ax.plot(
            x,
            y,
            color=style["line_color"],
            linestyle=style["line_style"],
            linewidth=line_thickness,
            marker=marker,
            markerfacecolor=style["marker_color"],
            markeredgecolor=style["marker_color"],
            label=name,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(border_thickness)
            spine.set_visible(True)

        for bg in backgrounds:
            if bg["enabled"]:
                ax.axvspan(bg["xmin"], bg["xmax"], color=bg["color"], alpha=bg["alpha"], zorder=0)

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        apply_formatter(ax.xaxis, x_format, x_scale)
        apply_formatter(ax.yaxis, y_format, y_scale)

        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.5, labelsize=axis_fontsize)
        ax.tick_params(axis="both", which="minor", direction="in", length=0, width=0)

        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        if plot_mode == "Stacked":
            ax.legend(loc=legend_position, frameon=False, fontsize=legend_fontsize)

    axes[-1].set_xlabel(xlabel, fontsize=axis_fontsize)

    if plot_mode == "Single":
        axes[0].legend(loc=legend_position, frameon=False, fontsize=legend_fontsize)
    else:
        # Use one shared y-label, matching the desktop app's stacked-plot intent.
        for ax in axes:
            ax.set_ylabel("")
        fig.text(0.015, 0.5, ylabel, va="center", rotation="vertical", fontsize=axis_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.985)

    if custom_text.strip():
        fig.text(
            text_x,
            1.0 - text_y,
            custom_text.strip(),
            fontsize=text_fontsize,
            ha="center",
            va="center",
        )

    fig.tight_layout(rect=[0.03 if plot_mode == "Stacked" else 0, 0, 1, 0.97])
    return fig


def figure_to_png(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", transparent=False)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Sidebar: files + preset
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    preset = st.selectbox(
        "Preset",
        ["UV-VIS", "UV-VIS (eV)", "XRD", "PL"],
        index=0,
    )

    allowed = ["csv"] if preset in ("UV-VIS", "UV-VIS (eV)", "PL") else ["xy"]
    multiple = preset not in ("UV-VIS", "UV-VIS (eV)")
    uploaded_files = st.file_uploader(
        "Upload data file" if not multiple else "Upload data files",
        type=allowed,
        accept_multiple_files=multiple,
        help=(
            "UV-Vis uses the first uploaded CSV, matching PYRL V9.2."
            if preset.startswith("UV-VIS")
            else None
        ),
    )

    st.divider()
    plot_mode = st.radio("Plot mode", ["Single", "Stacked"], horizontal=True)


# Normalize uploader return type.
if uploaded_files is None:
    file_list = []
elif isinstance(uploaded_files, list):
    file_list = uploaded_files
else:
    file_list = [uploaded_files]


# -----------------------------------------------------------------------------
# Parse uploaded data
# -----------------------------------------------------------------------------
datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
parse_errors = []

if file_list:
    try:
        if preset in ("UV-VIS", "UV-VIS (eV)"):
            datasets.update(parse_uvvis(file_list[0].getvalue(), energy=(preset == "UV-VIS (eV)")))
        elif preset == "XRD":
            for f in file_list:
                try:
                    datasets.update(parse_xrd(f.getvalue(), f.name))
                except Exception as exc:
                    parse_errors.append(str(exc))
        elif preset == "PL":
            for f in file_list:
                try:
                    datasets.update(parse_pl(f.getvalue(), f.name))
                except Exception as exc:
                    parse_errors.append(str(exc))
    except Exception as exc:
        parse_errors.append(str(exc))

for error in parse_errors:
    st.error(error)


# -----------------------------------------------------------------------------
# Main tabs — mirrors the desktop app's organization
# -----------------------------------------------------------------------------
general_tab, text_tab, limit_tab, misc_tab, background_tab, giwaxs_tab = st.tabs(
    ["General", "Text", "Limit", "Misc", "Background", "GIWAXS"]
)

with general_tab:
    if not datasets:
        st.info("Choose a preset and upload a compatible file to begin.")
        selected = []
        styles = {}
        graph_title = ""
    else:
        default_selected = list(datasets.keys())
        selected = st.multiselect("Datasets", list(datasets.keys()), default=default_selected)

        default_title = Path(file_list[-1].name).stem if file_list else ""
        graph_title = st.text_input("Graph title", value=default_title)

        styles = {}
        if selected:
            st.subheader("Line style")
            cols = st.columns(2)
            for i, name in enumerate(selected):
                with cols[i % 2].expander(name, expanded=False):
                    line_color = st.color_picker("Line color", "#000000", key=f"line_color_{preset}_{name}")
                    line_style = st.selectbox(
                        "Line style",
                        ["-", "--", ":", "-."],
                        index=0,
                        key=f"line_style_{preset}_{name}",
                    )
                    marker = st.selectbox(
                        "Marker",
                        ["None", "o", "s", "^", "v", "D", "x", "+"],
                        index=0,
                        key=f"marker_{preset}_{name}",
                    )
                    marker_color = st.color_picker(
                        "Marker color",
                        "#000000",
                        key=f"marker_color_{preset}_{name}",
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
    text_y = st.slider("Vertical position", 0.0, 1.0, 0.5, 0.01, help="Same top-to-bottom convention as your Tkinter canvas.")

with limit_tab:
    c1, c2 = st.columns(2)
    with c1:
        x_min_text = st.text_input("X Min", value="")
        y_min_text = st.text_input("Y Min", value="")
    with c2:
        x_max_text = st.text_input("X Max", value="")
        y_max_text = st.text_input("Y Max", value="")

with misc_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_scale = st.selectbox("X-axis scale", ["linear", "log"], index=0)
        x_format = st.selectbox("X-axis format", ["number", "scientific", "engineering", "percent"], index=0)
        border_thickness = st.number_input("Border thickness", 0.5, 5.0, 1.0, 0.1)
        line_thickness = st.number_input("Line thickness", 0.1, 10.0, 2.0, 0.1)
    with c2:
        y_scale = st.selectbox("Y-axis scale", ["linear", "log"], index=0)
        y_format = st.selectbox("Y-axis format", ["number", "scientific", "engineering", "percent"], index=0)
        axis_fontsize = st.number_input("Axis/tick font size", 6, 36, 16, 1)
        title_fontsize = st.number_input("Title font size", 6, 36, 18, 1)
    with c3:
        fig_width = st.number_input("Figure width (in)", 2.0, 20.0, 6.7, 0.5)
        fig_height = st.number_input("Figure height (in)", 2.0, 20.0, 5.5, 0.5)
        legend_fontsize = st.number_input("Legend font size", 8, 30, 16, 1)
        legend_position = st.selectbox(
            "Legend position",
            ["upper right", "lower right", "upper left", "lower left", "best"],
            index=0,
        )

with background_tab:
    st.caption("v0.1 supports up to three optional x-range highlight blocks.")
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
            bg_alpha = st.slider("Transparency", 0.0, 1.0, 0.2, 0.05, key=f"bg_alpha_{i}")
            backgrounds.append(
                {"enabled": enabled, "xmin": bg_xmin, "xmax": bg_xmax, "color": bg_color, "alpha": bg_alpha}
            )

with giwaxs_tab:
    st.info(
        "GIWAXS is intentionally not migrated in v0.1. The tab is reserved so the web app can keep the same overall layout. "
        "Next we can move Qimage / Qphi / CirAvg / in-situ workflows here."
    )


# -----------------------------------------------------------------------------
# Plot preview + download
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Plot preview")

if datasets and selected:
    try:
        x_min = optional_float(x_min_text)
        x_max = optional_float(x_max_text)
        y_min = optional_float(y_min_text)
        y_max = optional_float(y_max_text)

        fig = build_figure(
            datasets=datasets,
            selected=selected,
            preset=preset,
            styles=styles,
            plot_mode=plot_mode,
            title=graph_title,
            fig_width=float(fig_width),
            fig_height=float(fig_height),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_scale=x_scale,
            y_scale=y_scale,
            x_format=x_format,
            y_format=y_format,
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
elif datasets:
    st.warning("Select at least one dataset in the General tab.")
else:
    st.caption("Your plot will appear here after you upload data.")


with st.expander("v0.1 compatibility notes"):
    st.markdown(
        """
- **UV-VIS / UV-VIS (eV):** uses the same paired-column layout as PYRL V9.2 and the first uploaded CSV.
- **XRD:** uses `.xy` files, skips the first 3 rows, then plots columns 1 and 2.
- **PL:** uses the first 2 CSV columns and retains integer x/wavelength points, matching V9.2.
- **Not migrated yet:** CD/gCD, CV, EL, EIS, EL Time, EL Spectrum, CIE, CP-EL/gEL, CP-PL/gCPPL, TR CP-EL, TRPL, FTIR, and GIWAXS.
- The web version uses browser file uploads instead of direct filesystem paths. Your original Tkinter file remains untouched.
        """
    )
