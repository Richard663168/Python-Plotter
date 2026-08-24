# PYRL Web v1.0

PYRL Web is a browser-based plotting and data-processing tool for spectroscopy, LED/device characterization, polarization-resolved measurements, electrochemistry, and time-resolved photoluminescence.

The app is designed around the file formats produced by the instruments and workflows used with the original desktop PYRL. Upload your exported data, choose the matching preset, select the datasets to display, adjust plotting options, and export a 300 dpi PNG.

## Supported presets and data sources

| PYRL preset | Measurement / source | Instrument or system | Typical upload | Notes |
|---|---|---|---|---|
| **UV-VIS** | UV-Vis absorbance | Cary 5000 UV-Vis-NIR spectrophotometer | `.csv` | Uses the paired-column Cary export format used by PYRL. |
| **UV-VIS (eV)** | UV-Vis absorbance plotted vs energy | Cary 5000 UV-Vis-NIR spectrophotometer | `.csv` | Same measurement as UV-VIS; PYRL converts wavelength using `E = 1240 / λ`. |
| **XRD** | Powder/thin-film X-ray diffraction | Bruker D8 Powder Diffractometer | `.xy` | Reads the XRD export convention used by desktop PYRL. |
| **CD** | Circular dichroism | JASCO J-1500 CD Spectrometer | `.csv` | Reads the `XYDATA` region of JASCO exports. `bkg.csv` can be included for background/pair processing. |
| **gCD** | Dissymmetry factor derived from CD and absorbance | JASCO J-1500 + Cary 5000 | CD `.csv` + UV-Vis `.csv` | Derived quantity; not a separate instrument measurement. Requires matching CD and UV-Vis datasets. |
| **PL** | Steady-state photoluminescence | Edinburgh FLS1000 Fluorescence and Lifetime Spectrometer | `.csv` | Uses wavelength/intensity data. |
| **CP-PL** | Circularly polarized / polarization-resolved PL | Polarization-resolved PL setup | `.csv` | PYRL allows the X and Y columns to be selected for each file. Instrument assignment is intentionally kept generic because CP-PL may be collected with different polarization-resolved setups. |
| **gCPPL** | PL dissymmetry factor derived from LCP/RCP PL | Derived from CP-PL | CP-PL `.csv` | PYRL pairs LCP/RCP files and calculates the dissymmetry spectrum. |
| **CV** | Cyclic voltammetry | Autolab electrochemical workstation | `.csv`, `.xls`, `.xlsx` | Series/model independent in PYRL. Uses the original desktop-PYRL column convention. |
| **EL** | Electroluminescence / J-V-L / EQE device data | Enlitech LQ-50X | `.txt`, `.tsv` | Supports selectable X, Y, and optional secondary Y axes. |
| **EL Time** | Time-dependent LED/device measurements | Enlitech LQ-50X | `.txt`, `.tsv` | Includes optional decay-ratio normalization. |
| **EL Spectrum** | Electroluminescence spectra | Enlitech LQ-50X | spectrum `.txt`/`.tsv` + optional EL table | The EL table can be used to display Voltage, Current, J, or L in the legend. |
| **CIE** | CIE 1931 chromaticity derived from EL data | Enlitech LQ-50X | EL table `.txt`, `.tsv` | Uses the x/y chromaticity columns from the EL table; not a separate measurement. |
| **EIS** | Electrochemical impedance spectroscopy | Autolab electrochemical workstation | `.csv`, `.txt`, `.xls`, `.xlsx` | Series/model independent in PYRL. Default Nyquist axes are Z' and -Z''. |
| **FTIR** | Fourier-transform infrared spectroscopy | Bruker Vertex Vacuum FT-IR or Bruker Hyperion FT-IR | `.dpt` | Two-column wavenumber/absorbance data. Includes absorbance-to-transmission and arbitrary-unit options. |
| **CP-EL** | Circularly polarized / polarization-resolved EL | Enlitech LQ-50X + polarization optics | `.txt`, `.tsv` | First row is skipped; first two columns are wavelength and intensity. |
| **gEL** | EL dissymmetry factor derived from LCP/RCP EL | Derived from CP-EL | CP-EL `.txt`, `.tsv` | PYRL pairs RCP/LCP files and calculates `2(IL - IR) / (IL + IR)`. |
| **TR CP-EL** | Repeated/time-resolved polarization-resolved EL scans | Enlitech LQ-50X + polarization optics | multiple `.txt`, `.tsv` | Supports a 2D wavelength slice vs scan number and a 3D scan/wavelength/intensity view. |
| **TRPL** | Time-resolved photoluminescence | Edinburgh FLS1000 Fluorescence and Lifetime Spectrometer | `.csv` | Skips the first 10 settings rows, normalizes from the PL maximum, and performs biexponential fitting with monoexponential fallback. |

## Derived presets

Several PYRL presets are calculated from measured data rather than being separate instrument measurements:

- **UV-VIS (eV)** — converts UV-Vis wavelength to photon energy.
- **gCD** — combines CD and UV-Vis absorbance data.
- **gCPPL** — calculated from paired LCP/RCP CP-PL spectra.
- **CIE** — uses chromaticity coordinates contained in the EL measurement table.
- **gEL** — calculated from paired LCP/RCP CP-EL spectra.

## Basic use

1. Select the appropriate **Preset** in the sidebar.
2. Upload the exported instrument file(s).
3. Select the datasets you want to plot.
4. Adjust line colors, line styles, markers, labels, limits, scales, and other formatting options.
5. Preview the figure in the browser.
6. Download the finished figure as a **300 dpi PNG**.

The color controls include the original 30-color PYRL palette, with stable default colors assigned to datasets.

## File naming conventions for processed polarization data

Some processed presets depend on filenames so that matching measurements can be paired automatically.

- **gCPPL:** `LCP` and `RCP` should appear in the first underscore-separated filename block.
- **gEL:** `LCP` and `RCP` should appear at the end of the first underscore-separated filename block, following the desktop PYRL convention.
- **TR CP-EL:** files are grouped by the prefix before the first underscore and ordered using an `_P<number>` tag.
- **gCD:** matching identifiers are used to associate the processed CD traces with the appropriate UV-Vis absorbance data.

## Notes on instrument compatibility

PYRL Web is optimized for the export formats listed above. Data from another instrument may still work if it follows the same column/file structure, but the preset names describe the formats that have been tested with the listed systems.

Autolab support is intentionally listed generically because the relevant EIS and CV file structures do not depend on a particular Autolab series in PYRL.

## Included in v1.0

- UV-VIS / UV-VIS (eV)
- XRD
- CD / gCD
- PL / CP-PL / gCPPL
- CV
- EL / EL Time / EL Spectrum / CIE
- EIS
- FTIR
- CP-EL / gEL / TR CP-EL
- TRPL

GIWAXS is intentionally not included in v1.0 and may be added later as a separate module.

## Deploy on Streamlit Community Cloud

Keep these files in the same GitHub repository:

- `streamlit_app.py`
- `requirements.txt`
- `README.md`

Deploy `streamlit_app.py` through Streamlit Community Cloud. When you commit updates to the GitHub repository, the deployed app will rebuild from the latest version.
