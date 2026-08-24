PYRL Web v0.1
==============

Files
-----
PYRL_Web_v0.1.py             Streamlit web app
requirements_PYRL_Web.txt    Python dependencies

What works in v0.1
------------------
- UV-VIS
- UV-VIS (eV)
- XRD (.xy)
- PL (.csv)
- Single and stacked plots
- Dataset selection
- Per-dataset line color, line style, marker, marker color
- Axis limits
- Linear/log scales
- Number/scientific/engineering/percent formatting
- Figure size, font sizes, line/border thickness, legend position
- Custom figure text
- Up to 3 background highlight blocks
- 300 dpi PNG download

Run locally
-----------
1. Open Terminal in the folder containing these files.
2. Optional but recommended: create a virtual environment.

   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements_PYRL_Web.txt

4. Start the app:

   streamlit run PYRL_Web_v0.1.py

5. Streamlit should open the app in your browser. If it does not, copy the localhost URL printed in Terminal into your browser.

Notes
-----
- Your original PYRL V9.2 Tkinter file is not changed.
- Browser apps cannot use Tkinter's native file/folder dialogs. Data is uploaded through the browser instead.
- GIWAXS is reserved as a tab but intentionally not migrated yet. It should be added after the common plotting workflow is stable.
