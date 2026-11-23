# Changelog

## 2.0.1: first revision of the paper

Accompanies the revised manuscript (IJGGC, JGGC-D-26-00854). Reproduces its model runs
exactly.

### Changes that affect results

- **Hydrostatic gradient raised from 10 to 10.5 MPa/km** (`H_GRAD`, `model/utils.py`), the
  gradient of saline formation water (about 0.465 psi/ft) rather than fresh water. The
  allowable pressure increase is smaller, so the assessed capacity of the 71 countries
  falls from 3,935 to 3,744 GtCO<sub>2</sub> at 80% of fracture pressure.
- **Drilling runs until the logistic curve reaches 99% of its carrying capacity**, for at
  least 100 years (`drilling_window`, `model/growth.py`). It previously stopped after a
  fixed 100 years, which cut late-starting and slow-growing countries off part-way up their
  curve. Results to 2100 are almost unchanged; storage after 2100 is larger, and runs now
  end in different years (2151 to 2648).

### New options

- `CO2LOGIX_PLIM` sets the pressure limit (fraction of fracture pressure, default 0.8), for
  the paper's 70% and 90% sensitivity runs. Their workbooks are named `..._plim70.xlsx` and
  `..._plim90.xlsx`.
- `CO2LOGIX_SCENARIOS`, `CO2LOGIX_COUNTRIES` and `CO2LOGIX_WORKERS` choose the scenarios,
  limit a run to some countries (for a quick test), and set the number of processes.
- A finished scenario is skipped, so a stopped run can be restarted. If any country fails,
  no workbook is written rather than a partial one.

### Simpler code and outputs (no change to results)

- One workbook per scenario with named sheets: annual injection by country (with and
  without the pressure limit), by basin, CO<sub>2</sub> by the year its well stopped,
  cumulative wells, and a summary per country.
- The gradients are defined once (`model/utils.py`), input paths work from any folder and
  on any operating system, and unused functions and options have been removed.
- A start delay now relabels a country's years instead of shifting its series, so nothing
  is lost at the end of a run.
- Compiled `__pycache__` files, the unused basemap and ArcGIS sidecar files (`.sbn`,
  `.sbx`, `.shp.xml`) have been removed.

## 2.0: initial release

Released with the submitted paper. Zenodo: https://doi.org/10.5281/zenodo.22047158.
