# CO2LOGIX 2.0

A global model of how fast CO<sub>2</sub> geological storage deployment over the next 100 years.

CO2LOGIX 2.0 asks how quickly can storage capacity be delivered? 
It combines three constraints:

1. **Pressure** — each basin can only absorb so much CO<sub>2</sub> before pore
   pressure approaches defined limits. Wells interfere with each other, and
   the model shuts a well in when the superposed pressure field around it
   exceeds this limit.
2. **Injectivity** — Single well injection rates are scaled by
   permeability–thickness and the allowable overpressure.
3. **Readiness** — a country that is already injecting scales up sooner and
   faster than one that has never drilled an appraisal well. Every country is
   graded into a readiness tier that sets both when its curve starts and how
   steeply it climbs.

The result is an annual injection profile (GtCO<sub>2</sub>/yr) per country from
2023 onwards, under a set of named scenarios.

---

## Requirements

Python 3.13+ with:

```
pip install geopandas pandas numpy CoolProp openpyxl
```

`CoolProp` supplies the brine and CO<sub>2</sub> equations of state (density,
viscosity, compressibility) at each basin's reference pressure and temperature.
`openpyxl` is the Excel writer backend. `geopandas` pulls in `shapely`,
`pyproj` and a shapefile reader.

## Quick start

```bash
python runCO2LOGIX.py
```

Input paths resolve from the repository root via `ROOT` in
[model/geology.py](model/geology.py#L21), so the script runs from any working
directory. It writes one workbook per scenario into [output/](output/).

Countries run **in parallel, one process per country**, using every available
core. BLAS thread counts are pinned to 1 at the top of
[runCO2LOGIX.py](runCO2LOGIX.py#L9-L14) — with a process-per-country model,
letting each worker spin up a full-core BLAS pool causes OpenBLAS allocation
failures. The arithmetic is elementwise, not BLAS-heavy, so this costs nothing.

## Repository layout

```
runCO2LOGIX.py          Entry point: scenario loop, parallel execution, Excel output
model/
  geology.py            Basin loading, capacity, fluid properties, injectivity
  growth.py             Readiness tiers, scenarios, logistic well schedule
  pressure.py           Nordbotten pressure solution, shut-in logic, capacity series
  utils.py              Newman compressibility, local distance matrix, fracture pressure
inputs/
  smith_shapefiles/     Global basin dataset, one shapefile per country
  other_models/         Detailed national models (UK, USA), used in preference
  misc_gis/             Basemap, not used by the model
output/                 Generated workbooks
```

