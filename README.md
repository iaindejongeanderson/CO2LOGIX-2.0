# CO2LOGIX 2.0

A global model of how fast CO<sub>2</sub> geological storage can be deployed.

CO2LOGIX 2.0 couples country-level industry growth with subsurface pressure. For each of
71 countries (217 basins and storage units) it schedules new injection wells along a
logistic drilling curve, places them at random in the country's basins, superposes their
pressure fields every year, and shuts in any well whose surroundings exceed the pressure
limit. The result is the annual CO<sub>2</sub> injection rate of each country from 2023
until its deployment saturates.

This is the code that accompanies:

> de Jonge-Anderson, I., Johnson, G., Alcalde, J. & Roberts, J.J. Global limits on
> CO<sub>2</sub> geological storage deployment using a coupled industry growth - pressure
> constrained model. *International Journal of Greenhouse Gas Control* (under review).

It reproduces the paper's model runs exactly. The results of every run, and the inputs and
readiness evidence for every country, are in the paper's Supplementary Data 1 and 2.

## Install

Python 3.13 or later.

```bash
pip install -r requirements.txt
```

## Run

A quick test (two small countries, one scenario, about a minute):

```bash
CO2LOGIX_COUNTRIES="UK,Norway" CO2LOGIX_SCENARIOS="reference" python runCO2LOGIX.py
```

All nine scenarios of the paper:

```bash
python runCO2LOGIX.py
```

On Windows PowerShell, set the options first, for example
`$env:CO2LOGIX_SCENARIOS = "reference"`, then run `python runCO2LOGIX.py`.

Each scenario writes `output/output_<scenario>.xlsx`. A scenario whose workbook already
exists is skipped, so a stopped run can be restarted; delete the workbook to run it again.

| Option (environment variable) | Default | Meaning |
| --- | --- | --- |
| `CO2LOGIX_SCENARIOS` | all nine | Comma-separated scenarios (see below) |
| `CO2LOGIX_PLIM` | `0.8` | Pressure limit as a fraction of fracture pressure |
| `CO2LOGIX_COUNTRIES` | all | Comma-separated countries, for a test (file names in `inputs/`) |
| `CO2LOGIX_WORKERS` | CPU count - 4 | Countries simulated in parallel |

Countries are simulated independently, one process each. The full runs are demanding: in
the paper each scenario took 13-15 minutes on 28 workers, and the Maximum scenarios needed
several GB of memory per worker, so they were run on 16.

### Scenarios

A scenario name is the deployment scenario, optionally followed by the scale-up assumption
(Table 3 of the paper):

| Name | Paper | Countries that deploy |
| --- | --- | --- |
| `limited`, `limited_slow`, `limited_fast` | Limited (Central / Slow / Fast) | Injecting and Pipeline (26) |
| `reference`, `reference_slow`, `reference_fast` | Reference (Central / Slow / Fast) | + Appraised (45) |
| `maximum`, `maximum_slow`, `maximum_fast` | Maximum (Central / Slow / Fast) | + Conceptual (71) |

Slow halves each country's growth rate and multiplies its start delay by 1.5; Fast
multiplies the growth rate by 1.5 and halves the delay.

### Reproducing the paper

1. The nine scenarios: `python runCO2LOGIX.py`.
2. The pressure-limit sensitivity (Section 3.4.2), on Reference (Central):
   `CO2LOGIX_PLIM=0.7 CO2LOGIX_SCENARIOS=reference python runCO2LOGIX.py`, and the same
   with `0.9`. These write `output_reference_plim70.xlsx` and `..._plim90.xlsx`.
3. The other sensitivity cases in Table 7 need no new runs, because countries are
   simulated independently and a start delay only moves a country's series in time:
   - *no start delays*: move each country's row in `annual_injection` back by its
     `Start delay (years)` (in `country_summary`) and add up the countries again;
   - *without the USA* (or the USA and Brazil): drop those rows and add up again.

As a check, Reference (Central) gives a global injection of 2.71 GtCO<sub>2</sub>/yr in 2050,
a peak of 53.7 GtCO<sub>2</sub>/yr in 2084 and 1,646.5 GtCO<sub>2</sub> stored by 2100.

## Output

Each workbook has these sheets. Rates are in GtCO<sub>2</sub>/yr, by calendar year, and
are zero before a country starts and after its last well stops.

| Sheet | Contents |
| --- | --- |
| `annual_injection` | Injection by country under the pressure limit, with a `GLOBAL` total row |
| `annual_injection_no_limit` | The same schedule with no wells shut in |
| `annual_by_basin` | Injection by basin (adds up to the country rows) |
| `stopped_by_basin` | CO<sub>2</sub> (GtCO<sub>2</sub>) stored by wells that stopped injecting in each year |
| `cumulative_wells` | Cumulative wells drilled by country |
| `country_summary` | Pressure-limited capacity, wells drilled and shut in, capacity lost to shut-ins, start delay, last year of injection |

## How the model works

| Step | Where | Paper |
| --- | --- | --- |
| Basins and their properties; national datasets for the UK and USA, the global dataset elsewhere | `model/geology.py`, `load_country_basins` | 2.2 |
| Pressure-limited capacity: pore volume x total compressibility x CO<sub>2</sub> density x allowable pressure increase (Bump & Hovorka, 2024) | `model/pressure.py`, `pressure_space` | 2.3, App. A |
| Injection rate per well: injectivity from permeability x thickness (Valluri et al., 2021, lower coefficient), times the allowable pressure increase, capped at 2 MtCO<sub>2</sub>/yr | `model/geology.py`, `add_reservoir_properties` | 2.1.3 |
| Readiness class, growth rate and start delay of each country | `model/growth.py`, `COUNTRY_TIER` | 2.3, Table 2 |
| Logistic curve: carrying capacity from capacity / rate / 25 years; midpoint from 2023 injection; drilling until 99% of capacity, for at least 100 years | `model/growth.py` | 2.1.1, 2.3 |
| Wells at random grid-cell centres (2 km grid, 1 km for small countries), moved to another basin if theirs is full | `model/pressure.py`, `run_model_welloff` | 2.2 |
| Pressure of each well (Nordbotten et al., 2005), superposed on the grid each year; a well is shut in when any cell within its radius of influence exceeds the limit; finished and shut-in wells keep their final pressure field | `model/pressure.py` | 2.1.2-2.1.3 |

Key constants: hydrostatic gradient 10.5 MPa/km, lithostatic 23 MPa/km, fracture pressure =
pore pressure + 0.75 x (vertical stress - pore pressure) (Zhang & Yin, 2017), geothermal
gradient 25 °C/km from 15 °C, pressure limit 80% of fracture pressure, well life 25 years
(`model/utils.py` and `model/geology.py`).

## Repository

```
runCO2LOGIX.py           Runs the scenarios and writes the workbooks
model/
  geology.py             Basins, reservoir and fluid properties, capacity, injection rate
  growth.py              Readiness classes, scenarios, logistic drilling schedule
  pressure.py            Capacity, pressure solution, superposition and shut-in
  utils.py               Constants, compressibility, fracture pressure
inputs/
  smith_shapefiles/      Basin properties for 71 countries, one shapefile per country
  other_models/          National datasets used instead for the UK and USA
```

## Input data

- `inputs/smith_shapefiles/`: basin-averaged properties from the global dataset of Gao,
  X., Kivi, I.R., Elizarev, M. & Krevor, S. (2026), Zenodo,
  https://doi.org/10.5281/zenodo.24084271, used in Smith, A., Hampson, G. & Krevor, S.
  (2024), *IJGGC* 137, 104220, joined to the basin outlines of Evenick, J.C. (2021),
  *Earth-Science Reviews* 215, 103564, and split by country.
- `inputs/other_models/UK`: UK saline aquifers from de Jonge-Anderson et al. (2026),
  *IJGGC* 151, 104608.
- `inputs/other_models/USA`: USA storage basins from Rodriguez Calzado, E., Bump, A.P. &
  Hovorka, S.D. (2026), *Gas Science and Engineering* 156, 206032.

Please cite these sources if you use the inputs.

## Versions

What changed since the version released with the submitted paper is in
[CHANGELOG.md](CHANGELOG.md). CO2LOGIX 1.0, the UK basin-scale model, is at
https://github.com/iaindejongeanderson/CO2LOGIX.

## Citation and licence

Please cite the paper above and this software (see [CITATION.cff](CITATION.cff); Zenodo:
https://doi.org/10.5281/zenodo.22047157, all versions). Released under the
[Creative Commons Attribution 4.0 licence](LICENSE).

Contact: Iain de Jonge-Anderson, University of Strathclyde
(iain.de-jonge-anderson@strath.ac.uk).
