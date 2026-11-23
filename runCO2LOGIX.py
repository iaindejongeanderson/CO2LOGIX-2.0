"""Run CO2LOGIX 2.0 for the paper's scenarios and write one workbook per scenario to output/.

    python runCO2LOGIX.py

Options, set as environment variables:
    CO2LOGIX_SCENARIOS   comma-separated scenarios (default: all nine), e.g. "reference,maximum_fast"
    CO2LOGIX_PLIM        pressure limit as a fraction of fracture pressure (default 0.8)
    CO2LOGIX_COUNTRIES   comma-separated countries for a quick test, e.g. "UK,Norway"
    CO2LOGIX_WORKERS     number of parallel processes (default: CPU count - 4)

Each country is simulated independently, in its own process.
"""
import os

# One BLAS thread per process: the work is elementwise, and many processes each
# starting a full-core BLAS pool oversubscribe memory. Set before numpy loads.
for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '1')

from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd

from model.geology import load_country_basins, add_reservoir_properties, INJ_YEARS, PLIM, ROOT
from model.growth import (ALL_SCENARIOS, drilling_window, injection_delay_years,
                          logistic_capacity_and_midpoint, scenario_countries, scenario_growth_rate)
from model.pressure import run_model_welloff

# First model year. The logistic curves are anchored to observed injection in this year.
START_YEAR = 2023

# National injection in 2023 (MtCO2/yr) from the London Register of Subsurface CO2
# Storage (Gao & Krevor, 2025). Other countries, including those whose projects have
# ended (Croatia 2015, Japan 2019), start from a single well after their delay.
REGISTER_SEED_MT = {
    'USA':          21.7,
    'Brazil':       13.0,
    'Canada':        3.2,
    'China':         2.7,
    'Australia':     1.6,
    'Norway':        0.8,
    'Saudi_Arabia':  0.8,
}


def run_country(scenario, c):
    """Simulate one country under one scenario. Returns None if it has no basin data."""
    S_gdf, grid_size = load_country_basins(c, PLIM)
    if S_gdf is None:
        return None

    params = {'plim': PLIM, 'inj_years': INJ_YEARS}

    # Grid of cell centres over the country's basins
    xmin, ymin, xmax, ymax = S_gdf.total_bounds
    params['gx'], params['gy'] = np.meshgrid(np.arange(xmin + grid_size / 2.0, xmax, grid_size),
                                             np.arange(ymin + grid_size / 2.0, ymax, grid_size))

    S_gdf = add_reservoir_properties(S_gdf, PLIM)

    # Logistic drilling curve
    params['k_growth'] = scenario_growth_rate(c, scenario)
    params['L'], params['t0'], _ = logistic_capacity_and_midpoint(
        S_gdf, params['k_growth'], REGISTER_SEED_MT.get(c, 0.0), INJ_YEARS)
    params['model_years'] = drilling_window(params['t0'], params['k_growth'])

    wells, n_shut_in, inj_no_shut_in, inj, inj_basin, stopped_basin = run_model_welloff(S_gdf, params)

    # Model year 1 is START_YEAR + the country's start delay; years before that are zero
    delay = injection_delay_years(c, scenario)
    years = np.arange(START_YEAR + delay, START_YEAR + delay + len(inj))
    calendar = np.arange(START_YEAR, years[-1] + 1)

    def on_calendar(values, index):  # MtCO2 -> GtCO2, on calendar years
        return pd.DataFrame(np.atleast_2d(values) / 1000, index=index, columns=years) \
                 .reindex(columns=calendar, fill_value=0.0)

    name_col = next((n for n in ('Basin_Name', 'NAME', 'Name', 'name', 'BASIN', 'Basin') if n in S_gdf), None)
    names = S_gdf[name_col].astype(str).to_numpy() if name_col else [''] * len(S_gdf)
    basin_index = pd.MultiIndex.from_arrays([[c] * len(S_gdf), names], names=['Country', 'Basin'])

    annual = on_calendar(inj, pd.Index([c], name='Country'))
    annual_no_shut_in = on_calendar(inj_no_shut_in, pd.Index([c], name='Country'))
    by_basin = on_calendar(inj_basin, basin_index)
    stopped = on_calendar(stopped_basin, basin_index)

    # Checks: basins add up to the country, and every tonne injected stops exactly once
    assert np.abs(by_basin.sum().to_numpy() - annual.iloc[0].to_numpy()).max() < 1e-9, c
    assert np.abs(stopped.sum(axis=1).to_numpy() - by_basin.sum(axis=1).to_numpy()).max() < 1e-9, c

    # Cumulative wells drilled, on the same calendar
    starts = pd.Series([w['start_year'] for w in wells], dtype='int64')
    cum_wells = starts.value_counts().sort_index().cumsum().reindex(range(1, len(inj) + 1)).ffill().fillna(0)
    wells_cum = pd.DataFrame([cum_wells.to_numpy()], index=pd.Index([c], name='Country'), columns=years) \
                  .reindex(columns=calendar, fill_value=0.0)

    summary = pd.DataFrame({
        'Capacity (GtCO2)': [S_gdf['MCO2'].sum()],
        'Wells drilled': [len(wells)],
        'Wells shut in': [n_shut_in],
        'Capacity lost to shut-ins (GtCO2)': [annual_no_shut_in.values.sum() - annual.values.sum()],
        'Start delay (years)': [delay],
        'Last year of injection': [int(annual.columns[annual.iloc[0] > 0].max()) if annual.values.any() else None],
    }, index=pd.Index([c], name='Country'))

    return dict(annual=annual, annual_no_shut_in=annual_no_shut_in, by_basin=by_basin,
                stopped=stopped, wells=wells_cum, summary=summary)


SHEETS = {  # output sheet name: result key (the sheets are described in README.md)
    'annual_injection':          'annual',
    'annual_injection_no_limit': 'annual_no_shut_in',
    'annual_by_basin':           'by_basin',
    'stopped_by_basin':          'stopped',
    'cumulative_wells':          'wells',
    'country_summary':           'summary',
}


if __name__ == '__main__':
    shapefiles = ROOT / 'inputs' / 'smith_shapefiles'
    countries = sorted(f.stem for f in shapefiles.iterdir() if f.suffix == '.shp')

    scenarios = [s.strip() for s in os.environ.get('CO2LOGIX_SCENARIOS', ','.join(ALL_SCENARIOS)).split(',') if s.strip()]
    only = [s.strip() for s in os.environ.get('CO2LOGIX_COUNTRIES', '').split(',') if s.strip()]
    unknown = sorted(set(only) - set(countries))
    if unknown:
        raise SystemExit(f'Unknown countries {unknown}; names are the files in {shapefiles}')

    n_workers = int(os.environ.get('CO2LOGIX_WORKERS', max(1, (os.cpu_count() or 1) - 4)))
    tag = ('' if abs(PLIM - 0.8) < 1e-9 else f'_plim{round(PLIM * 100)}') + ('_test' if only else '')
    print(f'Pressure limit {PLIM:.2f} of fracture pressure; {n_workers} workers; scenarios: {scenarios}')

    (ROOT / 'output').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for scenario in scenarios:
            out_fp = ROOT / 'output' / f'output_{scenario}{tag}.xlsx'
            if out_fp.exists():
                print(f'{out_fp.name} exists, skipping (delete it to run again)')
                continue

            run_countries = [c for c in scenario_countries(countries, scenario) if not only or c in only]
            print(f'\n--- {scenario}: {len(run_countries)} countries ---')

            futures = {executor.submit(run_country, scenario, c): c for c in run_countries}
            results, failed = [], []
            for future in as_completed(futures):
                c = futures[future]
                try:
                    r = future.result()
                except Exception as exc:
                    print(f'   !!! {c} failed: {exc}')
                    failed.append(c)
                    continue
                if r is not None:
                    results.append(r)
                    print(f'   {c} done (to {r["summary"]["Last year of injection"].iloc[0]})', flush=True)

            # A partial workbook would look complete to anything that reads it
            if failed:
                raise RuntimeError(f'{scenario}: {len(failed)} countries failed, nothing written: {sorted(failed)}')

            with pd.ExcelWriter(out_fp, engine='openpyxl') as writer:
                for sheet, key in SHEETS.items():
                    df = pd.concat([r[key] for r in results]).sort_index()
                    if key != 'summary':
                        df = df.sort_index(axis=1).fillna(0.0)  # countries end in different years
                    if key == 'annual':
                        df.loc['GLOBAL'] = df.sum()
                    df.to_excel(writer, sheet_name=sheet)
            print(f'Written {out_fp}')
