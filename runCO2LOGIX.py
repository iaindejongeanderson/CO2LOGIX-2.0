#%% Import required packages

# Must be set before numpy (and therefore its BLAS backend) is imported anywhere.
# Each worker process below gets its own full-core BLAS thread pool by default;
# with a process-per-country parallelism model that means (workers x BLAS threads)
# threads fighting over memory, which is what caused the OpenBLAS allocation
# failures. The math here is elementwise array work, not BLAS-heavy, so pinning
# every process to a single BLAS thread costs nothing and avoids the oversubscription.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import geopandas as gpd
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import CO2LOGIX modules
from model.pressure import run_model_welloff
from model.geology import load_country_basins, add_reservoir_properties, INJ_YEARS, PLIM, ROOT
from model.growth import scenario_growth_rate, injection_delay_years
from model.growth import logistic_capacity_and_midpoint
from model.growth import scenario_countries, split_scenario, SCENARIO_TIERS
from model.growth import SCENARIOS_THREE, SCENARIOS_FULL


# Run logic. The scenario names come from model.growth so the model and every
# reporting script work from one list.
#   three - the three main scenarios
#   full  - the same three plus their slow and fast rate variants
RUN_SCENARIOS = {'three': SCENARIOS_THREE, 'full': SCENARIOS_FULL}

run = 'three'


START_YEAR = 2023

# Numbers from the London Register of
# Subsurface CO2 Storage (Gao & Krevor, 2025). 2023 is the latest complete
# reporting year. # Countries absent from the register start from a single well
# after their assigned delay.

REGISTER_SEED_MT = {
    'USA':          21.7,
    'Brazil':       13.0,
    'Canada':        3.2,
    'China':         2.7,
    'Australia':     1.6,
    'Norway':        0.8,
    'Saudi_Arabia':  0.8,
}


#%% Simulation Function

def run_country_simulation(scenario, c, injection_delay):

    params = {}
    params['plim'] = PLIM


    # ── Load geological data ──────────────────────────────────────────────────

    S_gdf, params['grid_size'] = load_country_basins(c, params['plim'])

    if S_gdf is None:
        return None

    # Calculate MCO2 total for the country
    MCO2_val = float(S_gdf["MCO2"].sum())

    # ── Reservoir, fluid and injectivity properties ───────────────────────────

    params['model_years'] = 100
    params['inj_years']  = INJ_YEARS
    params['domain']     = 'open'
    params['rc']         = 10000

    params['xmin'], params['ymin'], params['xmax'], params['ymax'] = S_gdf.total_bounds

    x_coords = np.arange(params['xmin'] + params['grid_size'] / 2.0, params['xmax'], params['grid_size'])
    y_coords = np.arange(params['ymin'] + params['grid_size'] / 2.0, params['ymax'], params['grid_size'])
    params['gx'], params['gy'] = np.meshgrid(x_coords, y_coords)

    S_gdf = add_reservoir_properties(S_gdf, params['plim'])

    # ── Growth parameters ─────────────────────────────────────────────────────

    params['k_growth'] = scenario_growth_rate(c, scenario)

    params['L'], params['t0'], _ = logistic_capacity_and_midpoint(
        S_gdf, params['k_growth'], REGISTER_SEED_MT.get(c, 0.0), params['inj_years']
    )

    # ── Grid pressure reference lookup ────────────────────────────────────────

    x_flat      = params['gx'].ravel()
    y_flat      = params['gy'].ravel()
    grid_points = gpd.GeoDataFrame(
        {"cell_id": np.arange(x_flat.size, dtype=np.int32)},
        geometry=gpd.points_from_xy(x_flat, y_flat),
        crs=S_gdf.crs
    )

    grid_join = gpd.sjoin(
        grid_points,
        S_gdf[['geometry', 'p_frac', 'p_ref']],
        how='left',
        predicate='within'
    ).sort_values('cell_id')

    grid_join = grid_join.drop_duplicates(subset='cell_id', keep='first').set_index('cell_id')
    grid_join = grid_join.reindex(np.arange(x_flat.size, dtype=np.int32))

    params['p_frac_grid'] = grid_join['p_frac'].to_numpy().reshape(params['gx'].shape)
    params['p_ref_grid']  = grid_join['p_ref'].to_numpy().reshape(params['gx'].shape)
    
    S_gdf['country'] = c

    # ── Run model ─────────────────────────────────────────────────────────────

    
    # Well placement is seeded, so one run per country is deterministic and a
    # single iteration is all that is needed. The single-element lists and the
    # Iteration index level below are what remains of a Monte-Carlo loop; the
    # level is kept so the output sheets keep their shape.
    *_, drop_count, wells, capnodrop, capwithdrop = run_model_welloff(S_gdf, params)

    capacity_orig_runs = [pd.Series(capnodrop)]
    capacity_drop_runs = [pd.Series(capwithdrop)]
    drop_count_runs    = [drop_count]

    capacity_orig_df   = pd.concat(capacity_orig_runs, axis=1)
    capacity_drop_df   = pd.concat(capacity_drop_runs, axis=1)

    # ── Aggregate results (All Iterations) ────────────────────────────────────
    
    # Transpose so iterations become rows and time periods become columns
    cap_df      = capacity_drop_df.T 
    cap_orig_df = capacity_orig_df.T

    num_years = cap_df.shape[1]
    delay_years = injection_delay.get(c, 0)
    
    # Reset columns to integer temporarily for clean shifting
    cap_df.columns = range(num_years)
    cap_orig_df.columns = range(num_years)

    # The timeline has to be widened before the shift. DataFrame.shift drops
    # whatever falls off the right-hand edge, so shifting in place truncated
    # the tail of every delayed country's injection profile - and because the
    # profile length is set by the country's own well schedule, the cut landed
    # inside the reporting window for countries whose schedule finishes early.
    total_years = num_years + delay_years

    if delay_years > 0:
        cap_df      = cap_df.reindex(columns=range(total_years), fill_value=0.0)
        cap_orig_df = cap_orig_df.reindex(columns=range(total_years), fill_value=0.0)
        # Shift data to the right (simulating delay) and fill new gaps with 0
        cap_df = cap_df.shift(delay_years, axis=1).fillna(0)
        cap_orig_df = cap_orig_df.shift(delay_years, axis=1).fillna(0)

    # Reassign real years as column headers
    years = np.arange(START_YEAR, START_YEAR + total_years)
    cap_df.columns = years
    cap_orig_df.columns = years

    # Cumulative wells actually drilled and retained by the simulation
    # (last iteration), aligned to the same delayed timeline as cap_df
    starts = pd.Series([w['start_year'] for w in wells], dtype='int64')
    cum_wells = (starts.value_counts().sort_index().cumsum()
                 .reindex(range(1, total_years + 1)).ffill().fillna(0))
    if delay_years > 0:
        cum_wells = cum_wells.shift(delay_years).fillna(0)
    wells_series = pd.Series(cum_wells.to_numpy(), index=years, name=c)

    # Convert to Gigatonnes
    cap_df = cap_df / 1000
    cap_orig_df = cap_orig_df / 1000

    # Apply MultiIndex (Country, Iteration)
    cap_df.index = pd.MultiIndex.from_product([[c], cap_df.index], names=['Country', 'Iteration'])
    cap_orig_df.index = pd.MultiIndex.from_product([[c], cap_orig_df.index], names=['Country', 'Iteration'])

    return c, cap_df, cap_orig_df, drop_count_runs, MCO2_val, S_gdf, wells_series


#%% Main Execution Block

if __name__ == '__main__':
    
    # Global Setup
    folder = ROOT / 'inputs' / 'smith_shapefiles'
    countries = sorted({f.stem for f in folder.iterdir() if f.suffix == '.shp'})

    output_dir = ROOT / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_workers = os.cpu_count()
    print(f"Using a process pool of {n_workers} workers")

    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        # Iterate through scenarios
        if run not in RUN_SCENARIOS:
            raise SystemExit(f"unknown run mode {run!r}; "
                             f"expected one of {sorted(RUN_SCENARIOS)}")
        scenarios = RUN_SCENARIOS[run]

        for scenario in scenarios:

            print(f"\n--- Running Scenario: {scenario} ---")

            
            run_countries = scenario_countries(countries, scenario)
            print(f"{len(run_countries)} of {len(countries)} countries "
                  f"(tiers {SCENARIO_TIERS[split_scenario(scenario)[0]]})")

            
            injection_delay = {country: injection_delay_years(country, scenario)
                               for country in run_countries}

            
            output_fp = output_dir / f'output_{scenario}.xlsx'

            
            all_country_rows = []
            all_country_rows_orig = []
            drop_count_per_country = {}
            MCO2_per_country = {}
            all_basins = []
            all_well_data = []

            # ── Execute Simulations In Parallel (one process per country) ─────────

            futures = {
                executor.submit(run_country_simulation, scenario, c, injection_delay): c
                for c in run_countries
            }

            for future in as_completed(futures):
                c = futures[future]
                try:
                    result = future.result()

                    if result is not None:
                        c_ret, df_rows, df_orig_rows, drop_counts, MCO2_val, S_gdf_c, wells_s = result

                        all_country_rows.append(df_rows)
                        all_country_rows_orig.append(df_orig_rows)
                        drop_count_per_country[c_ret] = drop_counts
                        MCO2_per_country[c_ret] = MCO2_val
                        all_basins.append(S_gdf_c)
                        all_well_data.append(wells_s)

                        print(f'   -> {c_ret} complete')
                    else:
                        print(f'   -> {c} skipped (No geodata found or scenario filter applied)')

                except Exception as exc:
                    print(f'\n!!! FATAL ERROR PROCESSING {c}: {exc} !!!\n')
                    

            # ── Compile and write output ────────────────────────────────────────────

            if len(all_country_rows) > 0:
                df      = pd.concat(all_country_rows,      axis=0)
                df_orig = pd.concat(all_country_rows_orig, axis=0)

                num_iters = len(list(drop_count_per_country.values())[0])
                drop_count_df = pd.DataFrame.from_dict(
                    drop_count_per_country, orient='index',
                    columns=[f"iter_{i+1}" for i in range(num_iters)]
                )

                drop_count_df['Average Drop Count'] = drop_count_df.mean(axis=1)
                drop_count_df = drop_count_df[drop_count_df['Average Drop Count'] > 0]
                drop_count_df = drop_count_df.sort_values('Average Drop Count', ascending=False)

                end_year = START_YEAR + 100 - 1

                total_original_cap_per_iter    = df_orig.ffill(axis=1).cumsum(axis=1)[end_year]
                total_constrained_cap_per_iter = df.ffill(axis=1).cumsum(axis=1)[end_year]
                lost_capacity_per_iter         = total_original_cap_per_iter - total_constrained_cap_per_iter

                dropped_countries              = drop_count_df.index.tolist()
                lost_capacity_per_iter         = lost_capacity_per_iter[lost_capacity_per_iter.index.get_level_values('Country').isin(dropped_countries)]

                MCO2_series = pd.Series(MCO2_per_country, name="MCO2_GtCO2").reindex(df.index.get_level_values('Country').unique())

                with pd.ExcelWriter(output_fp, engine="openpyxl") as writer:
                    df.to_excel(writer,                                       sheet_name="annual_constrained")
                    df_orig.to_excel(writer,                                  sheet_name="annual_original")
                    lost_capacity_per_iter.to_frame("lost_capacity_GtCO2").to_excel(writer, sheet_name="lost_capacity")
                    drop_count_df.to_excel(writer,                            sheet_name="drop_counts")
                    MCO2_series.to_frame().to_excel(writer,                   sheet_name="MCO2_capacity")

                if len(all_well_data) > 0:
                    wells_df = pd.concat(all_well_data, axis=1)
                    with pd.ExcelWriter(output_fp, engine="openpyxl", mode='a') as writer:
                        wells_df.to_excel(writer, sheet_name="well_counts_per_country")

                

                print(f"Simulation fully complete for {scenario}. Output written to {output_fp}.")

            else:
                print(f"No countries were successfully processed for scenario: {scenario}. Output files were not created.")