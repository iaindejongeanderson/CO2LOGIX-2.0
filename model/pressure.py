"""Pressure-limited capacity and the pressure simulation with well shut-in (Sections 2.1.2-2.1.3)."""
import CoolProp.CoolProp as CP
import geopandas as gpd
import numpy as np
import pandas as pd

from model.growth import generate_well_schedule_logistic
from model.utils import local_distance_matrix, calculate_newman_cp_consolidated, estimate_frac_pres
from model.utils import H_GRAD, L_GRAD, T_GRAD, T_SURF

YEAR_S = 86400 * 365


def pressure_space(a, h, z, phi, plim):
    """Pressure-limited storage capacity (GtCO2) after Bump & Hovorka (2024).

    Pore volume x total compressibility x CO2 density x allowable pressure
    increase, where the allowable increase runs from hydrostatic pressure to
    `plim` x fracture pressure. a: area (m2), h: net thickness (m), z: depth (m),
    phi: porosity (fraction).
    """
    P_h = z / 1000 * H_GRAD                      # hydrostatic pressure, MPa
    P_l = z / 1000 * L_GRAD                      # lithostatic pressure, MPa
    P_lim = plim * estimate_frac_pres(P_l, P_h)  # pressure limit, MPa

    tref = T_GRAD * z / 1000 + T_SURF            # temperature at depth, degC

    c_r = calculate_newman_cp_consolidated(phi) * 1e6  # rock, 1/MPa
    c_w = CP.PropsSI('ISOTHERMAL_COMPRESSIBILITY', 'T', tref + 273.15, 'P', P_h * 1e6, 'Water') * 1e6  # brine, 1/MPa
    c_t = c_r + c_w

    rho_c = CP.PropsSI('D', 'T', tref + 273.15, 'P', P_h * 1e6, 'CO2')  # kg/m3

    V_pore = a * h * phi                         # m3
    return V_pore * c_t * rho_c * (P_lim - P_h) * 1e-12


def delta_p_nordbotten(r, psi, gamma, R, p_char_mpa):
    """Pressure buildup (MPa) around one well, after Nordbotten et al. (2005).

    Logarithmic in the brine region out to the radius of influence R (zero
    beyond), with the steady CO2-plume term (viscosity ratio gamma) inside the
    plume radius psi. p_char_mpa is the characteristic pressure Q mu_w / (2 pi k h).
    """
    PD = np.zeros_like(r, dtype=float)
    valid_r = ~np.isnan(r)
    in_plume = valid_r & (r <= psi)
    out_plume = valid_r & (r > psi) & (r <= R)

    PD[in_plume] = gamma * np.log(psi / r[in_plume]) + np.log(R / psi)
    PD[out_plume] = np.log(R / r[out_plume])

    PD = PD * p_char_mpa
    PD[np.isnan(r)] = np.nan
    return PD


def run_model_welloff(S_gdf, params):
    """Simulate one country: place and schedule wells, superpose their pressure
    fields each year and shut in any well whose surroundings exceed the limit.

    Returns (wells, drop_count, injection without shut-ins, injection with
    shut-ins, injection by basin, CO2 by basin and year its well stopped).
    Injection is in MtCO2/yr per model year; by-basin arrays are basins x years.
    """
    # ── Candidate well locations: every grid-cell centre inside a basin, in random order
    grid_points = gpd.GeoDataFrame(
        {"pt_id": np.arange(params['gx'].size, dtype=np.int32)},
        geometry=gpd.points_from_xy(params['gx'].ravel(), params['gy'].ravel()),
        crs=S_gdf.crs
    )
    cand_join = gpd.sjoin(grid_points, S_gdf[['geometry', 'well_fraction']],
                          how='inner', predicate='within').reset_index(drop=False)

    if params['L'] > len(cand_join):
        print(f"Warning: L ({params['L']}) exceeds available candidate cells ({len(cand_join)}).")
        params['L'] = len(cand_join)

    np.random.seed(42)  # reproducible well placement
    shuffle_idx = np.random.permutation(len(cand_join))
    candidate_locations = np.column_stack([cand_join.geometry.x.to_numpy()[shuffle_idx],
                                           cand_join.geometry.y.to_numpy()[shuffle_idx]])
    candidate_aquifer_idx = cand_join['index_right'].to_numpy(dtype=np.int32)[shuffle_idx]

    # ── Drilling schedule from the logistic curve
    wells, _ = generate_well_schedule_logistic(params['model_years'], params['L'], params['k_growth'],
                                               params['t0'], params['inj_years'],
                                               candidate_locations, [0])

    # ── Place each well in a basin that still has pressure-limited capacity.
    # If the basin at the next candidate is full, walk on to the next candidate
    # whose basin has room. Budgets only fall, so one cursor is enough.
    MCO2_budget = S_gdf['MCO2'].to_numpy(dtype=float).copy()
    aquifer_spent = np.zeros(len(S_gdf), dtype=float)
    kept_wells = []
    cand_cursor = 0

    for w in wells:
        while cand_cursor < len(candidate_aquifer_idx):
            aq = int(candidate_aquifer_idx[cand_cursor])
            inj_rate = float(S_gdf['inj_rate'].iloc[aq])
            well_contribution = inj_rate * params['inj_years'] * 0.001  # GtCO2 this well would store
            remaining = MCO2_budget[aq] - aquifer_spent[aq]
            if remaining > 0:
                aquifer_spent[aq] += min(well_contribution, remaining)
                w.update(w_id=len(kept_wells) + 1, aquifer_idx=aq,
                         p_frac=float(S_gdf['p_frac'].iloc[aq]), p_ref=float(S_gdf['p_ref'].iloc[aq]),
                         inj_rate=inj_rate,
                         x=candidate_locations[cand_cursor, 0], y=candidate_locations[cand_cursor, 1])
                kept_wells.append(w)
                cand_cursor += 1
                break
            cand_cursor += 1
        # a well with no basin left to go to is not drilled

    wells = kept_wells
    if len(wells) == 0:
        n_years = params['model_years']
        zeros = np.zeros(n_years)
        return [], 0, zeros, zeros, np.zeros((len(S_gdf), n_years)), np.zeros((len(S_gdf), n_years))

    D_arr = S_gdf['D'].to_numpy()
    Q_arr = S_gdf['Q'].to_numpy()
    phi_arr = S_gdf['phi'].to_numpy()
    h_arr = S_gdf['h'].to_numpy()
    omega_arr = S_gdf['omega'].to_numpy()
    gamma_arr = S_gdf['gamma'].to_numpy()
    p_c_arr = S_gdf['p_c'].to_numpy()

    well_shutdown_info = {}  # w_id -> first year off, and injection age at shut-in
    all_years = range(min(w['start_year'] for w in wells),
                      max(w['start_year'] for w in wells) + params['inj_years'] + 1)
    pressure_limit = params['plim'] * 100  # % of fracture pressure
    drop_count = 0

    wells_df = pd.DataFrame(wells)
    sy_arr = wells_df['start_year'].to_numpy()
    ey_arr = wells_df['end_year'].to_numpy()
    max_end_year = int(wells_df['end_year'].max())

    # ── Yearly loop
    for year in all_years:

        # Every well drilled so far contributes: active wells with their current
        # age, finished or shut-in wells held at their final pressure field
        sub = wells_df[(sy_arr <= year)]
        wells_in_year = []
        for well in sub.itertuples(index=False):
            is_post = year > well.end_year
            shutdown = well_shutdown_info.get(well.w_id, {'year': np.inf})
            wells_in_year.append({
                'w_id': well.w_id, 'x': well.x, 'y': well.y,
                'injection_age': (well.end_year - well.start_year + 1) if is_post else (year - well.start_year + 1),
                'active': (not is_post) and (year < shutdown['year']),
                'aquifer_idx': int(well.aquifer_idx), 'p_frac': well.p_frac, 'p_ref': well.p_ref,
            })

        if year > max_end_year and not any(w['active'] for w in wells_in_year):
            break

        pressure_ages = []
        for w in wells_in_year:
            if w['active']:
                pressure_ages.append(w['injection_age'])
            elif w['w_id'] in well_shutdown_info:
                pressure_ages.append(well_shutdown_info[w['w_id']]['age'])  # frozen at shut-in
            else:
                pressure_ages.append(params['inj_years'])                   # frozen at end of life
        t_s = np.array(pressure_ages, dtype=float) * YEAR_S

        # Superpose the pressure field of every well, cell by cell
        dP_out = np.zeros_like(params['gx'], dtype=float)
        for i, w in enumerate(wells_in_year):
            if t_s[i] <= 0:
                continue
            aq = w['aquifer_idx']
            R = np.sqrt(2.25 * D_arr[aq] * t_s[i])                              # radius of influence
            psi = np.exp(omega_arr[aq]) * np.sqrt(Q_arr[aq] * t_s[i] / (np.pi * phi_arr[aq] * h_arr[aq]))  # plume radius
            dist_local, mask_y, mask_x = local_distance_matrix(w['x'], w['y'], R, params['gx'], params['gy'])
            dP_out[np.ix_(mask_y, mask_x)] += delta_p_nordbotten(dist_local, psi, gamma_arr[aq], R, p_c_arr[aq])

        # Shut in (from next year) any active well with a cell above the limit
        # within its 25-year radius of influence
        for w in wells_in_year:
            if not w['active'] or w['w_id'] in well_shutdown_info:
                continue
            R_check = np.sqrt(2.25 * D_arr[w['aquifer_idx']] * params['inj_years'] * YEAR_S)
            dist_local, mask_y, mask_x = local_distance_matrix(w['x'], w['y'], R_check, params['gx'], params['gy'])
            p_pc = (dP_out[np.ix_(mask_y, mask_x)] + w['p_ref']) / w['p_frac'] * 100
            if p_pc.size > 0 and np.nanmax(p_pc) > pressure_limit:
                drop_count += 1
                well_shutdown_info[w['w_id']] = {'year': year + 1, 'age': w['injection_age']}

    # ── Injection series from each well's start and actual end year
    wells_df['shutdown_year'] = wells_df['w_id'].map(
        lambda wid: well_shutdown_info[wid]['year'] if wid in well_shutdown_info else np.nan)
    # shutdown_year is the first year off, so the last year on is shutdown_year - 1
    wells_df['actual_end_year'] = wells_df.apply(
        lambda row: min(row['end_year'], row['shutdown_year'] - 1)
        if pd.notna(row['shutdown_year']) else row['end_year'], axis=1)

    years_sim = np.array(list(all_years), dtype=int)
    sy = wells_df['start_year'].to_numpy()[:, None]
    ey = wells_df['end_year'].to_numpy()[:, None]
    ae = wells_df['actual_end_year'].to_numpy()[:, None]
    rates = wells_df['inj_rate'].to_numpy()[:, None]
    on_scheduled = (sy <= years_sim) & (ey >= years_sim)
    on_actual = (sy <= years_sim) & (ae >= years_sim)

    capacity_no_drop_mtpy = (on_scheduled * rates).sum(axis=0)
    capacity_with_drop_mtpy = (on_actual * rates).sum(axis=0)

    basin_idx = wells_df['aquifer_idx'].to_numpy(dtype=int)
    capacity_with_drop_by_basin = np.zeros((len(S_gdf), len(years_sim)))
    np.add.at(capacity_with_drop_by_basin, basin_idx, on_actual * rates)

    # CO2 stored by each well, booked to the year it stopped injecting
    well_mass = (on_actual * rates).sum(axis=1)
    end_idx = np.clip(np.searchsorted(years_sim, ae[:, 0]), 0, len(years_sim) - 1)
    stopped_mass_by_basin = np.zeros((len(S_gdf), len(years_sim)))
    np.add.at(stopped_mass_by_basin, (basin_idx, end_idx), well_mass)

    return (wells, drop_count, capacity_no_drop_mtpy, capacity_with_drop_mtpy,
            capacity_with_drop_by_basin, stopped_mass_by_basin)
