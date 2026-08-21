import numpy as np
import pandas as pd
from model.growth import generate_well_schedule_logistic
from model.utils import local_distance_matrix, calculate_newman_cp_consolidated
import geopandas as gpd
import CoolProp.CoolProp as CP

def pressure_space(a,h,z,phi,plim):

    # Gradients must stay in step with h_grad/l_grad/t_grad/t_surf in runCO2LOGIX
    P_h = z/1000 * 10 # hydrostatic pressure, MPa
    P_l = z/1000 * 23 # lithostatic pressure, MPa
    P_f = P_h + (0.75 * (P_l-P_h))
    P_lim = plim * P_f

    tref = 25 * z/1000 + 15 # reference temperature at aquifer depth (degC)

    c_r = calculate_newman_cp_consolidated(phi) * 1e6 # rock compressibility, MPa-1
    c_w = CP.PropsSI('ISOTHERMAL_COMPRESSIBILITY', 'T', tref + 273.15, 'P', P_h * 1e6, 'Water') * 1e6 # brine compressibility, MPa-1
    c_t = c_r + c_w # total compressibility, MPa-1

    rho_c = CP.PropsSI('D', 'T', tref + 273.15, 'P', P_h * 1e6, 'CO2')

    P_allow = P_lim-P_h
    V_pore = a*h*phi # pore volume, m3
    M_pres = V_pore*c_t*rho_c*P_allow # pressure capacity, Kg
    M_pres = M_pres * 1E-12 # gigatonnes CO2

    return M_pres
             
def delta_p_nordbotten(r, psi, gamma, R, rc, p_char_mpa, domain):
    
    # Initialize output array with zeros
    PD = np.zeros_like(r, dtype=float)

    # Define masks for where to perform each calculation
    
    # Mask where distance matrix contains a number
    valid_r = ~np.isnan(r)
    
    # Mask for region inside plume radius (r <= psi)
    in_plume = valid_r & (r <= psi)
    

    # Open boundary calculation
    
    if domain=='open':
        
        # mask for outside plume but within R
        
        out_plume = valid_r & (r > psi) & (r <= R)
        
        # PD within CO2 plume
        
        PD[in_plume] = gamma * np.log(psi/r[in_plume]) + np.log(R / psi)
        
        # PD outside CO2 plume
        
        PD[out_plume] = np.log(R / r[out_plume])
        
    # Closed boundary calculation
        
    if domain == 'closed':
        
        # mask for outside plume but within rc
        
        out_plume = valid_r & (r > psi) & (r <= rc)
        
        # PD within CO2 plume
        
        PD[in_plume] = gamma * np.log(psi/r[in_plume]) + np.log(rc / psi) + ((2 * R**2)/(2.25 * rc**2)) - (3/4)
        
        # PD outside plume
        
        PD[out_plume] = np.log(rc / r[out_plume]) + ((2 * R**2)/(2.25 * rc**2)) - (3/4)
    
    # multiply by characteristic pressure
    
    PD = PD * p_char_mpa

    PD[np.isnan(r)] = np.nan

    return PD


def run_model_welloff(S_gdf, params):
    
    # Candidate locations
    grid_points = gpd.GeoDataFrame(
        {"pt_id": np.arange(params['gx'].size, dtype=np.int32)},
        geometry=gpd.points_from_xy(params['gx'].ravel(), params['gy'].ravel()),
        crs=S_gdf.crs
    )

    cand_join = gpd.sjoin(
        grid_points,
        S_gdf[['geometry', 'well_fraction']],
        how='inner',
        predicate='within'
    ).reset_index(drop=False)   # keeps original grid point index as 'index'

    if params['L'] > len(cand_join):
        print(f"Warning: L ({params['L']}) exceeds available candidate cells ({len(cand_join)}). Consider reducing grid_size.")
        params['L'] = len(cand_join)

    np.random.seed(42)       # reproducible well placement across runs
    shuffle_idx = np.random.permutation(len(cand_join))

    params['candidate_locations'] = np.column_stack([
        cand_join.geometry.x.to_numpy()[shuffle_idx],
        cand_join.geometry.y.to_numpy()[shuffle_idx]
    ])

    params['candidate_aquifer_idx'] = cand_join['index_right'].to_numpy(dtype=np.int32)[shuffle_idx]


    used_index = [0]

    

    # Well schedule
    wells, wells_per_year = generate_well_schedule_logistic(
        params['model_years'],
        params['L'],
        params['k_growth'],
        params['t0'],
        params['inj_years'],
        params['candidate_locations'],
        used_index
    )
    
    
        
    # New - a check to make sure that wells are only placed in an aquifer
    # if there is space to do so

    MCO2_budget   = S_gdf['MCO2'].to_numpy(dtype=float).copy()  # MtCO2 per aquifer polygon
    aquifer_spent = np.zeros(len(S_gdf), dtype=float)            # MtCO2 committed so far

    kept_wells       = []
    wells_dropped = 0
    cand_cursor = 0          # cursor walks forward through candidates; never backtracks

    # loop extracts kept and dropped wells

    for i, w in enumerate(wells, 1):
        # Try to place this well starting from the preferred (area-weighted) candidate.
        # If that aquifer is full, walk forward to find the next candidate whose
        # aquifer still has room. Budgets only decrease, so once an aquifer is full
        # it remains full; a single cursor suffices.
        placed = False

        while cand_cursor < len(params['candidate_aquifer_idx']):
            aq = int(params['candidate_aquifer_idx'][cand_cursor])
            inj_rate = float(S_gdf['inj_rate'].iloc[aq])
            well_contribution = inj_rate * params['inj_years'] * 0.001    # MtCO2 this well would store
            remaining = MCO2_budget[aq] - aquifer_spent[aq]

            if remaining > 0:
                # Found an aquifer with room; place the well there
                aquifer_spent[aq] += min(well_contribution, remaining)
                w['w_id']        = len(kept_wells) + 1
                w['aquifer_idx'] = aq
                w['p_frac']      = float(S_gdf['p_frac'].iloc[aq])
                w['p_ref']       = float(S_gdf['p_ref'].iloc[aq])
                w['inj_rate']    = inj_rate
                # Update location to the candidate actually used (in case we skipped ahead)
                w['x'] = params['candidate_locations'][cand_cursor, 0]
                w['y'] = params['candidate_locations'][cand_cursor, 1]
                kept_wells.append(w)
                cand_cursor += 1
                placed = True
                break

            cand_cursor += 1

        if not placed:
            # Ran out of candidates; no aquifer with room
            wells_dropped += 1

    # if wells_dropped > 0:
    #     print(f"  MCO2 ceiling: {wells_dropped} wells removed before simulation "
    #           f"({len(kept_wells)} retained)")

    wells = kept_wells

    # Guard: if every well was removed by the MCO2 ceiling, return zeroed outputs
    if len(wells) == 0:
        n_years = params['model_years']
        zeros   = np.zeros(n_years)
        return [], [], [], [], 0, [], zeros, zeros

    # Main delta P calculation arrays
    D_arr     = S_gdf['D'].to_numpy()
    Q_arr     = S_gdf['Q'].to_numpy()
    phi_arr   = S_gdf['phi'].to_numpy()
    h_arr     = S_gdf['h'].to_numpy()
    omega_arr = S_gdf['omega'].to_numpy()
    gamma_arr = S_gdf['gamma'].to_numpy()
    p_c_arr   = S_gdf['p_c'].to_numpy()

    

    # Shutdown tracking
    well_shutdown_info = {}

    all_years = range(
        min(w['start_year'] for w in wells),
        max(w['start_year'] for w in wells) + params['inj_years'] + 1
    )

    # Outputs
    # Storing every yearly grid costs (grid cells x 8 bytes x ~125 years) —
    # multiple GB for large countries — so it is opt-in via params.
    store_grids = params.get('store_dP_grids', False)
    all_dP_grids = []
    pressure_limit = params['plim']*100
    max_dP_frac_pc = []
    drop_count = 0
    
    wells_df_full = pd.DataFrame(wells)
    sy_arr = wells_df_full['start_year'].to_numpy()
    ey_arr = wells_df_full['end_year'].to_numpy()
    
    max_end_year = int(wells_df_full['end_year'].max())

    # ---- Main yearly loop (FIX 2: build wells_in_year dynamically each year) ----
    for year in all_years:
        
        in_window = (sy_arr <= year) & (ey_arr >= year)
        post      = ey_arr < year
        contrib   = in_window | post
        sub       = wells_df_full[contrib]
        
        wells_in_year = []
        
        for well in sub.itertuples(index=False):
            is_post   = year > well.end_year
            shutdown  = well_shutdown_info.get(well.w_id, {'year': np.inf})
            is_active = (not is_post) and (year < shutdown['year'])

            wells_in_year.append({
                'w_id':          well.w_id,
                'x':             well.x,
                'y':             well.y,
                'start_year':    well.start_year,
                'end_year':      well.end_year,
                'injection_age': (well.end_year - well.start_year + 1) if is_post else (year - well.start_year + 1),
                'active':        is_active,
                'aquifer_idx':   int(well.aquifer_idx),
                'p_frac':        well.p_frac,
                'p_ref':         well.p_ref,
                })
            
        if year > max_end_year and not any(w['active'] for w in wells_in_year):
            break

        dP_out = np.zeros_like(params['gx'], dtype=float)

        xs = [w['x'] for w in wells_in_year]
        ys = [w['y'] for w in wells_in_year]
        aquifer_idxs = [int(w['aquifer_idx']) for w in wells_in_year]
        pressure_ages = []

        # Pressure ages (active wells grow; inactive wells freeze)
        for w in wells_in_year:
            if w['active']:
                pressure_ages.append(w['injection_age'])
            else:
                info = well_shutdown_info.get(w['w_id'], None)
                if info:
                    pressure_ages.append(info['age'])  # freeze at shutdown age
                else:
                    # naturally ended wells (or any inactive without shutdown record)
                    pressure_ages.append(params['inj_years'])

        t_s = np.array(pressure_ages, dtype=float) * 86400 * 365
        num_wells = len(xs)

        # Main dP calculation (superposition of all wells incl. frozen inactive)
        for i in range(num_wells):
            if t_s[i] <= 0:
                continue
            
            D = D_arr[aquifer_idxs[i]]
            Q = Q_arr[aquifer_idxs[i]]
            phi = phi_arr[aquifer_idxs[i]]
            h = h_arr[aquifer_idxs[i]]
            omega = omega_arr[aquifer_idxs[i]]
            gamma = gamma_arr[aquifer_idxs[i]]
            p_c = p_c_arr[aquifer_idxs[i]]

            R = np.sqrt(2.25 * D * t_s[i])
            csi = np.sqrt(Q * t_s[i] / (np.pi * phi * h))
            psi = np.exp(omega) * csi

            if params['domain'] == 'open':
                dist_local, mask_y, mask_x = local_distance_matrix(xs[i], ys[i], R, params['gx'], params['gy'])
            elif params['domain'] == 'closed':
                # preserving your prior behavior (you had open-style call here)
                dist_local, mask_y, mask_x = local_distance_matrix(xs[i], ys[i], R, params['gx'], params['gy'])
            else:
                raise ValueError("params['domain'] must be 'open' or 'closed'")

            dP_local = delta_p_nordbotten(
                dist_local,
                psi,
                gamma,
                R,
                params['rc'],
                p_c,
                params['domain'],
            )

            dP_out[np.ix_(mask_y, mask_x)] += dP_local

        # Shut-in logic: if local max in influence area exceeds pressure_limit, shut from next year
        for val in wells_in_year:
            w_id = val['w_id']
            if not val['active']:
                continue

            D = D_arr[int(val['aquifer_idx'])]
            # use maximum pressure influence radius (25y) for screening (your original logic)
            R_check = np.sqrt(2.25 * D * 25 * 86400 * 365)

            dist_local, mask_y, mask_x = local_distance_matrix(
                val['x'], val['y'], R_check, params['gx'], params['gy']
            )

            dP_local = dP_out[np.ix_(mask_y, mask_x)]

            # Normalize to fracture pressure
            dP_local = (dP_local + val['p_ref']) / val['p_frac'] * 100

            if dP_local.size > 0 and np.nanmax(dP_local) > pressure_limit:
                if w_id not in well_shutdown_info:
                    drop_count += 1
                    #print(f"well dropped: {w_id} at year {year}, age={val['injection_age']}")
                    well_shutdown_info[w_id] = {
                        'year': year + 1,          # first year it is OFF
                        'age': val['injection_age']  # freeze pressure at this age
                    }

        # Store outputs
        dP_out[dP_out == 0] = np.nan
        

        dP_frac_pc = ((dP_out + params['p_ref_grid']) / params['p_frac_grid']) * 100
        if store_grids:
            all_dP_grids.append(dP_frac_pc)
        max_dP_frac_pc.append(np.nanmax(dP_frac_pc))


    # Summarise actual injecting counts from shutdown-aware well table
    wells_df = wells_df_full

    wells_df['shutdown_year'] = wells_df['w_id'].map(
        lambda wid: well_shutdown_info[wid]['year'] if wid in well_shutdown_info else np.nan
    )

    # shutdown_year is first OFF year, so last ON year is shutdown_year - 1
    wells_df['actual_end_year'] = wells_df.apply(
        lambda row: min(row['end_year'], row['shutdown_year'] - 1)
        if pd.notna(row['shutdown_year']) else row['end_year'],
        axis=1
    )

    years_sim = np.array(list(all_years), dtype=int)

    # Capacity calculations while accounting for variable injection rate

    sy    = wells_df['start_year'].to_numpy()[:, None]        # (W, 1)
    ey    = wells_df['end_year'].to_numpy()[:, None]
    ae    = wells_df['actual_end_year'].to_numpy()[:, None]
    rates = wells_df['inj_rate'].to_numpy()[:, None]          # (W, 1)
    
    # Replace the two for-y loops with:
    actual_injecting              = ((sy <= years_sim) & (ae >= years_sim)).sum(axis=0).tolist()
    original_injecting_from_schedule = ((sy <= years_sim) & (ey >= years_sim)).sum(axis=0).tolist()

    # (W, T) boolean × (W, 1) rates → sum over W axis
    capacity_no_drop_mtpy   = (((sy <= years_sim) & (ey >= years_sim)) * rates).sum(axis=0)
    capacity_with_drop_mtpy = (((sy <= years_sim) & (ae >= years_sim)) * rates).sum(axis=0)

    return all_dP_grids, actual_injecting, original_injecting_from_schedule, max_dP_frac_pc, drop_count, wells,capacity_no_drop_mtpy,capacity_with_drop_mtpy

