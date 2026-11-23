"""Basins, reservoir and fluid properties, capacity and injection rate for each country (Sections 2.1.3 and 2.2)."""

import os
import geopandas as gpd
import numpy as np
import CoolProp.CoolProp as CP
from pathlib import Path

from model.pressure import pressure_space
from model.utils import estimate_frac_pres, calculate_newman_cp_consolidated
from model.utils import H_GRAD, L_GRAD, T_GRAD, T_SURF

# Repository root, so input paths resolve no matter which directory the model
# is launched from, on any operating system.
ROOT = Path(__file__).resolve().parent.parent
OTHER_MODELS_DIR = ROOT / 'inputs' / 'other_models'
SMITH_DIR        = ROOT / 'inputs' / 'smith_shapefiles'

# Pressure limit as a fraction of fracture pressure. 0.8 is the central case;
# the paper's sensitivity runs set CO2LOGIX_PLIM to 0.7 or 0.9. Read at import
# so the worker processes, which inherit the environment, agree.
PLIM = float(os.environ.get('CO2LOGIX_PLIM', 0.8))

# Injectivity coefficient of Valluri et al. (2021): the lower bound, in t/yr/psi
# per mD.ft, converted to t/yr/MPa per mD.m below (about 14)
ALPHA = 0.03

# Injection life of each well, years
INJ_YEARS = 25

# Grid cell size (m): 2 km where a country's basins exceed 10,000 km2, else 1 km
GRID_LARGE, GRID_SMALL, GRID_AREA_KM2 = 2000, 1000, 10000


def load_country_basins(c, plim=PLIM):
    """Read the basins of country `c` and their pressure-limited capacity.

    Uses a national dataset from inputs/other_models where one exists (UK: de
    Jonge-Anderson et al., 2026; USA: Rodriguez Calzado et al., 2026), and
    otherwise the global dataset of Gao et al. (2026) and Smith et al. (2024)
    in inputs/smith_shapefiles. Each dataset's columns are mapped to phi
    (porosity, fraction), h (net thickness, m), k_md (permeability, mD) and
    z (depth, m). `MCO2` (GtCO2) is the capacity of each basin.

    Returns (S_gdf, grid_size), or (None, None) if the country has no data.
    """
    filepath = OTHER_MODELS_DIR / f'{c}.shp'
    if filepath.is_file():
        S_gdf = gpd.read_file(filepath)

        if c == 'UK':
            S_gdf['phi']  = S_gdf['Poro_frac']
            S_gdf['h']    = S_gdf['Net_thk_m']
            S_gdf['k_md'] = S_gdf['Perm_mD']
            S_gdf['z']    = S_gdf['Depth_m']
            S_gdf['area__km2_'] = S_gdf["Area_m2"] / 1e6

        if c == 'USA':
            S_gdf['phi']  = S_gdf['MEAN_Poros'] / 100
            S_gdf['h']    = S_gdf['MEAN_NetPo']
            S_gdf['k_md'] = S_gdf['MEAN_Perme']
            S_gdf['z']    = S_gdf['MEAN_DF_m']
            S_gdf["Area_m2"] = S_gdf['Shape_Area']
            S_gdf['area__km2_'] = S_gdf['Shape_Area'] / 1e6

        S_gdf["MCO2"] = pressure_space(S_gdf["Area_m2"].to_numpy(), S_gdf["h"].to_numpy(),
                                       S_gdf["z"].to_numpy(), S_gdf["phi"].to_numpy(), plim)
        grid_size = GRID_LARGE if sum(S_gdf["area__km2_"]) > GRID_AREA_KM2 else GRID_SMALL
        return S_gdf, grid_size

    filepath = SMITH_DIR / f'{c}.shp'
    if filepath.is_file():
        S_gdf = gpd.read_file(filepath)   # phi in %, h in m, k in mD, z in m

        S_gdf["MCO2"] = pressure_space(S_gdf["a_km2"].to_numpy() * 1e6, S_gdf["h"].to_numpy(),
                                       S_gdf["z"].to_numpy(), S_gdf["phi"].to_numpy() / 100, plim)
        S_gdf['phi']  = S_gdf['phi'] / 100
        S_gdf['k_md'] = S_gdf['k']
        grid_size = GRID_LARGE if sum(S_gdf["a_km2"]) > GRID_AREA_KM2 else GRID_SMALL
        return S_gdf, grid_size

    return None, None


def add_reservoir_properties(S_gdf, plim=PLIM):
    """Attach fluid properties, pressures and the per-well injection rate to `S_gdf`.

    Adds `well_fraction` (each basin's share of the country's area), `p_ref`
    and `p_frac` (hydrostatic and fracture pressure, MPa), the inputs of the
    pressure solution (`D`, `Q`, `gamma`, `omega`, `p_c`) and `inj_rate`
    (MtCO2/yr per well, Eq. 2 of the paper, capped at 2).
    """
    S_gdf['area_m2']       = S_gdf.geometry.area
    S_gdf['well_fraction'] = S_gdf['area_m2'] / S_gdf['area_m2'].sum()

    S_gdf['k']   = S_gdf['k_md'] * 9.869233e-16                    # permeability, m2
    S_gdf['c_r'] = calculate_newman_cp_consolidated(S_gdf['phi'])  # rock compressibility, 1/Pa

    S_gdf['p_ref'] = H_GRAD * S_gdf['z'] / 1000
    S_gdf['t_ref'] = T_GRAD * S_gdf['z'] / 1000 + T_SURF

    S_gdf['c_w'] = CP.PropsSI('ISOTHERMAL_COMPRESSIBILITY', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'Water')

    S_gdf['u_w']   = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'Water')
    S_gdf['u_c']   = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')
    S_gdf['rho_c'] = CP.PropsSI('D', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')


    S_gdf['c_tot'] = S_gdf['phi'] * (S_gdf['c_r'] + S_gdf['c_w']) # total bulk storativity, Pa-1
    S_gdf['gamma'] = S_gdf['u_c'] / S_gdf['u_w']
    S_gdf['omega'] = ((S_gdf['u_c'] + S_gdf['u_w']) / (S_gdf['u_c'] - S_gdf['u_w']) *
                      np.log(np.sqrt(S_gdf['gamma'])) - 1.0)
    S_gdf['D']     = S_gdf['k'] / (S_gdf['c_tot'] * S_gdf['u_w'])

    S_gdf['sv']     = L_GRAD * (S_gdf['z'] / 1000)
    S_gdf['p_frac'] = estimate_frac_pres(S_gdf['sv'], S_gdf['p_ref'])

    # Injectivity J = alpha k h (Valluri et al., 2021) and injection rate
    # I = J x allowable pressure increase (Eqs 1-2 of the paper)
    S_gdf['kh']           = S_gdf['k_md'] * S_gdf['h']
    conversion_factor     = 3.28 * 145.038   # ft/m x psi/MPa
    alpha                 = ALPHA * conversion_factor
    S_gdf['J']            = alpha * S_gdf['kh']
    S_gdf['dP_allowable'] = (S_gdf['p_frac'] * plim) - S_gdf['p_ref']
    S_gdf['inj_rate']     = (S_gdf['J'] * S_gdf['dP_allowable'] / 1e6).clip(upper=2)
    S_gdf['Q']            = S_gdf['inj_rate'] * 1e9 / S_gdf['rho_c'] / 365 / 86400
    S_gdf['p_c']          = (S_gdf['Q'] * S_gdf['u_w']) / (2.0 * np.pi * S_gdf['h'] * S_gdf['k'])  / 1e6

    return S_gdf
