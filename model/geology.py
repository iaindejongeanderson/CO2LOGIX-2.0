"""Per-country geological setup shared by the full simulation and the
standalone growth-parameter export.

Reading a country's basins and deriving their reservoir, fluid and
injectivity properties is a prerequisite for both the pressure model and the
logistic parameters (L depends on MCO2 and inj_rate, t0 on the area-weighted
mean injection rate). Keeping it in one place means the exported parameters
cannot drift from the ones the model actually runs on.
"""

import geopandas as gpd
import numpy as np
import CoolProp.CoolProp as CP
from pathlib import Path

from model.pressure import pressure_space
from model.utils import estimate_frac_pres, calculate_newman_cp_consolidated

# Repository root, so input paths resolve no matter which directory the model
# is launched from.
ROOT = Path(__file__).resolve().parent.parent

# Input datasets, relative to ROOT. Built with pathlib rather than literal
# separators so the paths resolve on Linux and macOS as well as Windows.
OTHER_MODELS_DIR = ROOT / 'inputs' / 'other_models'
SMITH_DIR        = ROOT / 'inputs' / 'smith_shapefiles'

# Fraction of the fracture pressure available for injection
PLIM = 0.8

# Gradients must stay in step with those hard-coded in pressure_space
H_GRAD = 10   # hydrostatic, MPa/km
L_GRAD = 23   # lithostatic, MPa/km
T_GRAD = 25   # geothermal, degC/km
T_SURF = 15   # surface temperature, degC

# Injectivity coefficient (Valluri et al.), field units before conversion
ALPHA = 0.03

# Years a single well injects for, used to convert capacity into well-years
INJ_YEARS = 25


def load_country_basins(c, plim=PLIM):
    """Read the best available basin shapefile for country `c`.

    Preferentiates the UK and USA sepeate models over the global Smith et
    al. (2024) dataset. Returns (S_gdf, grid_size), or (None, None) if no
    geodata exists for the country. `MCO2` (GtCO2 of pressure-limited
    capacity) is attached per basin.
    """

    # Check for detailed model (e.g. UK aquifers)
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
            S_gdf['phi']  = S_gdf['MEAN_Poros']/100
            S_gdf['h']    = S_gdf['MEAN_NetPo']
            S_gdf['k_md'] = S_gdf['MEAN_Perme']
            S_gdf['z']    = S_gdf['MEAN_DF_m']
            S_gdf["Area_m2"] = S_gdf['Shape_Area']
            S_gdf['area__km2_'] = S_gdf['Shape_Area'] / 1e6

        S_gdf["MCO2"] = pressure_space( # Returns in Gigatonnes
            S_gdf["Area_m2"].to_numpy(),
            S_gdf["h"].to_numpy(),
            S_gdf["z"].to_numpy(),
            S_gdf["phi"].to_numpy(),
            plim
        )

        if sum(S_gdf["area__km2_"]) > 10000:
            grid_size = 2000
        else:
            grid_size = 1000

        return S_gdf, grid_size

    # Else use the Smith et al (2024) dataset
    filepath = SMITH_DIR / f'{c}.shp'

    if filepath.is_file():

        S_gdf = gpd.read_file(filepath)

        S_gdf["MCO2"] = pressure_space( # Returns in Gigatonnes
            S_gdf["a_km2"].to_numpy() * 1e6,
            S_gdf["h"].to_numpy(),
            S_gdf["z"].to_numpy(),
            S_gdf["phi"].to_numpy() / 100,
            plim
        )

        S_gdf['phi']  = S_gdf['phi'] / 100
        #S_gdf['h']    = S_gdf['mean_thick']
        S_gdf['k_md'] = S_gdf['k']
        #S_gdf['z']    = S_gdf['mean_reser']

        if sum(S_gdf["a_km2"]) > 10000:
            grid_size = 2000
        else:
            grid_size = 1000

        return S_gdf, grid_size

    return None, None


def add_reservoir_properties(S_gdf, plim=PLIM):
    """Attach fluid properties, well weighting and injectivity to `S_gdf`.

    Adds `well_fraction` (area weighting used to place wells) and `inj_rate`
    (MtCO2/yr per well), which together with `MCO2` set the logistic
    carrying capacity and midpoint.
    """

    S_gdf['area_m2']       = S_gdf.geometry.area
    S_gdf['well_fraction'] = S_gdf['area_m2'] / S_gdf['area_m2'].sum()

    # ── Pre-computations ──────────────────────────────────────────────────────

    S_gdf['k']     = S_gdf['k_md'] * 9.869233e-16
    S_gdf['c_r'] = calculate_newman_cp_consolidated(S_gdf['phi']) # in Pa-1

    # This WOULD output in Pa^-1

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

    # ── Injectivity (Valluri et al.) ──────────────────────────────────────────

    S_gdf['kh']           = S_gdf['k_md'] * S_gdf['h']
    conversion_factor     = 3.28 * 145.038
    alpha                 = ALPHA * conversion_factor
    S_gdf['J']            = alpha * S_gdf['kh']
    S_gdf['dP_allowable'] = (S_gdf['p_frac'] * plim) - S_gdf['p_ref']
    S_gdf['inj_rate']     = (S_gdf['J'] * S_gdf['dP_allowable'] / 1e6).clip(upper=2)
    S_gdf['Q']            = S_gdf['inj_rate'] * 1e9 / S_gdf['rho_c'] / 365 / 86400
    S_gdf['p_c']          = (S_gdf['Q'] * S_gdf['u_w']) / (2.0 * np.pi * S_gdf['h'] * S_gdf['k'])  / 1e6

    return S_gdf
