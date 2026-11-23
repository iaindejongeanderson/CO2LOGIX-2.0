"""Constants and small helpers shared by the geology and pressure modules."""
import numpy as np

# Pressure and temperature gradients (Table 1 of the paper). Defined once here
# so the capacity estimate and the reservoir properties cannot drift apart.
# The hydrostatic gradient is that of saline formation water (~0.465 psi/ft).
H_GRAD = 10.5  # hydrostatic, MPa/km
L_GRAD = 23    # lithostatic (vertical stress), MPa/km
T_GRAD = 25    # geothermal, degC/km
T_SURF = 15    # surface temperature, degC


def calculate_newman_cp_consolidated(phi_frac):
    """Pore volume compressibility (1/Pa) of consolidated sandstone from porosity
    (fraction), using the Newman (1973) correlation in SI units."""
    a = 1.4115e-8
    b = 55.8721
    c = 1.4286
    return a / (1 + b * phi_frac)**c


def local_distance_matrix(x, y, R, gx, gy):
    """Distances (m) from a well at (x, y) to the grid cells within R of it.

    Returns the distances and the row and column masks that place them in the
    full grid. A well sits at a cell centre, so its own cell is evaluated at
    0.1 m (the wellbore) rather than at zero.
    """
    mask_x = (gx[0, :] >= x - R) & (gx[0, :] <= x + R)
    mask_y = (gy[:, 0] >= y - R) & (gy[:, 0] <= y + R)

    gx_local = gx[np.ix_(mask_y, mask_x)]
    gy_local = gy[np.ix_(mask_y, mask_x)]

    dist_local = np.sqrt((gx_local - x)**2 + (gy_local - y)**2)
    dist_local[dist_local == 0] = 0.1
    return dist_local, mask_y, mask_x


def estimate_frac_pres(sv, pp):
    """Fracture pressure: pore pressure plus 0.75 x (vertical stress - pore
    pressure), the central coefficient of Zhang & Yin (2017)."""
    return pp + (0.75 * (sv - pp))
