import numpy as np

def calculate_newman_cp_consolidated(phi_frac):
    """
    Calculates pore volume compressibility (cp) for consolidated sandstone
    using the Newman (1973) correlation converted to SI units.

    Parameters:
    phi_frac (float): Porosity as a fraction (e.g., 0.15 for 15%)

    Returns:
    float: Pore volume compressibility (cp) in Pa^-1
    """
    # Constants converted for Pa^-1 and fractional porosity
    a = 1.4115e-8
    b = 55.8721
    c = 1.4286

    cp = a / (1 + b * phi_frac)**c
    return cp

def local_distance_matrix(x, y, R, gx, gy):

    # Mask of cells within R in x and y
    mask_x = (gx[0, :] >= x - R) & (gx[0, :] <= x + R)
    mask_y = (gy[:, 0] >= y - R) & (gy[:, 0] <= y + R)

    # Grid subset
    gx_local = gx[np.ix_(mask_y, mask_x)]
    gy_local = gy[np.ix_(mask_y, mask_x)]

    # Distance from (x, y) to local grid
    dist_local = np.sqrt((gx_local - x)**2 + (gy_local - y)**2)
    dist_local[dist_local==0]=0.1
    # Also return the offset (so you can place into the full dP grid)
    return dist_local, mask_y, mask_x

def estimate_frac_pres(sv,pp):
    # Based on the work of Zhang & Yin (2017) who found that on average
    # Fracture pressure was 0.75 x the difference between pp and sv
   # simple assumption that fracture pressure is 2/3 of the distance from phyd to sv
    return pp + (0.75 * (sv-pp))
