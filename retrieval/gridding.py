import numpy as np
from numba import jit

@jit
def grid_data(data, xradar, yradar, xgrid, ygrid, theta_3db=1.5, rmax=150e3, gatespacing=250):
    """
    This function grid the polar data onto a Cartesian grid.
    This gridding technique is made to
    properly handle the absence of data while other
    gridding techniques tend to propagate NaN values.
    Parameters:
    ===========
    data: <ny, nx>
        Data to grid
    xradar: <ny, nx>
        x-axis Cartesian coordinates array of the input data
    yradar: <ny, nx>
        y-axis Cartesian coordinates array of the input data
    xgrid: <ny_out, nx_out>
        x-axis Cartesian coordinates array for the output data
    ygrid: <ny_out, nx_out>
        y-axis Cartesian coordinates array for the output data
    theta_3db: float
        Maximum resolution angle in degrees for polar coordinates.
    rmax: float
        Maximum range of the data (same unit as x/y).
    gatespacing: float
        Gate-to-gate resolution (same unit as x/y).
    Returns:
    ========
    eth_out: <ny_out, nx_out>
        Gridded data.
    """
    if xradar.shape != data.shape:
        raise IndexError("Bad dimensions")

    if len(xgrid.shape) < len(xradar.shape):
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)

    data_out = np.zeros(xgrid.shape) + np.NaN

    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            cnt = 0
            zmax = 0
            xi = xgrid[j, i]
            yi = ygrid[j, i]

            if xi ** 2 + yi ** 2 > rmax ** 2:
                continue

            width = 0.5 * (np.sqrt(xi ** 2 + yi ** 2) * theta_3db * np.pi / 180)
            if width < gatespacing:
                width = gatespacing

            for k in range(data.shape[1]):
                for l in range(data.shape[0]):
                    xr = xradar[l, k]
                    yr = yradar[l, k]

                    if (
                        xr >= xi - width
                        and xr < xi + width
                        and yr >= yi - width
                        and yr < yi + width
                    ):
                        if data[l, k] > 0:
                            zmax = zmax + data[l, k]
                            cnt = cnt + 1

            if cnt != 0:
                data_out[j, i] = zmax / cnt

    return data_out