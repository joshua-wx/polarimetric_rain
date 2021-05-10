import numpy as np
from numba import jit
from scipy.spatial import cKDTree

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

def KD_nn_interp(data_in, x_in, y_in, x_out, y_out, nnearest = 15, maxdist = None):
    """
    Nearest neighbour interpolation using scipy KDTree
    data_in: ndarray of float with shape (n1, n2)
        Data values to interpolate in input coordinate space
    x_in: ndarray of float with shape (n1, n2)
        x values of input coordinate space (e.g., require conversion from polar to Catesian first)
    y_in: ndarray of float with shape (n1, n2)
        y values of input coordinate space
    x_out: ndarray of float with shape (n1a, n2a)
        x values of output coordinate space
    y_out: ndarray of float with shape (n1a, n2a)
        x values of output coordinate space
    nnearest: int
        maximum number of nearest neighbours to consider when filling NaN values
    maxdist: float (in units of Cartesian space)
        maximum distance of nearest neighbours to consider when filling NaN values
        
        
    Returns: ndarray of float with shape (n1a, n2a)
    """
    
    def _make_coord_arrays(x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)
        Parameters
        ----------
        x : ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and
            the length of the ndarray corresponding to the number of points
        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif type(x) == np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 2:
                pass
            else:
                raise Exception("Cannot deal wih 3-d arrays, yet.")
        return x

    #transform output coordinates into pairs of coordiantes
    coord_out = _make_coord_arrays([x_out.ravel(), y_out.ravel()])
    vals_in = data_in.ravel()

    #build KDTree
    tree = cKDTree(np.c_[x_in.ravel(), y_in.ravel()])

    #query tree using output coordinates
    dists, idx = tree.query(coord_out, k=nnearest+1)
    # avoid bug, if there is only one neighbor at all
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idx = idx[:, np.newaxis]
    # get first neighbour

    vals_out = vals_in[idx[:, 0]]
    dists_cp = dists[..., 0].copy()

    # iteratively fill NaN with next neighbours
    isnan = np.isnan(vals_out)
    nanidx = np.argwhere(isnan)[..., 0]
    if nnearest > 1 & np.count_nonzero(isnan):
        for i in range(nnearest - 1):
            vals_out[isnan] = vals_in[idx[:, i + 1]][isnan]
            dists_cp[nanidx] = dists[..., i + 1][nanidx]
            isnan = np.isnan(vals_out)
            nanidx = np.argwhere(isnan)[..., 0]
            if not np.count_nonzero(isnan):
                break

    #apply max distance
    if maxdist is not None:
        vals_out = np.where(dists_cp > maxdist, np.nan, vals_out)

    return np.reshape(vals_out, x_out.shape)
