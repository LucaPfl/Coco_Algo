

import numpy as np
from matplotlib import pyplot as plt


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def mwe_interpolate():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate
    x = np.arange(0, 10)
    y = np.exp(-x / 3.0)
    y[1] = np.nan
    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, 9, 0.1)
    ynew = f(xnew)  # use interpolation function returned by `interp1d`
    plt.plot(x, y, 'o', xnew, ynew, '-')


def mwe_interpolate_nans():
    """
    Interpolate with Nans
    Minimal Working Example (MWE)
    """
    # source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    y = np.array([1, 1, 1, np.nan, np.nan, 2, 2, np.nan, 0])
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    print(y.round(2))


if __name__ == '__main__':
    mwe_interpolate()
    mwe_interpolate_nans()
    plt.show()

