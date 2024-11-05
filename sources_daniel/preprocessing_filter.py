

import numpy as np
from scipy import signal
from pandas.core.nanops import nanmean as pd_nanmean


def apply_fir_filter(in_data, filter_coeffs):

    h = filter_coeffs
    in_buffer = np.ones(len(h)) * in_data[0]
    out = np.empty(len(in_data))

    for k in range(len(out)):
        in_buffer = np.roll(in_buffer, 1)
        in_buffer[0] = in_data[k]
        out[k] = np.dot(in_buffer, h)

    return out


def _pad_nans(x, head=None, tail=None):
    if head is None and tail is None:
        return x
    elif head and tail:
        return np.r_[[np.nan] * head, x, [np.nan] * tail]
    elif tail is None:
        return np.r_[[np.nan] * head, x]
    elif head is None:
        return np.r_[x, [np.nan] * tail]


def convolution_filter(x, filt, nsides=2):

    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt)/2.) - 1) or None
        trim_tail = int(np.ceil(len(filt)/2.) - len(filt) % 2) or None
    else:
        raise ValueError

    result = signal.convolve(x, filt, mode='valid')
    result = _pad_nans(result, trim_head, trim_tail)
    return result


def calc_period_filter_coefficients(period):
    if period % 2 == 0:
        filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
    else:
        filt = np.repeat(1.0 / period, period)
    return filt


def period_filter(x, period=350, two_sided=True):

    filt = calc_period_filter_coefficients(period)

    if two_sided:
        nsides = int(two_sided) + 1
    else:
        nsides = 1

    trend = convolution_filter(x, filt, nsides)

    return trend


def seasonal_mean(x, period):
    """
    Return means for each period in x. period is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    return np.array([pd_nanmean(x[i::period], axis=0) for i in range(period)])


def custom_decompose(x, period=350):

    trend = period_filter(x, period, two_sided=True)

    detrended = x - trend

    period_averages = seasonal_mean(detrended, period)

    period_averages -= np.mean(period_averages, axis=0)

    seasonal = np.tile(period_averages.T, len(x) // period + 1).T[:len(x)]

    resid = detrended - seasonal

    return trend, seasonal, resid




if __name__ == '__main__':
    print(calc_period_filter_coefficients(10))
