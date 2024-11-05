import numpy as np
from scipy import signal
from sources_daniel.custom_interpolation import nan_helper

def preprocess_with_butterworth_filter(y):

    #selection = np.where(y < 3500)
    #y[selection] = np.nan
    power_series_interpolated = np.copy(y)
    nans, x = nan_helper(power_series_interpolated)
    power_series_interpolated[nans] = np.interp(x(nans), x(~nans), power_series_interpolated[~nans])

    sig = power_series_interpolated
    sos = signal.butter(N=20, Wn=1, btype='lowpass', fs=100, output='sos')
    filtered_signal_3 = signal.sosfilt(sos, sig)

    #delay = 100
    #filtered_signal_3 = np.roll(filtered_signal_3, -delay)
    #filtered_signal_3[-delay:] = 0

    return filtered_signal_3

