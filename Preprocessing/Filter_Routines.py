# -*- coding: utf-8 -*-
"""
integration of filter algorithm
"""
import numpy as np
from scipy import signal

from sources_daniel.custom_interpolation import nan_helper
from sources_daniel.preprocessing_filter import period_filter


def preprocess_with_period_filter(power_series,plot_offset,filter_period,lower_cutoff_threshold=2000):

    orig_series = power_series.copy()

    filter_period_half = int(filter_period / 2)

    filtered_signal_2 = period_filter(power_series, period=filter_period, two_sided=False)
    filtered_2 = filtered_signal_2.astype(np.float)
    selection = np.where(power_series < lower_cutoff_threshold)
    # print('selection', selection)

    filtered_2[selection] = np.nan
    filtered_2 += plot_offset * np.ones_like(filtered_2)
    filtered_2 = np.roll(filtered_2, -filter_period_half)
    for i in range(int(filter_period_half)):
        filtered_2[i] = np.nan

    sel_indexes = []
    for i in range(len(filtered_2)):
        curr_i = i
        next_i = curr_i + 1
        if next_i == len(filtered_2):
            break
        if not np.isnan(filtered_2[curr_i]) and np.isnan(filtered_2[next_i]):
            sel_indexes.append(i)
    sel_signal = np.zeros_like(filtered_2)
    for i in sel_indexes:
        sel_signal[i] = 5000
        for j in range(10):
            sel_signal[i - j] = 5000
    mask = np.where(sel_signal == 5000)
    filtered_2[mask] = np.nan

    # print('filtered_2', filtered_2)
    filtered_2_interpolated = np.copy(filtered_2)
    nans, x = nan_helper(filtered_2)
    filtered_2_interpolated[nans] = np.interp(x(nans), x(~nans), filtered_2[~nans])

    mask = np.where(~np.isnan(filtered_2))
    first_index = mask[0][0]
    last_index = mask[0][-1]
    for i in range(first_index):
        filtered_2_interpolated[i] = np.nan
    for i in range(last_index, len(filtered_2)):
        filtered_2_interpolated[i] = np.nan

    first_index = first_index + 1
    zeroth_index = 0
    infty_index = len(orig_series) - 1
    first_value = np.max(orig_series)
    # first_non_nan_value = orig_series[first_index]
    first_non_nan_value = filtered_2_interpolated[first_index]
    x = np.arange(zeroth_index, first_index)
    t = first_value
    m = - (first_value - first_non_nan_value) / (first_index - zeroth_index)
    y = m * x + t
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(y)
    filtered_2_interpolated[zeroth_index:first_index] = y[:]

    last_value = orig_series[infty_index]
    last_non_nan_value = orig_series[last_index]

    t = last_value
    m = (last_non_nan_value - last_value) / (infty_index - last_index)
    x = np.arange(0, infty_index - last_index + 1)
    y = m * x + t
    filtered_2_interpolated[last_index:infty_index + 1] = y[:]

    filtered_2_interpolated = filtered_2_interpolated.reshape((-1, 1))

    return filtered_2_interpolated


def cut_CPS (Signal):
    c_Signal=Signal.copy()
    while c_Signal[0] == 0:
        c_Signal.pop(0)
    return c_Signal


def IIR_filter(Signal, alpha):
 f_Signal = np.zeros_like(Signal)  # Erstelle eine Liste mit Nullen der gleichen Länge wie das Signal
 for n in range(len(Signal)):
     if n == 0:
         f_Signal[n] = Signal[n]
     else:
         f_Signal[n] = alpha * Signal[n] + (1 - alpha) * f_Signal[n-1]
 return f_Signal


def preprocess_with_period_filter2(
        input_series,
        lower_cutoff_threshold=3500,
        filter_period=350,
        start_end_padding_enabled=True,
        padding_value=0,
        group_delay_compensation_enabled=True,
        group_delay_compensation='default',
    ):

    power_series = input_series.copy()  # avoid call by reference object
    selection = np.where(power_series < lower_cutoff_threshold)
    power_series[selection] = np.nan
    power_series_interpolated = np.copy(power_series)  # avoid call by reference object
    nans, x = nan_helper(power_series_interpolated)
    power_series_interpolated[nans] = np.interp(x(nans), x(~nans), power_series_interpolated[~nans])

    filter_period_half = int(filter_period / 2)
    filtered_signal = period_filter(power_series_interpolated, period=filter_period, two_sided=False)

    if group_delay_compensation_enabled:
        if group_delay_compensation == 'default':
            group_delay_compensation = filter_period_half
            filtered_signal = np.roll(filtered_signal, -group_delay_compensation)

    if start_end_padding_enabled:
        group_delay_compensation = filter_period_half
        filtered_signal[:group_delay_compensation] = padding_value
        filtered_signal[-group_delay_compensation:] = padding_value

    return filtered_signal


# def preprocess_with_butterworth_filter(df):

#     y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)
#     selection = np.where(y < 3500)
#     y[selection] = np.nan
#     power_series_interpolated = np.copy(y)
#     nans, x = nan_helper(power_series_interpolated)
#     power_series_interpolated[nans] = np.interp(x(nans), x(~nans), power_series_interpolated[~nans])

#     sig = power_series_interpolated
#     sos = signal.butter(N=10, Wn=1, btype='lowpass', fs=100, output='sos')
#     filtered_signal_3 = signal.sosfilt(sos, sig)

#     delay = 100
#     filtered_signal_3 = np.roll(filtered_signal_3, -delay)
#     filtered_signal_3[-delay:] = 0

#     return filtered_signal_3