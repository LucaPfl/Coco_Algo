

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import tools.read_data as rd
from tools.read_data import DataManager

from sources_daniel.custom_interpolation import nan_helper
from sources_daniel.preprocessing_filter import period_filter
from sources_daniel.preprocessing_filter import custom_decompose
from sources_daniel.preprocessing_filter import apply_fir_filter
from sources_daniel.preprocessing_filter import calc_period_filter_coefficients


def try_time_series_decomposition():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    file_index = 0
    df = test_list[file_index]
    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    dir_changed = np.roll(dir_changed, 10)
    for i in range(10):
        dir_changed[i] = 0
    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)

    selection = np.where(y < 3000)
    y[selection] = np.nan

    power_series = df['Send_HotAirFan_Power']
    # power_series['Send_HotAirFan_Power'] = y

    result, seaonal, resid, observed = period_filter(power_series, period=350)
    trend = result.astype(np.float)
    seaonal = seaonal.astype(np.float)
    resid = resid.astype(np.float)
    observed = observed.astype(np.float)

    result_2 = period_filter(power_series, period=350)

    filtered = result.to_numpy().astype(np.float)
    filtered[selection] = np.nan
    filtered = filtered + 400 * np.ones_like(filtered)

    filtered_2 = result_2.astype(np.float)
    filtered_2[selection] = np.nan
    filtered_2 = filtered_2 + 400 * np.ones_like(filtered_2)

    plt.figure()
    plt.plot(trend)
    plt.figure()
    plt.plot(seaonal)
    plt.figure()
    plt.plot(resid)
    plt.figure()
    plt.plot(observed)

    plt.figure()
    plt.plot()

    plt.figure()
    plt.plot(y)
    plt.plot(filtered)


def try_time_series_decomposition_2():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    file_index = 0
    df = test_list[file_index]
    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    dir_changed = np.roll(dir_changed, 10)
    for i in range(10):
        dir_changed[i] = 0
    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)

    selection = np.where(y < 3500)

    imputation_value = np.nan
    # imputation_value = np.mean(y[np.where(~np.isnan(y))])
    y[selection] = imputation_value

    power_series = df['Send_HotAirFan_Power']
    # power_series['Send_HotAirFan_Power'] = y

    result, seaonal, resid, observed = filter(power_series, period=350)
    # result, seaonal, resid, observed = filter(y, period=350)
    trend = result.astype(np.float)
    seaonal = seaonal.astype(np.float)
    resid = resid.astype(np.float)
    observed = observed.astype(np.float)

    # result_2 = period_filter(power_series, period=350)
    result_2 = period_filter(y, period=350)

    filtered = result.to_numpy().astype(np.float)
    # filtered = result.astype(np.float)
    filtered[selection] = np.nan
    filtered = filtered + 400 * np.ones_like(filtered)

    filtered_2 = result_2.astype(np.float)
    filtered_2[selection] = np.nan
    filtered_2 = filtered_2 + 400 * np.ones_like(filtered_2)

    plt.figure()
    plt.plot(trend)
    plt.figure()
    plt.plot(seaonal)
    plt.figure()
    plt.plot(resid)
    plt.figure()
    plt.plot(observed)

    plt.figure()
    plt.plot()

    plt.figure()
    plt.plot(y)
    plt.plot(filtered)
    plt.plot(filtered_2)


def try_time_series_decomposition_3():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    file_index = 0
    df = test_list[file_index]
    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    dir_changed = np.roll(dir_changed, 10)
    for i in range(10):
        dir_changed[i] = 0
    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)

    selection = np.where(y < 3500)

    # imputation_value = np.nan
    imputation_value = np.mean(y[np.where(~np.isnan(y))])
    y[selection] = imputation_value

    # power_series = df['Send_HotAirFan_Power']
    # power_series['Send_HotAirFan_Power'] = y
    power_series = y

    result, seaonal, resid, observed = filter(power_series, period=350)
    # result, seaonal, resid, observed = filter(y, period=350)

    trend = result.astype(np.float)
    seaonal = seaonal.astype(np.float)
    resid = resid.astype(np.float)
    observed = observed.astype(np.float)

    plt.figure()
    plt.plot(power_series)
    plt.plot(trend)
    plt.plot(seaonal)
    plt.plot(resid)

    trend, seaonal, resid = custom_decompose(power_series, period=350)
    plt.figure()
    plt.plot(power_series)
    plt.plot(trend)
    plt.plot(seaonal)
    plt.plot(resid)


def get_data():
    data_manager = DataManager()
    test_list, labels = data_manager.get_data()
    # print(len(test_list))
    # print(type(test_list))
    # print(test_list[0])
    # test_list[2].plot()
    # plt.show()
    # # print(list(test_list[0].columns))
    #
    # # print(labels)
    # # print(len(test_list))
    # input('>>')

    # Get single test
    clear_test_list = []

    for t, test in enumerate(test_list):

        # -- RESAMPLE ------
        SAMPLING_RATE = 5
        test = test.iloc[::int(SAMPLING_RATE)]
        # -- We remove directly the first sample after each speed change (1-0, 0-1)
        # We assume that blower can stabilize faster than 5 seconds

        # Create a column indicating speed change and
        change = np.zeros(len(test))
        for n in range(1,len(test)):
            if test.iloc[n][rd.FAN_DIRECTION] != test.iloc[n-1][rd.FAN_DIRECTION]:
                change[n] = 1

        test['change'] = change
        test = test[test['change'] == 0]

        # Remove all samples with speed changes and with low fan speed
        test = test[test[rd.FAN_SPEED]> 1000]
        test = test[test[rd.FAN_SPEED]< 1800]  # For outliers, like in test 40
        test = test[test[rd.FAN_POWER]> 2000]  # Still some outliers on power, this value is far aways from typical minimum values

        # -------- TEMPERATURE FILTER ------------
        test = test[test[rd.TEMPERATURE]>= 150]

        #--------- STEPS FILTER ------------------
        # separate data between both states of fan direction
        test_dir_0 = test[test[rd.FAN_DIRECTION] == 0]
        test_dir_1 = test[test[rd.FAN_DIRECTION] == 1]

        # calculate the average value on each state and estimated offset.
        average_0 = test_dir_0[rd.FAN_POWER].mean()
        average_1 = test_dir_1[rd.FAN_POWER].mean()
        offset_estimated = average_1 - average_0

        # correct the power signal removing the estimated offset to the samples on up fan direction
        # ! this operation assumes that fan direction signal is always 0-1, otherwise it will fail
        test[rd.FAN_POWER] = test[rd.FAN_POWER] - (test[rd.FAN_DIRECTION] * offset_estimated)

        # plt.plot(test[rd.FAN_POWER])
        # plt.title(str(f'{t} {labels[rd.FILE_NAME][t]}'))
        # plt.show()

        clear_test_list.append(test)

    return clear_test_list


def try_time_series_decomposition_4():

    def compensate_delay(array, delay, amplitude=0):
        tmp = np.copy(array)
        tmp = np.roll(tmp, -delay)
        tmp[-delay:] = amplitude * np.ones((delay,))
        return tmp

    # period = 350
    # period = int(350 / 2)
    # period = int(350 / 4)
    period = int(350 / 8)
    # period = int(350 / 16)

    delay = int(period/2) + 1

    clear_test_list = get_data()

    datas = []
    plt.figure()
    for test in clear_test_list:
        data = test[rd.FAN_POWER].to_numpy(dtype='float')

        mask = np.where(data < 3500)
        data[mask] = np.mean(data)

        datas.append(data)
        plt.plot(data)
        # power_filt = FIR_filter(test[rd.FAN_POWER].to_numpy(dtype='float'))
        # plt.plot(power_filt - power_filt[0], linewidth=1.0)
    # plt.title('Power corrected with FIR filter')

    filter_coeffs = calc_period_filter_coefficients(period=period)
    trends = []
    trend_comps = []
    detrends = []
    errors = []
    for data in datas:
        trend = apply_fir_filter(data, filter_coeffs)
        trends.append(trend)
        trend_comp = compensate_delay(trend, delay, np.mean(data[:-delay]))
        trend_comps.append(trend_comp)
        detrends.append(data-trend_comp)
        errors.append(np.mean(np.square(data[:-delay]-trend_comp[:-delay])))

    plt.figure()
    for i in [130]:
        plt.plot(datas[i], label='data')
        # plt.plot(trends[i], label='trends')
        plt.plot(trend_comps[i], label='trends compensated')
        plt.plot(detrends[i], label='detrends')
    plt.legend()
    plt.title('before after - index: {}, mse: {}'.format(i, errors[i]))

    # plt.figure()
    # for i in range(len(datas)):
    #     plt.plot(trends[i] - trends[i][0], linewidth=1.0)
    # plt.title('trend')

    # plt.figure()
    # for i in range(len(datas)):
    #     plt.plot(detrends[i] - detrends[i][0], linewidth=1.0)
    # plt.title('detrend')

    # trends, seasonals, resids = [], [], []
    # for data in datas:
    #     trend, seaonal, resid = custom_decompose(data, period=period)
    #     trends.append(trend)
    #     seasonals.append(seaonal)
    #     resids.append(resid)

    # plt.figure()
    # for i in range(len(datas)):
    #     plt.plot(seasonals[i])
    #
    # plt.figure()
    # for i in range(len(datas)):
    #     plt.plot(resids[i])


def try_out():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    file_index = 110
    df = test_list[file_index]
    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    dir_changed = np.roll(dir_changed, 10)
    for i in range(10):
        dir_changed[i] = 0
    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)

    selection = np.where(y < 3000)
    y[selection] = np.nan

    power_series = df['Send_HotAirFan_Power']
    # power_series['Send_HotAirFan_Power'] = y

    plot_offset = 400
    filter_period = 350
    filter_period_half = int(filter_period/2)

    # filtered_signal, _, _, _ = filter(power_series, period=filter_period, two_sided=True)
    # filtered = filtered_signal.to_numpy().astype(np.float)
    # filtered[selection] = np.nan
    # filtered += plot_offset * np.ones_like(filtered)
    # # filtered = np.roll(filtered, -filter_period_half)

    filtered_signal_2 = period_filter(power_series, period=filter_period, two_sided=False)
    filtered_2 = filtered_signal_2.astype(np.float)
    filtered_2[selection] = np.nan
    filtered_2 += plot_offset * np.ones_like(filtered_2)
    filtered_2 = np.roll(filtered_2, -filter_period_half)

    plt.figure()
    plt.plot(y, label='fan power')
    plt.plot(filtered_2, label='filtered_2')
    plt.legend()


def try_out_2():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    # dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    # dir_changed = np.copy(dir)
    # roll_n_samples = 10
    # dir_changed = np.roll(dir_changed, roll_n_samples)
    # for i in range(roll_n_samples):
    #     dir_changed[i] = 0

    file_index = 1
    df = test_list[file_index]
    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    roll_n_samples = 10
    dir_changed = np.roll(dir_changed, roll_n_samples)
    for i in range(roll_n_samples):
        dir_changed[i] = 0
    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)

    # selection = np.where(y < 4300)
    # selection = np.where(y < 3900)
    selection = np.where(y < 3500)
    # selection = np.where(y < 3000)
    y[selection] = np.nan
    power_series = df['Send_HotAirFan_Power']
    # power_series['Send_HotAirFan_Power'] = y

    plot_offset = 400
    filter_period = 350
    filter_period_half = int(filter_period/2)

    # filtered_signal, _, _, _ = filter(power_series, period=filter_period, two_sided=True)
    # filtered = filtered_signal.to_numpy().astype(np.float)
    # filtered[selection] = np.nan
    # filtered += plot_offset * np.ones_like(filtered)
    # # filtered = np.roll(filtered, -filter_period_half)

    filtered_signal_2 = period_filter(power_series, period=filter_period, two_sided=False)
    filtered_2 = filtered_signal_2.astype(np.float)
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
            sel_signal[i-j] = 5000
    mask = np.where(sel_signal == 5000)
    filtered_2[mask] = np.nan

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

    plt.figure()
    plt.plot(y, label='fan power')
    # plt.plot(sel_signal, label='sel_signal')
    # plt.plot(filtered_2, label='filtered_2')
    plt.plot(filtered_2_interpolated, label='filtered_2_interpolated')
    plt.legend()


if __name__ == '__main__':

    # try_time_series_decomposition()
    # try_time_series_decomposition_2()
    # try_time_series_decomposition_3()
    # try_time_series_decomposition_4()
    # try_out()
    try_out_2()

