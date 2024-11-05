

import numpy as np
import matplotlib.pyplot as plt

from tools.read_data import DataManager


def index_reduce(input_array, index_list):
    return input_array[index_list]


def index_expand(output_array, index_list, reduced_list):
    run_j = -1
    for i in range(len(output_array)):
        if i in index_list:
            run_j += 1
            output_array[i] = reduced_list[run_j]
    return output_array


def mwe_reduce_expand():
    a = np.array([str(x) for x in range(10)])
    mask = [3, 4, 7, 8]
    b = a[mask]
    print(a)
    print(b)
    c = np.copy(a)
    for i in range(len(c)):
        if i in mask:
            c[i] = np.nan
    print(c)
    run_j = -1
    for i in range(len(c)):
        if i in mask:
            run_j += 1
            c[i] = b[run_j]
    print(c)
    a = np.array([str(x) for x in range(10)])
    mask = [3, 4, 7, 8]
    b = index_reduce(a, mask)
    print(a)
    print(b)
    c = np.copy(a)
    for i in range(len(c)):
        if i in mask:
            c[i] = np.nan
    print(c)
    c = index_expand(c, mask, b)
    print(c)


def mwe_split_signal_stable():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    file_index = 10
    df = test_list[file_index]

    dir = df['CPM1_ConvFanDirection'].to_numpy().astype(np.float)
    dir_changed = np.copy(dir)
    dir_changed = np.roll(dir_changed, 10)
    for i in range(10):
        dir_changed[i] = 0

    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)
    x = np.arange(0, len(y))
    print(dir.shape)
    print(x.shape)

    selection_upper = np.where(dir_changed <= 0)
    selection_lower = np.where(dir_changed > 0)
    x_upper = np.copy(x)
    x_lower = np.copy(x)
    y_upper = np.copy(y)
    y_lower = np.copy(y)
    y_upper[selection_upper] = np.nan
    y_lower[selection_lower] = np.nan

    selection = np.where(y_lower < 3000)
    y_lower[selection] = np.nan
    selection = np.where(y_upper < 3000)
    y_upper[selection] = np.nan

    plt.figure()
    plt.plot(y)
    plt.title('signal')

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    ax[0].plot(y)
    ax[0].plot(dir_changed * 6000)
    ax[1].plot(y_upper)
    ax[2].plot(y_lower)

    # upper signal parts
    mask = np.argwhere(~np.isnan(y_upper))
    mask = list(mask.reshape((-1,)))
    y_upper_dropped = index_reduce(y_upper, mask)
    x_upper_dropped = np.arange(len(y_upper_dropped))

    # print(list(x_upper_dropped))
    # print(list(y_upper_dropped))
    # input('>>1>>')

    # ...

    x_upper_interpolated = np.copy(x_upper)
    y_upper_interpolated = index_expand(np.zeros_like(y_upper), mask, y_upper_dropped)
    select = np.where(y_upper_interpolated == 0)
    y_upper_interpolated[select] = np.nan

    # lower signal parts
    mask = np.argwhere(~np.isnan(y_lower))
    mask = list(mask.reshape((-1,)))
    y_lower_dropped = index_reduce(y_lower, mask)
    x_lower_dropped = np.arange(len(y_lower_dropped))

    # print(list(x_lower_dropped))
    # print(list(y_lower_dropped))
    # input('>>2>>')

    # ...

    x_lower_interpolated = np.copy(x_lower)
    y_lower_interpolated = index_expand(np.zeros_like(y_lower), mask, y_lower_dropped)
    select = np.where(y_lower_interpolated == 0)
    y_lower_interpolated[select] = np.nan

    plt.figure()
    plt.plot(x_upper_dropped, y_upper_dropped)
    plt.plot(x_lower_dropped, y_lower_dropped)
    plt.title('splitted')

    plt.figure()
    plt.plot(x_upper_interpolated, y_upper_interpolated)
    plt.plot(x_lower_interpolated, y_lower_interpolated)
    plt.title('interpolated')


if __name__ == '__main__':
    mwe_reduce_expand()
    mwe_split_signal_stable()
    plt.show()

