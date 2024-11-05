

import numpy as np
import matplotlib.pyplot as plt

from tools.read_data import DataManager

from sources_daniel.applied_functions import preprocess_with_butterworth_filter
from sources_daniel.applied_functions import preprocess_with_period_filter


def compare_period_filter_and_butterworth():

    data_manager = DataManager()
    test_list, labels = data_manager.get_data()

    selected_file_index = 3
    df = test_list[selected_file_index]
    print(list(df.columns))

    filtered_period_filter_output = preprocess_with_period_filter(df)

    y = df['Send_HotAirFan_Power'].to_numpy().astype(np.float)
    selection = np.where(y < 3500)
    y[selection] = np.nan
    fan_power_signal = y

    filtered_butterworth_filter_output = preprocess_with_butterworth_filter(df)

    plt.figure()
    plt.plot(fan_power_signal, label='Fan power raw signal')
    plt.plot(filtered_period_filter_output, label='Period-Filter with group delay of 175 samples')
    plt.plot(filtered_butterworth_filter_output, label='Butterworth-Filter with group delay of 100 samples')
    plt.xlabel('power signal')
    plt.xlabel('samples')
    plt.title('Effect of Preprocessing Filters')
    plt.legend()


if __name__ == '__main__':

    compare_period_filter_and_butterworth()

    plt.show()

