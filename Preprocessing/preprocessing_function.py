# -*- coding: utf-8 -*-
"""
Pre processing & search function
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sources_daniel.custom_interpolation import nan_helper
from scipy import signal

def calc_period_filter_coefficients(period):
    if period % 2 == 0:
        filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
    else:
        filt = np.repeat(1.0 / period, period)
    return filt

class cStaticVariables:

    def __init__(self):

        # self.filter_coefficients = [0.024459527665758870, 0.032313653069918326, 0.040091373545339488, 0.047522730992687549,
        #     0.054343397861395588, 0.060306153728622261, 0.065191736934949049, 0.068818548462512730,
        #     0.071050738051570750, 0.071804279374490687,
        #     0.071050738051570750, 0.068818548462512730, 0.065191736934949049, 0.060306153728622261,
        #     0.054343397861395588, 0.047522730992687549, 0.040091373545339488, 0.032313653069918326,
        #     0.024459527665758870]
        
        




        # Beispielkoeffizienten
        #self.filter_coefficients=calc_period_filter_coefficients(2)
        #self.filter_coefficients = signal.firwin(numtaps=150, cutoff=0.00000000000001, window='hamming')
        self.filter_coefficients = [1,0]
        #self.filter_coefficients = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.fir_in_buffer = np.zeros(len(self.filter_coefficients))

        self.state = "Idle"
        self.fan_direction_prev = 0

        self.accum_direction = [0, 0]
        self.n_samples = [0, 0]
        self.average = [0, 0]
        self.offset_estimated = 0
        
        
        
class cStaticVariables_1:

    def __init__(self,number_discarded_samples):

        # self.filter_coefficients = [0.024459527665758870, 0.032313653069918326, 0.040091373545339488, 0.047522730992687549,
        #     0.054343397861395588, 0.060306153728622261, 0.065191736934949049, 0.068818548462512730,
        #     0.071050738051570750, 0.071804279374490687,
        #     0.071050738051570750, 0.068818548462512730, 0.065191736934949049, 0.060306153728622261,
        #     0.054343397861395588, 0.047522730992687549, 0.040091373545339488, 0.032313653069918326,
        #     0.024459527665758870]
        

        # Beispielkoeffizienten
        #self.filter_coefficients=calc_period_filter_coefficients(2)
        #self.filter_coefficients = signal.firwin(numtaps=150, cutoff=0.00000000000001, window='hamming')
        self.filter_coefficients = [1,0]
        
        
        self.discarded_samples=number_discarded_samples
        self.counter_discarded=0
        self.fir_in_buffer = np.zeros(len(self.filter_coefficients))

        self.state = "Idle"
        self.fan_direction_prev = 0

        self.accum_direction = [0, 0]
        self.n_samples = [0, 0]
        self.average = [0, 0]
        self.offset_estimated = 0

def resample_BLDC_data(df):
    keep_index = np.arange(0, len(df['Send_HotAirFan_Power']) ,5) # Index der gewünschten Zeilen
    df_resampled = df.loc[keep_index]
    return df_resampled

def temperature_comp(df):
    Temp_K=df['CPM1_TempPT']+273
    df['Send_HotAirFan_Power']=df['Send_HotAirFan_Power']*(1/Temp_K)

def cut_BLDC_data_b(df):
    #cut data at the beginning 
    
    #calculate Temp_Set_Point & find first index
    Temp_Set_Value=df['CPM1_TempSetPoint']
    
    Temp_Set_Point=max(df['CPM1_TempSetPoint'])
    first_value = df.loc[df['CPM1_TempSetPoint'] == Temp_Set_Point, 'CPM1_TempSetPoint'].idxmin()

    # Alle Werte bis zum gefundenen Index löschen und Index neu generieren
    df_cut = df.drop(range(first_value)).reset_index(drop=True)

    return df_cut


def cut_BLDC_data(df):
 # Cut data at the beginning & end
 Temp_Set_Value=df['CPM1_TempSetPoint']
 Temp_Set_Value=np.where(np.isnan(Temp_Set_Value), 0, Temp_Set_Value)
 # Calculate Temp_Set_Point & find first index
 Temp_Set_Point = max(Temp_Set_Value)
 first_value = df.loc[df['CPM1_TempSetPoint'] == Temp_Set_Point, 'CPM1_TempSetPoint'].head(1).idxmin()
 
 # Cut data at the end
 # Calculate Temp_Set_Point & find last index
 last_value = df.loc[df['CPM1_TempSetPoint'] == Temp_Set_Point, 'CPM1_TempSetPoint'].tail(1).idxmax()

 # Remove rows before the first index and after the last index, and reset the index
 df_cut2 = df.loc[first_value:last_value].reset_index(drop=True)
 
 return df_cut2

def filter_valid_range(df,valid_bound):
    return df
    

def Filter_Accumulated_Average_old(df,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION=False):
    # Reset all used variable when a new baking process is started
    INVALID_SAMPLE = -1
    Clean_Power_Signal = []
    myStaticVariables = cStaticVariables()
    
    # Transform df to variables
    ConvFanDirection = df['CPM1_ConvFanDirection'].to_numpy(dtype='float')
    TempPT = df['CPM1_TempPT'].to_numpy(dtype='float')
    Oxygen = df['Send_Oxygen'].to_numpy(dtype='float')
    HotAirFan_Speed = df['Send_HotAirFan_Speed'].to_numpy(dtype='float')
    HotAirFan_Power = df['Send_HotAirFan_Power'].to_numpy(dtype='float')

    # B2.- Every 5 seconds
    for t in range(0, len(ConvFanDirection)):

        # B3.- Call the cleaning function with current sensors data
        result = Clean_BLDC_Power_Signal(HotAirFan_Power[t], HotAirFan_Speed[t], ConvFanDirection[t], TempPT[t],
                                         myStaticVariables,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION)

        # B4.- If there is a valid sample store it. Otherwise ignore it
        if result != INVALID_SAMPLE:
            Clean_Power_Signal.append(result)
        else:
            Clean_Power_Signal.append(0)
            #Clean_Power_Signal.append(np.nan)
            first_valid_t = t + 1
    Clean_Power_Signal = np.array(Clean_Power_Signal)
    Clean_Power_Signal_interp = np.copy(Clean_Power_Signal)  # avoid call by reference object
    nans, x = nan_helper(Clean_Power_Signal_interp)
    Clean_Power_Signal_interp[nans] = np.interp(x(nans), x(~nans), Clean_Power_Signal_interp[~nans])
    #return Clean_Power_Signal
    return Clean_Power_Signal_interp

def Clean_BLDC_Power_Signal_old(fan_power, fan_speed, fan_direction, temperature, s,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION):
    """
    :param fan_power: sample of Send_HotAirFan_Power signal
    :param fan_speed: sample of Send_HotAirFan_Speed signal
    :param fan_direction: sample of CPM1_ConvFanDirection signal
    :param temperature: sample of CPM1_TempPT signal
    :param s: class storing the static variables used
    :return:
    """
    INVALID_SAMPLE = -1
    is_valid_sample = True

    fan_direction = int(fan_direction)  # convert to integer so we can use it as index

    # 0.- Detection of change in fan direction ------------------------------
    direction_change = False
    if fan_direction != s.fan_direction_prev:
        direction_change = True
        s.fan_direction_prev = fan_direction

    # 1.- Invalid samples filter: outliers ---------------------------------
    # 1A.- Remove outliers, samples with values out of valid range
    if fan_speed >= ACCEPTANCE_LIMIT['speed_high'] or fan_speed <= ACCEPTANCE_LIMIT['speed_low']:
        is_valid_sample = False
        #print('invalid speed: ' + str(fan_speed))
    if fan_power < ACCEPTANCE_LIMIT['power_min']:
        is_valid_sample = False
        #print('invalid power: ' + str(fan_power))

    # 1B .- Remove 1 sample after the fan change assuming power is not yet stable
    if direction_change is True:
        s.is_valid_sample = False

    # 2.- Temperature filter -------------------------------------
    # Keep only samples when temperature is over a certain threshold
    if temperature < MINIMUM_TEMPERATURE:
        is_valid_sample = False
        #print('invalid temp: ' + str(temperature))

    # 3.- State of the cleaner.
    # If have no valid sample yet, keep on IDLE and return INVALID_SAMPLE
    # On first valid sample set state to running. It will feed a sample on each call after
    if s.state == "Idle":
        if is_valid_sample:
            s.state = "Running"
            s.fir_in_buffer += fan_power        # FIR Filter: on first sample set the buffer to current input to avoid filter delay
        else:
            return INVALID_SAMPLE

    # 4.- Step filter - remove offset variation due to fan direction changes

    # Update offset between both states as difference between average values on both states
    # Offset is only valid when I have calculated on both states. Otherwise make it 0
    if direction_change is True:
        if s.n_samples[0] > 0 and s.n_samples[1] > 0:
            s.offset_estimated = s.average[1] - s.average[0]

    if is_valid_sample:
        s.accum_direction[fan_direction] += fan_power       # accumulate value on the corresponding direction
        s.n_samples[fan_direction] += 1                     # increase accumulated samples number
        s.average[fan_direction] = s.accum_direction[fan_direction] / s.n_samples[fan_direction]  # offset in this direction as average of accumulated value

        # Correct offset in one direction
        if fan_direction == 1:
            fan_power = fan_power - s.offset_estimated

    # 5.- Noise removal - FIR Filter -----------------------------
    # 5.1 - Manage input buffer
    s.fir_in_buffer = np.roll(s.fir_in_buffer, 1)   # Shift the input buffer on sample
    if is_valid_sample:
        s.fir_in_buffer[0] = fan_power              # If valid sample add it to buffer
    else:
        s.fir_in_buffer[0] = s.fir_in_buffer[1]     # If invalid sample repeat latest input
        #s.fir_in_buffer[0] = np.nan                  # If invalid sample output=nan

    # 5.2.- Apply filtering
    out = np.dot(s.fir_in_buffer, s.filter_coefficients)
    if TEMPERATURE_COMPENSATION==True:
        out = out*(temperature+273)
    return out


def Filter_Accumulated_Average(df,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION=False,number_discarded_samples=5):
    # Reset all used variable when a new baking process is started
    INVALID_SAMPLE = -1
    Clean_Power_Signal = []
    myStaticVariables = cStaticVariables_1(number_discarded_samples)
    
    # Transform df to variables
    ConvFanDirection = df['CPM1_ConvFanDirection'].to_numpy(dtype='float')
    TempPT = df['CPM1_TempPT'].to_numpy(dtype='float')
    Oxygen = df['Send_Oxygen'].to_numpy(dtype='float')
    HotAirFan_Speed = df['Send_HotAirFan_Speed'].to_numpy(dtype='float')
    HotAirFan_Power = df['Send_HotAirFan_Power'].to_numpy(dtype='float')

    # B2.- Every 5 seconds
    for t in range(0, len(ConvFanDirection)):

        # B3.- Call the cleaning function with current sensors data
        result = Clean_BLDC_Power_Signal(HotAirFan_Power[t], HotAirFan_Speed[t], ConvFanDirection[t], TempPT[t],
                                         myStaticVariables,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION)

        # B4.- If there is a valid sample store it. Otherwise ignore it
        if result != INVALID_SAMPLE:
            Clean_Power_Signal.append(result)
        else:
            Clean_Power_Signal.append(0)
            #Clean_Power_Signal.append(np.nan)
            first_valid_t = t + 1
    Clean_Power_Signal = np.array(Clean_Power_Signal)
    Clean_Power_Signal_interp = np.copy(Clean_Power_Signal)  # avoid call by reference object
    nans, x = nan_helper(Clean_Power_Signal_interp)
    Clean_Power_Signal_interp[nans] = np.interp(x(nans), x(~nans), Clean_Power_Signal_interp[~nans])
    #return Clean_Power_Signal
    return Clean_Power_Signal_interp

def Clean_BLDC_Power_Signal(fan_power, fan_speed, fan_direction, temperature, s,MINIMUM_TEMPERATURE,ACCEPTANCE_LIMIT,TEMPERATURE_COMPENSATION):
    """
    :param fan_power: sample of Send_HotAirFan_Power signal
    :param fan_speed: sample of Send_HotAirFan_Speed signal
    :param fan_direction: sample of CPM1_ConvFanDirection signal
    :param temperature: sample of CPM1_TempPT signal
    :param s: class storing the static variables used
    :return:
    """
    INVALID_SAMPLE = -1
    is_valid_sample = True

    fan_direction = int(fan_direction)  # convert to integer so we can use it as index

    # 0.- Detection of change in fan direction ------------------------------
    direction_change = False
    if fan_direction != s.fan_direction_prev:
        direction_change = True
        s.fan_direction_prev = fan_direction

    # 1.- Invalid samples filter: outliers ---------------------------------
    # 1A.- Remove outliers, samples with values out of valid range
    if fan_speed >= ACCEPTANCE_LIMIT['speed_high'] or fan_speed <= ACCEPTANCE_LIMIT['speed_low']:
        is_valid_sample = False
        #print('invalid speed: ' + str(fan_speed))
    if fan_power < ACCEPTANCE_LIMIT['power_min']:
        is_valid_sample = False
        #print('invalid power: ' + str(fan_power))

    # 1B .- initialization of the counter to discard samples after direction change
    if direction_change is True:
        s.counter_discarded=s.discarded_samples
        #print('direction_change: ' + str(s.counter_discarded))

    # 1C .- Remove samples after the fan change until counter = 0
    if s.counter_discarded != 0:
        is_valid_sample = False
        s.counter_discarded -=1
        #print('direction_change: ' + str(s.counter_discarded))
        
    # 2.- Temperature filter -------------------------------------
    # Keep only samples when temperature is over a certain threshold
    if temperature < MINIMUM_TEMPERATURE:
        is_valid_sample = False
        #print('invalid temp: ' + str(temperature))

    # 3.- State of the cleaner.
    # If have no valid sample yet, keep on IDLE and return INVALID_SAMPLE
    # On first valid sample set state to running. It will feed a sample on each call after
    if s.state == "Idle":
        if is_valid_sample:
            s.state = "Running"
            s.fir_in_buffer += fan_power        # FIR Filter: on first sample set the buffer to current input to avoid filter delay
        else:
            return INVALID_SAMPLE

    # 4.- Step filter - remove offset variation due to fan direction changes

    # Update offset between both states as difference between average values on both states
    # Offset is only valid when I have calculated on both states. Otherwise make it 0
    if direction_change is True:
        if s.n_samples[0] > 0 and s.n_samples[1] > 0:
            s.offset_estimated = s.average[1] - s.average[0]

    if is_valid_sample:
        s.accum_direction[fan_direction] += fan_power       # accumulate value on the corresponding direction
        s.n_samples[fan_direction] += 1                     # increase accumulated samples number
        s.average[fan_direction] = s.accum_direction[fan_direction] / s.n_samples[fan_direction]  # offset in this direction as average of accumulated value

        # Correct offset in one direction
        if fan_direction == 1:
            fan_power = fan_power - s.offset_estimated

    # 5.- Noise removal - FIR Filter -----------------------------
    # 5.1 - Manage input buffer
    s.fir_in_buffer = np.roll(s.fir_in_buffer, 1)   # Shift the input buffer on sample
    if is_valid_sample:
        s.fir_in_buffer[0] = fan_power              # If valid sample add it to buffer
    else:
        s.fir_in_buffer[0] = s.fir_in_buffer[1]     # If invalid sample repeat latest input
        #s.fir_in_buffer[0] = np.nan                  # If invalid sample output=nan

    # 5.2.- Apply filtering
    out = np.dot(s.fir_in_buffer, s.filter_coefficients)
    if TEMPERATURE_COMPENSATION==True:
        out = out*(temperature+273)
    return out


def get_nom_speed(PWM):
    rpm_speed = {
 6: 800,
 7: 800,
 8: 800,
 9: 1125,
 10: 1125,
 11: 1125,
 12: 1250,
 13: 1250,
 14: 1250,
 15: 1375,
 16: 1375,
 17: 1375,
 18: 1500,
 19: 1500,
 20: 1500,
 21: 1625,
 22: 1625,
 23: 1625,
 24: 1750,
 25: 1750,
 26: 1750,
 27: 1875,
 28: 1875,
 29: 1875,
 30: 2000,
 31: 2000,
 32: 2000,
 33: 2125,
 34: 2125,
 35: 2125,
 36: 2250,
 37: 2250,
 38: 2250,
 39: 2500,
 40: 2500,
 41: 2500,
 42: 2625,
 43: 2625,
 44: 2625
}

    speed = rpm_speed[PWM]
    return speed