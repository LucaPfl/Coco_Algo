# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:14:28 2024

@author: BenderT
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def Check_Temperature(df,Offset_Temp=10):
    Temp_Set_Point=max(df['CPM1_TempSetPoint'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['CPM1_TempPT'], color='red', linewidth=1,label='temperature')
    plt.axhline(y=Temp_Set_Point, color='black', linestyle='--',label='set temperature',linewidth=1)# Add labels and title
    plt.axhline(y=Temp_Set_Point-Offset_Temp, color='black', linestyle='--',label='set temperature',linewidth=1)# Add labels and title
    plt.xlabel('samples')
    plt.ylabel('temperature (°C)')
    plt.legend()
    plt.show()
    #plt.close()
    

def search_gradient_algorithm(Power_Data, min_samples, review_samples, stop_samples,filename=''):
    x_array = np.arange(review_samples)
    power_gradient=np.zeros(len(Power_Data))
    first_non_nan_value = next((index for index, value in enumerate(Power_Data) if not math.isnan(value)), None)
    
    if first_non_nan_value+review_samples > min_samples:
        start_sample=first_non_nan_value+review_samples
    else:
        start_sample=min_samples

    for t_step in range(start_sample, len(Power_Data)):
        y_array=Power_Data[t_step-review_samples:t_step]
        coefficients = np.polyfit(x_array, y_array, 1)
        power_gradient[t_step] = coefficients[0] 
   
    end_sample=1
    end_flag=False
    end_counter=0
    for t_step in range(min_samples, len(power_gradient)):
        if power_gradient[t_step] * power_gradient[t_step-1] < 0:
            end_sample=t_step
            end_flag=True
            break

    
    # -- Plot
    plt.figure(figsize=(10, 6))
    plt.plot(power_gradient, color='blue', linewidth=1,label='slope of regression')
    plt.plot(end_sample,power_gradient[end_sample],marker='o', color='black',linestyle='', linewidth=0.5,label='end point')
    # Add labels and title
    plt.xlabel('samples')
    plt.ylabel('slope')
    plt.suptitle('Gradient-Evaluation  |  ' + filename)
    plt.text(0.5, 1.05, 'startsample: '+str(start_sample)+'  |  endsample: '+str(end_sample)+'  |  min samples: '+str(min_samples)+'  |  regression samples: '+str(review_samples) + '  |   successful: ' + str(end_flag), ha='center', va='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.show()
    #plt.close()
    
    
    
    
    
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax2 = ax1.twinx()  # Erstelle eine zweite y-Achse

    # p1,=ax1.plot(Power_Data, color='green', label='power data')
    # p2,=ax2.plot(Counter_s_abs, marker='*',color='red',linestyle='', linewidth=0.5,label='rising counter')
    # p3, =ax1.plot(end_sample,Power_Data[end_sample],marker='o', color='black', linestyle='',linewidth=0.5,label='end point')
    # ax1.legend(handles=[p1, p2,p3],loc='lower left')
    # ax1.set_xlabel('samples')
    # ax1.set_ylabel('power signal', color='black')
    # ax2.set_ylabel('stop signal (#)', color='black')

    # title_text = 'stopsample: '+str(stop_samples)+'  |  endsample: '+str(end_sample)+'  |  min samples: '+str(min_samples)+'  |  review samples: '+str(review_samples) +'  |   sucessful: ' + str(end_flag)
    # plt.title(title_text, ha='center')
    # plt.show()
    # plt.close()
    return end_sample, end_flag

def search_lambda_algorithm(Power_Data, min_samples, review_samples, stop_samples,filename=''):
    Counter_Trend_s = np.zeros(len(Power_Data))
    Counter_Trend_s_a = np.zeros(len(Power_Data))
    Counter_Trend_f = np.zeros(len(Power_Data))
    Counter_s_abs = np.zeros(len(Power_Data))
    Counter_f_abs = np.zeros(len(Power_Data))
    Counter_dez = np.zeros(len(Power_Data))
    Leistung_Diff = np.zeros(len(Power_Data))
    Counter_s = 0
    Counter_f = 0
    Counter_s_a = 0
    Leistung_Trend = np.zeros(len(Power_Data))
    end_flag=False
    first_non_nan_value = next((index for index, value in enumerate(Power_Data) if not math.isnan(value)), None)
    if first_non_nan_value+review_samples > min_samples:
        start_sample=first_non_nan_value+review_samples
    else:
        start_sample=min_samples

    for t_step in range(start_sample, len(Power_Data)):
        Leistung_Diff[t_step] = Power_Data[t_step] - Power_Data[t_step-review_samples]  # Leistungsdifferenz berechnen <0 = fallender Verlauf
        if Power_Data[t_step] >= Power_Data[t_step-review_samples]:  # steigender Verlauf
            Leistung_Trend[t_step] = 1  # 1=steigender Verlauf
            Counter_s += 1
            if Leistung_Trend[t_step-1] == 1:
                Counter_Trend_s[t_step] = Counter_Trend_s[t_step-1] + 1
                Counter_s_a += 1
            else:
                Counter_Trend_s[t_step] = 0
        else:
            Leistung_Trend[t_step] = -1  # -1=fallender Verlauf
            Counter_f += 1
            if Leistung_Trend[t_step-1] == -1:
                Counter_Trend_f[t_step] = Counter_Trend_f[t_step-1] + 1
            else:
                Counter_Trend_f[t_step] = 0
        Counter_s_abs[t_step] = Counter_s
        Counter_f_abs[t_step] = Counter_f
        Counter_Trend_s_a[t_step] = Counter_s_a
        Counter_dez[t_step] = Counter_Trend_s[t_step] - Counter_Trend_f[t_step]
        if Counter_s > stop_samples:
            end_sample = t_step
            end_flag=True
            break
        else:
            end_sample = t_step
    
    Counter_Trend_s = np.where(Counter_Trend_s == 0, np.nan, Counter_Trend_s)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Erstelle eine zweite y-Achse

    p1,=ax1.plot(Power_Data, color='green', label='power data')
    p2,=ax2.plot(Counter_Trend_s, marker='*',color='red',linestyle='', linewidth=0.5,label='rising counter')
    p3, =ax1.plot(end_sample,Power_Data[end_sample],marker='o', color='black', linestyle='',linewidth=0.5,label='end point')
    ax1.legend(handles=[p1, p2,p3],loc='lower left')
    ax1.set_xlabel('samples')
    ax1.set_ylabel('power signal', color='black')
    ax2.set_ylabel('stop signal (#)', color='black')
    plt.suptitle('Lambda-Evaluation  |  ' + filename)
    title_text = 'stopsample: '+str(stop_samples)+'  |  endsample: '+str(end_sample)+'  |  min samples: '+str(min_samples)+'  |  review samples: '+str(review_samples) +'  |   successful: ' + str(end_flag)
    plt.title(title_text, ha='center')
    plt.show()
    #plt.close()
    return end_sample, end_flag

def search_moving_average(Power_Data, Min_Time, stop_signal, samples_short, samples_long,filename,folder_plot='./result_plot/'):
 # Comparison of two moving averages for finding the end of the baking process
 # Minimum Time is shifted to minimum possible point
 # 10.3.2024: last change
 # 16.7. nan value added in first non zero value
 
# Create a sample array
    if len(Power_Data.shape)==1:
        data=Power_Data.reshape(-1,1)
    else:
        data=Power_Data
            

# Calculate the short moving average
    window_size = samples_short
# Pad the array with NaN values at the beginning
    #padded_data = np.pad(data, (window_size-1, 0), mode='constant', constant_values=np.nan)
    padded_data = np.pad(data, ((window_size-1,0), (0, 0)), mode='constant', constant_values=np.nan)
    
# Calculate the moving average
    moving_average = np.convolve(padded_data.flatten(), np.ones(window_size)/window_size, mode='valid')
    moving_average = np.nan_to_num(moving_average)
    ma_short=moving_average

# Calculate the long moving average
    window_size = samples_long
# Pad the array with NaN values at the beginning
    padded_data = np.pad(data, ((window_size-1,0), (0, 0)), mode='constant', constant_values=np.nan)
# Calculate the moving average
    moving_average = np.convolve(padded_data.flatten(), np.ones(window_size)/window_size, mode='valid')
    moving_average = np.nan_to_num(moving_average)
    ma_long=moving_average

    end_sample = 1 # if no end value is found
    end_flag = False
    diff_ma = ma_long - ma_short # calculate difference of moving average
    counter = 0
    counter_a = np.zeros(diff_ma.size)

# search for first non zero value
    #first_non_zero_value = next((index for index, value in enumerate(Power_Data) if value != 0), None)
    first_non_zero_value = next((index for index, value in enumerate(Power_Data) if value != 0 and not math.isnan(value)), None)
# calculate searching start point

    if first_non_zero_value + samples_long <= Min_Time:
        start_sample=Min_Time
    else:
        start_sample=first_non_zero_value + samples_long 

 # search for negative difference
    for z in range(start_sample, diff_ma.size):
         if diff_ma[z] <= 0:
             counter += 1
             if counter >= stop_signal and end_flag == 0:
                 end_sample = z
                 end_flag = True
         else:
                 counter = 0
                 counter_a[z] = counter

# Plot figure    
    # fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16, 8))
    # ax1.plot(ma_short, color='black', linewidth=0.5)
    # ax1.plot(ma_long, color='blue', linewidth=0.5)
    # ax1.plot(Power_Data, color='red', linewidth=0.5)
    # ax2.plot(diff_ma, color='blue', linewidth=1,label= 'difference long-short')
    # ax1.legend(['short (' +str(samples_short)+ ' samples)', 'long (' +str(samples_long)+ ' samples)','original','endpoint'])
    # ax1.set_xlabel('samples (5s)')
    # ax1.set_ylabel('power (mW)')
    
    
    # ax2.plot(end_sample,diff_ma[end_sample], marker='o', color='black', linewidth=0.5)
    # ax2.legend(['difference long-short'])
    # ax2.set_ylim([-20, 20])
    # ax2.set_xlabel('samples (5s)')
    # ax2.set_ylabel('power (mW)')
    # #plt.subplots_adjust(wspace=0.4) 
    # plt.suptitle(filename, ha='center')
    # plt.text(0.5, 0.93, 'startsample: '+str(start_sample)+'  |  endsample: '+str(end_sample)+'  |  endtime: '+str(round(end_sample*5/60,2))+'min', ha='center', va='center', transform=fig.transFigure)
    # plt.savefig(folder_plot + filename[:-4] + '_end_average.png',dpi=400)
    # plt.show()
    # plt.close()
    
    
    x_index_diff=np.arange(start_sample,len(diff_ma))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Erstelle eine zweite y-Achse
    p1, =ax1.plot(Power_Data, color='green', label='power data')
    p2, =ax1.plot(ma_short, color='black', linewidth=1, label='short (' +str(samples_short)+ ' samples)')
    p3, =ax1.plot(ma_long, color='red', linewidth=1,label= 'long (' +str(samples_long)+ ' samples)')
    p4, =ax1.plot(end_sample,Power_Data[end_sample], marker='o',linestyle='', color='black', linewidth=0.5,label= 'endpoint')
    p5, =ax2.plot(x_index_diff,diff_ma[start_sample:], color='blue', linewidth=1,label= 'difference long-short')
    ax2.plot(end_sample,diff_ma[end_sample], marker='o', color='black', linewidth=0.5,label= 'endpoint')
    ax1.legend(handles=[p1, p2,p3,p4,p5],loc='lower left')
    ax1.set_xlabel('samples')
    ax1.set_ylabel('signal', color='black')
    ax2.set_ylabel('diff signal', color='black')
    plt.suptitle('Average-Evaluation  |  ' + filename)
    title_text = 'startsample: '+str(start_sample)+'  |  endsample: '+str(end_sample)+'  |  endtime: '+str(round(end_sample*5/60,2))+'min''  |   successful: ' + str(end_flag)
    plt.title(title_text, ha='center')
    plt.show()
    #plt.close()
              
    return end_sample, end_flag