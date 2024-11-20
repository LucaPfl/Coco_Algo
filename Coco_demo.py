import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import customtkinter
import threading
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from Preprocessing.evaluation_function import search_moving_average, check_saddle_point_occurred
from Preprocessing.Filter_Routines import preprocess_with_period_filter2
from Preprocessing.preprocessing_function import (
    cut_BLDC_data_b, cut_BLDC_data, resample_BLDC_data, cStaticVariables,
    Filter_Accumulated_Average, temperature_comp, get_nom_speed
)


class CocoReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coco Reader")
        self.root.geometry("1400x800")  # Larger display for better spacing
        self.root.configure(bg="#f4f4f4")  # Light background for freshness
        self.stop = False

        # Placeholder for the writer and reader threads
        self.writer_thread = None
        self.reader_thread = None

        # Main frame for layout organization
        main_frame = tk.Frame(self.root, bg="#f4f4f4", padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(5, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Heading Label
        title_label = tk.Label(
            main_frame, text="Coco Reader Data Analysis", font=("Helvetica", 16, "bold"),
            bg="#f4f4f4", fg="#333"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        # Create a frame for the upper section (Buttons, Inputs, and Info)
        upper_frame = tk.Frame(main_frame, bg="#f4f4f4", padx=10, pady=10)
        upper_frame.grid(row=1, column=0, columnspan=4, pady=10)

        # Configure weights for centering the upper frame
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Left: Buttons and Input Fields
        action_frame = tk.Frame(upper_frame, bg="#f4f4f4", padx=10, pady=10, bd=1, relief="solid")
        action_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        # Right: Status and File Path
        info_frame = tk.Frame(action_frame, bg="#f4f4f4", padx=10, pady=10, bd=1, relief="solid")
        info_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Mockup and Start Buttons
        self.start_mock_button = tk.Button(
            info_frame, text="Mockup", command=self.start_mock_thread,
            bg="#008080", fg="white", font=("Helvetica", 7, "bold"),
            width=5, height=1
        )
        self.start_mock_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.start_button = tk.Button(
            info_frame, text="Start", command=self.start_thread,
            bg="#008080", fg="white", font=("Helvetica", 12, "bold"),
            width=15, height=2
        )
        self.start_button.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Input Fields in the same frame
        input_frame = tk.LabelFrame(action_frame, text="Parameter Einstellungen", bg="#f4f4f4",
                                    font=("Helvetica", 12, "bold"), padx=10, pady=10)
        input_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.label1 = tk.Label(input_frame, text="Min Samples:", bg="#f4f4f4", anchor="w")
        self.label1.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.min_samples = tk.Entry(input_frame)
        self.min_samples.insert(0, 500)
        self.min_samples.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.label2 = tk.Label(input_frame, text="Stop Signal:", bg="#f4f4f4", anchor="w")
        self.label2.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.stop_signal = tk.Entry(input_frame)
        self.stop_signal.insert(0, 2)
        self.stop_signal.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.label3 = tk.Label(input_frame, text="Samples Short:", bg="#f4f4f4", anchor="w")
        self.label3.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.samples_short = tk.Entry(input_frame)
        self.samples_short.insert(0, 20)
        self.samples_short.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        self.label4 = tk.Label(input_frame, text="Samples Long:", bg="#f4f4f4", anchor="w")
        self.label4.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.samples_long = tk.Entry(input_frame)
        self.samples_long.insert(0, 50)
        self.samples_long.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        self.label5 = tk.Label(input_frame, text="Filter Period 1:", bg="#f4f4f4", anchor="w")
        self.label5.grid(row=4, column=0, padx=10, pady=5, sticky="e")
        self.filter_period_1 = tk.Entry(input_frame)
        self.filter_period_1.insert(0, 350)
        self.filter_period_1.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Status Label
        self.status_label = tk.Label(
            info_frame, text="Status: Bereit", fg="red", font=("Helvetica", 15),
            bg="#f4f4f4", anchor="w"
        )
        self.status_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # File Path Selection
        self.path_label = tk.Label(info_frame, text="Dateipfad:", bg="#f4f4f4", anchor="w")
        self.path_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.path_entry = tk.Entry(info_frame, width=40)
        self.path_entry.insert(0, r'K:\EWB_070_Steam_Tech\011_measurements\read_in_test\MeasITtestlive.csv')
        self.path_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.browse_button = tk.Button(info_frame, text="Durchsuchen", command=self.browse_path)
        self.browse_button.grid(row=1, column=2, padx=5, pady=5)

        # Configure resizing for path entry
        info_frame.grid_columnconfigure(1, weight=1)

        # Bottom: Plots
        self.plot_frame1 = tk.Frame(main_frame, bg="#f4f4f4", bd=1, relief="sunken", padx=10, pady=10)
        self.plot_frame1.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.plot_frame2 = tk.Frame(main_frame, bg="#f4f4f4", bd=1, relief="sunken", padx=10, pady=10)
        self.plot_frame2.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        # Create the first plot
        self.figure1 = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.figure1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plot_frame1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create the second plot
        self.figure2 = Figure(figsize=(5, 4), dpi=100)
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        # Bind the window's close button to the close_app function
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def browse_path(self):
        path = filedialog.askopenfilename()
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, path)

    def update_plot(self, data, datatime, plot=1, saddle_point_time=None):
        """Updates the plot with new data and optionally adds a saddle point marker."""
        if plot == 1:
            self.ax1.clear()
            self.ax1.plot(datatime, data, color="#004466", label="Power Signal", linewidth=1.5)
            self.ax1.set_title("Power Signal over Time", fontsize=14, fontweight='bold', color="#333")
            self.ax1.set_xlabel("Time in min", fontsize=12)
            self.ax1.set_ylabel("Power Signal", fontsize=12)
            self.ax1.grid(True, linestyle="--", alpha=0.5)
            self.ax1.legend()

            # If a saddle point is provided, add a vertical line at the saddle point
            if saddle_point_time is not None:
                self.ax1.axvline(x=saddle_point_time, color="red", linestyle="--", label="Saddle Point")
                self.ax1.legend()
        
            self.canvas1.draw()
        elif plot == 2:
            try:
                self.ax2.clear()
                self.ax2.plot(datatime, data, color="#004466", label="Clean Power Signal", linewidth=1.5)
                self.ax2.set_title("Power Signal over Time", fontsize=14, fontweight='bold', color="#333")
                self.ax2.set_xlabel("Time in min", fontsize=12)
                self.ax2.set_ylabel("Power Signal", fontsize=12)
                self.ax2.grid(True, linestyle="--", alpha=0.5)
                self.ax2.legend()
                # If a saddle point is provided, add a vertical line at the saddle point
                if saddle_point_time is not None:
                    self.ax2.axvline(x=saddle_point_time, color="red", linestyle="--", label="Saddle Point")
                    self.ax2.legend()
                self.canvas2.draw()
            except:
                print('plot 2 not ready yet')


    def dataReader(self, mock=False, datapath=r"K:\EWB_070_Steam_Tech\011_measurements\Coco_Read_In\MeasITtestlive.csv"):
        motor_sample_rate = 1
        temperature_compensation = False
        min_samples = int(self.min_samples.get()) // motor_sample_rate
        stop_signal = int(self.stop_signal.get())
        samples_short = int(self.samples_short.get()) * 5 // motor_sample_rate
        samples_long = int(self.samples_long.get()) * 5 // motor_sample_rate
        acceptance_limit = {
            "power_min": 2500,
            "speed_low": 10,
            "speed_high": 11640,
            "temp_min": 20,
        }
        filter_period_1 = int(self.filter_period_1.get())
        lower_cutoff_threshold = 0
        start_end_padding_enabled = False
        padding_value = 0
        group_delay_compensation_enabled = False

        time.sleep(2)
        while not self.stop:
            try:
                if mock:
                    df = pd.read_csv(datapath)
                else:
                    df = pd.read_csv(datapath, delimiter=';', on_bad_lines='skip', decimal=',')
                df.fillna(-1, inplace=True)

                # Offset Correction with Temp set point
                Temp_Minimum_Value = max(df['CPM1_TempSetPoint']) - acceptance_limit['temp_min']
                Clean_Power_Signal = Filter_Accumulated_Average(df, Temp_Minimum_Value, acceptance_limit, temperature_compensation)
                df["cps"] = Clean_Power_Signal
                Clean_Power_Signal_C = np.trim_zeros(Clean_Power_Signal, 'f')

                valid_indices = np.where(Clean_Power_Signal != 0)[0]
                
                df['timestamp_datetime'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                df['time_delta'] = (df['timestamp_datetime'] - (df['timestamp_datetime'].iloc[0])).dt.total_seconds() / 60
                try:
                    df_timestemp_clean = df['timestamp'].iloc[valid_indices].reset_index(drop=True)

                    df_timestemp_clean['timestamp_clean_datetime'] = pd.to_datetime(df_timestemp_clean, format='%d.%m.%Y %H:%M:%S', errors='coerce')
                    df_timestemp_clean['time_delta_clean'] = (df_timestemp_clean['timestamp_clean_datetime'] - (df['timestamp_datetime'].iloc[0])).dt.total_seconds() / 60
                except:
                    print('2 not ready yet')    

                if len(Clean_Power_Signal_C) == 0:
                    self.status_label.config(text="No power data or data out of bounds", fg="red")
                    print(": No power data or data out of bounds")
                else:
                    # Filter and smooth the signal
                    y = np.array(Clean_Power_Signal_C)
                    Clean_Power_Signal_Smooth = preprocess_with_period_filter2(
                        y, lower_cutoff_threshold, filter_period_1,
                        start_end_padding_enabled, padding_value, group_delay_compensation_enabled
                    )

                    # Check for saddle point occurrence
                    saddle_occurred, end_sample_ma = check_saddle_point_occurred(
                        Clean_Power_Signal_Smooth, min_samples, stop_signal, samples_short, samples_long
                    )

                    
                    # Pass the timestamp of the saddle point (if found)
                    saddle_point_time = None
                    if saddle_occurred:
                        # Convert the sample index of the saddle point to timestamp
                        saddle_point_time = df_timestemp_clean['time_delta_clean'].iloc[end_sample_ma]
                        print(f"Saddle Point Found at Sample: {end_sample_ma}")

                    self.update_plot(df['Send_HotAirFan_Power'], df['time_delta'], plot=1, saddle_point_time=saddle_point_time)

                    #self.update_plot(df['Send_HotAirFan_Power'], matplotlib.dates.date2num(pd.to_datetime(df['time_delta'], format='%d.%m.%Y %H:%M:%S')), plot=1, saddle_point_time=saddle_point_time1)
                    self.update_plot(Clean_Power_Signal_Smooth,df_timestemp_clean['time_delta_clean'] , plot=2,saddle_point_time = saddle_point_time)
                    print(df)
                    # Update status message if saddle point is found
                    if saddle_occurred:
                        self.status_label.config(
                            text=f"Saddle Point Found at Sample: {end_sample_ma}", fg="#008080"
                        )
                        self.stop = True
                    else:
                        self.status_label.config(text="Searching...", fg="blue")
                        print("Searching...")
            except Exception as e:
                print("Could not read the file this time")
                print(e)
            time.sleep(2)


    def dataWriter(self, file_path_from=r'K:\EWB_070_Steam_Tech\011_measurements\read_in_test\MeasITtestlive.csv',
                   file_path_to=r"K:\EWB_070_Steam_Tech\011_measurements\Coco_Read_In\MeasITtestlive.csv",
                   time_shift=100):
        #df = pd.read_csv(file_path_from, delimiter=';', skiprows=3, decimal=',')
        df = pd.read_csv(file_path_from, delimiter=';', on_bad_lines='skip', decimal=',')
        counter = 1
        while not self.stop:
            print("Writing data...")
            df.iloc[:counter].to_csv(file_path_to)
            counter += time_shift
            time.sleep(1)

    def start_mock_thread(self):
        if not self.writer_thread or not self.writer_thread.is_alive():
            self.writer_thread = threading.Thread(target=self.dataWriter, kwargs={'time_shift': 60}, daemon=True)
            self.writer_thread.start()

        if not self.reader_thread or not self.reader_thread.is_alive():
            self.reader_thread = threading.Thread(target=self.dataReader, kwargs={'mock': True}, daemon=True)
            self.reader_thread.start()

    def start_thread(self):
        self.stop=False
        if not self.reader_thread or not self.reader_thread.is_alive():
            self.reader_thread = threading.Thread(target=self.dataReader,kwargs={'datapath':str(self.path_entry.get()),'mock': False}, daemon=True)
            self.reader_thread.start()

    def close_app(self):
        print("Closing application...")
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join()

        self.root.destroy()


# Initialize and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CocoReaderApp(root)
    root.mainloop()
