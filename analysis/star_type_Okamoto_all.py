# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:08:48 2024

@author: david

goal: Same idea as in prot_Okamoto_all.py but instead of the rotational period the star type
(<--> effective temperature of the star) is used to create different groups.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def wait_times_energies(Okamoto_df, short_KIC):
    Okamoto_wait_times = []
    Okamoto_energies = []
    Okamoto_energies1 = []
    Okamoto_energies2 = []
    Okamoto_flare_length = []
    
    for KIC in short_KIC:
        Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    
        for i in range(np.size(Okamoto_KIC.values[:,0])-1):
            Okamoto_wait_times.append(Okamoto_KIC.values[i+1,1]-Okamoto_KIC.values[i,1])
            Okamoto_energies1.append(Okamoto_KIC.values[i,3])
            Okamoto_energies2.append(Okamoto_KIC.values[i+1,3])
            Okamoto_flare_length.append(Okamoto_KIC.values[i,2])
            Okamoto_energies.append(Okamoto_KIC.values[i,3])
        if np.size(Okamoto_KIC.values[:,2])>0:
            Okamoto_flare_length.append(Okamoto_KIC.values[-1,2])
            Okamoto_energies.append(Okamoto_KIC.values[-1,3])
    return Okamoto_wait_times, Okamoto_energies1, Okamoto_energies2, Okamoto_energies, Okamoto_flare_length

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13, 1, 6]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy', 'Teff', 'Prot'])

# lists of star types
# type_A = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/stellar parameters/A-type.csv")
# type_F = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/stellar parameters/F-type.csv")
# type_G = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/stellar parameters/G-type.csv")
# type_K = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/stellar parameters/K-type.csv")
# type_M = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/stellar parameters/M-type.csv")

# Okamoto_data_A = Okamoto_df[Okamoto_df['kepler_id'].isin(type_A.values[:,0])]
# Okamoto_data_F = Okamoto_df[Okamoto_df['kepler_id'].isin(type_F.values[:,0])]
# Okamoto_data_G = Okamoto_df[Okamoto_df['kepler_id'].isin(type_G.values[:,0])]
# Okamoto_data_K = Okamoto_df[Okamoto_df['kepler_id'].isin(type_K.values[:,0])]
# Okamoto_data_M = Okamoto_df[Okamoto_df['kepler_id'].isin(type_M.values[:,0])]


Okamoto_data_G = Okamoto_df[Okamoto_df['Teff'].between(5400,6000)]
Okamoto_data_K = Okamoto_df[Okamoto_df['Teff'].between(5000,5400)]

Okamoto_wait_times_G, Okamoto_energies1_G, Okamoto_energies2_G, Okamoto_energies_G, Okamoto_flare_length_G = wait_times_energies(Okamoto_data_G, short_KIC)

Okamoto_wait_times_K, Okamoto_energies1_K, Okamoto_energies2_K, Okamoto_energies_K, Okamoto_flare_length_K = wait_times_energies(Okamoto_data_K, short_KIC)


# =============================================================================
# Histogram
# =============================================================================
def plot_histogram(data, plot_filter, bins_set, range_array):
    
    # compute f2: measurements are outside the selected range for plotting
    allf1 = []
    plot_label = []
    for index in range(len(data)):
        filter1 = 0
        for item in data[index]:
            if item < range_array[0] or item > range_array[1]:
                filter1 +=1
        f1 = 100 * (filter1/len(data[index]))
        allf1.append(f1)
        label = plot_filter[index] + r': F$_1$ = ' + str(round(f1,3)) + '%'
        plot_label.append(label)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))

    ax.hist(data, bins = bins_set, range = range_array, label = plot_label, stacked = True)
    xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
    ax.set_xlabel(xlabeltxt)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title("Histogram for wait times of flares")
    ax.grid(True)
    plt.show()

    # histogram for each filter individually
    # for index in range(len(data)):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    #     ax.hist(data[index], bins = bins_set, range = range_array, label = plot_label[index])
    #     xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
    #     ax.set_xlabel(xlabeltxt)
    #     ax.set_ylabel('Frequency')
    #     ax.legend()
    #     titletxt = "Histogram for wait times of "+plot_filter[index]+" flares"
    #     ax.set_title(titletxt)
    #     ax.grid(True)
    #     plt.show()
    #     plt.close()

# histogram for all
data = [Okamoto_wait_times_G, Okamoto_wait_times_K]
plot_filter = ["G-type", "K-Type"]

# histogram parameter 1000
bins_set = int((max(Okamoto_wait_times_G))/5)
range_array = (0, 1000)

# plot_histogram(data, plot_filter, bins_set, range_array)

# histogram parameter 200
bins_set = int((max(Okamoto_wait_times_G))/20)
range_array = (0, 200)

plot_histogram(data, plot_filter, bins_set, range_array)

# histogram parameter 50
bins_set = int((max(Okamoto_wait_times_G))/20)
range_array = (0, 50)

plot_histogram(data, plot_filter, bins_set, range_array)

# =============================================================================
# Energies
# =============================================================================

def plot_energies(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time to next flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Wait times")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in [0,1]:
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    #     ax.scatter(data[index], energies[index], s=1, label = plot_filter[index])
    #     ax.set_yscale('log')
    #     ax.set_xscale('log')
    #     ax.set_xlabel('wait time to next flare [days]')
    #     ax.set_ylabel('flare energy [ergs]')
    #     ax.legend()
    #     titletxt = "Energies and Wait times for "+plot_filter[index]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[index]), np.median(energies[index]))
    #     ax.set_title(titletxt)
    #     ax.grid(True)
    #     plt.show()

def plot_energies2(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black", marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time before flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Wait times")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in [0,1]:
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    #     ax.scatter(data[index], energies[index], s=1, label = plot_filter[index])
    #     ax.set_yscale('log')
    #     ax.set_xscale('log')
    #     ax.set_xlabel('wait time before flare [days]')
    #     ax.set_ylabel('flare energy [ergs]')
    #     ax.legend()
    #     titletxt = "Energies and Wait times for "+plot_filter[index]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[index]), np.median(energies[index]))
    #     ax.set_title(titletxt)
    #     ax.grid(True)
    #     plt.show()
    

# newX = np.logspace(0, 3, base=10)
# newY = np.logspace(33, 36, base = 10)

energies1 = [Okamoto_energies1_G, Okamoto_energies1_K]

plot_energies(data, energies1, plot_filter)

# energies2
data2 = [Okamoto_wait_times_G, Okamoto_wait_times_K]
energies2 = [Okamoto_energies2_G, Okamoto_energies2_K]

plot_energies2(data2, energies2, plot_filter)

# =============================================================================
# flare length
# =============================================================================

def plot_flare_length(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    for i in range(len(data)):
        ax.scatter(data[i], energies[i],  s=1, label = plot_filter[i])
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('flare duration [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Flare duration")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in [0,1]:
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    #     ax.scatter(data[index], energies[index], s=1, label = plot_filter[index])
    #     ax.set_yscale('log')
    #     # ax.set_xscale('log')
    #     ax.set_xlabel('flare duration [days]')
    #     ax.set_ylabel('flare energy [ergs]')
    #     ax.legend()
    #     titletxt = "Energies and Flare duration for "+plot_filter[index]
    #     ax.set_title(titletxt)
    #     ax.grid(True)
    #     plt.show()

flare_length = [Okamoto_flare_length_G, Okamoto_flare_length_K]
energies = [Okamoto_energies_G, Okamoto_energies_K]

plot_flare_length(flare_length, energies, plot_filter)