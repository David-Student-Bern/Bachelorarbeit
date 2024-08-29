# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:25:31 2024

@author: david
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 06:49:02 2024

@author: david
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")
short_KIC = short_KIC[:30]

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 36, usecols=[0, 11, 12, 13]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy'])

flatwrm_data_1 = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_1/flatwrm_output_985_erg_0-30.txt", skiprows = 1)
flatwrm_1_df = pd.DataFrame(
    flatwrm_data_1,
    columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])

flatwrm_data_2 = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_3/flatwrm_output_985_erg_0-30_3.txt", skiprows = 1)
flatwrm_2_df = pd.DataFrame(
    flatwrm_data_2,
    columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])


all_O = 0
common_O_f1 = 0
common_O_f2 = 0

common_f1_f2 = 0

common_O_f1_f2 = 0


flatwrm_1_wait_times = []
flatwrm_1_flare_length = []
flatwrm_1_energies = []
flatwrm_1_energies1 = []
flatwrm_1_energies2 = []

flatwrm_2_wait_times = []
flatwrm_2_flare_length = []
flatwrm_2_energies = []
flatwrm_2_energies1 = []
flatwrm_2_energies2 = []

flatwrm_12_wait_times = []
flatwrm_12_flare_length = []
flatwrm_12_energies = []
flatwrm_12_energies1 = []
flatwrm_12_energies2 = []


for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    flatwrm_1_KIC = flatwrm_1_df.loc[flatwrm_1_df['kepler_id'] == KIC]
    flatwrm_2_KIC = flatwrm_2_df.loc[flatwrm_2_df['kepler_id'] == KIC]
    
    for O_fpt in Okamoto_KIC['flare_peak_time']:
        O_f1 = False
        O_f2 = False
        all_O += 1
        #test if in flatwrm test1
        for i in range(np.size(flatwrm_1_KIC.values[:,0])):
            if (flatwrm_1_KIC.values[i,1] <= (O_fpt+2400000) <= flatwrm_1_KIC.values[i,2]):
                common_O_f1 += 1
                O_f1 = True
                break
        
        #test if in flatwrm test2
        for i in range(np.size(flatwrm_2_KIC.values[:,0])):
            if (flatwrm_2_KIC.values[i,1] <= (O_fpt+2400000) <= flatwrm_2_KIC.values[i,2]):
                common_O_f2 += 1
                O_f2 = True
                break
        
        if O_f1 and O_f2:
            common_O_f1_f2 += 1
    
    for f1_fpt in flatwrm_1_KIC["t_max"]:
        #test if in flatwrm
        for i in range(np.size(flatwrm_2_KIC.values[:,0])):
            if (flatwrm_2_KIC.values[i,1] <= (f1_fpt) <= flatwrm_2_KIC.values[i,2]):
                common_f1_f2 += 1
                flatwrm_12_flare_length.append(flatwrm_2_KIC.values[i,2]-flatwrm_2_KIC.values[i,1])
                flatwrm_12_energies.append(flatwrm_2_KIC.values[i,6])
                if i < np.size(flatwrm_2_KIC.values[:,0])-1:
                    flatwrm_12_wait_times.append(flatwrm_2_KIC.values[i+1,3]-flatwrm_2_KIC.values[i,3])
                    flatwrm_12_energies1.append(flatwrm_2_KIC.values[i,6])
                    flatwrm_12_energies2.append(flatwrm_2_KIC.values[i+1,6])
    
    for i in range(np.size(flatwrm_1_KIC.values[:,0])):
        flatwrm_1_flare_length.append(flatwrm_1_KIC.values[i,2]-flatwrm_1_KIC.values[i,1])
        flatwrm_1_energies.append(flatwrm_1_KIC.values[i,6])
    for i in range(np.size(flatwrm_2_KIC.values[:,0])):
        flatwrm_2_flare_length.append(flatwrm_2_KIC.values[i,2]-flatwrm_2_KIC.values[i,1])
        flatwrm_2_energies.append(flatwrm_2_KIC.values[i,6])
    
    for i in range(np.size(flatwrm_1_KIC.values[:,0])-1):
        flatwrm_1_wait_times.append(flatwrm_1_KIC.values[i+1,3]-flatwrm_1_KIC.values[i,3])
        flatwrm_1_energies1.append(flatwrm_1_KIC.values[i,6])
        flatwrm_1_energies2.append(flatwrm_1_KIC.values[i+1,6])
    for i in range(np.size(flatwrm_2_KIC.values[:,0])-1):
        if flatwrm_2_KIC.values[i,2]-flatwrm_2_KIC.values[i,1] > 0.2:
            print("flatwrm: KIC = {:.0f}, tstart = {:.2f}, dur = {:.3f}".format(flatwrm_2_KIC.values[i,0], flatwrm_2_KIC.values[i,1], flatwrm_2_KIC.values[i,2]-flatwrm_2_KIC.values[i,1]))
        flatwrm_2_wait_times.append(flatwrm_2_KIC.values[i+1,3]-flatwrm_2_KIC.values[i,3])
        flatwrm_2_energies1.append(flatwrm_2_KIC.values[i,6])
        flatwrm_2_energies2.append(flatwrm_2_KIC.values[i+1,6])

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
    ax.legend(loc = "upper left")
    ax.set_title("Energies and Wait times")
    ax.grid(True)
    plt.show()

def plot_energies2(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time before flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend(loc = "upper left")
    ax.set_title("Energies and Wait times")
    ax.grid(True)
    plt.show()

def plot_flare_length(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('flare duration [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Flare duration")
    ax.grid(True)
    plt.show()

data = [flatwrm_1_wait_times, flatwrm_2_wait_times, flatwrm_12_wait_times]
plot_filter = ["only Flatwrm-1", "only Flatwrm-2", "Flatwrm-1&2"]
energies1 = [flatwrm_1_energies1, flatwrm_2_energies1, flatwrm_12_energies1]


plot_energies(data, energies1, plot_filter)

# energies2
energies2 = [flatwrm_1_energies2, flatwrm_2_energies2, flatwrm_12_energies2]

plot_energies2(data, energies2, plot_filter)

# flare length
flare_length = [flatwrm_1_flare_length, flatwrm_2_flare_length, flatwrm_12_flare_length]
energies = [flatwrm_1_energies, flatwrm_2_energies, flatwrm_12_energies]

plot_flare_length(flare_length, energies, plot_filter)