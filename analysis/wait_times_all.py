# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 07:08:45 2024

@author: david

goal: create wait time/energy plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy'])

flatwrm_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_31-265/flatwrm_output_985_erg_all.txt", skiprows = 1)
flatwrm_df = pd.DataFrame(
    flatwrm_data,
    columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])

AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/final-flares_all.csv")

AFDc_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/flares-candidates_all.csv")

Okamoto_wait_times = []
flatwrm_wait_times = []
AFD_wait_times = []
AFDc_wait_times = []
AFDc_wait_times2 = []

Okamoto_energies = []
flatwrm_energies = []
AFD_energies = []
AFDc_energies = []

Okamoto_energies2 = []
flatwrm_energies2 = []
AFD_energies2 = []
AFDc_energies2 = []
for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    flatwrm_KIC = flatwrm_df.loc[flatwrm_df['kepler_id'] == KIC]
    AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == int(KIC)]
    AFDc_KIC = AFDc_df.loc[AFDc_df['kepler_id'] == int(KIC)]
    
    for i in range(np.size(flatwrm_KIC.values[:,0])-1):
        flatwrm_wait_times.append(flatwrm_KIC.values[i+1,3]-flatwrm_KIC.values[i,3])
        flatwrm_energies.append(flatwrm_KIC.values[i,6])
        flatwrm_energies2.append(flatwrm_KIC.values[i+1,6])
    for i in range(np.size(Okamoto_KIC.values[:,0])-1):
        Okamoto_wait_times.append(Okamoto_KIC.values[i+1,1]-Okamoto_KIC.values[i,1])
        Okamoto_energies.append(Okamoto_KIC.values[i,3])
        Okamoto_energies2.append(Okamoto_KIC.values[i+1,3])
    for i in range(np.size(AFD_KIC.values[:,0])-1):
        AFD_wait_times.append(AFD_KIC.values[i+1,19]-AFD_KIC.values[i,19])
        AFD_energies.append(AFD_KIC.values[i,9])
        AFD_energies2.append(AFD_KIC.values[i+1,9])
    for i in range(np.size(AFDc_KIC.values[:,0])-1):
        if AFDc_KIC.values[i,9] > 0:
            AFDc_wait_times.append(AFDc_KIC.values[i+1,19]-AFDc_KIC.values[i,19])
            AFDc_energies.append(AFDc_KIC.values[i,9])
        if AFDc_KIC.values[i+1,9] > 0:
            AFDc_wait_times2.append(AFDc_KIC.values[i+1,19]-AFDc_KIC.values[i,19])
            AFDc_energies2.append(AFDc_KIC.values[i+1,9])


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
    ax.grid(True)
    ax.hist(data, bins = bins_set, range = range_array, density = True, label = plot_label)
    xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
    ax.set_xlabel(xlabeltxt)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title("Histogram for wait times of flares")
    plt.show()

    # histogram for each filter individually
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for index in range(len(data)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.grid(True)
        ax.hist(data[index], bins = bins_set, color = colors[index], range = range_array, label = plot_label[index])
        xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
        ax.set_xlabel(xlabeltxt)
        ax.set_ylabel('Frequency')
        ax.legend()
        titletxt = "Histogram for wait times of "+plot_filter[index]+" flares"
        ax.set_title(titletxt)
        plt.show()
        plt.close()

# histogram for all
# data = [Okamoto_wait_times, flatwrm_wait_times, AFDc_wait_times, AFD_wait_times]
# plot_filter = ["Okamoto", "Flatwrm", "AFD_c", "AFD_f"]

data = [Okamoto_wait_times, flatwrm_wait_times, AFD_wait_times]
plot_filter = ["Okamoto", "Flatwrm", "AFD_f"]

# histogram parameter 1000
bins_set = int((max(Okamoto_wait_times))/5)
range_array = (0, 1000)

# plot_histogram(data, plot_filter, bins_set, range_array)

# histogram parameter 200
bins_set = int((max(flatwrm_wait_times))/20)
range_array = (0, 200)

# plot_histogram(data, plot_filter, bins_set, range_array)

# histogram parameter 50
bins_set = int((max(flatwrm_wait_times))/20)
range_array = (0, 50)

# plot_histogram(data, plot_filter, bins_set, range_array)

# =============================================================================
# Energies
# =============================================================================

def plot_energies(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    # for i in range(len(data)):
    ax.grid(True)
    for i in [1,0,2]:
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    # for i in range(len(data)):
    for i in [1,0,2]:
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time to next flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Wait times")
    plt.show()
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    # energies for each filter individually
    for index in [0,1]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.grid(True)
        ax.scatter(data[index], energies[index], s=1, color = colors[index], label = plot_filter[index])
        ax.scatter(np.median(data[index]), np.median(energies[index]), s=30, c = colors[index], edgecolors = "black",  marker = "X")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('wait time to next flare [days]')
        ax.set_ylabel('flare energy [ergs]')
        # ax.legend()
        titletxt = "Energies and Wait times for "+plot_filter[index]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[index]), np.median(energies[index]))
        ax.set_title(titletxt)
        plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.grid(True)
    ax.scatter(data[2], energies[2], s=1, color = colors[2], label = plot_filter[2])
    ax.scatter(np.median(data[2]), np.median(energies[2]), s=30, c = colors[2], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1, max(data[2])*1.3])
    ax.set_xlabel('wait time to next flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    # ax.legend()
    titletxt = "Energies and Wait times for "+plot_filter[2]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[2]), np.median(energies[2]))
    ax.set_title(titletxt)
    plt.show()

def plot_energies2(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    ax.grid(True)
    for i in [1,0,2]:
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    for i in [1,0,2]:
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time before flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Wait times")
    plt.show()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    # energies for each filter individually
    for index in [0,1]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.grid(True)
        ax.scatter(data[index], energies[index], s=1, color = colors[index], label = plot_filter[index])
        ax.scatter(np.median(data[index]), np.median(energies[index]), s=30, c = colors[index], edgecolors = "black",  marker = "X")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('wait time before flare [days]')
        ax.set_ylabel('flare energy [ergs]')
        # ax.legend()
        titletxt = "Energies and Wait times for "+plot_filter[index]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[index]), np.median(energies[index]))
        ax.set_title(titletxt)
        plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.grid(True)
    ax.scatter(data[2], energies[2], s=1, color = colors[2], label = plot_filter[2])
    ax.scatter(np.median(data[2]), np.median(energies[2]), s=30, c = colors[2], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1, max(data[2])*1.3])
    ax.set_xlabel('wait time before flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    # ax.legend()
    titletxt = "Energies and Wait times for "+plot_filter[2]+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(np.median(data[2]), np.median(energies[2]))
    ax.set_title(titletxt)
    plt.show()

# energies = [Okamoto_energies, flatwrm_energies, AFDc_energies, AFD_energies]
energies = [Okamoto_energies, flatwrm_energies, AFD_energies]

plot_energies(data, energies, plot_filter)

# energies2
# data2 = [Okamoto_wait_times, flatwrm_wait_times, AFDc_wait_times2, AFD_wait_times]
# energies2 = [Okamoto_energies2, flatwrm_energies2, AFDc_energies2, AFD_energies2]

data2 = [Okamoto_wait_times, flatwrm_wait_times, AFD_wait_times]
energies2 = [Okamoto_energies2, flatwrm_energies2, AFD_energies2]

plot_energies2(data2, energies2, plot_filter)
