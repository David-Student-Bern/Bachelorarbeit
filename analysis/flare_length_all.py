# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:36:26 2024

@author: david

goal: create flare duration/energy plots
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

Okamoto_flare_length = []
flatwrm_flare_length = []
AFD_flare_length = []
AFDc_flare_length = []


Okamoto_energies = []
flatwrm_energies = []
AFD_energies = []
AFDc_energies = []

for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    flatwrm_KIC = flatwrm_df.loc[flatwrm_df['kepler_id'] == KIC]
    AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == int(KIC)]
    AFDc_KIC = AFDc_df.loc[AFDc_df['kepler_id'] == int(KIC)]
    
    for i in range(np.size(flatwrm_KIC.values[:,0])):
        if flatwrm_KIC.values[i,2]-flatwrm_KIC.values[i,1] > 0.2:
            print("flatwrm: KIC = {:.0f}, tstart = {:.2f}, dur = {:.3f}".format(flatwrm_KIC.values[i,0], flatwrm_KIC.values[i,1], flatwrm_KIC.values[i,2]-flatwrm_KIC.values[i,1]))
        flatwrm_flare_length.append(flatwrm_KIC.values[i,2]-flatwrm_KIC.values[i,1])
        flatwrm_energies.append(flatwrm_KIC.values[i,6])
    for i in range(np.size(Okamoto_KIC.values[:,0])):
        Okamoto_flare_length.append(Okamoto_KIC.values[i,2])
        Okamoto_energies.append(Okamoto_KIC.values[i,3])
    for i in range(np.size(AFD_KIC.values[:,0])):
        # if AFD_KIC.values[i,6]-AFD_KIC.values[i,5] > 0.2:
        #     print("AFD: KIC = {:.0f}, tstart = {:.2f}".format(AFD_KIC.values[i,1], AFD_KIC.values[i,5]))
        AFD_flare_length.append(AFD_KIC.values[i,6]-AFD_KIC.values[i,5])
        AFD_energies.append(AFD_KIC.values[i,9])
    for i in range(np.size(AFDc_KIC.values[:,0])):
        if AFDc_KIC.values[i,9] > 0 and (AFDc_KIC.values[i,6]-AFDc_KIC.values[i,5])< 1:
            AFDc_flare_length.append(AFDc_KIC.values[i,6]-AFDc_KIC.values[i,5])
            AFDc_energies.append(AFDc_KIC.values[i,9])

def plot_energies(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    # for i in range(len(data)):
    for i in [0,1,3]:
        ax.scatter([x *24 for x in data[i]], energies[i],  s=1, c = colors[i], label = plot_filter[i])
    # for i in range(len(data)):
    for i in [0,1,3]:
        ax.scatter(np.median(data[i])*24, np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('flare duration [hours]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Energies and Flare duration")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    for index in [0,1,2,3]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.scatter([x *24 for x in data[index]], energies[index], color = colors[index], s=1, label = plot_filter[index])
        ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_xlabel('flare duration [hours]')
        ax.set_ylabel('flare energy [ergs]')
        ax.legend()
        titletxt = "Energies and Flare duration for "+plot_filter[index]
        ax.set_title(titletxt)
        ax.grid(True)
        plt.show()

    
    # overlap with okamoto flares
    # energies for each filter individually
    for index in [1,2,3]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.scatter([x *24 for x in data[0]], energies[0], s=1, c = "tab:blue", label = plot_filter[0])
        ax.scatter([x *24 for x in data[index]], energies[index], s=1, c = colors[index], label = plot_filter[index])
        # ax.scatter(data[0], energies[0], s=1, c = "tab:blue", label = plot_filter[0])
        ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_xlabel('flare duration [hours]')
        ax.set_ylabel('flare energy [ergs]')
        ax.legend()
        titletxt = "Energies and Flare duration for "+plot_filter[index]
        ax.set_title(titletxt)
        ax.grid(True)
        plt.show()

data = [Okamoto_flare_length, flatwrm_flare_length, AFDc_flare_length, AFD_flare_length]
plot_filter = ["Okamoto", "Flatwrm", "AFD_c", "AFD_f"]
energies = [Okamoto_energies, flatwrm_energies, AFDc_energies, AFD_energies]

plot_energies(data, energies, plot_filter)