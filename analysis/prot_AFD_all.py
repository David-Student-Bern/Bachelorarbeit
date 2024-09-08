# -*- coding: utf-8 -*-
"""
Created on Tue May  7 07:11:46 2024

@author: david

goal: create different plots for AFD including different ranges in rotational period
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wait_times_energies(Okamoto_df, short_KIC):
    Okamoto_wait_times = []
    Okamoto_energies = []
    Okamoto_energies1 = []
    Okamoto_energies2 = []
    Okamoto_flare_length = []
    
    for KIC in short_KIC:
        Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    
        for i in range(np.size(Okamoto_KIC.values[:,0])-1):
            Okamoto_wait_times.append(Okamoto_KIC.values[i+1,19]-Okamoto_KIC.values[i,19])
            Okamoto_energies1.append(Okamoto_KIC.values[i,9])
            Okamoto_energies2.append(Okamoto_KIC.values[i+1,9])
            Okamoto_flare_length.append(Okamoto_KIC.values[i,6]-Okamoto_KIC.values[i,5])
            Okamoto_energies.append(Okamoto_KIC.values[i,9])
        if np.size(Okamoto_KIC.values[:,2])>0:
            Okamoto_flare_length.append(Okamoto_KIC.values[-1,6]-Okamoto_KIC.values[-1,5])
            Okamoto_energies.append(Okamoto_KIC.values[-1,9])
    return Okamoto_wait_times, Okamoto_energies1, Okamoto_energies2, Okamoto_energies, Okamoto_flare_length

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

# =============================================================================
# Okamoto
# =============================================================================
Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13, 1, 6]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy', 'Teff', 'Prot'])

# Okamoto_data = [0,0,0,0,0,0]

# Okamoto_data[0] = Okamoto_df[Okamoto_df['Prot'].between(0,2)]
# Okamoto_data[1] = Okamoto_df[Okamoto_df['Prot'].between(2,5)]
# Okamoto_data[2] = Okamoto_df[Okamoto_df['Prot'].between(5,15)]
# Okamoto_data[3] = Okamoto_df[Okamoto_df['Prot'].between(15,25)]
# Okamoto_data[4] = Okamoto_df[Okamoto_df['Prot'].between(25,35)]
# Okamoto_data[5] = Okamoto_df[Okamoto_df['Prot'].between(35,100)]

# =============================================================================
# Flatwrm
# =============================================================================
# flatwrm_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_31-265/flatwrm_output_985_erg_all.txt", skiprows = 1)
# Flatwrm_df = pd.DataFrame(
#     flatwrm_data,
#     columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])

# Flatwrm_Prot = []

# for KIC in short_KIC:
#     Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
#     Flatwrm_KIC = Flatwrm_df.loc[Flatwrm_df['kepler_id'] == KIC]
#     for i in range(np.size(Flatwrm_KIC.values[:,0])):
#         Flatwrm_Prot.append(Okamoto_KIC.values[0,5])

# Okamoto_data = [0,0,0,0,0,0]

# Okamoto_data[0] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(0,2)].values[:,0])))]
# Okamoto_data[1] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(2,5)].values[:,0])))]
# Okamoto_data[2] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(5,15)].values[:,0])))]
# Okamoto_data[3] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(15,25)].values[:,0])))]
# Okamoto_data[4] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(25,35)].values[:,0])))]
# Okamoto_data[5] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(35,100)].values[:,0])))]

# =============================================================================
# AFD
# =============================================================================
AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/final-flares_all.csv")

AFD_Prot = []

for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == int(KIC)]
    for i in range(np.size(AFD_KIC.values[:,0])):
        AFD_Prot.append(Okamoto_KIC.values[0,5])

Okamoto_data = [0,0,0,0,0,0]

Okamoto_data[0] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(0,2)].values[:,0])))]
Okamoto_data[1] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(2,5)].values[:,0])))]
Okamoto_data[2] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(5,15)].values[:,0])))]
Okamoto_data[3] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(15,25)].values[:,0])))]
Okamoto_data[4] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(25,35)].values[:,0])))]
Okamoto_data[5] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(35,100)].values[:,0])))]


Okamoto_wait_times = [0,0,0,0,0,0]
Okamoto_energies1 = [0,0,0,0,0,0]
Okamoto_energies2 = [0,0,0,0,0,0]
Okamoto_energies = [0,0,0,0,0,0]
Okamoto_flare_length = [0,0,0,0,0,0]

for i in range(len(Okamoto_data)):
    Okamoto_wait_times[i] , Okamoto_energies1[i], Okamoto_energies2[i], Okamoto_energies[i], Okamoto_flare_length[i] = wait_times_energies(Okamoto_data[i], short_KIC)

plot_filter = ["Prot: [0,2]", "Prot: [2,5]", "Prot: [5,15]", "Prot: [15,25]", "Prot: [25,35]", "Prot: >35"]

# =============================================================================
# statistical information
# =============================================================================
for i in range(len(Okamoto_wait_times)):
    # text = plot_filter[i] + ' wait time: mean = {:.2f} days, median = {:.2f} days, stdev = {:.4f}'.format(np.mean(Okamoto_wait_times[i]), np.median(Okamoto_wait_times[i]), np.std(Okamoto_wait_times[i]))
    text = plot_filter[i] + ' wait time: mean = {:.2f} days +- {:.2f} days, stdev = {:.2f} days and median = {:.2f} days'.format(np.mean(Okamoto_wait_times[i]), np.std(Okamoto_wait_times[i])/np.sqrt(np.size(Okamoto_wait_times[i])), np.std(Okamoto_wait_times[i]), np.median(Okamoto_wait_times[i]))
    print(text)

for i in range(len(Okamoto_wait_times)):
    text = plot_filter[i] + ' duration: mean = {:.2f} hours +- {:.3f} hours, stdev = {:.2f} hours and median = {:.2f} hours'.format(np.mean(Okamoto_flare_length[i])*24, 24*np.std(Okamoto_flare_length[i])/np.sqrt(np.size(Okamoto_flare_length[i])), 24*np.std(Okamoto_flare_length[i]), 24*np.median(Okamoto_flare_length[i]))
    print(text)


for i in range(len(Okamoto_energies)):
    # text = plot_filter[i] + ' energy: mean = {:.1E} ergs, median = {:.1E} ergs, stdev = {:.4E}'.format(np.mean(Okamoto_energies[i]), np.median(Okamoto_energies[i]), np.std(Okamoto_energies[i]))
    text = plot_filter[i] + ' energy: mean = {:.3E} +- {:.1E} ergs, stdev = {:.1E} ergs, median = {:.1E} ergs'.format(np.mean(Okamoto_energies[i]), np.std(Okamoto_energies[i])/np.sqrt(np.size(Okamoto_energies[i])), np.std(Okamoto_energies[i]),np.median(Okamoto_energies[i]))
    print(text)

for i in range(len(Okamoto_data)):
    subset =  Okamoto_data[i]
    pivot = subset.pivot_table(index = ['kepler_id'], aggfunc = 'size')
    pivot_array = np.array(pivot.to_list())
    # print("number of flares: ",int(np.sum(pivot_array)),", mean = ",np.mean(pivot_array),", std = ", np.std(pivot_array))
    print("#stars: {:.1f}, # flares: {:d}, mean = {:.2f} , err = {:.2f}, std = {:.2f}".format(np.size(pivot_array), int(np.sum(pivot_array)), np.mean(pivot_array), np.std(pivot_array)/np.size(pivot_array), np.std(pivot_array)))

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
        if len(data[index]) == 0:
            f1 = 0
        else:
            f1 = 100 * (filter1/len(data[index]))
        allf1.append(f1)
        label = plot_filter[index] + ': '+str(len(data[index]))+' gaps'+'\n'+r'F$_1$ = ' + str(round(f1,3)) + '%'
        plot_label.append(label)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))

    ax.hist(data, bins = bins_set, range = range_array, label = plot_label, stacked = True)
    xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
    ax.set_xlabel(xlabeltxt)
    ax.set_ylabel('Frequency')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("AFD: Histogram for wait times of flares")
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

# histogram parameter 1000
bins_set = int(1000/5)
range_array = (0, 1000)

# plot_histogram(Okamoto_wait_times, plot_filter, bins_set, range_array)

# histogram parameter 200
bins_set = int(1000/20)
range_array = (0, 200)

# plot_histogram(Okamoto_wait_times, plot_filter, bins_set, range_array)

# histogram parameter 50
bins_set = int(1000/20)
range_array = (0, 50)

# plot_histogram(Okamoto_wait_times, plot_filter, bins_set, range_array)

# =============================================================================
# new histogram
# =============================================================================
def plot_histogram2(data, plot_filter, bins_set, range_array):
    
    # compute f2: measurements are outside the selected range for plotting
    allf1 = []
    plot_label = []
    for index in range(len(data)):
        filter1 = 0
        for item in data[index]:
            if item < range_array[0] or item > 400:
                filter1 +=1
        if len(data[index]) == 0:
            f1 = 0
        else:
            f1 = 100 * (filter1/len(data[index]))
        allf1.append(f1)
        label = plot_filter[index] + ': '+str(len(data[index]))+' gaps'+r', F$_1$ = ' + str(round(f1,3)) + '%'
        plot_label.append(label)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))

    ax.hist(data, bins = bins_set, range = range_array, label = plot_label, stacked = True)
    xlabeltxt = "wait times [days]"
    ax.set_xlabel(xlabeltxt, fontsize = 14)
    ax.set_ylabel('Frequency', fontsize = 14)
    ax.set_xlim([0, 400])
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center right', fontsize = 12)
    ax.set_title("AFD: Histogram for wait times", fontsize = 14)
    ax.grid(True)
    plt.show()

def plot_chist2(data, plot_filter, bins_set, range_array):
    
    # compute f2: measurements are outside the selected range for plotting
    allf1 = []
    plot_label = []
    for index in range(len(data)):
        filter1 = 0
        for item in data[index]:
            if item < range_array[0] or item > 600:
                filter1 +=1
        if len(data[index]) == 0:
            f1 = 0
        else:
            f1 = 100 * (filter1/len(data[index]))
        allf1.append(f1)
        label = plot_filter[index] +'\n'+r'F$_1$ = ' + str(round(f1,3)) + '%'
        plot_label.append(label)
    
    temp = plt.hist(data, bins = bins_set, range = range_array, cumulative = -1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

    hist, bin_edges = temp[0], temp[1]
    bin_center = (bin_edges[:-1] + bin_edges[1:])/2
    for i in range(np.size(plot_label)):
        ax.plot(bin_center, hist[i], ".", label = plot_label[i])
    xlabeltxt = "wait times [days]"
    ax.set_xlabel(xlabeltxt, fontsize = 14)
    ax.set_ylabel('Frequency', fontsize = 14)
    ax.set_yscale('log')
    ax.set_xlim([0, 600])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
    ax.set_title("AFD: Cumulative Distribution for wait times", fontsize = 14)
    ax.grid(True)
    plt.show()

# histogram parameter 1000
bins_set = int(1000/5)
range_array = (0, 1000)

# plot_histogram2(Okamoto_wait_times, plot_filter, bins_set, range_array)

# plot_chist2(Okamoto_wait_times, plot_filter, bins_set, range_array)

# =============================================================================
# Energies
# =============================================================================

def plot_energies(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        labeltxt = plot_filter[i] + ': \n'+ str(len(data[i])) + ' gaps'
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = labeltxt)
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time to next flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("AFD: Energies and Wait times")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in range(len(data)):
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
        labeltxt = plot_filter[i] + ': \n'+ str(len(data[i])) + ' gaps'
        ax.scatter(data[i], energies[i],  s=1, c = colors[i], label = labeltxt)
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('wait time before flare [days]')
    ax.set_ylabel('flare energy [ergs]')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("AFD: Energies and Wait times")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in range(len(data)):
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


# plot_energies(Okamoto_wait_times, Okamoto_energies1, plot_filter)

# energies2
# plot_energies2(Okamoto_wait_times, Okamoto_energies2, plot_filter)

# =============================================================================
# flare length
# =============================================================================

def plot_flare_length(data, energies, plot_filter):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    
    # for i in range(len(data)):
    #     labeltxt = plot_filter[i] + ': \n'+ str(len(data[i])) + ' events'
    #     ax.scatter(data[i], energies[i],  s=14-2*i, label = labeltxt)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i in range(len(data)):
        labeltxt = plot_filter[i] + ': \n'+ str(len(data[i])) + ' events'
        ax.scatter(data[i], energies[i],  s=14-2*i, c = colors[i], label = labeltxt)
    for i in range(len(data)):
        ax.scatter(np.median(data[i]), np.median(energies[i]), s=30, c = colors[i], edgecolors = "black",  marker = "X")
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('flare duration [days]')
    ax.set_ylabel('flare energy [ergs]')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("AFD: Energies and Flare duration")
    ax.grid(True)
    plt.show()

    # energies for each filter individually
    # for index in range(len(data)):
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

# plot_flare_length(Okamoto_flare_length, Okamoto_energies, plot_filter)

# =============================================================================
# Prot plots
# =============================================================================
from sklearn.metrics import r2_score

def plot_prot(data, energies):
    
    # "polyfit"
    x = np.log10(data)
    y = np.log10(energies)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    
    xp = np.linspace(min(x), max(x), 1000)
    
    # r-squared
    print("deg 1: r2_score = ", r2_score(y, p1(x)))
    print("deg 3: r2_score = ", r2_score(y, p(x)))
    
    R2_adj_1 = 1- (((1-r2_score(y, p1(x)))*(np.size(x)-1))/(np.size(x)-1-1))
    R2_adj_3 = 1- (((1-r2_score(y, p(x)))*(np.size(x)-1))/(np.size(x)-3-1))
    print("deg 1: adj = ", R2_adj_1)
    print("deg 3: adj = ", R2_adj_3)
    label1 = "poly of deg 1, R$^2$ = {:.3f}".format(R2_adj_1)
    label3 = "poly of deg 1, R$^2$ = {:.3f}".format(R2_adj_3)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.scatter(data, energies,  s=2)
    ax.plot(10**xp, 10**(p1(xp)), 'm--', label = label1)
    ax.plot(10**xp, 10**(p(xp)), 'r-', label = label3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'P$_{rot}$ [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("AFD: Energies and rotational Period")
    ax.grid(True)
    plt.show()

energies = AFD_df.values[:,9]

plot_prot(AFD_Prot, energies.astype(float))