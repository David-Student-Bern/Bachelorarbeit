# -*- coding: utf-8 -*-
"""
Created on Tue May  7 07:11:46 2024

@author: david
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import gamma
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gamma

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

# =============================================================================
# Okamoto
# =============================================================================
Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13, 1, 6]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy', 'Teff', 'Prot'])

Okamoto_data = [0,0,0,0,0,0]

Okamoto_data[0] = Okamoto_df[Okamoto_df['Prot'].between(0,2)]
Okamoto_data[1] = Okamoto_df[Okamoto_df['Prot'].between(2,5)]
Okamoto_data[2] = Okamoto_df[Okamoto_df['Prot'].between(5,15)]
Okamoto_data[3] = Okamoto_df[Okamoto_df['Prot'].between(15,25)]
Okamoto_data[4] = Okamoto_df[Okamoto_df['Prot'].between(25,35)]
Okamoto_data[5] = Okamoto_df[Okamoto_df['Prot'].between(35,100)]

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
# for i in range(len(Okamoto_wait_times)):
#     text = plot_filter[i] + ' wait time: mean = {:.2f} days, median = {:.2f} days'.format(np.mean(Okamoto_wait_times[i]), np.median(Okamoto_wait_times[i]))
#     print(text)

# for i in range(len(Okamoto_energies)):
#     text = plot_filter[i] + ' energy: mean = {:.1E} ergs, median = {:.1E} ergs'.format(np.mean(Okamoto_energies[i]), np.median(Okamoto_energies[i]))
#     print(text)


def plot_dist(data, plot_filter, bins_set, range_array):
    
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
    
    for i in range(len(data)-3):
        counts, bins = np.histogram(data[i], bins = range_array[1], range = range_array, density = True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        def exp_dist(k, lamb):
            '''poisson function, parameter lamb is the fit parameter'''
            # return poisson.pmf(k, lamb)
            return lamb * np.exp(-lamb*k)
            # return lamb**k * np.exp(-lamb*k)/factorial(k)
        
        def rayleigh(k, lamb):
            return k/lamb**2 *np.exp(-k**2/(2*lamb**2))
        
        def chi(x, k):
            return x**(k-1) * np.exp(-x**2/2)/(2**(k/2 -1) * gamma(k/2))
        
        def burr(x, c, k, lamb):
            return c*k/lamb * (x/lamb)**(c-1)/((1+(x/lamb)**c)**(k+1))
    
    
        # fit with curve_fit
        exp_parameters, cov_matrix = curve_fit(exp_dist, bin_centers, counts)
        # ray_parameters, cov_matrix = curve_fit(rayleigh, bin_centers, counts)
        # chi_parameters, cov_matrix = curve_fit(chi, bin_centers, counts)
        burr_parameters, cov_matrix = curve_fit(burr, bin_centers, counts)
        x_plot = np.arange(0, range_array[1])
        
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.hist(data[i], bins = range_array[1], density = True, range = range_array, label = plot_label[i], stacked = True)
        ax.plot(x_plot, exp_dist(x_plot, *exp_parameters), '.', label='exponential')
        # ax.plot(x_plot, rayleigh(x_plot, *ray_parameters), '.', label='rayleigh')
        # ax.plot(x_plot, chi(x_plot, *chi_parameters), '.', label='chi')
        ax.plot(x_plot, burr(x_plot, *burr_parameters), '.', label='burr')
        xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
        ax.set_xlabel(xlabeltxt)
        ax.set_ylabel('Frequency')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title("Okamoto: Probability mass function")
        ax.grid(True)
        plt.show()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        ax.hist(data[i], bins = range_array[1], density = True, range = range_array, label = plot_label[i], cumulative = True)
        ax.plot(x_plot, np.cumsum(exp_dist(x_plot, *exp_parameters)), '.', label='Fit result')
        # ax.plot(x_plot, np.cumsum(rayleigh(x_plot, *ray_parameters)), '.', label='rayleigh')
        # ax.plot(x_plot, np.cumsum(chi(x_plot, *chi_parameters)), '.', label='chi')
        ax.plot(x_plot, np.cumsum(burr(x_plot, *burr_parameters)), '.', label='burr')
        xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
        ax.set_xlabel(xlabeltxt)
        ax.set_ylabel('Frequency')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title("Okamoto: Cummulative distribution function")
        ax.grid(True)
        plt.show()

# histogram parameter 1000
bins_set = int(1000/5)
range_array = (0, 1000)

# plot_dist(Okamoto_wait_times, plot_filter, bins_set, range_array)

# histogram parameter 200
bins_set = int(1000/20)
range_array = (0, 200)

# plot_dist(Okamoto_wait_times, plot_filter, bins_set, range_array)

# histogram parameter 50
bins_set = int(1000/20)
range_array = (0, 50)

# plot_dist(Okamoto_wait_times, plot_filter, bins_set, range_array)

# =============================================================================
# Energies
# =============================================================================

def energy_dist(energies):
    
    weights = np.ones_like(energies)/len(energies)
   
    counts, bins = np.histogram(energies, bins = 100, range = (np.min(energies), np.max(energies)), weights=weights)
    counts = counts/np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    def norm_dist(x, u, sigm):
        return np.exp(-0.5*((x-u)/sigm)**2)/(sigm*np.sqrt(2*np.pi))
    def exp_dist(k, lamb):
        return lamb * np.exp(-lamb*k)
    
    def inv_gauss(x, u, lamb):
        return np.sqrt(lamb/(2*np.pi*x**3)) * np.exp(- lamb*(x-u)**2/(2*u**2 * x))
    
    # def rayleigh(k, lamb):
    #     return k/lamb**2 *np.exp(-k**2/(2*lamb**2))
    
    # def chi(x, k):
    #     return x**(k-1) * np.exp(-x**2/2)/(2**(k/2 -1) * gamma(k/2))
    
    # def burr(x, c, k, lamb):
    #     return c*k/lamb * (x/lamb)**(c-1)/((1+(x/lamb)**c)**(k+1))
    
    
    # fit with curve_fit
    # norm_parameters, cov_matrix = curve_fit(norm_dist, bin_centers, counts)
    # exp_parameters, cov_matrix = curve_fit(exp_dist, bin_centers, counts)
    igauss_parameters, cov_matrix = curve_fit(inv_gauss, bin_centers, counts)
    # ray_parameters, cov_matrix = curve_fit(rayleigh, bin_centers, counts)
    # chi_parameters, cov_matrix = curve_fit(chi, bin_centers, counts)
    # burr_parameters, cov_matrix = curve_fit(burr, bin_centers, counts)
    x_plot = np.linspace(np.min(energies), np.max(energies),1000)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.hist(energies, bins = 100, range = (np.min(energies), np.max(energies)), label = "flare energies [erg]", weights=weights)
    ax.plot(bin_centers, counts, 'r.')
    # ax.plot(x_plot, norm_dist(x_plot, *norm_parameters), '.', label='normal')
    # ax.plot(x_plot, exp_dist(x_plot, *exp_parameters), '.', label='exponential')
    ax.plot(x_plot, inv_gauss(x_plot, *igauss_parameters), '.', label='inverse gauss')
    # ax.plot(x_plot, rayleigh(x_plot, *ray_parameters), '.', label='rayleigh')
    # ax.plot(x_plot, chi(x_plot, *chi_parameters), '.', label='chi')
    # ax.plot(x_plot, burr(x_plot, *burr_parameters), '.', label='burr')
    ax.set_xlabel('flare energy [ergs]')
    ax.set_ylabel('Frequency')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Okamoto: Probability mass function")
    ax.grid(True)
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.hist(energies, bins = 100, range = (np.min(energies), np.max(energies)), label = "flare energies [erg]", cumulative = True, weights=weights)
    # ax.plot(x_plot, np.cumsum(norm_dist(x_plot, *norm_parameters)), '.', label='normal')
    # ax.plot(x_plot, np.cumsum(exp_dist(x_plot, *exp_parameters)), '.', label='exponential')
    ax.plot(x_plot, np.cumsum(inv_gauss(x_plot, *igauss_parameters)), '.', label='inverse gauss')
    # ax.plot(x_plot, np.cumsum(rayleigh(x_plot, *ray_parameters)), '.', label='rayleigh')
    # ax.plot(x_plot, np.cumsum(chi(x_plot, *chi_parameters)), '.', label='chi')
    # ax.plot(x_plot, np.cumsum(burr(x_plot, *burr_parameters)), '.', label='burr')
    ax.set_xlabel('flare energy [ergs]')
    ax.set_ylabel('Frequency')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Okamoto: Cummulative distribution function")
    ax.grid(True)
    plt.show()

energy_dist(Okamoto_df.values[:,3])
# energy_dist(np.log10(Okamoto_df.values[:,3]))

# =============================================================================
# Prot plots
# =============================================================================

def plot_prot(data, energies):
    
    # "polyfit"
    x = np.log10(data)
    y = np.log10(energies)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    
    xp = np.linspace(min(x), max(x), 1000)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    ax.scatter(data, energies,  s=2)
    ax.plot(10**xp, 10**(p1(xp)), 'm--', label = "poly of deg 1")
    ax.plot(10**xp, 10**(p(xp)), 'r-', label = "poly of deg 3")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'P$_{rot}$ [days]')
    ax.set_ylabel('flare energy [ergs]')
    ax.legend()
    ax.set_title("Okamoto: Energies and rotational Period")
    ax.grid(True)
    plt.show()

# plot_prot(Okamoto_df.values[:,5], Okamoto_df.values[:,3])