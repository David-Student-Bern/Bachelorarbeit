# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:09:42 2024

@author: david
"""

"""
Goal: find those stars that only Okamoto reported as flaring
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import gamma
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gamma

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

flatwrm_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_31-265/flatwrm_output_985_erg_all.txt", skiprows = 1)
Flatwrm_df = pd.DataFrame(
    flatwrm_data,
    columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])


Flatwrm_data = [0,0,0,0,0,0]

Flatwrm_data[0] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(0,2)].values[:,0])))]
Flatwrm_data[1] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(2,5)].values[:,0])))]
Flatwrm_data[2] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(5,15)].values[:,0])))]
Flatwrm_data[3] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(15,25)].values[:,0])))]
Flatwrm_data[4] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(25,35)].values[:,0])))]
Flatwrm_data[5] = Flatwrm_df[Flatwrm_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(35,100)].values[:,0])))]


AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/final-flares_all.csv")


AFD_data = [0,0,0,0,0,0]

AFD_data[0] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(0,2)].values[:,0])))]
AFD_data[1] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(2,5)].values[:,0])))]
AFD_data[2] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(5,15)].values[:,0])))]
AFD_data[3] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(15,25)].values[:,0])))]
AFD_data[4] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(25,35)].values[:,0])))]
AFD_data[5] = AFD_df[AFD_df['kepler_id'].isin(np.ndarray.tolist(np.unique(Okamoto_df[Okamoto_df['Prot'].between(35,100)].values[:,0])))]

# Flatwrm Okamoto
print("\n------------FLATW'RM------------\n")
for i in range(len(Okamoto_data)):
    print("subset ",i)
    Okamoto_subset =  Okamoto_data[i]
    # Okamoto_pivot = Okamoto_subset.pivot_table(index = ['kepler_id'], aggfunc = 'size')
    # Okamoto_pivot_array = np.array(Okamoto_pivot.to_list())
    
    Flatwrm_subset =  Flatwrm_data[i]
    # Flatwrm_pivot = Flatwrm_subset.pivot_table(index = ['kepler_id'], aggfunc = 'size')
    # Flatwrm_pivot_array = np.array(Flatwrm_pivot.to_list())
    
    # AFD_subset =  AFD_data[i]
    # AFD_pivot = AFD_subset.pivot_table(index = ['kepler_id'], aggfunc = 'size')
    # AFD_pivot_array = np.array(AFD_pivot.to_list())
    Okamoto_KIC = Okamoto_subset['kepler_id'].unique()
    for KIC in Okamoto_KIC:
        Flatwrm_KIC = Flatwrm_subset.loc[Flatwrm_subset['kepler_id'] == KIC]
        # AFD_KIC = AFD_subset.loc[AFD_subset['kepler_id'] == int(KIC)]
        if len(Flatwrm_KIC) == 0:
            print(int(KIC))
        # if len(AFD_KIC) == 0:
        #     print(int(KIC))

# Flatwrm Okamoto
print("\n------------AFD------------\n")
for i in range(len(Okamoto_data)-1):
    counter = 0
    print("subset ",i)
    Okamoto_subset =  Okamoto_data[i]
    Okamoto_pivot = Okamoto_subset.pivot_table(index = ['kepler_id'], aggfunc = 'size')
    
    AFD_subset =  AFD_data[i]
    
    Okamoto_KIC = Okamoto_subset['kepler_id'].unique()
    for KIC in Okamoto_KIC:
        # Flatwrm_KIC = Flatwrm_subset.loc[Flatwrm_subset['kepler_id'] == KIC]
        AFD_KIC = AFD_subset.loc[AFD_subset['kepler_id'] == int(KIC)]
        # if len(Flatwrm_KIC) == 0:
        #     print(int(KIC))
        if len(AFD_KIC) == 0:
            print(int(KIC))
            counter += 1
    print("---> total = ",counter)