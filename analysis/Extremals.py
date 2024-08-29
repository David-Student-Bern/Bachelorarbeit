# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:27:27 2024

@author: david
"""

"""
Goal: 

    find the flares with the highest energies and the longest duration and create 
    a new list for them so that they can be analysed further
    
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

# adding flare duration
dur= np.ndarray.tolist(flatwrm_df.values[:,2]-flatwrm_df.values[:,1])
flatwrm_df['duration'] = dur

AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/final-flares_all.csv")

Number = 10

# =============================================================================
# Flatwrm extremals
# =============================================================================
fw = flatwrm_df.sort_values(by=['duration'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/Flatwrm_dur.csv')

fw = flatwrm_df.sort_values(by=['energy'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/Flatwrm_energy.csv')

# =============================================================================
# Okamoto extremals
# =============================================================================
fw = Okamoto_df.sort_values(by=['flare_duration'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/Okamoto_dur.csv')

fw = Okamoto_df.sort_values(by=['energy'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/Okamoto_energy.csv')

# =============================================================================
# Flatwrm extremals
# =============================================================================
fw = AFD_df.sort_values(by=['flare_duration'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/AFD_dur.csv')

fw = AFD_df.sort_values(by=['energy'], ascending = False)
fw_d = fw.head(Number)
fw_d.to_csv('Extremals/AFD_energy.csv')

