# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 06:49:02 2024

@author: david
"""

import numpy as np
import pandas as pd

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")
short_KIC = short_KIC[:30]

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 36, usecols=[0, 11, 12, 13]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy'])

flatwrm_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_1/flatwrm_output_985_erg_0-30.txt", skiprows = 1)
flatwrm_df = pd.DataFrame(
    flatwrm_data,
    columns = ["kepler_id","t_start", "t_end", "t_max", "flux_max", "raw_integral", "energy", "fit_amp", "fit_fwhm", "fit_t_start", "fit_t_end", "fit_t_max", "fit_integral", "fit_stdev"])

AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/final-flares.csv")

AFDc_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/flares-candidates.csv")

all_O = 0
common_O_f = 0
common_O_A = 0
common_O_Ac = 0

common_f_A = 0
common_f_Ac = 0

common_O_f_A = 0
common_O_f_Ac = 0


for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    flatwrm_KIC = flatwrm_df.loc[flatwrm_df['kepler_id'] == KIC]
    AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == int(KIC)]
    AFDc_KIC = AFDc_df.loc[AFDc_df['kepler_id'] == int(KIC)]
    
    for O_fpt in Okamoto_KIC['flare_peak_time']:
        O_f = False
        O_A = False
        O_Ac = False
        all_O += 1
        #test if in flatwrm
        for i in range(np.size(flatwrm_KIC.values[:,0])):
            if (flatwrm_KIC.values[i,1] <= (O_fpt+2400000) <= flatwrm_KIC.values[i,2]):
                common_O_f += 1
                O_f = True
                break
        
        #test if in AFD
        for i in range(np.size(AFD_KIC.values[:,0])):
            if (AFD_KIC.values[i,17] <= O_fpt <= AFD_KIC.values[i,18]):
                common_O_A += 1
                O_A = True
                break
        
        #test if in AFD_candidates
        for i in range(np.size(AFDc_KIC.values[:,0])):
            if (AFDc_KIC.values[i,17] <= O_fpt <= AFDc_KIC.values[i,18]):
                common_O_Ac += 1
                O_Ac = True
                break
        if O_f and O_A:
            common_O_f_A += 1
            common_O_f_Ac += 1
        elif O_f and O_Ac:
            common_O_f_Ac += 1
    
    for A_fpt in AFD_KIC['flare_peak_BJD']:
        #test if in flatwrm
        for i in range(np.size(flatwrm_KIC.values[:,0])):
            if (flatwrm_KIC.values[i,1] <= (A_fpt+2400000) <= flatwrm_KIC.values[i,2]):
                common_f_A += 1
                break
    for Ac_fpt in AFDc_KIC['flare_peak_BJD']:
        #test if in flatwrm
        for i in range(np.size(flatwrm_KIC.values[:,0])):
            if (flatwrm_KIC.values[i,1] <= (Ac_fpt+2400000) <= flatwrm_KIC.values[i,2]):
                common_f_Ac += 1
                break

