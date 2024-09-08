# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:23:55 2024

@author: david
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from pathlib import Path

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13, 1, 6]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy', 'Teff', 'Prot'])

flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*/*llc.fits")

# figure 2 in thesis
# flare_files = flare_files[1612:1630]
# flare_files = flare_files[2311:2328]
# flare_files = flare_files[2124:2141]
# flare_files = flare_files[1167:1185]

# plot files with flares and flares
for filename in flare_files:
    with fits.open(filename, mode="readonly") as hdulist:
        # Read in the "BJDREF" which is the time offset of the time array.
        bjdrefi = hdulist[1].header['BJDREFI'] 
        bjdreff = hdulist[1].header['BJDREFF']

        # Read in the columns of data.
        times = hdulist[1].data['time'] 
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

        KIC = hdulist[1].header['KEPLERID']

    # Convert the time array to full BJD by adding the offset back in.
    bjds = times + bjdrefi + bjdreff 

    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    if np.size(Okamoto_KIC.values[:,1]) == 0:
        continue
    Peak_times = Okamoto_KIC.values[:,1] + 2400000
    Flare_dur = Okamoto_KIC.values[:,2]

    
    plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/Okamoto_flares_thesis/'+str(KIC)
    Path(plots_path).mkdir(parents=True, exist_ok=True)

    for k in range(np.size(Peak_times)):
        if bjds[0]<=Peak_times[k]<=bjds[-1]:
            plt.figure(figsize=(9,4))
            data_bjd = []
            data_pdcsap = []
            for i in range(np.size(bjds)):
                if (Peak_times[k]-5*Flare_dur[k])<=bjds[i]<=(Peak_times[k]+5*Flare_dur[k]):
                    data_bjd.append(bjds[i])
                    data_pdcsap.append(pdcsap_fluxes[i])
            plt.plot(data_bjd, data_pdcsap, '.b', label='PDCSAP Flux')
            
            flare_bjd = []
            flare_pdcsap = []
            for i in range(np.size(bjds)):
                if (Peak_times[k]-0.25*Flare_dur[k])<=bjds[i]<=(Peak_times[k]+0.75*Flare_dur[k]):
                    flare_bjd.append(bjds[i])
                    flare_pdcsap.append(pdcsap_fluxes[i])
            plt.plot(flare_bjd, flare_pdcsap, '-r', label='flare')
            titletxt = str(KIC) + ': Kepler Light Curve - Flares'
            plt.title(titletxt)
            plt.legend()
            plt.xlabel('Time (days)')
            plt.ylabel('Flux (electrons/second)')
            titletxt = "Okamoto_flares_thesis/"+str(KIC)+"/Okamoto_"+str(KIC)+"_"+str(Peak_times[k])+".png"
            plt.savefig(titletxt)
            # plt.show()
            plt.clf()
    plt.close('all')
    