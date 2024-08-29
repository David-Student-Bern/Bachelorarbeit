# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:59:43 2024

@author: david
"""

"""
Goal: 

    plot the flares that were filtered by Extremals.py and display
    flare energy and flare duration

"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from pathlib import Path

# =============================================================================
# Okamoto
# =============================================================================

Okamoto_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Okamoto_dur.csv")
# Okamoto_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Okamoto_energy.csv")


Okamoto_KIC = Okamoto_df.values[:,1]
Okamoto_KIC = np.array(Okamoto_KIC)
Okamoto_KIC = Okamoto_KIC.astype(int)

for i in Okamoto_KIC:
    text = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*"+str(i)+"*/*llc.fits"
    flare_files = glob.glob(text)
    
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
        Peak_times = Okamoto_KIC.values[:,2] + 2400000
        Flare_dur = Okamoto_KIC.values[:,3]
        Flare_energy = Okamoto_KIC.values[:,4]

        
        plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Okamoto_dur'
        # plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Okamoto_energy'
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
                titletxt = str(KIC) + ": Kepler Light Curve - Okamoto Flares\n energy = {:.2E} erg, duration = {:.1f}h".format(Flare_energy[k], 24*Flare_dur[k])
                plt.title(titletxt)
                plt.legend()
                plt.xlabel('Time (days)')
                plt.ylabel('Flux (electrons/second)')
                titletxt = "Extremals/Okamoto_dur"+"/Okamoto_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                # titletxt = "Extremals/Okamoto_energy"+"/Okamoto_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                plt.savefig(titletxt)
                # plt.show()
                plt.clf()
        plt.close('all')


# =============================================================================
# Flatwrm
# =============================================================================
"""

# Flatwrm_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Flatwrm_dur.csv")
Flatwrm_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Flatwrm_energy.csv")


a = Flatwrm_df.values[:,1]
a = np.array(a)
a = a.astype(int)
a = set(a)
a = np.array(list(a))

# plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Flatwrm_dur'
plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/Flatwrm_energy'
Path(plots_path).mkdir(parents=True, exist_ok=True)

for i in a:
    text = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*"+str(i)+"*/*llc.fits"
    flare_files = glob.glob(text)
    
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
        
        Flatwrm_KIC = Flatwrm_df.loc[Flatwrm_df['kepler_id'] == KIC]
        Peak_times = Flatwrm_KIC.values[:,4]
        Start_times = Flatwrm_KIC.values[:,2]
        End_times = Flatwrm_KIC.values[:,3]
        Flare_dur = Flatwrm_KIC.values[:,15]
        Flare_energy = Flatwrm_KIC.values[:,7]

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
                    if (Start_times[k])<=bjds[i]<=(End_times[k]):
                        flare_bjd.append(bjds[i])
                        flare_pdcsap.append(pdcsap_fluxes[i])
                plt.plot(flare_bjd, flare_pdcsap, '-r', label='flare')
                titletxt = str(KIC) + ": Kepler Light Curve - FLATW'RM Flares\n energy = {:.2E} erg, duration = {:.1f}h".format(Flare_energy[k], 24*Flare_dur[k])
                plt.title(titletxt)
                plt.legend()
                plt.xlabel('Time (days)')
                plt.ylabel('Flux (electrons/second)')
                # titletxt = "Extremals/Flatwrm_dur"+"/Flatwrm_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                titletxt = "Extremals/Flatwrm_energy"+"/Flatwrm_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                plt.savefig(titletxt)
                # plt.show()
                plt.clf()
        plt.close('all')

"""

# =============================================================================
# AFD
# =============================================================================

"""

AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/AFD_dur.csv")
# AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/AFD_energy.csv")


a = AFD_df.values[:,2]
a = np.array(a)
a = a.astype(int)
a = set(a)
a = np.array(list(a))

plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/AFD_dur'
# plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/analysis/Extremals/AFD_energy'
Path(plots_path).mkdir(parents=True, exist_ok=True)

for i in a:
    text = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*"+str(i)+"*/*llc.fits"
    flare_files = glob.glob(text)
    
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
        
        AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == KIC]
        Peak_times = AFD_KIC.values[:,20] + 2400000
        Start_times = AFD_KIC.values[:,18] + 2400000
        End_times = AFD_KIC.values[:,19] + 2400000
        Flare_dur = AFD_KIC.values[:,17]
        Flare_energy = AFD_KIC.values[:,10]

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
                    if (Start_times[k])<=bjds[i]<=(End_times[k]):
                        flare_bjd.append(bjds[i])
                        flare_pdcsap.append(pdcsap_fluxes[i])
                plt.plot(flare_bjd, flare_pdcsap, '-r', label='flare')
                titletxt = str(KIC) + ": Kepler Light Curve - AFD Flares\n energy = {:.2E} erg, duration = {:.1f}h".format(Flare_energy[k], 24*Flare_dur[k])
                plt.title(titletxt)
                plt.legend()
                plt.xlabel('Time (days)')
                plt.ylabel('Flux (electrons/second)')
                titletxt = "Extremals/AFD_dur"+"/AFD_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                # titletxt = "Extremals/AFD_energy"+"/AFD_"+str(KIC)+"_"+str(Peak_times[k])+".png"
                plt.savefig(titletxt)
                # plt.show()
                plt.clf()
        plt.close('all')


"""