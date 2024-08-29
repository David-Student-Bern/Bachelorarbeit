# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:23:55 2024

@author: david
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Date of flare peak; BJD-2400000
Dates = np.loadtxt("aaList.txt", skiprows = 35, usecols = [11])

# Duration of flare
Dur = np.loadtxt("aaList.txt", skiprows = 35, usecols = [12])

# Kepler Input Catalog identifier
KIC = np.loadtxt("aaList.txt", dtype = str, skiprows = 35, usecols = [0])

# testing only first KIC (1028018)

test_KIC = KIC[0:8]
test_Dates = Dates[0:8] + 2400000
test_Dur = Dur[0:8]

flare_files = np.loadtxt("test_flares.txt", dtype = str, skiprows = 0, usecols = [0])

# plot files with flares and flares
for filename in flare_files:
    with fits.open(filename, mode="readonly") as hdulist:
        # Read in the "BJDREF" which is the time offset of the time array.
        bjdrefi = hdulist[1].header['BJDREFI'] 
        bjdreff = hdulist[1].header['BJDREFF']

        # Read in the columns of data.
        times = hdulist[1].data['time'] 
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    # Convert the time array to full BJD by adding the offset back in.
    bjds = times + bjdrefi + bjdreff 

    plt.figure(figsize=(9,4))

    # Plot the time, uncorrected and corrected fluxes.
    plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux') 
    plt.plot(bjds, pdcsap_fluxes, '-b', label='PDCSAP Flux') 
    for k in range(np.size(test_Dates)):
        if bjds[0]<=test_Dates[k]<=bjds[-1]:
            flare_bjd = []
            flare_sap = []
            flare_pdcsap = []
            for i in range(np.size(bjds)):
                if (test_Dates[k]-test_Dur[k])<=bjds[i]<=(test_Dates[k]+test_Dur[k]):
                    flare_bjd.append(bjds[i])
                    flare_sap.append(sap_fluxes[i])
                    flare_pdcsap.append(pdcsap_fluxes[i])
            plt.plot(flare_bjd, flare_sap, '-r', label='flare')
            plt.plot(flare_bjd, flare_pdcsap, '-r', label='flare')
    plt.title('Kepler Light Curve')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Flux (electrons/second)')
    plt.show()

    for k in range(np.size(test_Dates)):
        if bjds[0]<=test_Dates[k]<=bjds[-1]:
            plt.figure(figsize=(9,4))
            data_bjd = []
            data_sap = []
            data_pdcsap = []
            for i in range(np.size(bjds)):
                if (test_Dates[k]-10*test_Dur[k])<=bjds[i]<=(test_Dates[k]+10*test_Dur[k]):
                    data_bjd.append(bjds[i])
                    data_sap.append(sap_fluxes[i])
                    data_pdcsap.append(pdcsap_fluxes[i])
            plt.plot(data_bjd, data_sap, '-k', label='SAP Flux')
            plt.plot(data_bjd, data_pdcsap, '-b', label='PDCSAP Flux')
            
            flare_bjd = []
            flare_sap = []
            flare_pdcsap = []
            for i in range(np.size(bjds)):
                if (test_Dates[k]-test_Dur[k])<=bjds[i]<=(test_Dates[k]+test_Dur[k]):
                    flare_bjd.append(bjds[i])
                    flare_sap.append(sap_fluxes[i])
                    flare_pdcsap.append(pdcsap_fluxes[i])
            plt.plot(flare_bjd, flare_sap, '-r', label='flare')
            plt.plot(flare_bjd, flare_pdcsap, '-r', label='flare')
            plt.title('Kepler Light Curve - Flares')
            plt.legend()
            plt.xlabel('Time (days)')
            plt.ylabel('Flux (electrons/second)')
            plt.show()