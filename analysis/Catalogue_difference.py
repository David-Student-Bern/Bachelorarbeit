# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:47:39 2024

@author: david
"""

"""
Goal:
    determine the difference between the effective temperatures and roational periods
    used with each survey
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from pathlib import Path

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0,1,3]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'Teff', 'Radius'])

AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/final-flares_all.csv")

# AFD candidates
# AFD_df = pd.read_csv("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/flares-candidates_all.csv")


Teff_Ok_FW = []
Teff_Ok_AFD = []
Teff_AFD_FW = []

Radius_Ok_FW = []
Radius_Ok_AFD = []
Radius_AFD_FW = []

for i in range(np.size(short_KIC)):
    text = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*"+str(int(short_KIC[i]))+"*/*llc.fits"
    flare_files = glob.glob(text)
    
    filename = flare_files[0]
    with fits.open(filename, mode="readonly") as hdulist:
        # Read in the "BJDREF" which is the time offset of the time array.
        
        FW_Teff = hdulist[0].header['TEFF'] #[K] Effective temperature
        FW_Radius = hdulist[0].header['RADIUS'] #[solar radii] stellar radius
    
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == short_KIC[i]]
    Okamoto_Teff = Okamoto_KIC.values[0,1]
    Okamoto_Radius = Okamoto_KIC.values[0,2]
    
    Teff_Ok_FW.append((Okamoto_Teff-FW_Teff))
    Radius_Ok_FW.append((Okamoto_Radius-FW_Radius))
    
    AFD_KIC = AFD_df.loc[AFD_df['kepler_id'] == short_KIC[i]]
    if np.size(AFD_KIC.values[:,1]) > 0:
        AFD_Teff = AFD_KIC.values[0,11]
        AFD_Radius = AFD_KIC.values[0,12]
        
        Teff_Ok_AFD.append((Okamoto_Teff-AFD_Teff))
        Radius_Ok_AFD.append((Okamoto_Radius-AFD_Radius))
        
        Teff_AFD_FW.append((FW_Teff-AFD_Teff))
        Radius_AFD_FW.append((FW_Radius-AFD_Radius))

# Statistics
print("------Difference Okamoto - FLATW'RM:------\n")
print("effectice Temperature:")
print("  Different {:d}/{:d}".format(np.size(Teff_Ok_FW) - sum(abs(x) <= 1e-10for x in Teff_Ok_FW), np.size(Teff_Ok_FW)))
print("  mean: {:.2f}".format(np.mean(Teff_Ok_FW)))
print("  std: {:.2f}".format(np.std(Teff_Ok_FW)))
print("  max: {:.2f}".format(np.max(Teff_Ok_FW)))
print("  min: {:.2f}\n".format(np.min(Teff_Ok_FW)))
print("stellar Radius in solar radii:")
print("  Different {:d}/{:d}".format(np.size(Radius_Ok_FW) - sum(abs(x) <= 1e-10for x in Radius_Ok_FW), np.size(Radius_Ok_FW)))
print("  mean: {:.2f}".format(np.mean(Radius_Ok_FW)))
print("  std: {:.2f}".format(np.std(Radius_Ok_FW)))
print("  max: {:.2f}".format(np.max(Radius_Ok_FW)))
print("  min: {:.2f}\n\n".format(np.min(Radius_Ok_FW)))

print("------Difference Okamoto - AFD:------\n")
print("effectice Temperature:")
print("  Different {:d}/{:d}".format(np.size(Teff_Ok_AFD) - sum(abs(x) <= 1e-10for x in Teff_Ok_AFD), np.size(Teff_Ok_AFD)))
print("  mean: {:.2f}".format(np.mean(Teff_Ok_AFD)))
print("  std: {:.2f}".format(np.std(Teff_Ok_AFD)))
print("  max: {:.2f}".format(np.max(Teff_Ok_AFD)))
print("  min: {:.2f}\n".format(np.min(Teff_Ok_AFD)))
print("stellar Radius in solar radii:")
print("  Different {:d}/{:d}".format(np.size(Radius_Ok_AFD) - sum(abs(x) <= 1e-10for x in Radius_Ok_AFD), np.size(Radius_Ok_AFD)))
print("  mean: {:.2f}".format(np.mean(Radius_Ok_AFD)))
print("  std: {:.2f}".format(np.std(Radius_Ok_AFD)))
print("  max: {:.2f}".format(np.max(Radius_Ok_AFD)))
print("  min: {:.2f}\n\n".format(np.min(Radius_Ok_AFD)))

print("------Difference FLATW'RM - AFD:------\n")
print("effectice Temperature:")
print("  Different {:d}/{:d}".format(np.size(Teff_AFD_FW) - sum(abs(x) <= 1e-10for x in Teff_AFD_FW), np.size(Teff_AFD_FW)))
print("  mean: {:.2f}".format(np.mean(Teff_AFD_FW)))
print("  std: {:.2f}".format(np.std(Teff_AFD_FW)))
print("  max: {:.2f}".format(np.max(Teff_AFD_FW)))
print("  min: {:.2f}\n".format(np.min(Teff_AFD_FW)))
print("stellar Radius in solar radii:")
print("  Different {:d}/{:d}".format(np.size(Radius_AFD_FW) - sum(abs(x) <= 1e-10for x in Radius_AFD_FW), np.size(Radius_AFD_FW)))
print("  mean: {:.2f}".format(np.mean(Radius_AFD_FW)))
print("  std: {:.2f}".format(np.std(Radius_AFD_FW)))
print("  max: {:.2f}".format(np.max(Radius_AFD_FW)))
print("  min: {:.2f}".format(np.min(Radius_AFD_FW)))