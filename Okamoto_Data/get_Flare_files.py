# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:15:35 2024

@author: david
"""
from astropy.io import fits
# import matplotlib.pyplot as plt
import numpy as np
import glob


# Date of flare peak; BJD-2400000
Dates = np.loadtxt("aaList.txt", skiprows = 35, usecols = [11])

# Duration of flare
# Dur = np.loadtxt("aaList.txt", skiprows = 35, usecols = [12])

# Kepler Input Catalog identifier
# KIC = np.loadtxt("aaList.txt", dtype = str, skiprows = 35, usecols = [0])

# All .fits filenames
filenames = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*/*lc.fits")

# testing only first KIC (1028018)

# test_KIC = KIC[0:8]
test_Dates = Dates[0:8] + 2400000
# test_Dur = Dur[0:8]

flare_files = []

# create list with filenames that contain a flare
# for filename in filenames:
#     # print("\n", filename)
#     with fits.open(filename, mode="readonly") as hdulist:
#         # Read in the "BJDREF" which is the time offset of the time array.
#         bjdrefi = hdulist[1].header['BJDREFI']
#         bjdreff = hdulist[1].header['BJDREFF']
#         # print("bjdrefi: ", bjdrefi)
#         # print("bjdreff: ", bjdreff)

#         # Read in the columns of data.
#         times = hdulist[1].data['time']

#     # Convert the time array to full BJD by adding the offset back in.
#     bjds = times + bjdrefi + bjdreff 
#     # print("starttime: ", bjds[0])
#     # print("endtime: ", bjds[-1])
#     for D in test_Dates:
#         if bjds[0]<=D<=bjds[-1]:
#             if filename not in flare_files:
#                 flare_files.append(filename)

# # print flare files into .txt file
# with open('test_flares.txt', 'w') as file:
#     for filename in flare_files:
#         file.write(filename)
#         file.write("\n")