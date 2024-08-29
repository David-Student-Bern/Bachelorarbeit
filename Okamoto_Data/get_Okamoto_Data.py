# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:34:40 2024

@author: david
"""
import numpy as np

import time as tt

# from astroquery.mast import Mast
from astroquery.mast import Observations


# =============================================================================
# create a list with all KIC for all Okamoto flares
# =============================================================================


# List = np.loadtxt("aaList.txt", skiprows = 35, usecols = [0,11,12])

# KIC = np.loadtxt("aaList.txt", dtype = str, skiprows = 35, usecols = [0]) # Kepler Input Catalog identifier
# # KIC = List[:,0]
# Date = List[:,1] # Date of flare peak; BJD-2400000
# Dur = List[:,2] # Duration of flare

# jd = Date[0] + 2400000.5

# # get date
# tm = Time(str(jd), format='jd')
# print(tm.jd)
# print(tm.to_datetime())
# print(tm.iso)

# short_KIC = []
# for tn in KIC:
#     if tn not in short_KIC:
#         short_KIC.append(tn)

# # save all KIC in one file
# with open("KIC_list.txt", 'w') as file:
#     for i in range(np.size(short_KIC)):
#         file.write(str(short_KIC[i]) + "\n")

# =============================================================================
# download all long cadence light curve (*.llc) files for the flares in
# Okamoto's catalogue
# =============================================================================


short_KIC = np.loadtxt("KIC_list.txt", dtype = str)

starti = 200
endi = 300

start_time = tt.time()
for tn in short_KIC[starti:endi]:
    if len(tn)==7:
        keplerObs = Observations.query_criteria(target_name='kplr00'+str(tn), obs_collection='Kepler')
        keplerProds = Observations.get_product_list(keplerObs)
        yourProd = Observations.filter_products(keplerProds, extension='lc.fits',mrp_only=False)

        Observations.download_products(yourProd, mrp_only = False, cache = False)
    elif len(tn)==8:
        keplerObs = Observations.query_criteria(target_name='kplr0'+str(tn), obs_collection='Kepler')
        keplerProds = Observations.get_product_list(keplerObs)
        yourProd = Observations.filter_products(keplerProds, extension='lc.fits',mrp_only=False)

        Observations.download_products(yourProd, mrp_only = False, cache = False)
    else:
        print("error, size =", len(tn))
end_time = tt.time()
print("download time for ", np.size(short_KIC[starti:endi]), " objects: ", end_time-start_time)
print("estimated duration for all: ",  np.size(short_KIC)/np.size(short_KIC[starti:endi])* (end_time-start_time))