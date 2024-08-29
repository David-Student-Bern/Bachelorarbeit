# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:09:22 2024

@author: david
"""
from astropy.io import fits
# from astropy.table import Table 
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# files that produce the following error:
# =============================================================================
#     ValueError: The optimizer is not designed to search for 
#     for periods larger than the data baseline. 
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler\kplr001028018_lc_Q111111111111111111\kplr001028018-2009131105131_llc.fits"
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler\kplr003217852_lc_Q111111111111111111\kplr003217852-2009131105131_llc.fits"

# =============================================================================
# files that contain many flare events
# =============================================================================
"""final parameters:
  period  = 3.0493
  degree  = 4
  fwhm    = 0.0000
  #flares = 26"""
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler\kplr002303352_lc_Q011111111111111111\kplr002303352-2009350155506_llc.fits "

"""final parameters:
  period  = 3.0325
  degree  = 7
  fwhm    = 0.0000
  #flares = 16"""
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler\kplr002303352_lc_Q011111111111111111\kplr002303352-2010355172524_llc.fits"

"""final parameters:
  period  = 3.0472
  degree  = 5
  fwhm    = 0.0000
  #flares = 23"""
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler\kplr002303352_lc_Q011111111111111111\kplr002303352-2013011073258_llc.fits"

# =============================================================================
# files that contain 1 flare and fit to -99
# =============================================================================
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr002012690_lc_Q011111111111111111\kplr002012690-2010265121752_llc.fits "

# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr002012690_lc_Q011111111111111111\kplr002012690-2013131215648_llc.fits "

# =============================================================================
# other
# =============================================================================
# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr002012690_lc_Q011111111111111111\kplr002012690-2013011073258_llc.fits"

# filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr004585486_lc_Q011111111111111111\kplr004585486-2009166043257_llc.fits"

filename = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr010537061_lc_Q011111111111111111/kplr010537061-2012277125453_llc.fits"

# os.chdir(file_path)

with fits.open(filename, mode="readonly") as hdulist:
    # Read in the "BJDREF" which is the time offset of the time array.
    bjdrefi = hdulist[1].header['BJDREFI'] 
    bjdreff = hdulist[1].header['BJDREFF']

    # Read in the columns of data.
    times = hdulist[1].data['time'] 
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
    
    KIC = hdulist[1].header['KEPLERID']

# Convert the time array to full BJD by adding the offset back in.
bjds = times + bjdrefi + bjdreff 

plt.figure(figsize=(9,4))

# Plot the time, uncorrected and corrected fluxes.
# plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux') 
plt.scatter(bjds, pdcsap_fluxes, s=2, c = "b", label='PDCSAP Flux') 

titletxt = 'Kepler Light Curve, KIC='+str(KIC)
plt.title(titletxt)
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Flux (electrons/second)')
plt.show()

# =============================================================================
# close ups for flare
# =============================================================================
"""

# add start and end time of the flare
t_start = 2455422.4930961
t_end = 2455422.5544

# flare
f_bjds = []
f_flux = []
for i in range(np.size(bjds)):
    if((t_start - 0.00*(t_end-t_start)) <= bjds[i] <=(t_end + 0.00*(t_end-t_start))):
        print(i)
        f_bjds.append(bjds[i])
        f_flux.append(pdcsap_fluxes[i])

c_bjds = []
c_flux = []
for i in range(np.size(bjds)):
    if((t_start - 10*(t_end-t_start)) < bjds[i] < (t_end + 10*(t_end-t_start))):
        c_bjds.append(bjds[i])
        c_flux.append(pdcsap_fluxes[i])

plt.figure(figsize=(9,4))

# Plot the time, uncorrected and corrected fluxes.
# plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux') 
plt.scatter(c_bjds, c_flux, s=2, c = "b", label='PDCSAP Flux') 
plt.scatter(f_bjds, f_flux, s=2, c = "r", label='flare') 

plt.title('Kepler Light Curve')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Flux (electrons/second)')
plt.show()

plt.figure(figsize=(9,4))
# Plot the time, uncorrected and corrected fluxes.
# plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux') 
plt.scatter(bjds, pdcsap_fluxes, s=2, c = "b", label='PDCSAP Flux')
plt.scatter(f_bjds, f_flux, s=2, c = "r", label='flare') 

plt.title('Kepler Light Curve')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Flux (electrons/second)')
plt.show()
"""