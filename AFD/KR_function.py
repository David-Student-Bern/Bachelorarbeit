# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:20:30 2024

@author: david
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# =============================================================================
# plot of the Kepler pixel Response function
# =============================================================================
KpRF = pd.read_csv('KpRF.txt')  # Kepler Instrument Response Function (high resolution)
l_nm = KpRF.values[:,0]
tr = KpRF.values[:,1]  # Transmission

plt.figure()
plt.plot(l_nm, tr, "k-")
plt.xlabel(r"Wavelenth $\lambda$ [nm]")
plt.ylabel("Response")
plt.title("Kepler Resonse Function")
plt.grid()
plt.show()

# =============================================================================
# plot of flare energies observed with Kepler compared to a black body
# =============================================================================
sigma = 5.6704 * 10 ** -5  # Stefan_Boltzmann constant(erg cm^-2 s^-1 k^-4)
Tflare = 9000  # temperature of the flare =9000K
h = 6.62606 * 10 ** -27  # plank's constant (cm^2 g s^-2)
c = 2.99792 * 10 ** 10  # speed of light cm (s^-1)
k = 1.38064 * 10 ** -16  # Boltzmann's constant (g s^-2 k^-1)
Teff = 5777

# Black Body
def Blambda(T, lamb):
    return 2*h*c**2/(lamb**5) * 1/ (np.exp(h*c/(lamb*k*T)) -1)

lam = np.linspace(1,2000,200) #wavelength in nm

# BB with Kepler response function
l = l_nm * 1e-7
n = len(l)

rb1 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the star)
rb2 = []
rb3 = []

for i in range(n):
    rb1.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * Teff))) - 1)))
    rb2.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * 8000))) - 1)))
    rb3.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * 10000))) - 1)))

plt.figure()
plt.plot(lam, Blambda(5777, lam*1e-7 ), "k-", label = "BB at T = 5777 K")
plt.plot(l*1e7, rb1, "k--", label = "Kepler at T = 5777 K")

plt.plot(lam, Blambda(8000, lam*1e-7 ), "r-", label = "BB at T = 8000 K")
plt.plot(l*1e7, rb2, "r--", label = "Kepler at T = 8000 K")

plt.plot(lam, Blambda(10000, lam*1e-7 ), "b-", label = "BB at T = 10000 K")
plt.plot(l*1e7, rb3, "b--", label = "Kepler at T = 10000 K")
plt.xlabel(r"Wavelenth $\lambda$ [nm]")
plt.ylabel(r"Spectral Radiance [erg/cm$^2$/s/sr/A]")
plt.grid()
plt.legend()
plt.show()

# =============================================================================
# Flare luminosity at different temperatures and different flaring areas
# =============================================================================
Ftemp = np.linspace(6000,10000, 600)
Rsol = 6.96342 * 1e10 #cm
Tsol = 5777

rb1 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the star)

for i in range(n-1):
    rb1.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * Tsol))) - 1)))

intsol = np.trapz(rb1, l[:-1])

intflare = []
for j in range(np.size(Ftemp)):
    rb2 = []
    for i in range(n-1):
        rb2.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * Ftemp[j]))) - 1)))
    
    intflare.append(np.trapz(rb2, l[:-1]))
# intflare = np.trapz(rb2, l[:-1])

F1 = sigma * Ftemp**4 * 0.05 * np.pi*Rsol**2 * intsol /(intflare - intsol)
F2 = sigma * Ftemp**4 * 0.1 * np.pi*Rsol**2 * intsol /(intflare - intsol)
F3 = sigma * Ftemp**4 * 0.2 * np.pi*Rsol**2 * intsol /(intflare - intsol)
# F1 = sigma * Ftemp**4 * 0.05 * np.pi*Rsol**2 * intsol /(intsol - intflare)
# F2 = sigma * Ftemp**4 * 0.1 * np.pi*Rsol**2 * intsol /(intsol - intflare)
# F3 = sigma * Ftemp**4 * 0.2 * np.pi*Rsol**2 * intsol /(intsol - intflare)
# F4 = sigma * Ftemp**4 * 0.4 * np.pi*Rsol**2 * intsol /(intflare - intsol)

plt.figure()
plt.plot(Ftemp, F1, "b-.", label = r"$\Delta F/ F = 0.05$")
plt.plot(Ftemp, F2, "k--", label = r"$\Delta F/ F = 0.1$")
plt.plot(Ftemp, F3, "r-.", label = r"$\Delta F/ F = 0.2$")
# plt.plot(Ftemp, F4, "g--", label = r"$\Delta F/ F = 0.4$")
plt.xlabel(r"Temperature [K]")
plt.ylabel(r"Luminosity [erg/s]")
plt.grid()
plt.legend()
plt.show()


# =============================================================================
# Difference between Althukair and correct
# =============================================================================
plt.figure()
plt.plot(Ftemp, (intflare - intsol)/intflare , "k-")
plt.xlabel(r"Temperature [K]")
plt.ylabel(r"$L_{flare,old} / L_{flare,new}$")
plt.grid()
# plt.legend()
plt.show()

# =============================================================================
# Lucias formula
# =============================================================================
plt.figure()
plt.plot(Ftemp, intflare /(intflare - intsol), "k-")
# plt.plot(Ftemp, F4, "g--", label = r"$\Delta F/ F = 0.4$")
plt.xlabel(r"Temperature [K]")
plt.ylabel(r"$L_{flare,old} / L_{flare,new}$")
plt.grid()
# plt.legend()
plt.show()

# =============================================================================
# Difference to Lucias formula
# =============================================================================
anna = [a*b for a,b in zip(intflare,intflare)]
ben = [a*b for a,b in zip((intflare - intsol),(intflare - intsol))]
carly = [a/b for a,b in zip(anna,ben)]
plt.figure()
plt.plot(Ftemp, carly, "k-")
# plt.plot(Ftemp, F4, "g--", label = r"$\Delta F/ F = 0.4$")
plt.xlabel(r"Temperature [K]")
plt.ylabel(r"$L_{flare,Lucia} / L_{flare,new}$")
plt.grid()
# plt.legend()
plt.show()