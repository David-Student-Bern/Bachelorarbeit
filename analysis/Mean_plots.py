# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:43:23 2024

@author: david
"""

"""
    plotting mean values for wait time, duration and flare energy in different ranges
    of rotational period
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def exp_func(x, m, t):
    return m * np.exp(- t * x)

def quad_func(x, a, b):
    return a * x**2 + b

def log_func(x, a, b):
    return a * 10**x + b

prot_edge = [0,2,5,15,25,35]
prot_width = [2,3,10,10,10,10]

prot_center = [1., 3.5, 10., 20., 30., 40.]
x_err = [1, 1.5, 5, 5, 5, 5]
x_err2 = [[0.8, 1.3, 4.8, 4.8, 4.8, 4.8], [1.2, 1.7, 5.2, 5.2, 5.2, 5.2]]
x_err3 = [[1.2, 1.7, 5.2, 5.2, 5.2, 5.2], [0.8, 1.3, 4.8, 4.8, 4.8, 4.8]]

offset = 0.2*np.ones_like(prot_center)

# =============================================================================
# Flare energy
# =============================================================================

Ok_energy = np.array([11.99, 6.34, 2.14, 1.61, 1.43, 0.65]) * 1e34
Ok_energy_err = np.array([4.9, 2.5, 1, 1, 1.6, 1.5]) * 1e33

FW_energy = np.array([11.2, 4.32, 4.5, 2.02, 0.848, 0.92]) * 1e34
FW_energy_err = np.array([14, 2.5, 220, 8.5, 8.3, 1.7]) * 1e33

AFD_energy = np.array([28.8, 9.07, 4.52, 3.04, 3.06, 1.89]) * 1e34
AFD_energy_err = np.array([22, 3.6, 1.8, 3.2, 3.3, 3.2]) * 1e33

plt.figure()
# plt.bar(prot_edge, Ok_energy, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center, Ok_energy, yerr = Ok_energy_err, xerr = x_err, fmt='.', color = 'tab:blue', capsize = 5, label = "Okamoto")

# plt.bar(prot_edge, FW_energy, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center-offset, FW_energy, yerr = FW_energy_err, xerr = x_err2, fmt='.', color = 'tab:orange', capsize = 5, label = "FLATW'RM")

# plt.bar(prot_edge, AFD_energy, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center+offset, AFD_energy, yerr = AFD_energy_err, xerr = x_err3, fmt='.', color = 'tab:green', capsize = 5, label = "AFD")
plt.grid()
plt.xlabel(r"$P_{rot}$ [days]")
plt.ylabel("Energy [ergs]")
plt.legend()
plt.yscale("log")
plt.ylim((3*1e33, 4*1e35))
plt.xlim((-2,44))
plt.title("Mean Energies for different rotational Periods")
plt.show()

# =============================================================================
# Flare duration
# =============================================================================
Ok_dur = np.array([3.58, 3.16, 2.61, 2.22, 2.05, 1.44])
Ok_dur_err = np.array([6, 4, 5, 9, 13, 0]) * 0.01

FW_dur = np.array([1.75, 1.76, 1.74, 1.75, 1.9, 2.06])
FW_dur_err = np.array([1, 1, 2, 2, 3, 7]) * 0.01

AFD_dur = np.array([2.46, 1.99, 1.9, 1.79, 1.89, 1.47])
AFD_dur_err = np.array([5, 2, 3, 6, 9, 0]) * 0.01

plt.figure()
# plt.bar(prot_edge, Ok_dur, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center, Ok_dur, yerr = Ok_dur_err, xerr = x_err, fmt='.', color = 'tab:blue', capsize = 5, label = "Okamoto")

# plt.bar(prot_edge, FW_dur, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center-offset, FW_dur, yerr = FW_dur_err, xerr = x_err2, fmt='.', color = 'tab:orange', capsize = 5, label = "FLATW'RM")

# plt.bar(prot_edge, AFD_dur, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center+offset, AFD_dur, yerr = AFD_dur_err, xerr = x_err3, fmt='.', color = 'tab:green', capsize = 5, label = "AFD")

plt.xlabel(r"$P_{rot}$ [days]")
plt.ylabel("Flare Duration [h]")
plt.legend()
plt.grid()
plt.ylim((1, 4))
plt.xlim((-2,44))
plt.title("Mean Flare Duration for different rotational Periods")
plt.show()

# =============================================================================
# Wait Time
# =============================================================================
Ok_WT = np.array([42.4, 48.1, 89.7, 211., 329.])
Ok_WT_err = np.array([2.7, 3.2, 6.3, 24, 57])

FW_WT = np.array([20.64, 22.62, 57.4, 68.3, 57.5, 80])
FW_WT_err = np.array([0.88, 0.94, 2.1, 3.5, 4.8, 21])

AFD_WT = np.array([59.5, 61.5, 108.9, 214, 329])
AFD_WT_err = np.array([4.5, 4.9, 8.4, 23, 56])

plt.figure()
# plt.bar(prot_edge[:-1], Ok_WT, align = 'edge', width = prot_width[:-1], color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center[:-1], Ok_WT, yerr = Ok_WT_err, xerr = x_err[:-1], fmt='.', color = 'tab:blue', capsize = 5, label = "Okamoto")

# plt.bar(prot_edge, FW_WT, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center-offset, FW_WT, yerr = FW_WT_err, xerr = x_err, fmt='.', color = 'tab:orange', capsize = 5, label = "FLATW'RM")

# plt.bar(prot_edge[:-1], AFD_WT, align = 'edge', width = prot_width[:-1], color = 'none', edgecolor = 'gray')
plt.errorbar(prot_center[:-1] +offset[:-1], AFD_WT, yerr = AFD_WT_err, xerr = [[1.2, 1.7, 5.2, 5.2, 5.2], [0.8, 1.3, 4.8, 4.8, 4.8]], fmt='.', color = 'tab:green', capsize = 5, label = "AFD")
plt.grid()
plt.xlabel(r"$P_{rot}$ [days]")
plt.ylabel("Flare Wait Time [days]")
plt.legend()
plt.yscale("log")
plt.xlim((-2,44))
plt.title("Mean Wait Time for different rotational Periods")
plt.show()

# =============================================================================
# flares per star
# =============================================================================
Ok_fps = np.array([29.78, 21.35, 5.66, 2.23, 1.65, 1.00])
Ok_fps_err = np.array([1.01, 0.61, 0.10, 0.04, 0.05, 0.00])

FW_fps = np.array([66.73, 59.45, 19.32, 18.58, 21.58, 16.67])
FW_fps_err = np.array([1.96, 1.41, 0.25, 0.17, 0.48, 4.37])

AFD_fps = np.array([20.82, 16.46, 5.62, 2.94, 2.09, 1])
AFD_fps_err = np.array([0.77, 0.53, 0.11, 0.07, 0.07, 0.00])

plt.figure()
# plt.bar(prot_edge, Ok_fps, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
# plt.scatter(prot_center, Ok_fps, marker='o', color = 'tab:blue', label = "Okamoto")
plt.errorbar(prot_center, Ok_fps, yerr = Ok_fps_err, xerr = x_err, fmt='.', color = 'tab:blue', capsize = 5, label = "Okamoto")

# plt.bar(prot_edge, FW_fps, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
# plt.scatter(prot_center, FW_fps, marker='s', color = 'tab:orange', label = "FLATW'RM")
plt.errorbar(prot_center-offset, FW_fps, yerr = FW_fps_err, xerr = x_err2, fmt='.', color = 'tab:orange', capsize = 5, label = "FLATW'RM")

# plt.bar(prot_edge, AFD_fps, align = 'edge', width = prot_width, color = 'none', edgecolor = 'gray')
# plt.scatter(prot_center, AFD_fps, marker='d', color = 'tab:green', label = "AFD")
plt.errorbar(prot_center+offset, AFD_fps, yerr = AFD_fps_err, xerr = x_err3, fmt='.', color = 'tab:green', capsize = 5, label = "AFD")
plt.grid()
plt.xlabel(r"$P_{rot}$ [days]")
plt.ylabel("Flare per Star")
plt.legend()
plt.yscale("log")
plt.xlim((-2,44))
plt.title("Flares per Star for different rotational Periods")
plt.show()