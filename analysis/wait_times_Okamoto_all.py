# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:02:13 2024

@author: david
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy'])



Okamoto_wait_times = []
Okamoto_energies = []
Okamoto_energies2 = []

for KIC in short_KIC:
    Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]

    for i in range(np.size(Okamoto_KIC.values[:,0])-1):
        Okamoto_wait_times.append(Okamoto_KIC.values[i+1,1]-Okamoto_KIC.values[i,1])
        Okamoto_energies.append(Okamoto_KIC.values[i,3])
        Okamoto_energies2.append(Okamoto_KIC.values[i+1,3])
        # if (Okamoto_KIC.values[i+1,1]-Okamoto_KIC.values[i,1]) < 0:
        #     print('kepler_id: ', Okamoto_KIC.values[i,0])
        #     print('peak_time: ', Okamoto_KIC.values[i,1])

# =============================================================================
# statistical data
# =============================================================================
mean_wt = np.mean(Okamoto_wait_times)
mean_e = np.mean(Okamoto_energies)
mean_e2 = np.mean(Okamoto_energies2)

stdev_wt = np.std(Okamoto_wait_times)
stdev_e = np.std(Okamoto_energies)
stdev_e2 = np.std(Okamoto_energies2)

median_wt = np.median(Okamoto_wait_times)
median_e = np.median(Okamoto_energies)
median_e2 = np.median(Okamoto_energies2)

# =============================================================================
# Histogram
# =============================================================================


# histogram for all
data = Okamoto_wait_times
plot_filter = "Okamoto"

# histogram parameter 1000
bins_set = int((max(Okamoto_wait_times))/5)
range_array = (0, 1000)

filter1 = 0
for item in data:
    if item < range_array[0] or item > range_array[1]:
        filter1 +=1
f1 = 100 * (filter1/len(data))
plot_label = plot_filter + r': F$_1$ = ' + str(round(f1,3)) + '%'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.hist(data, bins = bins_set, range = range_array, label = plot_label)
xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
ax.set_xlabel(xlabeltxt)
ax.set_ylabel('Frequency')
ax.legend()
ax.set_title("Histogram for wait times for Okamoto")
ax.grid(True)
plt.show()

# histogram parameter 200
bins_set = int((max(Okamoto_wait_times))/20)
range_array = (0, 200)

filter1 = 0
for item in data:
    if item < range_array[0] or item > range_array[1]:
        filter1 +=1
f1 = 100 * (filter1/len(data))
plot_label = plot_filter + r': F$_1$ = ' + str(round(f1,3)) + '%'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.hist(data, bins = bins_set, range = range_array, label = plot_label)
xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
ax.set_xlabel(xlabeltxt)
ax.set_ylabel('Frequency')
ax.legend()
ax.set_title("Histogram for wait times for Okamoto")
ax.grid(True)
plt.show()

# histogram parameter 500
bins_set = int((max(Okamoto_wait_times))/20)
range_array = (0, 50)

filter1 = 0
for item in data:
    if item < range_array[0] or item > range_array[1]:
        filter1 +=1
f1 = 100 * (filter1/len(data))
plot_label = plot_filter + r': F$_1$ = ' + str(round(f1,3)) + '%'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.hist(data, bins = bins_set, range = range_array, label = plot_label)
xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
ax.set_xlabel(xlabeltxt)
ax.set_ylabel('Frequency')
ax.legend()
ax.set_title("Histogram for wait times for Okamoto")
ax.grid(True)
plt.show()

# =============================================================================
# Energies
# =============================================================================

newX = np.logspace(0, 3, base=10)
newY = np.logspace(33, 36, base = 10)

# Let's fit an exponential function.  
# This looks like a line on a lof-log plot.

# def myComplexFunc(x, a, b, c):
#     return a * np.power(x, b) + c
# popt1, pcov1 = curve_fit(myComplexFunc, data, Okamoto_energies)
# popt2, pcov2 = curve_fit(myComplexFunc, data, Okamoto_energies2)

# plot statistical data
O_mean_wt = mean_wt * np.ones_like(newY)
O_mean_e = mean_e * np.ones_like(newX)
# O_mean_e2 = mean_e2 * np.ones_like(newX)
O_median_wt = median_wt * np.ones_like(newY)
O_median_e = median_e * np.ones_like(newX)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.scatter(data, Okamoto_energies, s=1, label = "time to next flare")
ax.scatter(data, Okamoto_energies2, s=1, label = "time before flare")
# ax.plot(newX, myComplexFunc(newX, *popt1), 'b-', label = "({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*popt1))
# ax.plot(newX, myComplexFunc(newX, *popt2), 'r-', label= "({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*popt2))

#statistics
# ax.plot(O_mean_wt, newY, 'g-', label = 'mean wait time')
# ax.plot(newX, O_mean_e, 'k-', label = 'mean energy')
ax.plot(O_median_wt, newY, 'g--', label = 'median wait time')
ax.plot(newX, O_median_e, 'k--', label = 'median energy')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('wait time [days]')
ax.set_ylabel('flare energy [ergs]')
ax.legend()
titletxt = "Energies and Wait times for "+plot_filter
ax.set_title(titletxt)
ax.grid(True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.scatter(data, Okamoto_energies, s=1, label = plot_filter)
ax.plot()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('wait time to next flare [days]')
ax.set_ylabel('flare energy [ergs]')
ax.legend()
titletxt = "Energies and Wait times for "+plot_filter+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(median_wt, median_e)
ax.set_title(titletxt)
ax.grid(True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
ax.scatter(data, Okamoto_energies2, s=1, label = plot_filter)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('wait time before flare [days]')
ax.set_ylabel('flare energy [ergs]')
ax.legend()
titletxt = "Energies and Wait times for "+plot_filter+"\n"+"median_wt: {:.1f} days, median_e: {:.1E} ergs" .format(median_wt, median_e2)
ax.set_title(titletxt)
ax.grid(True)