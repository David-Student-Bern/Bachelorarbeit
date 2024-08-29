# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:13:29 2024

curtesies to the stack overflow communitiy
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
# import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

def wait_times_energies(Okamoto_df, short_KIC):
    Okamoto_wait_times = []
    Okamoto_energies = []
    Okamoto_energies1 = []
    Okamoto_energies2 = []
    Okamoto_flare_length = []
    
    for KIC in short_KIC:
        Okamoto_KIC = Okamoto_df.loc[Okamoto_df['kepler_id'] == KIC]
    
        for i in range(np.size(Okamoto_KIC.values[:,0])-1):
            Okamoto_wait_times.append(Okamoto_KIC.values[i+1,1]-Okamoto_KIC.values[i,1])
            Okamoto_energies1.append(Okamoto_KIC.values[i,3])
            Okamoto_energies2.append(Okamoto_KIC.values[i+1,3])
            Okamoto_flare_length.append(Okamoto_KIC.values[i,2])
            Okamoto_energies.append(Okamoto_KIC.values[i,3])
        if np.size(Okamoto_KIC.values[:,2])>0:
            Okamoto_flare_length.append(Okamoto_KIC.values[-1,2])
            Okamoto_energies.append(Okamoto_KIC.values[-1,3])
    return Okamoto_wait_times, Okamoto_energies1, Okamoto_energies2, Okamoto_energies, Okamoto_flare_length


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    # end
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

short_KIC = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/KIC_list.txt")

# =============================================================================
# Okamoto
# =============================================================================
Okamoto_data = np.loadtxt("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/aaList.txt", skiprows = 35, usecols=[0, 11, 12, 13, 1, 6]) # Kepler Input Catalog identifier
Okamoto_df = pd.DataFrame(
    Okamoto_data,
    columns = ['kepler_id', 'flare_peak_time', 'flare_duration', 'energy', 'Teff', 'Prot'])

Okamoto_data = [0,0,0,0,0,0]

Okamoto_data[0] = Okamoto_df[Okamoto_df['Prot'].between(0,2)]
Okamoto_data[1] = Okamoto_df[Okamoto_df['Prot'].between(2,5)]
Okamoto_data[2] = Okamoto_df[Okamoto_df['Prot'].between(5,15)]
Okamoto_data[3] = Okamoto_df[Okamoto_df['Prot'].between(15,25)]
Okamoto_data[4] = Okamoto_df[Okamoto_df['Prot'].between(25,35)]
Okamoto_data[5] = Okamoto_df[Okamoto_df['Prot'].between(35,100)]

Okamoto_wait_times = [0,0,0,0,0,0]
Okamoto_energies1 = [0,0,0,0,0,0]
Okamoto_energies2 = [0,0,0,0,0,0]
Okamoto_energies = [0,0,0,0,0,0]
Okamoto_flare_length = [0,0,0,0,0,0]

for i in range(len(Okamoto_data)):
    Okamoto_wait_times[i] , Okamoto_energies1[i], Okamoto_energies2[i], Okamoto_energies[i], Okamoto_flare_length[i] = wait_times_energies(Okamoto_data[i], short_KIC)

plot_filter = ["Prot: [0,2]", "Prot: [2,5]", "Prot: [5,15]", "Prot: [15,25]", "Prot: [25,35]", "Prot: >35"]

Okamoto_wait_times_all , Okamoto_energies1_all, Okamoto_energies2_all, Okamoto_energies_all, Okamoto_flare_length_all = wait_times_energies(Okamoto_df, short_KIC)

# Load data from statsmodels datasets
# data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
# data = Okamoto_wait_times
# data = [Okamoto_wait_times_all]
data = [Okamoto_df.values[:,3]]

# histogram parameter 1000
# bins_set = int(1000/5)
# range_array = (0, 1000)

# histogram parameter 200
bins_set = int(1000/20)
range_array = (0, 200)

# histogram parameter 50
# bins_set = int(1000/20)
# range_array = (0, 50)

# compute f2: measurements are outside the selected range for plotting
allf1 = []
plot_label = []
for index in range(len(data)):
    filter1 = 0
    for item in data[index]:
        if item < range_array[0] or item > range_array[1]:
            filter1 +=1
    if len(data[index]) == 0:
        f1 = 0
    else:
        f1 = 100 * (filter1/len(data[index]))
    allf1.append(f1)
    label = plot_filter[index] + ': '+str(len(data[index]))+' gaps'+'\n'+r'F$_1$ = ' + str(round(f1,3)) + '%'
    plot_label.append(label)

# Plot for comparison
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
# ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
ax.hist(data[0], bins = range_array[1], density = True, range = range_array, label = plot_label[0])

# Save plot limits
dataYLim = 200

# Find best fit distribution
best_distibutions = best_fit_distribution(data[0], 200, ax)
best_dist = best_distibutions[0]

# Update plots
# ax.set_ylim(dataYLim)
ax.set_title(u'All Fitted Distributions')
xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
ax.set_xlabel(xlabeltxt)
ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist[0], best_dist[1])

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
ax.hist(data[0], bins = range_array[1], density = True, range = range_array, label = plot_label[0])

param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)

ax.set_title(u'Best fit distribution \n' + dist_str)
xlabeltxt = "wait times [days]\n"+ r"F$_1$% refers to portion of the total received measurements"+"\n" +"that lie outside the range of values on X-axis of this plot."
ax.set_xlabel(xlabeltxt)
ax.set_ylabel('Frequency')