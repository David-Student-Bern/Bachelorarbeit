# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:02:31 2024

@author: david
"""
import matplotlib.pyplot as plt
import numpy as np

Okamoto_pt = 2455614.15
Okamoto_dur = 0.1

FW_pt = 2455614.1549
FW_se = [2455614.1140, 2455614.1958]

AFD_pt = 2455614.154901321
AFD_se = [2455614.11403342274, 2455614.195769320424]



plt.figure(figsize=(6,2))
plt.plot(Okamoto_pt, 3, color = "tab:blue", marker = 'o', label = "Okamoto")
plt.plot(np.linspace(Okamoto_pt-Okamoto_dur, Okamoto_pt+Okamoto_dur, 1000), np.linspace(3,3,1000), color = "tab:blue", linestyle = '--')

plt.plot(FW_pt, 2, color = "tab:orange", marker = 'o', label = "FLATW'RM")
plt.plot(FW_se[0], 2, color = "tab:orange", marker = '>')
plt.plot(FW_se[1], 2, color = "tab:orange", marker = '<')
plt.plot(np.linspace(FW_se[0], FW_se[1], 1000), np.linspace(2,2,1000), color = "tab:orange", linestyle = '-')

plt.plot(AFD_pt, 1, color = "tab:green", marker = 'o',  label = "AFD")
plt.plot(AFD_se[0], 1, color = "tab:green", marker = '>')
plt.plot(AFD_se[1], 1, color = "tab:green", marker = '<')
plt.plot(np.linspace(AFD_se[0], AFD_se[1], 1000), np.linspace(1,1,1000), color = "tab:green", linestyle = '-')
plt.ylim((0.5,3.5))
plt.yticks([], [])
plt.legend()
# plt.grid()
plt.xlabel("Time (days)")
plt.title("Comparison of times for different surveys \n for one flare for KIC 7264671")
plt.show()