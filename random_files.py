# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:31:40 2024

@author: david
"""

# import the module
import shutil
import glob
import numpy as np
import random

# =============================================================================
# files 0-30
# =============================================================================

# flatwrm 0-30 test 1
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_1/all_lightcurves/*.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_1/all_lightcurves/00random2/"

# flatwrm 0-30 test 2
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_2/*/*_1.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0-30_2/00random/"

# AFD 0-30 test
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/*/*flare.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/000random/"

# AFDc 0-30 test
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/*/*.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_0-30_4/000random_c/"

# =============================================================================
# files 0-265
# =============================================================================

# Okamoto all
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/Okamoto_flares/*/*.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/Okamoto_flares/00random/"

# flatwrm all
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_31-265/*/*_1.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_31-265/00random/"

# AFD all
# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/*/*flare.png")
# destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/00random/"

# AFDc all
flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/*/*.png")
destination_directory = "C:/Users/david/Documents/David/Unibe/Bachelorarbeit/AFD/Kepler_Plots_all/00random_c/"

# =============================================================================
# random generator
# =============================================================================

for i in range(np.size(flare_files)):
# Specify the path of the file you want to copy
    rand = random.random()
    if rand < 0.02:
        file_to_copy = flare_files[i]

# Use the shutil.copy() method to copy the file to the destination directory
        shutil.copy(file_to_copy, destination_directory)
