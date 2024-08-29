# Bachelorarbeit

This repository contains all the code used for my Bachelor Thesis and flare catalogues. It includes code from other sources, namely:

- Notebook `kepler_lightcurve.ipynb` (Downloading and Using Kepler Data to Plot a Light Curve) by Thomas Dutkiewicz: [Github page](https://github.com/spacetelescope/notebooks/blob/master/notebooks/MAST/Kepler/Kepler_Lightcurve/kepler_lightcurve.ipynb)
  
- FLATW'RM (FLAre deTection With Ransac Method) is a code that uses machine learning method to detect flares by Vida and Roettenbacher: [paper](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.163V/abstract) and [Github page](https://github.com/vidakris/flatwrm/blob/master/README.md).

- AFD (Python script for Automated Flare Detection and flare analysis) by Althukair and Tsiklauri: [paper](https://arxiv.org/abs/2212.10224) and [Github page](https://github.com/akthukair/AFD)


My thesis explains the modifications made to the two flare-finding algorithms in detail. Below is a short description of the structure of this repository and how to use the Code.

## Structure of Repository

- AFD
  - `AFD.py`

    **Description:** AFD is a Python script for Automated Flare Detection and flare analysis, used on Kepler's light curves long cadence data. For more details, read the accompanying [paper](https://arxiv.org/abs/2212.10224).

    **Author:** Original Code by Althukair and Tsiklauri, modified by Schwarz

    **Usage:**
    1. add the directory to light curve files by changing the parameter `file_list`
    2. add a catalogue with stellar parameters by changing `s_p`
    3. add the directory where you want the light curve plots saved by changing `plots_path`
    4. run script
    5. check `errors.csv` and debug
    6. safe the output files (`errors.csv`, `final-flares.csv`, `flares-candidates.csv`) in a different directory as they will be overwritten in the next run

  - `KpRF.txt`

    **Description:** File containing the Kepler Instrument Response Function (high resolution)
    
  - `KR_function.py`
 
  - stellar parameters

    **Description:** Directory containing stellar parameters from some A-, F-, G-, K- and M-type stars downloaded from the from the [Q1-Q17 (DR25)](https://exoplanetarchive.ipac.caltech.edu/docs/Q1Q17-DR25-KOIcompanion.html) stellar and planet catalogue

  - Kepler_Plots_all
 
    **Description:** Contains the results from the final survey. The code was run for different ranges of light curve files (first to 1000th file, 1001st file to 2000th file, ...) to minimise the runtime for a stint and thus minimise the impact of a crash. The files were then merged into the final list. The survey initially included one suspected binary candidate (KIC 007772296), which was later removed, resulting in two different formats for the final list:
    
    - `*_all.csv` is the final list used for the analysis
    - `*_all_plus.csv` includes the flares found for the suspected binary candidate (KIC 007772296)

- flatwrm-master
  - `final_flatwrm.py`

    **Description:** Code that uses machine learning method to detect flares. For more details, read the `README.md` in this directory or the accompanying [paper](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.163V/abstract).

    **Author:** Original Code by Vida and Roettenbacher, modified by Schwarz

    **Usage:**
    1. add the directory to light curve files by changing the parameter `flare_files`
    2. add the directory where you want the light curve plots saved by changing `plots_path`
    3. name the output file by changing `fout`
    4. check the initial parameters (`debug, fit_events, magnitude, flarepoints, sigma, period, degree, fwhm`)
    5. run script
    6. check output file for errors and debug

 
  - `aflare.py`
 
    **Description:** Analytic flare model
 
  - `KpRF.txt`
 
    **Description:** File containing the Kepler Instrument Response Function (high resolution)
 
  - Kepler_31-265
 
    **Description:** Contains the results from the final survey. The code had to be run multiple times because of crashes. After each crash, the run continued at the last KIC before the crash. The files were then merged into the final list. The survey initially included one suspected binary candidate (KIC 007772296), which was later removed, resulting in two different formats for the final list:

    - `*_all.txt` is the final list used for the analysis
    - `*_all_plus.txt` includes the flares found for the suspected binary candidate (KIC 007772296)
    
- Okamoto_Data

  - `aaList.txt`

    **Description:** File containing flares found by [Okamoto et al.](https://iopscience.iop.org/article/10.3847/1538-4357/abc8f5) *without* the suspected binary candidate (KIC 007772296)

  - `aaList_plus.txt`
 
    **Description:** Original file containing flares found by [Okamoto et al.](https://iopscience.iop.org/article/10.3847/1538-4357/abc8f5)
    
  - `final_plot_flare.py`

    **Description:** Creates plots for all the flares in `aaList.txt`

    **Usage:**
    1. change the parameters `short_KIC, Okamoto_data, flare_files, plots_path` to adjust for the local directories
    2. run script

  - `get_Okamoto_Data.py`
 
    **Description:**
    1. Creates a list with all KIC for all the flares found by Okamoto et al. --> `KIC_list.txt `
    2. Downloads all light curve files for those KIC into a new directory: mastDownload\Kepler\
  
  - `KIC_list.txt`
 
    **Description:** List with all KIC for all the flares found by Okamoto et al. created by `get_Okamoto_Data.py` *without* the suspected binary candidate (KIC 007772296)
  - `KIC_list_plus.txt`
 
    **Description:** List with all KIC for all the flares found by Okamoto et al. created by `get_Okamoto_Data.py`

   - `plotting_light_curves.py`
 
     **Description:** Generates figures showing the whole light curve from individual .fits files.
 
- analysis

  **Description:** Many different scripts were created to analyse various aspects of the flares and create plots that illustrate them by eithher comparing the catalogue made by Okamoto and al. to the catalogues made by the FLATW'RM and the AFD algorithms or for each catalogue individually. The numbers and figures created with these scripts can be found in my thesis. 

  **Usage:** To use these scripts, the path to the input files must be adapted to the local directory

  - `analysis_all.py`
 
    **Description:**  Find all common flares for the different surveys and create Venn diagrams
    
  - `Catalogue_difference.py`
 
    **Description:** Determine the difference between the effective temperatures and rotational periods used with each survey
    
  - `Extremals.py` 
 
    **Description:** Find the flares with the highest energies and the longest duration and create a new list for them so that they can be analysed further
    
  - `Extremals_plots.py`
 
    **Description:** Plot the flares that were filtered by Extremals.py and display flare energy and flare duration
    
  - `flare_length_all.py`
 
    **Description:** Create flare duration/energy plots
    
  - `Flare_time_comp.py`
 
    **Description:** Compares the different flare times one flare for KIC 7264671 returned by Okamoto, FLATW'RM and AFD
    
  - `Mean_plots.py`
 
    **Description:** Plotting mean values for wait time, duration and flare energy in different ranges of rotational period
    
  - `Okamoto_flares.py`
 
    **Description:** Find those stars that only Okamoto reported as flaring
    
  - `prot_AFD_all.py`
 
    **Description:** Create different plots for AFD including different ranges in rotational period
    
  - `prot_Flatwrm_all.py`
 
    **Description:** Create different plots for FLATW'RM including different ranges in rotational period
    
  - `prot_Okamoto_all.py`
 
    **Description:** Create different plots for Okamoto including different ranges in rotational period
    
  - `star_type_Okamoto_all.py`
 
    **Description:** Same idea as in prot_Okamoto_all.py but instead of the rotational period the star type (<--> effective temperature of the star) is used to create different groups.
    
  - `wait_times_all.py`
 
    **Description:** Create wait time/energy plots
    
- `random_files.py`

  **Description:** Algorithm used for the random flare evaluation: Create a subgroup of all flare .png files by using a random number generator
