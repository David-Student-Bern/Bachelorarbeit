# Bachelorarbeit

This repository contains all the code used for my Bachelor Thesis and flare catalogues. It includes code from other sources, namely:

- Notebook `kepler_lightcurve.ipynb` (Downloading and Using Kepler Data to Plot a Light Curve) by Thomas Dutkiewicz: [Github page](https://github.com/spacetelescope/notebooks/blob/master/notebooks/MAST/Kepler/Kepler_Lightcurve/kepler_lightcurve.ipynb)
  
- FLATW'RM (FLAre deTection With Ransac Method) is a code that uses machine learning method to detect flares by Vida and Roettenbacher: [paper](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.163V/abstract) and [Github page](https://github.com/vidakris/flatwrm/blob/master/README.md).

- AFD (Python script for Automated Flare Detection and flare analysis) by Althukair and Tsiklauri: [paper](https://arxiv.org/abs/2212.10224) and [Github page](https://github.com/akthukair/AFD)


My thesis explains the modifications made to the two flare-finding algorithms in detail. Below is a short description of the structure of this repository and how to use the Code.

## Structure of Repository

- `random_files.py`
- AFD
  - `AFD.py`

    **Description:** AFD is a Python script for Automated Flare Detection and flare analysis, used on Kepler's light curves long cadence data.

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
 
    **Description:** Contains the results from the final survey. The survey initially included one suspected binary candidate (KIC 007772296), which was later removed, resulting in two different formats for `final-flares.csv`, `flares-candidates.csv`:
    
    - `*_all.csv` is the final list used for the analysis
    - `*_all_plus.csv` includes the flares found for the suspected binary candidate (KIC 007772296)
- flatwrm-master
- Okamoto_Data
- analysis
