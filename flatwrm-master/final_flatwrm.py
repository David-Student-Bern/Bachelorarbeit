# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:48:34 2024

@author: david
"""

# for flatwrm
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from gatspy.periodic import LombScargleFast


from sklearn import linear_model
# from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV

from aflare import aflare1
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps

# added be david
from astropy.io import fits
# import os.path
import glob
import time as tt
from pathlib import Path

# =============================================================================
# Test goal:  generate flatwrm flares in one script and save them in one file
# =============================================================================

# total run time
tstime = tt.time()

class PolynomialRANSAC(BaseEstimator):
    '''
    scikit-learn estimator that enables tuning of polynomial degrees 
    for linear regression
    '''

    def __init__(self, deg=None):
        self.deg = deg

    def fit(self, X, y, deg=None):
        #Adding random_state=0 for consistency in consecutive runs
        self.model = linear_model.RANSACRegressor(random_state=0)
        self.model.fit(np.vander(X, N=self.deg + 1), y)

    def predict(self, x):
        return self.model.predict(np.vander(x, N=self.deg + 1))

    @property
    def coef_(self):
        return self.model.coef_
    @property
    def inlier_mask_(self):
        return self.model.inlier_mask_

def SelectDegree(time,flux,window,seed=42, n_iterate=5, debug=False):
    if debug:
        print("Selecting best degree with",n_iterate,"samples...")
    np.random.seed(seed)
    best_degree = []
#	for i in range(n_iterate):
    i = 0
    watchdog = 0
    while i < n_iterate:
        t0 = np.random.random() * ( time.max()-time.min() ) + time.min()
        ind = np.where( (time > t0) * (time < t0+window) )
        t = time[ind]
        f = flux[ind]

        t_mean = np.mean(t)
        t_std = np.std(t)
        t_scale = (t - t_mean) / t_std

        grid = GridSearchCV(PolynomialRANSAC(),
                            param_grid={'deg':np.arange(1,15)}, 
                            scoring='neg_median_absolute_error', 
                            cv=KFold(n_splits=3,shuffle=True),
                            verbose=0, n_jobs=1)

        if watchdog > 50:
            print("SelectDegree is running in circles...")
            best_degree=8
            break

        try:
            grid.fit(t_scale, f)
        except ValueError:
            watchdog +=1
            if debug:
                #This can get really messy with long gaps...
                pass
                #print("I really shouldn't be here. Is this a data gap?",watchdog)
            continue


        if debug:
            print("{:2d}: {:4d} LC points, best polynomial degree: {:2d}".format( i+1, np.size(ind), grid.best_params_['deg']))
        best_degree.append( grid.best_params_['deg'] )
        i+=1

    degree = int(np.median(best_degree))
    if debug:
        print("Using polynomials of degree {:2d}".format( degree ))
    return degree


def FindPeriod(time,flux, minper=0.1,maxper=30,debug=False):
    '''
    Finds period in time series data using Lomb-Scargle method
    '''
    pgram = LombScargleFast(fit_period=True,\
                            optimizer_kwds={"quiet": not debug})
    # Ensure that maxper is not larger than the baseline of the data
    data_baseline = max(time) - min(time)
    maxper = min(maxper,data_baseline)
    
    pgram.optimizer.period_range = (minper, maxper)
    pgram.fit(time,flux)
    if debug:
        print("Best period:", pgram.best_period)
    return pgram.best_period


def FindFlares(time,flux, period, window_var=1.5,
               shift_var=4., degree=0, detection_sigma=3.,
               detection_votes=3, returnbinary=True, N3=2, 
               debug=False):
    '''
    Finds flare candidates using machine learning algorithm. 
    Currently the random sample consensus (RANSAC) method is
    implemented, as this yields a fast and robust fit. 
    The algorithm takes a given window (1.5*P_rot by default) 
    and fits a polynomial of given degree. Using RANSAC estimate 
    of the inlier points the standard deviation is calculated, 
    and the flare candites are selected.
    Since the polynomial fit might overfit the light curve at the
    ends (or RANSAC select these as outliers), this selection is 
    done multiple times by shifting the window, and only those flare 
    candidate points are kept, which get at least a given number of 
    vote.

    Parameters
    ----------
    time: numpy array
        Light curve time array
    flux: numpy array
        Light curve flux array
    window_var: float, optional
        Determines the size of fit window in period units (default: 1.5)
    shift_var: float, optional
        Determines window shift, portion of window (default: 3.)
    degree: int, optional
        Size of the Vandermonde matrix for the fit, determines 
        the order of fitted polynomial (default: 10)
    detection_sigma: float, optional
        Detection level of flare candidates in 
        np.stdev(flux - model) units (default: 3.)
    detection_votes: int, optional
        Number of votes to accept a flare candidate. If shift_var is
        changed, consider changing this, too. (default: 2)
    returnbinary: bool, optional
        If True the return value is a boolean mask with flare points
        marked. Otherwise flare start/end points are returned.
    N3:	int,optional
        Number of consecutive candidate points needed for a flare event
    
    Returns
    -------
    returnbinary == True:
        boolean array with flare points flagged
    returnbinary == False:
        two arrays with flare start/end indices
    '''

    if debug:
        print("Using period:", period)

    #We define the window to be fitted. Too short windows
    #might be overruled by long flares, too long ones might have
    #poor fits. 
    window = window_var * period
    shift = window / shift_var

    isflare = np.zeros_like( time )
    #We put the first windows before the light curve so not to miss
    #the early events
    t0 = np.min(time) - window + shift
    i = 0

    #You probably don't want to call this again, but who knows...
    if degree == 0:
        degree = SelectDegree(time,flux,period*1.5,debug=debug)

    #Originally the built-in RANSAC was used, but grid search for polynomial
    #degree is not possible that way...
    #regressor = linear_model.RANSAC(random_state=0)
    regressor = PolynomialRANSAC(deg=degree)

    while t0 < np.max(time):
        #degree = 10
        ind = np.where( (time > t0) * (time < t0+window) )
        #If we find a gap, move on
        if np.size(ind) <= degree+2:
            t0 += shift
            continue

        t = time[ind]
        f = flux[ind]

        #Machine learning estimators might behave badly 
        #if the data is not normalized
        t_mean = np.mean(t)
        t_std = np.std(t)
        t_scale = (t - t_mean) / t_std
        
        #Polynomial fit is achieved by feeding a Vandermonde matrix
        #to the regressor
        #regressor.fit( np.vander(t_scale, degree), f )

        #With the custom polynomial regressor we can just input time/flux
        regressor.fit( t_scale, f )
        

        #RANSAC outlier estimation is not trustworthy
        #flare_mask = np.logical_not(regressor.inlier_mask_)

        #model = regressor.predict( np.vander((t-t_mean)/t_std, degree) )
        model = regressor.predict( t_scale )

        #We won't use the outlier for statistics:
        stdev = np.std( (f - model)[regressor.inlier_mask_] )
        #Then select flare candidate points over given
        #sigma limit for this segment 
        flare_mask =  f > model + detection_sigma*stdev 

        #Each candidate gets a vote for the given time segment
        isflare[ind[0][0] : ind[0][-1]+1] += flare_mask 

#DEBUG Use this to plot the fitted model
#	t_plot = np.linspace(np.min(time[ind]),np.max(time[ind]), 1000)
#	f_plot = regressor.predict( np.vander(t_plot-t_mean,degree) )
#	f_plot = regressor.predict( np.vander((t_plot-t_mean)/t_std, degree) )

#NOTE: this is too much even for debug mode :)
#		if debug:	
#			print "Segment ",i,"\tCandidate points: ",\
#				  np.size(np.where(flare_mask==True))

        #Move on to the next segment
        t0 += shift
        i+=1

#DEBUG Use this to stop the algorithm somewhere for inspection
#		if i>=111:
#			break
#		if np.size(ind) == 0:
#			break

#	return isflare


    ##############################################
    # Thankfully taking this part from appaloosa #
    # https://github.com/jradavenport/appaloosa  #
    ##############################################
    #
    #We'll keep only candidates with enough votes
    ctmp = np.where( isflare >= detection_votes )

    cindx = np.zeros_like(flux)
    cindx[ctmp] = 1

    # Need to find cumulative number of points that pass "ctmp"
    # Count in reverse!
    ConM = np.zeros_like(flux)
    # this requires a full pass thru the data -> bottleneck
    for k in range(2, len(flux)):
        ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])

    # these only defined between dl[i] and dr[i]
    # find flare start where values in ConM switch from 0 to >=N3
    istart_i = np.where((ConM[1:] >= N3) &
                        (ConM[0:-1] - ConM[1:] < 0))[0] + 1

    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i] - 1)


    #we add an extra point for better energy estimation...
    #... but only if it's not too far away
    for j in range( len(istart_i) ):
        try:
            if np.abs( time[ int(istart_i[j]) ] - time[ int(istart_i[j]-1) ] ) < \
                2*np.abs( time[ int(istart_i[j]+1) ] - time[ int(istart_i[j]) ] ):
                istart_i[j] -= 1
        except IndexError:
            pass
    
    # added by David
    # removes flares if they are too close to the start or end of the dataset
    # but also if they are too close to a gap
    # and if there is a gap after the first point of the flare
    
    filtered_istart_i = []
    filtered_istop_i = []
    for j in range( len(istart_i) ):
        # print("istart ", int(istart_i[j]))
        # print("istop ", int(istop_i[j]))
        #start less that a flare duration before flare
        if ( time[ int(istart_i[j]) ] - (time[ int(istop_i[j]) ] - time[ int(istart_i[j]) ]) ) < time[0]:
            # print("filter 1: {:.4f}".format(time[ int(istart_i[j]) ]))
            continue
        #end less that a flare duration ahead of flare
        elif ( time[ int(istop_i[j]) ] + (time[ int(istop_i[j]) ] - time[ int(istart_i[j]) ]) ) > time[-1]:
            # print("filter 4: {:.4f}".format(time[ int(istart_i[j]) ]))
            continue
        #gap less that a flare duration + 2 points before flare
        elif ( time[ int(istart_i[j]) ] - (time[ int(istop_i[j]+2) ] - time[ int(istart_i[j]) ]) ) > time[int(istart_i[j]) - (int(istop_i[j]+2) - int(istart_i[j])) +1]:
            # print("filter 2: {:.4f}".format(time[ int(istart_i[j]) ]))
            continue
        #gap less that a flare duration + 2 points ahead of flare
        elif ( time[ int(istop_i[j]) ] + (time[ int(istop_i[j]) ] - time[ int(istart_i[j]-2) ]) ) < time[int(istop_i[j]) + (int(istop_i[j]) - int(istart_i[j]-2)) -2]:
            print("filter 3: {:.4f}".format(time[ int(istart_i[j]) ]))
            continue
        #if there is a gap after the first point of the flare
        elif np.abs( time[ int(istart_i[j])+1 ] - time[ int(istart_i[j]) ] ) > 2*np.abs( time[ int(istart_i[j])+2 ] - time[ int(istart_i[j])+1 ] ):
            # print("filter 5: {:.4f}".format(time[ int(istart_i[j]) ]))
            continue
        # if there is a significant gap between the start and end point (30 min = 0.020833 days)
        elif time[int(istop_i[j])]-time[int(istart_i[j])] > 2*0.0209 * ( int(istop_i[j]) - int(istart_i[j]) + 1):
            continue
        # if the number of flarepoints after the peakpoint are higher than before the peak point
        # but only if there are at least 5 flare points
        elif int(istop_i[j])-int(istart_i[j])>10:
            maxind = int(istart_i[j]) + np.argmax(flux[ int(istart_i[j]):int(istop_i[j])+1 ])
            if 2*(maxind - int(istart_i[j])) > (int(istop_i[j]) - maxind):
                continue
            else:
                filtered_istart_i.append(int(istart_i[j]))
                filtered_istop_i.append(int(istop_i[j]))
        # 2* is too much? semms to work, therfore no (?)
        elif int(istop_i[j])-int(istart_i[j])>4:
            maxind = int(istart_i[j]) + np.argmax(flux[ int(istart_i[j]):int(istop_i[j])+1 ])
            if (maxind - int(istart_i[j])) > (int(istop_i[j]) - maxind):
                # print("maxind: ", maxind)
                # print("istart: ", int(istart_i[j]), " = ",istart_i[j])
                # print("istop: ", int(istop_i[j]), " = ",istop_i[j])
                # print("time: ", time[int(istart_i[j])])
                continue
            else:
                filtered_istart_i.append(int(istart_i[j]))
                filtered_istop_i.append(int(istop_i[j]))
        # if standard deviation ?
        else:
            filtered_istart_i.append(int(istart_i[j]))
            filtered_istop_i.append(int(istop_i[j]))

    istart_i = np.array(filtered_istart_i, dtype='int') 
    istop_i = np.array(filtered_istop_i, dtype='int')
    
    

    if returnbinary is False:
        return istart_i, istop_i
    else:
        bin_out = np.zeros_like(flux, dtype='int')
        for k in range(len(istart_i)):
            bin_out[istart_i[k]:istop_i[k]+1] = 1
        return np.array(bin_out, bool)

    #############
    # </thanks> #
    #############


def FitFlare(time,flux,istart,istop,period,KIC,window_var=1.5, degree=10, debug=False, domodel=True):
    
    midflare = (time[istart] + time[istop]) / 2.
    window_mask = (time > midflare - period*window_var/2.) \
                * (time < midflare + period*window_var/2.)
    t = time[window_mask]
    f = flux[window_mask]

    #We should never get here, but if we have two points 
    #across a gap for some reason...
    if np.size(t) == 0:
        t = np.linspace(0,1,5)
        f = np.ones(5)
        model = np.ones_like(f)
        fx, fy, fy0 = t, model, f-model
        degree = 1
        if domodel:
            popt1 = np.array([np.nan, np.nan, np.nan])
            stdev=np.nan
            return fx, fy, fy0, popt1, stdev, [0, 5]
        else:
            return fx, fy, fy0, 0, 0, [0, 5]

    # original code
    start = (np.abs( t-time[istart] )).argmin()
    stop = (np.abs( t-time[istop] )).argmin()

    t_mean = np.mean(t)
    t_std = np.std(t)
    t_scale = (t - t_mean) / t_std
    
    #You probably don't want to call this again, but who knows...
    if degree == 0:
        SelectDegree(time,flux,period*1.5,debug=True,n_iterate=5)

    regressor = PolynomialRANSAC(deg=degree)

    regressor.fit(t_scale, f)
    model = regressor.predict(t_scale)            

    # added by David
    if debug:
        plt.figure(figsize=(9,4))
        plt.scatter(t[start:stop+1], f[start:stop+1], c = "k", marker = "X", label = "flare")
        plt.scatter(t, f, s=1, c = "b", label = "f:data") 
        plt.scatter(t, model, s=1, c = "r", label = "model")
        titletxt = KIC + ': flux vs model'
        plt.title(titletxt)
        plt.xlabel('Time (days)')
        plt.ylabel('Flux (electrons/second)')
        plt.legend()
        plot_path = "Kepler_31-265/"+KIC+"/"+"{:.4f}".format(time[istart])+"_1.png"
        plt.savefig(plot_path)
        # plt.show()
        plt.clf()

        # plt.figure(figsize=(9,4))
        # plt.scatter(t[regressor.inlier_mask_], (f - model)[regressor.inlier_mask_], s=1, c = "b", label = "data-model") 
        # plt.title('fy0 inlier')
        # plt.xlabel('Time (days)')
        # plt.ylabel('Flux (electrons/second)')
        # plt.legend()
        # # plt.show()
        # plt.clf()
    #====================================

    #Time, flare-free LC and unspotted LC with flare
    fx, fy, fy0 = t, model, f-model
    
    if domodel:
        #We also save the stdev for calculating start/end times
        stdev = np.std( (f - model)[regressor.inlier_mask_] )
        
        #====================================
        # added by David
        # remove outliers in the model
        new_fx = []
        new_f = []
        new_fy = []
        new_fy0 = []
        newstart = start
        newstop = stop
        for i in range(np.size(fx)):
            # +3 so that even points that are not declared as flare-points,
            # but may be part of the exponential decay are not eliminated
            if (start<=i<=stop+3):
                new_fx.append(fx[i])
                new_f.append(f[i])
                new_fy.append(fy[i])
                new_fy0.append(fy0[i])
            else:
                if abs(fy0)[i]<3*stdev:
                    new_fx.append(fx[i])
                    new_f.append(f[i])
                    new_fy.append(fy[i])
                    new_fy0.append(fy0[i])
                else:
                    if (start>i):
                        newstart -= 1
                        newstop -= 1
                

        if debug:
            plt.figure(figsize=(9,4))
            plt.scatter(new_fx[newstart:newstop+1], new_f[newstart:newstop+1], c = "k", marker = "X", label = "flare")
            plt.scatter(new_fx, new_f, s=1, c = "b", label = "f:data") 
            plt.scatter(new_fx, new_fy, s=1, c = "r", label = "model") 
            titletxt = KIC + ': flux vs model, non-flare outlier removed'
            plt.title(titletxt)
            plt.xlabel('Time (days)')
            plt.ylabel('Flux (electrons/second)')
            plt.legend()
            plot_path = "Kepler_31-265/"+KIC+"/"+"{:.4f}".format(time[istart])+"_2.png"
            plt.savefig(plot_path)
            plt.show()
            plt.clf()

            # plt.figure(figsize=(9,4))
            # plt.scatter(new_fx, new_fy0, s=1, c = "b", label = "data-model") 
            # plt.title('fy0 after removing non-flare outlier')
            # plt.xlabel('Time (days)')
            # plt.ylabel('Flux (electrons/second)')
            # plt.legend()
            # # plt.show()
            # plt.clf()
        #====================================
        
        try:
            global fwhm 		#I know, there is a special place in hell for this...
            if fwhm == 0: 		#not defined as command-line argument
                fwhm = 1./24 	#Selected by educated random guessing
        except NameError:
            fwhm = 1./24		#If calling just this function this might be handy

        #First try: the input is used as peak
        #tpeak = time[ind]
        #Second try: the maximum of the selection is considered as peak. 
        #But what if there are other events around?
        #ftpeak = fx[ np.argmin( np.max(fy0) - fy0) ]
        #tpeak = time[ np.argmin( np.abs( time - ftpeak ) ) ]
        tpeak = np.average( time[istart:istop+1] )

        #Same goes for the amplitude:
        ampl = np.max(new_fy0) # modified by David
        #ampl = np.max( flux[istart:istop+1] )

        pguess = (tpeak, fwhm, ampl)

        
        
        if False:
            print("fx = ", fx)
            print("fy0 = ", fy0)
        

        try:
            popt1, pcov = curve_fit(aflare1, new_fx, new_fy0, p0=pguess)  # modified by David
        except ValueError:
            # tried to fit bad data, so just fill in with NaN's
            # shouldn't happen often
            popt1 = np.array([np.nan, np.nan, np.nan])
        except RuntimeError:
            # could not converge on a fit with aflare
            # fill with bad flag values
            popt1 = np.array([-99., -99., -99.])
        if debug:
            print("Initial guess:",pguess)
            print("Fitted result:",popt1)
        return new_fx, new_fy, new_fy0, popt1, stdev, [newstart, newstop]  # modified by David
    else:
        return fx, fy, fy0, 0, 0, [start, stop]



#calculate flare energy
def FlareEnergy(Teff,flare_amp,time,flux,istart,istop,R):

    normalized_flux = flux / float(flux[0])
    flare_amp = flare_amp / float(flux[0])
    #maxind = istart + np.argmax(flux[ istart:istop+1 ])
    

    ########################## energy calculation in erg ###############################

    sigma = 5.6704 * 10 ** -5  # Stefan_Boltzmann constant(erg cm^-2 s^-1 k^-4)
    Tflare = 10000  # temperature of the flare =9000K
    h = 6.62606 * 10 ** -27  # plank's constant
    c = 2.99792 * 10 ** 10  # speed of light
    k = 1.38064 * 10 ** -16  # Boltzmann's constant

    KpRF = pd.read_csv('KpRF.txt')  # Kepler Instrument Response Function (high resolution)
    l = KpRF.lam * (10 ** -7)  # lambda in cm
    tr = KpRF.transmission  # Transmission

    n = len(l)

    rb1 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the star)
    rb2 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the flare)

    for i in range(n - 1):
        rb1.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * Teff))) - 1)))
        rb2.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((np.exp(h * c / (l[i] * k * Tflare))) - 1)))

    s1 = np.trapz(rb1, l[:-1])
    s2 = np.trapz(rb2, l[:-1])

    Af = []  ## Area of the flare
    Lf = []  ## Luminosity of the flare

    for i in range(len(normalized_flux)):
        af = flare_amp * np.pi * (R ** 2) * (s1 / s2)
        Af.append(flare_amp * np.pi * (R ** 2) * (s1 / s2))
        Lf.append(sigma * (Tflare ** 4) * af)

    flare_energy = np.trapz(np.array(Lf[istart:istop + 1]),time[istart: istop + 1])
    flare_energy_per_second = flare_energy * 24 * 60 * 60
    return flare_energy_per_second


# =============================================================================
# Generate output (modified function from flatwrm)
# =============================================================================
def GenerateOutput(time,flux,istart,istop,period,outputfile,KIC,Teff,R, degree=10,\
                   fit_events=True,debug=False):

    for i in range(len(istart)):
        t_start = time[ istart[i] ]
        t_stop = time[ istop[i]+1 ]
        #index of maximum LC point from the selection
        maxind = istart[i] + np.argmax(flux[ istart[i]:istop[i]+1 ])
        

        #For determining the flare energy, we have to integrate the light curve
        fx, fy, fy0, popt1, stdev, ind = \
            FitFlare(time,flux,istart[i],istop[i],period,KIC, degree=degree,debug=debug,domodel=fit_events)

        raw_integral = simps( fy0[ind[0]:ind[1]+1 ], fx[ ind[0]:ind[1]+1 ] )

        t_max = time[ maxind ]  #We also save the time of maximum

        #For precise start/end time and amplitude we fit the event
        #NOTE: funny fits yield funny results
        if fit_events:
            flare_t = np.linspace(np.min(fx), np.max(fx), int((fx[-1]-fx[0])*10000))
            flare_f = aflare1(flare_t, popt1[0], popt1[1], popt1[2] )
            # flare_f = -aflare(flare_t, [popt1[0], popt1[1], popt1[2]] )
            
            # plotting the fitted flare to the data
            if debug:
                plt.figure(figsize=(9,4))
                plt.scatter(flare_t, flare_f, s=1, c = "b", label = "fit") 
                plt.scatter(fx, fy0, s=2, c = "c", label = "flux-model") 
                plt.scatter(fx[ind[0]:ind[1]+1], fy0[ind[0]:ind[1]+1], s=2, c = "r", label = "flarepoints") 
                titletxt = KIC +': fit events'
                plt.title(titletxt)
                plt.xlabel('Time (days)')
                plt.ylabel('Flux (electrons/second)')
                plot_path = "Kepler_31-265/"+KIC+"/"+"{:.4f}".format(time[istart[i]])+"_3.png"
                plt.legend()
                plt.savefig(plot_path)
                # plt.show()
                plt.clf()
            
            amp = np.max(flare_f)
            fit_t_max = flare_t[ np.argmin( np.abs(amp - flare_f) ) ]

            fx_maxind = np.argmin( np.abs( fx - time[maxind] ) )
            lc_amp = flux[maxind] - fy[fx_maxind]
            
            fit_int = simps( flare_f, flare_t )

            energy = FlareEnergy(Teff,lc_amp,time,flux,istart[i],istop[i],R)
            # print("energy: ", energy)

            #Flare event is defined where it is above noise level
            event_ind = np.where(flare_f > stdev )
            
            # print("fit amp: ", amp)
            # print("stdev:   ", stdev)

            #Filters for strange detections:
            if ( fx[ind[-1]] - fx[ind[0]] ) / ( ind[-1]-ind[0] ) > 10*(fx[-1] - fx[0] ) / len(fx):
                #NOTE: long gap during the event -> pass
                continue
            if lc_amp < 0 :
                #NOTE: negative amplitude (WHY?) -> pass
                continue
        
            if np.size(event_ind) > 0: 
                fit_t_start = flare_t[ event_ind[0][0] ]
                fit_t_stop = flare_t[ event_ind[0][-1] ]

                outstring="{:<14s}".format(KIC)+\
                          "{:<14.4f}".format(t_start)+\
                          "{:<14.4f}".format(t_stop)+\
                          "{:<14.4f}".format(t_max)+\
                          "{:<14.4f}".format(lc_amp)+\
                          "{:<14.8f}".format(raw_integral)+\
                          "{:<14.4E}".format(energy)+\
                          "{:<14.4f}".format(amp)+\
                          "{:<14.4f}".format(popt1[1])+\
                          "{:<14.4f}".format(fit_t_start)+\
                          "{:<14.4f}".format(fit_t_stop)+\
                          "{:<14.4f}".format(fit_t_max)+\
                          "{:<18.8f}".format(fit_int)+\
                          "{:<14.8f}".format(stdev)
            else:
                print("\n====================\n-99")
                print("max(flare_f):", max(flare_f))
                print("stdev", stdev)
                print("====================\n")
                #Honestly, we should never get here...
                outstring="{:<14s}".format(KIC)+\
                          "{:<14.4f}".format(t_start)+\
                          "{:<14.4f}".format(t_stop)+\
                          "{:<14.4f}".format(t_max)+\
                          "{:<14.4f}".format(lc_amp)+\
                          "{:<14.8f}".format(raw_integral)+\
                          "{:<14.4E}".format(energy)+\
                          "{:<14.4f}".format(-99)+\
                          "{:<14.4f}".format(-99)+\
                          "{:<14.4f}".format(-99)+\
                          "{:<14.4f}".format(-99)+\
                          "{:<14.4f}".format(-99)+\
                          "{:<14.8f}".format(-99)+\
                          "{:<14.8f}".format(stdev)
                pass
            

        else:
            #without the fit we can give only a crude estimate on the 
            #amplitude based on the neighbors
            lc_amp = flux[maxind]-np.median(flux[maxind-10:maxind+10])

            outstring="{:<14s}".format(KIC)+\
                      "{:<14.4f}".format(t_start)+\
                      "{:<14.4f}".format(t_stop)+\
                      "{:<14.4f}".format(t_max)+\
                      "{:<14.4f}".format(lc_amp)+\
                      "{:<14.8f}".format(raw_integral)

        outputfile.write(outstring+"\n")

# =============================================================================
# all KIC files
# =============================================================================

flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/*/*llc.fits")

# flare_files = flare_files[3689:]

# =============================================================================
# One KIC
# =============================================================================

# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr003128078_lc_Q011111011101110111/*llc.fits")


# =============================================================================
# One lightcurve
# =============================================================================

# flare_files = glob.glob("C:/Users/david/Documents/David/Unibe/Bachelorarbeit/Okamoto_Data/mastDownload/Kepler/kplr008936408_lc_Q011111111111111111/kplr008936408-2010174085026_llc.fits ")

# =============================================================================
# flatwrm initial parameters
# =============================================================================
debug = True
fit_events = True
magnitude = False


# create output file
fout = open("flatwrm_output_985_erg.txt", "w")
if fit_events:
    header = "{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}".format(\
        "KIC",\
        "t_start",\
        "t_end",\
        "t_max",\
        "flux_max",\
        "raw_integral",\
        "energy",\
        "fit_amp",\
        "fit_fwhm",\
        "fit_t_start",\
        "fit_t_end",\
        "fit_t_max",\
        "fit_integral",\
        "fit_stdev")

else:
    header = "{:14}{:14}{:14}{:14}{:14}{:14}{:14}".format("KIC","#t_start","t_end","t_max","lc_amp","raw_integral","energy")

fout.write(header + "\n")

# =============================================================================
mean_flux_err = []
mean_deg = []

# extract time and flux
for j in range(np.size(flare_files)):
    print("---------------------------------------------------------------\n")
    print("file:\n", flare_files[j], "\n")
    stime = tt.time()
    with fits.open(flare_files[j], mode="readonly") as hdulist:
        # Read in the "BJDREF" which is the time offset of the time array.
        bjdrefi = hdulist[1].header['BJDREFI']
        bjdreff = hdulist[1].header['BJDREFF']

        # Read in the columns of data.
        times = hdulist[1].data['time'] 
        # sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        pdcsap_err = hdulist[1].data['PDCSAP_FLUX_ERR']
        
        Teff = hdulist[0].header['TEFF'] #[K] Effective temperature
        radius = hdulist[0].header['RADIUS'] #[solar radii] stellar radius

        # KIC = hdulist[1].header['KEPLERID']

    # Convert radius from stellar radius in solar radii to cm
    R = 6.957 * radius * 10 ** 10  # star's radius(cm)

    # Convert the time array to full BJD by adding the offset back in.
    bjds = times + bjdrefi + bjdreff 
    
    # remove nan values
    mask = np.isnan(pdcsap_fluxes)
    time = bjds[~mask]
    flux = pdcsap_fluxes[~mask]
    flux_err = pdcsap_err[~mask]
    
    mean_flux_err.append(np.mean(flux_err))
    
    # KIC
    KIC = flare_files[j][126:135]
# =============================================================================
#     flatwrm code
# =============================================================================
    
    # reinitialize initial parameters after each run
    flarepoints = 2
    sigma = 3
    period=0.
    degree=0
    fwhm = 0.
    
    if magnitude:
        flux = 10**(-0.4*flux)
    
    # minper = 3.011*(time[1]-time[0]), maxper = time[-1]-time[0]
    if period == 0:
        period = FindPeriod(time, flux, minper = 3.011*(time[1]-time[0]), maxper = 40, debug=debug)
    window_var = 1.5
    
    #Optionally you can use the code interactively to get a flare mask:
    #isflare = FindFlares(time,flux, period)
    #If time is not a concern, the search can be run multiple times
    #to remove false positives:
    #isflare = FindFlares(time,flux, period) *\
    #          FindFlares(time,flux, period)


    if degree == 0:
        degree = SelectDegree(time,flux,period*window_var,debug=debug)
    
    #by david
    if degree < 9:
        sigma = 4
    if degree < 8:
        flarepoints = 3
    if degree < 5:
        sigma = 5

    istart, istop = FindFlares(time,flux, period,
                               returnbinary=False,
                               N3=flarepoints,
                               degree=degree, 
                               detection_sigma=sigma,
                               debug=debug)
    
    #statistics:
    mean_deg.append(degree)

    # makes sure that there is a directory for the plots
    
    # !!!! 
    # replace the plot path for the plots
    # !!!!
    
    plots_path = 'C:/Users/david/Documents/David/Unibe/Bachelorarbeit/flatwrm-master/Kepler_0/'+KIC
    Path(plots_path).mkdir(parents=True, exist_ok=True)
    
    # Generate output
    GenerateOutput(time,flux,istart,istop,period,fout,KIC,Teff,R,\
                       fit_events=fit_events,degree=degree,debug=debug,)
    etime = tt.time()
    print("  final parameters:\n")
    print("    period  = {:.4f}".format(period))
    print("    degree  = {:d}".format(degree))
    print("    sigma   = {:d}".format(sigma))
    print("    fp      = {:d}".format(flarepoints))
    # print("    fwhm    = {:.4f}".format(fwhm))
    print("    #flares = {:d}\n".format(np.size(istart)))
    print("\n","duration: {:.2f} seconds = {:.4f} minutes \n".format((etime-stime),(etime-stime)/60))

#total mean flux error
# print("total mean flux error:", np.mean(mean_flux_err))
#mean degree
print("total mean degree:", np.mean(mean_deg))

# total run time
tetime = tt.time()
print("===============================================================\n")
print("total run time: {:.2f} min = {:.4f} h".format((tetime-tstime)/60,(tetime-tstime)/3600))

fout.close()