import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm, gumbel_r, genpareto
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

class Bootstrap_fit:
    n_bootstraps = 200
    
    def __init__(self, data):
        self.data = data

    def fit_genextreme(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=gev.fit(self.data)
        
        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GEV distribution to the sample using MLE method
                params_bs = gev.fit(sample, method="MLE", loc=paramEsts[1], scale=paramEsts[2])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)
        
        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper)).T
        
        return paramEsts, paramCIs

    def fit_gumbel(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=gumbel_r.fit(self.data)
        
        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GUMBEL distribution
                params_bs = gumbel_r.fit(sample, loc=paramEsts[0], scale=paramEsts[1])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                print(f"Bootstrap sample {j} failed: {e}")
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)

        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper)).T

        return paramEsts, paramCIs


    def fit_genpareto(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=genpareto.fit(self.data,method="MLE",loc=np.mean(self.data),scale=np.std(self.data))
        
        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GPD distribution
                params_bs = genpareto.fit(sample,method="MLE",loc=paramEsts[1],scale=paramEsts[2])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                print(f"Bootstrap sample {j} failed: {e}")
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)

        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper)).T

        Z_alpha_half = 1.96
        
        # Calculate standard errors from the confidence intervals
        standard_errors = []
        for ci in paramCIs:
            # CI = [lower_bound, upper_bound]
            lower_bound, upper_bound = ci
            SE = (upper_bound - lower_bound) / (2 * Z_alpha_half)
            standard_errors.append(SE)

        # Convert to a numpy array for easy access
        standard_errors = np.array(standard_errors)
        return paramEsts, paramCIs, standard_errors

def tsEasyParseNamedArgs(args, argStruct):
    avlArgs = fieldnames(argStruct)
    for ia in M[1 : length(avlArgs)]:
        argName = avlArgs[I[ia]]
        argIndx = find(strcmpi(args, argName))
        if _not(isempty(argIndx)):
            val = args[I[argIndx + 1]]
            argStruct[argName] = copy(val)

    return argStruct
    

def tsEvaPlotReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr):
    # Default argument values

    minReturnPeriodYears = 5
    maxReturnPeriodYears = 1000
    confidenceAreaColor = [189/255, 252/255, 201/255]  # light green
    confidenceBarColor = [34/255, 139/255, 34/255]      # dark green
    returnLevelColor = 'k'
    xlabel = 'return period (years)'
    ylabel = 'return levels (m)'
    ylim = None
    dtSampleYears = 1
    ax = None

    # Compute return periods and their corresponding periods in dt
    returnPeriodsInYears = np.logspace(np.log10(minReturnPeriodYears), np.log10(maxReturnPeriodYears), num=100)
    returnPeriodsInDts = returnPeriodsInYears / dtSampleYears

    # Compute return levels and errors
    returnLevels, returnLevelsErrs = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr, returnPeriodsInDts)
    
    # Confidence intervals
    supRLCI = returnLevels + 2 * returnLevelsErrs
    infRLCI = returnLevels - 2 * returnLevelsErrs
    
    # Determine Y limits based on the confidence intervals
    if ylim is not None:
        minRL, maxRL = min(ylim), max(ylim)
    else:
        minRL, maxRL = min(infRLCI), max(supRLCI)

        
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))  # Similar to figure position and size
    else:
        fig = None

    # Area plot for confidence intervals
    ax.fill_between(returnPeriodsInYears, infRLCI[0], supRLCI[0], color=confidenceAreaColor, label='Confidence Area', zorder=1)

    # Plot return levels
    ax.plot(returnPeriodsInYears, returnLevels[0], color=returnLevelColor, linewidth=3, label='Return Levels', zorder=2)

    # Plot confidence bars
    ax.plot(returnPeriodsInYears, supRLCI[0], color=confidenceBarColor, linewidth=2, label='Upper CI', zorder=3)
    ax.plot(returnPeriodsInYears, infRLCI[0], color=confidenceBarColor, linewidth=2, label='Lower CI', zorder=3)

    # Set plot scale to logarithmic for x-axis
    ax.set_xscale('log')

    # Set axis limits
    ax.set_xlim([minReturnPeriodYears, maxReturnPeriodYears])
    ax.set_ylim([minRL[0], maxRL[0]])

    # Labeling and formatting
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.grid(True, which='both', zorder=4)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16)


    # Finalizing plot appearance
    if fig:
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Return plot handles (mimicking phandles in MATLAB)
    phandles = {
        'fig': fig,
        'ax': ax,
        'confidence_area': ax.collections[0],  # first area collection
        'return_levels': ax.lines[0],  # first line plot (return levels)
        'upper_CI': ax.lines[1],  # second line plot (upper CI)
        'lower_CI': ax.lines[2],  # third line plot (lower CI)
    }

    return phandles

def tsEvaPlotReturnLevelsGEVFromAnalysisObj(
    nonStationaryEvaParams, timeIndex
):
    epsilon = nonStationaryEvaParams['GEVstat']['parameters']['epsilon']
    sigma = np.mean(nonStationaryEvaParams['GEVstat']['parameters']['sigma'])
    mu = np.mean(nonStationaryEvaParams['GEVstat']['parameters']['mu'])
    dtSampleYears = nonStationaryEvaParams['GEVstat']['parameters']['timeDeltaYears']
    epsilonStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['epsilonErr']
    sigmaStdErr = np.mean(nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErr'])
    muStdErr = np.mean(nonStationaryEvaParams['GEVstat']['paramErr']['muErr'])
    phandles = tsEvaPlotReturnLevelsGEV(
        epsilon,
        sigma,
        mu,
        epsilonStdErr,
        sigmaStdErr,
        muStdErr
    )
    return phandles

                
def tsEvaPlotReturnLevelsGPDFromAnalysisObj(
    nonStationaryEvaParams, timeIndex
):
    epsilon = nonStationaryEvaParams['GPDstat']['parameters']['epsilon']
    sigma = nonStationaryEvaParams['GPDstat']['parameters']['sigma']
    threshold = nonStationaryEvaParams['GPDstat']['parameters']['threshold']
    thStart = nonStationaryEvaParams['GPDstat']['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams['GPDstat']['parameters']['timeHorizonEnd']
    timeHorizonInYears = round((thEnd-thStart)/ 365.2425)
    nPeaks = nonStationaryEvaParams['GPDstat']['parameters']['nPeaks']
    
    epsilonStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['epsilonErr']
    sigmaStdErr = np.mean(nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErr'])
    thresholdStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErr']

    phandles = tsEvaPlotReturnLevelsGPD(
        epsilon,
        sigma,
        threshold,
        epsilonStdErr,
        sigmaStdErr,
        thresholdStdErr,
        nPeaks,
        timeHorizonInYears
    )
    return phandles

def tsEvaPlotReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr,thresholdStdErr,nPeaks,timeHorizonInYears):
    # Default argument values

    minReturnPeriodYears = 5
    maxReturnPeriodYears = 1000
    confidenceAreaColor = [189/255, 252/255, 201/255]  # light green
    confidenceBarColor = [34/255, 139/255, 34/255]      # dark green
    returnLevelColor = 'k'
    xlabel = 'return period (years)'
    ylabel = 'return levels (m)'
    ylim = None
    dtSampleYears = 1
    ax = None

    # Compute return periods and their corresponding periods in dt
    returnPeriodsInYears = np.logspace(np.log10(minReturnPeriodYears), np.log10(maxReturnPeriodYears), num=100)
    returnPeriodsInDts = returnPeriodsInYears / dtSampleYears

    # Compute return levels and errors (this should call your custom function)
    returnLevels, returnLevelsErrs = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, timeHorizonInYears, returnPeriodsInDts)

    # Confidence intervals
    supRLCI = returnLevels + 2 * returnLevelsErrs
    infRLCI = returnLevels - 2 * returnLevelsErrs

    # Determine Y limits based on the confidence intervals
    if ylim is not None:
        minRL, maxRL = min(ylim), max(ylim)
    else:
        minRL, maxRL = min(infRLCI), max(supRLCI)

        
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))  # Similar to figure position and size
    else:
        fig = None

    # Area plot for confidence intervals
    ax.fill_between(returnPeriodsInYears, infRLCI[0], supRLCI[0], color=confidenceAreaColor, label='Confidence Area', zorder=1)

    # Plot return levels
    ax.plot(returnPeriodsInYears, returnLevels[0], color=returnLevelColor, linewidth=3, label='Return Levels', zorder=2)

    # Plot confidence bars
    ax.plot(returnPeriodsInYears, supRLCI[0], color=confidenceBarColor, linewidth=2, label='Upper CI', zorder=3)
    ax.plot(returnPeriodsInYears, infRLCI[0], color=confidenceBarColor, linewidth=2, label='Lower CI', zorder=3)

    # Set plot scale to logarithmic for x-axis
    ax.set_xscale('log')

    # Set axis limits
    ax.set_xlim([minReturnPeriodYears, maxReturnPeriodYears])
    ax.set_ylim([minRL[0], maxRL[0]])

    # Labeling and formatting
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.grid(True, which='both', zorder=4)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16)


    # Finalizing plot appearance
    if fig:
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Return plot handles (mimicking phandles in MATLAB)
    phandles = {
        'fig': fig,
        'ax': ax,
        'confidence_area': ax.collections[0],  # first area collection
        'return_levels': ax.lines[0],  # first line plot (return levels)
        'upper_CI': ax.lines[1],  # second line plot (upper CI)
        'lower_CI': ax.lines[2],  # third line plot (lower CI)
    }

    return phandles

def tsEvaComputeTimeRP(params, RPiGEV, RPiGPD):
    paramx = pd.DataFrame(params, index=[0]).T
    qxV = 1 - np.exp(-((1 + paramx[0]['epsilonGEV'] * (RPiGEV - paramx[0]['muGEV']) / paramx[0]['sigmaGEV']) ** (-1 / paramx[0]['epsilonGEV'])))
    returnPeriodGEV = 1 / qxV
    X0 = paramx[0]['nPeaks'] / paramx[0]['SampleTimeHorizon']
    qxD = ((1 + paramx[0]['epsilonGPD'] * (RPiGPD - paramx[0]['thresholdGPD']) / paramx[0]['sigmaGPD']) ** (-1 / paramx[0]['epsilonGPD']))
    returnPeriodGPD = 1 / (X0 * qxD)

    return returnPeriodGEV, returnPeriodGPD

def tsEvaComputeReturnPeriodsGEV(epsilon, sigma, mu, BlockMax):
    nyr = len(BlockMax)
    # Compute the empirical return period:
    empval = empdis(BlockMax, nyr)
    my = [i for i, val in enumerate(empval['Q']) if val in BlockMax]
    PseudoObs = empval.iloc[my]

    # uniforming dimensions of yp, sigma, mu
    try:
        npars = len(sigma) if sigma is np.ndarray else 1
    except:
        raise Exception(f"tsEvaComputeReturnPeriodsGEV: unsupported type for sigma parameter: {type(sigma)}")
    nt = len(BlockMax)
    Bmax = np.tile(BlockMax, npars).reshape(npars, nt)
    sigma_ = np.empty([npars, nt])
    sigma_.fill(sigma)
    mu_ = np.empty([npars, nt])
    mu_.fill(mu)

    # estimating the return levels
    qx = 1 - np.exp(-(1 + epsilon * (Bmax - mu_) / sigma_)**(-1 / epsilon))
    GevPseudo = qx
    returnPeriods = 1 / qx
    return GevPseudo, returnPeriods, PseudoObs

def tsEvaComputeReturnPeriodsGPD(epsilon, sigma, threshold, peaks, nPeaks, peaksID, sampleTimeHorizon):
    X0 = nPeaks / sampleTimeHorizon
    nyr = sampleTimeHorizon

    peaksj = np.random.normal(peaks, 2)  # Jitter function replaced with numpy random normal
    peakAndId = pd.DataFrame({'peaksID': peaksID, 'peaksj': peaksj})
    peaksjid = sorted(peaksID, key=lambda x: peaksj.any())

    # Compute the empirical return period:
    empval = empdis(peaksj, nyr)  # Assuming empdis is a function defined elsewhere
    PseudoObs = empval
    PseudoObs['peaksID'] = peaksjid
    PseudoObs = PseudoObs.sort_values(by='peaksID')

    # Uniform dimensions of variables
    try:
        npars = len(sigma) if sigma is np.ndarray else 1
    except:
        raise Exception(f"tsEvaComputeReturnPeriodsGEV: unsupported type for sigma parameter: {type(sigma)}")    
    nt = len(peaks)
    Peakx = np.tile(PseudoObs['Q'],npars).reshape((npars, nt))
    sigma_ = np.empty([npars, nt])
    sigma_.fill(sigma)

    threshold_ = np.empty([npars, nt])
    threshold_.fill(threshold)

    # Estimate the return levels
    # Here I have all the pseudo observations of every annual extreme for every timestep
    qx = (((1 + epsilon * (Peakx - threshold_) / sigma_)**(-1 / epsilon)))
    GpdPseudo = qx
    GpdPseudo[GpdPseudo < 0] = 0
    returnPeriods = 1 / (X0 * qx)

    return GpdPseudo, returnPeriods, PseudoObs

def tsEvaComputeReturnLevelsGEV(epsilon,sigma,mu,epsilonStdErr,sigmaStdErr,muStdErr,returnPeriodsInDts):
    returnPeriodsInDts=np.matrix(returnPeriodsInDts)
    returnPeriodsInDtsSize = returnPeriodsInDts.shape
    if returnPeriodsInDtsSize[0] > 1:
        returnPeriodsInDts = returnPeriodsInDts.H
    yp = -np.log(1 - (1 / returnPeriodsInDts))
    npars = len(sigma) if sigma is np.ndarray else 1
    nt = len(returnPeriodsInDts) if returnPeriodsInDts is np.ndarray else 1
    yp = np.tile(yp, (npars, 1))
    yp=np.array(yp)
    sigma_ = np.tile(sigma, (nt, 1)).transpose()
    sigmaStdErr_ = np.tile(sigmaStdErr, (nt, 1)).transpose()

    mu_ = np.tile(mu, (nt, 1)).transpose()
    muStdErr_ = np.tile(muStdErr, (nt, 1)).transpose()
    if epsilon != 0:
        returnLevels = np.array(mu_ - ((sigma_ / epsilon) * (1 - (yp ** (-epsilon)))))
        dxm_mu = 1
        dxm_sigma = np.array((1/epsilon) * (1 - (yp ** (-epsilon))))
        dxm_epsilon = np.array(((sigma_ / (epsilon**2)) * (1 - (yp ** (-epsilon)))) - (((sigma_ / epsilon) * np.log(yp)) * (yp ** (-epsilon))))
        returnLevelsErr = np.array(((((dxm_mu * muStdErr_) ** 2) + ((dxm_sigma * sigmaStdErr_) ** 2))+ ((dxm_epsilon * epsilonStdErr) ** 2)) ** 0.5)
    else:
        returnLevels = np.array(mu_ - (sigma_ * np.log(yp)))
        dxm_u = 1
        dxm_sigma = np.array(np.log(yp))
        returnLevelsErr = np.array((((dxm_u * muStdErr_) ** 2) + ((dxm_sigma * sigmaStdErr_) ** 2)) ** 0.5)

    return returnLevels, returnLevelsErr

def tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, timeIndex=-1):

    epsilon = nonStationaryEvaParams['GEVstat']['parameters']['epsilon']
    epsilonStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['epsilonErr']
    epsilonStdErrFit = epsilonStdErr
    epsilonStdErrTransf = 0
    nonStationary = 'sigmaErrTransf' in nonStationaryEvaParams['GEVstat']['paramErr']
    
    if timeIndex > 0:
        sigma = nonStationaryEvaParams['GEVstat']['parameters']['sigma']
        mu = nonStationaryEvaParams['GEVstat']['parameters']['mu']
        sigmaStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErr']
        if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErrFit']
        if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErrTransf']
        muStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['muErr']
        if 'muErrFit' in nonStationaryEvaParams: muStdErrFit = nonStationaryEvaParams['GEVstat']['paramErr']['muErrFit']
        if 'muErrTransf' in nonStationaryEvaParams: muStdErrTransf = nonStationaryEvaParams['GEVstat']['paramErr']['muErrTransf']
    else:
        sigma = nonStationaryEvaParams['GEVstat']['parameters']['sigma']
        mu = nonStationaryEvaParams['GEVstat']['parameters']['mu']
        sigmaStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErr']
        muStdErr = nonStationaryEvaParams['GEVstat']['paramErr']['muErr']
        
        if nonStationary:
            if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErrFit']
            if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams['GEVstat']['paramErr']['sigmaErrTransf']
            if 'muErrFit' in nonStationaryEvaParams: muStdErrFit = nonStationaryEvaParams['GEVstat']['paramErr']['muErrFit']
            if 'muErrTransf' in nonStationaryEvaParams: muStdErrTransf = nonStationaryEvaParams['GEVstat']['paramErr']['muErrTransf']

    returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr, returnPeriodsInYears)
    
    
#    if nonStationary:
#        returnLevels, returnLevelsErrFit = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErrFit, sigmaStdErrFit, muStdErrFit, returnPeriodsInYears)
#        returnLevels, returnLevelsErrTransf = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErrTransf, sigmaStdErrTransf, muStdErrTransf, returnPeriodsInYears)
#    else:
#        returnLevelsErrFit = returnLevelsErr
#        returnLevelsErrTransf = [0] * returnLevelsErr.size
    return returnLevels, returnLevelsErr


def tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, sampleTimeHorizon, returnPeriods):
    X0 = nPeaks / sampleTimeHorizon
    XX = X0 * np.array(returnPeriods)
    if isinstance(sigma, np.ndarray):
        npars = len(sigma)
    else:
        npars = 1

    nt = len(returnPeriods)
    XX_ = np.tile(XX, (npars, 1))
    XXlog_=np.log(XX_)
    sigma_ = np.tile(sigma, (nt, 1)).transpose()
    sigmaStdErr_ = np.tile(sigmaStdErr, (nt, 1)).transpose()
    threshold_ = np.tile(threshold, (nt, 1)).transpose()
    thresholdStdErr_ = np.tile(thresholdStdErr, (nt, 1)).transpose()
    
    if epsilon != 0:
        # estimating the return levels
        returnLevels = threshold_ + sigma_ / epsilon * ((XX_)**epsilon - 1)
        # estimating the error
        # estimating the differential of returnLevels to the parameters
        # !! ASSUMING NON ZERO ERROR ON THE THRESHOLD AND 0 ERROR ON THE PERCENTILE.
        # THIS IS NOT COMPLETELY CORRECT BECAUSE THE PERCENTILE DEPENDS ON THE
        # THRESHOLD AND HAS THEREFORE AN ERROR RELATED TO THAT OF THE
        # THRESHOLD
        dxm_u = 1
        dxm_sigma = 1 / epsilon * ((XX_)**epsilon - 1)
        dxm_epsilon = -sigma_ / epsilon**2 * ((XX_)**epsilon - 1) + sigma_ / epsilon * np.log(XX_) * (XX_)**epsilon
        
        returnLevelsErr = ((dxm_u * thresholdStdErr_)**2 + (dxm_sigma * sigmaStdErr_)**2 + (dxm_epsilon * epsilonStdErr)**2)**0.5
    else:
        returnLevels = threshold_ + sigma_ * np.log(XX_)
        dxm_u = 1
        dxm_sigma = np.log(XX_)
        
        returnLevelsErr = ((dxm_u * thresholdStdErr_)**2 + (dxm_sigma * sigmaStdErr_)**2)**0.5
        
    return returnLevels, returnLevelsErr

def tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, timeIndex=-1):
    epsilon = nonStationaryEvaParams['GPDstat']['parameters']['epsilon']
    epsilonStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['epsilonErr']
    epsilonStdErrFit = epsilonStdErr
    epsilonStdErrTransf = 0
    thStart = nonStationaryEvaParams['GPDstat']['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams['GPDstat']['parameters']['timeHorizonEnd']
    timeHorizonInYears = round((thEnd-thStart)/ 365.2425)
    nPeaks = nonStationaryEvaParams['GPDstat']['parameters']['nPeaks']
    nonStationary = "sigmaErrTransf" in nonStationaryEvaParams['GPDstat']['paramErr']
    if timeIndex > 0:
        sigma = nonStationaryEvaParams['GPDstat']['parameters']['sigma']
        threshold = nonStationaryEvaParams['GPDstat']['parameters']['threshold']
        sigmaStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErr']
        if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErrFit']
        if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErrTransf']
        thresholdStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErr']
        thresholdStdErrFit = 0
        if 'thresholdErrTransf' in nonStationaryEvaParams: thresholdStdErrTransf = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErrTransf']
    else:
        sigma = nonStationaryEvaParams['GPDstat']['parameters']['sigma']
        threshold = nonStationaryEvaParams['GPDstat']['parameters']['threshold']
        sigmaStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErr']
        if nonStationary:
            thresholdStdErr = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErr']
            
            if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErrFit']
            if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams['GPDstat']['paramErr']['sigmaErrTransf']
            if 'thresholdErrFit' in nonStationaryEvaParams: thresholdStdErrFit = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErrFit']
            if 'thresholdErrTransf' in nonStationaryEvaParams: thresholdStdErrTransf = nonStationaryEvaParams['GPDstat']['paramErr']['thresholdErrTransf']
        else:
            thresholdStdErr = 0
            
    returnLevels,returnLevelsErr = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, timeHorizonInYears, returnPeriodsInYears)

#    if nonStationary:
#        returnLevelsErrFit = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErrFit, sigmaStdErrFit, [0]*len(thresholdStdErr), nPeaks, timeHorizonInYears, returnPeriodsInYears)
#        returnLevelsErrTransf = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErrTransf, sigmaStdErrTransf, thresholdStdErrTransf, nPeaks, timeHorizonInYears, returnPeriodsInYears)
#    else:
#        returnLevelsErrFit = returnLevelsErr
#        returnLevelsErrTransf = [0]*returnLevelsErr.size
 
    return returnLevels, returnLevelsErr#, returnLevelsErrFit,returnLevelsErrTransf

def tsEvaComputeRLsGEVGPD(nonStationaryEvaParams, RPgoal, timeIndex, trans=None):

    # GEV
    epsilonGEV = nonStationaryEvaParams[0]['parameters']['epsilonGEV']
    sigmaGEV = np.mean(nonStationaryEvaParams[0]['parameters']['sigmaGEV'][timeIndex])
    muGEV = np.mean(nonStationaryEvaParams[0]['parameters']['muGEV'][timeIndex])
    dtSampleYears = nonStationaryEvaParams[0]['parameters']['timeDeltaYears']

    # GPD
    epsilonGPD = -nonStationaryEvaParams[1]['parameters']['epsilonGPD']
    sigmaGPD = np.mean(nonStationaryEvaParams[1]['parameters']['sigmaGPD'][timeIndex])
    thresholdGPD = np.mean(nonStationaryEvaParams[1]['parameters']['thresholdGPD'][timeIndex])
    nPeaks = nonStationaryEvaParams[1]['parameters']['nPeaks']
    thStart = nonStationaryEvaParams[1]['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams[1]['parameters']['timeHorizonEnd']
    sampleTimeHorizon = round((thEnd - thStart).dt.days / 365.2425)

    if nonStationaryEvaParams[0]['method'] == "No fit":
        print("Could not fit EVD to this pixel")
        ParamGEV = np.array([epsilonGEV, sigmaGEV, muGEV, None, None, None])
        ParamGPD = np.array([epsilonGPD, sigmaGPD, thresholdGPD, None, None, None, nPeaks, sampleTimeHorizon])
        return {'Fit': 'No fit', 'Params': [ParamGEV, ParamGPD]}
    else:
        epsilonStdErrGEV = nonStationaryEvaParams[0]['paramErr']['epsilonGEVErr']
        sigmaStdErrGEV = np.mean(nonStationaryEvaParams[0]['paramErr']['sigmaGEVErr'][timeIndex])
        muStdErrGEV = np.mean(nonStationaryEvaParams[0]['paramErr']['muGEVErr'][timeIndex])

        epsilonStdErrGPD = nonStationaryEvaParams[1]['paramErr']['epsilonGPDErr']
        sigmaStdErrGPD = np.mean(nonStationaryEvaParams[1]['paramErr']['sigmaGPDErr'][timeIndex])
        thresholdStdErrGPD = np.mean(nonStationaryEvaParams[1]['paramErr']['thresholdGPDErr'][timeIndex])

        if trans == "rev":
            sigmaGEV = -sigmaGEV
            muGEV = -muGEV
            sigmaGPD = -sigmaGPD
            thresholdGPD = -thresholdGPD

        returnLevelsGEV,returnLevelsGEVErr = tsEvaComputeReturnLevelsGEV(epsilonGEV, sigmaGEV, muGEV, epsilonStdErrGEV, sigmaStdErrGEV, muStdErrGEV, RPgoal)
        returnLevelsGPD,returnLevelsGPDErr = tsEvaComputeReturnLevelsGPD(epsilonGPD, sigmaGPD, thresholdGPD, epsilonStdErrGPD, sigmaStdErrGPD, thresholdStdErrGPD, nPeaks, sampleTimeHorizon, RPgoal)

        rlevGEV = returnLevelsGEV
        rlevGPD = returnLevelsGPD
        errGEV = returnLevelsGEVErr
        errGPD = returnLevelsGPDErr

        ParamGEV = [epsilonGEV, sigmaGEV, muGEV, epsilonStdErrGEV, sigmaStdErrGEV, muStdErrGEV]
        ParamGPD = [epsilonGPD, sigmaGPD, thresholdGPD, epsilonStdErrGPD, sigmaStdErrGPD, thresholdStdErrGPD, nPeaks, sampleTimeHorizon]

        return 'Fitted',RPgoal,rlevGEV,rlevGPD,errGEV,errGPD, ParamGEV, ParamGPD

def tsTimeSeriesToPointData(ms, pot_threshold, pot_threshold_error):
    # Filter the time series by the threshold
    ms1 = ms[ms[:, 1] > pot_threshold]
    percentile = (1 - ms1.shape[0] / ms.shape[0]) * 100

    pointData = {}
    pointData['completeSeries'] = ms1

    pot_data = {
        'threshold': pot_threshold,
        'thresholdError': pot_threshold_error,
        'percentile': percentile,
        'peaks': ms1[:, 1],
        'ipeaks': np.arange(1, ms1.shape[0] + 1),
        'sdpeaks': ms1[:, 0]
    }
    pointData['POT'] = pot_data

    # These need to be implemented or imported
    ret_annual = tsEvaComputeAnnualMaxima(ms1)

    pointData['annualMax']=ret_annual['annualMax']
    pointData['annualMaxTimeStamp']=ret_annual['annualMaxDate']
    pointData['annualMaxIndexes']=ret_annual['annualMaxIndx']

    ret_monthly = tsEvaComputeMonthlyMaxima(ms1)

    pointData['monthlyMax']=ret_monthly['monthlyMax']
    pointData['monthlyMaxTimeStamp']=ret_monthly['monthlyMaxDate']
    pointData['monthlyMaxIndx']=ret_monthly['monthlyMaxIndx']
    
    # Compute the years range
    datetime_series_with_offset = pd.to_datetime(np.array(ms1[:, 0]) - 719529, unit='D', origin='unix')  + pd.Timedelta(hours=1)
    yrs = np.unique(datetime_series_with_offset.year)
    yrs = yrs - np.min(yrs)
    pointData['years'] = np.arange(np.nanmin(yrs), np.nanmax(yrs) + 1)

    pointData['Percentiles'] = [0]

    return pointData

def tsEvaSampleData(ms, meanEventsPerYear, minEventsPerYear, minPeakDistanceInDays, tail=None):
    pctsDesired = [90, 95, 99, 99.9]
    args = {'meanEventsPerYear': meanEventsPerYear,
            'minEventsPerYear': minEventsPerYear,
            'potPercentiles': [50, 70] + list(range(85, 98, 2))}
    meanEventsPerYear = args['meanEventsPerYear']
    minEventsPerYear = args['minEventsPerYear']
    potPercentiles = args['potPercentiles']

    if tail is None:
        raise ValueError("tail for POT selection needs to be 'high' or 'low'")

    POTData = tsGetPOT(ms, potPercentiles, meanEventsPerYear, minEventsPerYear, minPeakDistanceInDays, tail)

    vals = np.nanquantile(ms[:, 1], [x / 100 for x in pctsDesired])
    
    percentiles = {'percentiles': pctsDesired, 'values': vals}

    pointData = {}
    pointData['completeSeries'] = ms
    pointData['POT'] = POTData
    pointDataA = tsEvaComputeAnnualMaxima(ms)
    pointDataM = tsEvaComputeMonthlyMaxima(ms)

    datetime_series_with_offset = pd.to_datetime(np.array(ms[:, 0]) - 719529, unit='D', origin='unix')  + pd.Timedelta(hours=1)
    yrs = np.unique(datetime_series_with_offset.year)
    yrs=yrs-np.min(yrs)
    pointData['years'] = list(range(min(yrs), max(yrs) + 1))
    pointData['Percentiles'] = percentiles
    pointData['annualMax'] = pointDataA['annualMax']
    pointData['annualMaxDate'] = pointDataA['annualMaxDate']
    pointData['annualMaxIndx'] = pointDataA['annualMaxIndx']
    pointData['monthlyMax'] = pointDataM['monthlyMax']
    pointData['monthlyMaxDate'] = pointDataM['monthlyMaxDate']
    pointData['monthlyMaxIndx'] = pointDataM['monthlyMaxIndx']

    return pointData

def tsGetPOT(ms, pcts, desiredEventsPerYear, minEventsPerYear, minPeakDistanceInDays, tail):
    if minPeakDistanceInDays == -1:
        raise ValueError("label parameter 'minPeakDistanceInDays' must be set")

    ms[np.isnan(ms[:, 1]), 1] = 0
    dt1 = tsEvaGetTimeStep(ms[:, 0])
    dt = dt1.astype(float)
    
    minPeakDistance = minPeakDistanceInDays / dt
     
    nyears = round((np.max(ms[:, 0]) - np.min(ms[:, 0])) / 365.25)
        
    if len(pcts) == 1:
        pcts = [pcts - 3, pcts]
        desiredEventsPerYear = -1
    
    numperyear = np.empty(len(pcts))
    minnumperyear = np.empty(len(pcts))
    thrsdts = np.empty(len(pcts))
    gpp = np.empty(len(pcts))
    gpp[:]=np.nan
    devpp = np.empty(len(pcts))
    devpp[:]=np.nan
    trip = None
    perfpen = 0

    for ipp in range(len(pcts)):
        thrsdt = np.percentile(ms[:, 1], pcts[ipp])
        thrsdts[ipp] = thrsdt
        ms[np.isnan(ms[:, 1])] = -9999
        minEventsPerYear = 1

        if tail == "high":
            shape_bnd = [-0.5, 1]
            locs,pks = find_peaks(ms[:, 1], height=thrsdt,distance=minPeakDistance)
            
        if tail == "low":
            shape_bnd = [-1.5, 0]
            locs,pks = declustpeaks(data = ms[:, 1] ,minpeakdistance = minPeakDistance ,minrundistance = minRunDistance, qt=thrsdt)

        peaks=pks['peak_heights']

        numperyear[ipp] = len(peaks) / nyears
        
        nperYear=tsGetNumberPerYear(ms,locs);
        minnumperyear[ipp]=np.nanmin(nperYear);

        if ipp>0 and numperyear[ipp] < minEventsPerYear and minnumperyear[ipp]<minEventsPerYear:
            break
        
    diffNPerYear = np.mean(np.diff(np.nan_to_num(numperyear[::-1])))
    
    if diffNPerYear == 0:
        diffNPerYear = 1
    
    thresholdError = np.mean(np.diff(np.nan_to_num(thrsdts))) / diffNPerYear / 2
    indexp = ipp

    if indexp is not None:
        thrsd = np.quantile(ms[:, 1], pcts[indexp] / 100)
        pct = pcts[indexp]
    else:
        thrsd = 0
        pct = None

    if tail == "high":
        locs,pks = find_peaks(ms[:, 1], distance=minPeakDistance, height=thrsdt)
    if tail == "low":
        locs,pks = declustpeaks(data=ms[:, 1], minpeakdistance=minPeakDistance, minrundistance=minRunDistance, qt=thrsd)
    peaks=pks['peak_heights']
    
    ms_Q=ms[:, 1]
    ms_time=ms[:, 0]+3600
    peaks = np.array(ms_Q[locs])
    st = locs[0]
    end = locs[len(locs)-1]

    POTdata = {
        'threshold': thrsd,
        'thresholdError': thresholdError,
        'percentile': pct,
        'peaks': peaks,
        'stpeaks': st,
        'endpeaks': end,
        'ipeaks': locs,
        'time': np.array(ms_time[locs]),
    }

    return POTdata

def tsEvaGetTimeStep(times):
    df = np.diff(times)  # Compute the differences between consecutive times
    dt = np.min(df)      # Find the minimum difference
    if dt == 0:
        df = df[df != 0]  # Remove zeros from the differences
        dt = np.min(df)   # Find the minimum difference again
    return dt

def tsEvaStationary(time_and_series, **kwargs):
    # Default arguments
    args = {
        'minPeakDistanceInDays': -1,
        'potEventsPerYear': 5,
        'gevMaxima': 'annual',
        'gevType': 'GEV',  # can be 'GEV' or 'Gumbel'
        'doSampleData': True,
        'potThreshold': np.nan,
        'evdType': ['GEV', 'GPD']
    }

    
    # Parse the named arguments (you can replace with your argument parser)
    for key, value in kwargs.items():
        if key in args:
            args[key] = value

    tail="high"
    minEventsPerYear=1
    minPeakDistanceInDays = args['minPeakDistanceInDays']
    if minPeakDistanceInDays == -1:
        raise ValueError("label parameter 'minPeakDistanceInDays' must be set")
    
    potEventsPerYear = args['potEventsPerYear']
    gevMaxima = args['gevMaxima']
    gevType = args['gevType']
    doSampleData = args['doSampleData']
    evdType = args['evdType']
    potThreshold = args['potThreshold']

    stationaryEvaParams = []
    
    print("Executing stationary EVA...")
    
    # Data sampling
    if doSampleData:
        # Removing NaN values
        time_and_series = time_and_series[~np.isnan(time_and_series[:, 1])]
                
        # Replace tsEvaSampleData with a custom data sampling function
        pointData = tsEvaSampleData(time_and_series, potEventsPerYear,minEventsPerYear,minPeakDistanceInDays,tail)
    else:
        if np.isnan(potThreshold):
            raise ValueError("If doSampleData==False, you need to provide a value for the potThreshold.")
        pointData = tsTimeSeriesToPointData(time_and_series, potThreshold, 0)
    
    # Call to tsEVstatistics for GEV fitting
    evaAlphaCI = 0.68  # Approximation of 68% confidence interval
    EVmeta, EVdata, isValid = tsEVstatistics(pointData, alphaCI=evaAlphaCI, gevMaxima=gevMaxima, gevType=gevType, evdType=evdType)
    if not isValid:
        return None, False
    
    # GEV Parameters and Errors
    if ('GEV' in evdType):
        epsilonGevX = -EVdata['GEVstat']['parameters']['xi']
        errEpsilonX = -epsilonGevX - EVdata['GEVstat']['paramCIs']['xici'][0]
        sigmaGevX = EVdata['GEVstat']['parameters']['sigma']
        errSigmaGevX = sigmaGevX - EVdata['GEVstat']['paramCIs']['sigci'][0]
        muGevX = EVdata['GEVstat']['parameters']['mu']
        errMuGevX = muGevX - EVdata['GEVstat']['paramCIs']['muci'][0]
            
        gevParams = {
            'epsilon': epsilonGevX,
            'sigma': sigmaGevX,
            'mu': muGevX,
            'timeDelta': 365.25 if gevMaxima == 'annual' else 365.25 / 12,
            'timeDeltaYears': 1 if gevMaxima == 'annual' else 1 / 12
        }
        gevParamStdErr = {
            'epsilonErr': errEpsilonX,
            'sigmaErr': errSigmaGevX,
            'muErr': errMuGevX
        }
        
        gevObj = {
            'method': EVdata['GEVstat']['method'],
            'parameters': gevParams,
            'paramErr': gevParamStdErr,
            'objs': {'monthlyMaxIndexes': pointData['monthlyMaxIndx']}
        }
    if ('GPD' in evdType):
        # Estimating the non-stationary GPD parameters
        epsilonPotX = EVdata['GPDstat']['parameters']['shape']
        errEpsilonPotX = epsilonPotX - EVdata['GPDstat']['paramCIs'][0][0]
        
        sigmaPotX = EVdata['GPDstat']['parameters']['sigma']
        errSigmaPotX = sigmaPotX - EVdata['GPDstat']['paramCIs'][2][0]
        thresholdPotX = EVdata['GPDstat']['parameters']['threshold']
        errThresholdPotX = pointData['POT']['thresholdError']
        percentilePotX = EVdata['GPDstat']['parameters']['percentile']
        nPotPeaks = EVdata['GPDstat']['parameters']['peaks']

        
        timeStamps = time_and_series[:, 0]
        dt = tsEvaGetTimeStep(timeStamps)
        minPeakDistance = minPeakDistanceInDays / dt
        dtSample = (timeStamps[-1] - timeStamps[0]) / len(timeStamps)
        dtPotX = max(dtSample, dtSample * minPeakDistance)

        potParams = {
            'epsilon': epsilonPotX,
            'sigma': sigmaPotX,
            'threshold': thresholdPotX,
            'percentile': percentilePotX,
            'timeDelta': dtPotX,
            'timeDeltaYears': dtPotX / 365.2425,
            'timeHorizonStart': np.min(timeStamps),
            'timeHorizonEnd': np.max(timeStamps),
            'nPeaks': nPotPeaks
        }

        potParamStdErr = {
            'epsilonErr': errEpsilonPotX,
            'sigmaErr': errSigmaPotX,
            'thresholdErr': errThresholdPotX
        }
        
        potObj = {
            'method': EVdata['GPDstat']['method'],
            'parameters': potParams,
            'paramErr': potParamStdErr,
            'objs': []
        }
        
    stationaryEvaParams={'GEVstat':gevObj,'GPDstat':potObj}
        
    return stationaryEvaParams

def log_likelihood(params, data):
    shape, loc, scale = params
    return np.sum(genpareto.logpdf(data, shape, loc, scale))

#def tsEVstatistics(pointData, alphaCI=0.95, gevMaxima='annual', gevType='GEV', evdType=['GEV', 'GPD'], shape_bnd=[-0.5, 1]):
def tsEVstatistics(pointData, alphaCI=0.95, gevMaxima='annual', gevType='GEV', evdType=['GEV', 'GPD'], shape_bnd=[-0.5, 1]):
    
    # Create empty data structures
    EVmeta = {}
    EVdata = {}
    isValid = True

    minvars=1
    minGEVSample = 10
    if alphaCI is None:
        alphaCI = 0.95
    if gevMaxima is None:
        gevMaxima = 'annual'
    if gevType is None:
        gevType = 'GEV'
    if evdType is None:
        evdType = ['GEV', 'GPD']

    # Define Tr vector
    Tr = [5, 10, 20, 50, 100, 200, 500, 1000]
    EVmeta['Tr'] = Tr
    nyears = len(pointData['annualMax'])

    imethod = 1
    methodname = 'GEVstat'
    paramEsts = np.zeros(3)
    paramCIs = np.full((2, 3), np.nan)
    rlvls = np.zeros(len(Tr))

    # GEV statistics
    if (('GEV' in evdType) and pointData['annualMax'] is not None):
        if gevMaxima == 'annual':
            tmpmat = np.array(pointData['annualMax'])
        elif gevMaxima == 'monthly':
            tmpmat = np.array(pointData['monthlyMax'])
        else:
            raise ValueError(f'Invalid gevMaxima type: {gevMaxima}')

        iIN = ~np.isnan(tmpmat)
        
        if sum(iIN) >= minGEVSample:
            tmp = tmpmat[iIN]
            # Perform GEV/Gumbel fitting and computation of return levels
            if gevType == "GEV":
                stdfit = True
                # Try to fit GEV with bounded shape parameters and stderr, reduces the constraints if no fit.
                try:
#                    fit  = gev.fit(tmp, method="MLE",loc=np.mean(tmp),scale=np.std(tmp))
                    gev_instance = Bootstrap_fit(tmp)
#                    params,gev_confidence_interval = gev_instance.fit_genextreme()
                    params,paramCL = gev_instance.fit_genextreme()
                    paramEsts = {'xi': params[0], 'mu': params[1], 'sigma': params[2]}
                    paramCIs = {'xici': paramCL[0],'muci': paramCL[1], 'sigci': paramCL[2]}

                    alphaCIx = 1 - alphaCI
                    
                except Exception as e:
                    stdfit = False
                    print("Not able to fit GEV stderr")
                        
            elif gevType == "Gumbel":
                gumbel_instance = Bootstrap_fit(tmp)
                params,paramCL = gumbel_instance.fit_gumbel()

                paramEsts = {'xi': 0, 'mu': params[0], 'sigma': params[1]}
                paramCIs = {'xici': [0,0],'muci': paramCL[0], 'sigci': paramCL[1]}
                alphaCIx = 1 - alphaCI
                
            else:    
                print("tsEVstatistics: invalid gevType ",gevType," Can be only GEV or Gumbel")
                methodname = 'No fit'
                rlvls = None
                paramEsts = None
                paramCIs = None
                
    Tr_inv=  [1 - 1 / x for x in Tr]  
    if gevType == "GEV":
        rlvls = gev.ppf(Tr_inv, paramEsts['xi'], loc=paramEsts['mu'], scale=paramEsts['sigma'])
    if gevType == "Gumbel":
        rlvls = gumbel_r.ppf(Tr_inv, loc=paramEsts['mu'], scale=paramEsts['sigma'])

    EVdata['GEVstat'] = {
        'method': methodname,
        'values': rlvls,
        'parameters': paramEsts,
        'paramCIs': paramCIs
    }
    
    # Create output structures for GEV statistics
    # GPD statistics
    imethod = 2
    methodname = 'GPDstat'
    paramEsts = np.zeros(6)
    paramCIs = np.full((2, 3), np.nan)
    rlvls = np.zeros(len(Tr))
    if ('GPD' in evdType) and (pointData['annualMax'] is not None):
        # Perform GPD fitting and computation of return levels
        ik = 1
        d1 = pointData['POT']['peaks']-pointData['POT']['threshold']
        gpd_instance = Bootstrap_fit(d1)
        paramEsts,paramCIs,se = gpd_instance.fit_genpareto()
        alphaCIx = 1 - alphaCI

        ksi = paramEsts[0] # shape
        sgm = paramEsts[2] # scale
        
        # FINO QUI
        if (ksi < -.5):
            probs = [alphaCIx / 2, 1 - alphaCIx / 2]
            
            kci = norm.ppf(probs, ksi, se[2])
            kci[kci < -1] = -1
        
            # Compute the CI for sigma using a normal approximation for log(sigmahat)
            # and transform back to the original scale.
            lnsigci = norm.ppf(probs, np.log(sgm), se[2] / sgm)
            paramCIs = [kci, np.exp(lnsigci)]
        
        # Create output structures for GPD statistics
        # Assign values to paramEstsall
        paramEstsall = {'sigma':sgm, 'shape': ksi, 'threshold': pointData['POT']['threshold'], 'length':len(d1), 'peaks':len(pointData['POT']['peaks']), 'percentile':pointData['POT']['percentile']}
        
        # Assign values to rlvls
        for i in range(len(Tr)):
            rlvls[i] = pointData['POT']['threshold'] + (sgm/ksi) * ((((len(d1)/len(pointData['POT']['peaks']))*(1/Tr[i]))**(-ksi))-1)

        EVdata['GPDstat'] = {
            'method': methodname,
            'values': rlvls,
            'parameters': paramEstsall,
            'paramCIs': paramCIs
        }
    else:
        methodname = 'No fit'
        ik = 1
        th = pointData['POT']['threshold']
        d1 = pointData['POT']['peaks']
        paramEstsall = [pointData['POT']['pars'][0], pointData['POT']['pars'][1],
                        pointData['POT']['threshold'], len(d1),
                        len(pointData['POT']['peaks']), pointData['POT']['percentile']]
        EVdata['GPDstat'] = {
            'method': methodname,
            'values': None,
            'parameters': paramEstsall,
            'paramCIs': None
        }
        print("could not estimate GPD: bounded distribution")
        
        # Return outputs
    return EVmeta, EVdata, isValid

def tsGetNumberPerYear(ms, locs):
    
    # Get all years
    sdfull = np.arange(np.nanmin(ms[:, 0]), np.nanmax(ms[:, 0]))
 
    # Make full time vector
    sdfull2 = sdfull - min(sdfull)

    # Round to years
    sdfull2_dt = pd.to_datetime(sdfull2, utc=True, unit='D', origin='unix')
    yearss = np.unique(sdfull2_dt.year-1970)
    
    sdser=ms[:,0] - min(ms[:,0])
    sdser_dt = pd.to_datetime(sdser, utc=True, unit='D', origin='unix')
    yearsser, icmser = np.unique(sdser_dt.year-1970, return_inverse=True)

    sdp = ms[locs, 0]

    sdp_dt = pd.to_datetime(sdp-min(sdfull), utc=True, unit='D', origin='unix')
    yearssp, icm = np.unique(sdp_dt.year-1970, return_inverse=True)

    # Prepare the series time vector
    df1 = pd.DataFrame({'icm': icm, 'sdp': sdp})
    
    ecoount = df1.groupby('icm').size().values
    
    df2 = pd.DataFrame({'icmser': icmser, 'sdser': sdser})
    ecoountSeries = df2.groupby('icmser').size().values
    
    iexcl=ecoountSeries/max(ecoountSeries)<0.5;
    
    nperYear = np.full(len(yearss), np.nan)
    nperYear[yearsser]=0;
    nperYear[yearssp]=ecoount;
    for i in range(len(iexcl)):
        if iexcl[i]==True:
            nperYear[yearsser[i]]=np.nan;
    
    return nperYear

def tsEvaComputeMonthlyMaxima(time_and_series):
    time_stamps = time_and_series[:, 0]
    tmvec = pd.to_datetime(np.array(time_stamps) - 719529, unit='D', origin='unix')
    srs = time_and_series[:, 1]

    yrs = tmvec.year
    mnts = tmvec.month
    mnttmvec = pd.DataFrame({'yrs': yrs, 'mnts': mnts})
    vals_indxs = np.arange(0, len(srs))
    
    monthly_max_indx = mnttmvec.groupby(['yrs', 'mnts']).apply(lambda x: find_max(x.index, srs)).reset_index(name='valsIndxs')
    monthly_max_indx['valsIndxs'] = monthly_max_indx['valsIndxs'].astype(int)
    monthly_max_indx = monthly_max_indx.sort_values(by=['yrs', 'mnts'])['valsIndxs']
    monthly_max = srs[monthly_max_indx]
    monthly_max_date = time_stamps[monthly_max_indx]

    ret=pd.DataFrame({'monthlyMax': monthly_max, 'monthlyMaxDate': monthly_max_date, 'monthlyMaxIndx': monthly_max_indx})
    return ret

def tsEvaComputeAnnualMaxima(time_and_series):
    time_stamps = time_and_series[:, 0]
    tmvec = pd.to_datetime(np.array(time_stamps) - 719529, unit='D', origin='unix')
    srs = time_and_series[:, 1]
    years = tmvec.year
    srs_indices = range(0, len(srs))
    unique_years = np.unique(years)
    df = pd.DataFrame({'years': years, 'srs_indices': srs_indices, 'srs': srs})
    annual_max_indx = df.groupby('years').apply(lambda group: find_max(group['srs_indices'].values, df['srs'].values))
    annual_max = [srs[i] for i in annual_max_indx]
    annual_max_date = time_stamps[annual_max_indx]

    ret=pd.DataFrame({'annualMax': annual_max, 'annualMaxDate': annual_max_date, 'annualMaxIndx': annual_max_indx})
    return ret

def find_max(indices, srs):
    if len(indices) == 0:
        return None
    return indices[np.argmax(srs[indices])] 



def declust(data, threshold, minrundistance):
    # Identify the peaks that exceed the threshold
    data=np.array(data)
    peaks = data > threshold
    n = len(data)
    cnr=0
    en=0
    ien=0
    thExceedances = []
    clusters = []
    sizes = []
    cluster_peaks = []
    clusterMaxima = []
    isClusterMax = []
    exceedanceTimes = []
    InterExceedTimes = []
    InterCluster = []
    intcl = []
    
    last_peak = 0  # Initialize to ensure the first peak is included
    for i in range(n):
        if (i >= minrundistance-1):
            ien = ien + 1
        if peaks[i]:
            if (en > 0):
                InterExceedTimes.append(ien)
                ien=0
            thExceedances.append(data[i])
            exceedanceTimes.append(i+1)
            if (i - last_peak > minrundistance):
                sizes.append(en)
                cnr = cnr + 1
                en=0
            last_peak = i
            en = en + 1
            clusters.append(cnr+1);
    cnr = cnr+1

    n=len(clusters)-1
    for i in range(n):
        cluster_peaks.append(thExceedances[i])
        if clusters[i+1] > clusters[i]:
            clusterMaxima.append(max(cluster_peaks))
            cluster_peaks=[]
            i=i-1
        elif i==n-1:
            cluster_peaks.append(thExceedances[i+1])
            clusterMaxima.append(max(cluster_peaks))
            
    j=0
    for i in range(len(thExceedances)-1):
        if (InterExceedTimes[i]>minrundistance):
            InterCluster.append(True)
        else:
            InterCluster.append(False)
        if (j<len(clusterMaxima)):
            if (thExceedances[i]==clusterMaxima[j]):
                isClusterMax.append(True)
                j = j+1
            else:
                isClusterMax.append(False)
        else:
            isClusterMax.append(False)
    isClusterMax.append(False)

    peakev=pd.DataFrame({
        'thExceedances': pd.Series(thExceedances),
        'clusters': pd.Series(clusters),
        'sizes': pd.Series(sizes),
        'clusterMaxima': pd.Series(clusterMaxima),
        'isClusterMax': pd.Series(isClusterMax),
        'exceedanceTimes': pd.Series(exceedanceTimes),
        'InterExceedTimes': pd.Series(InterExceedTimes),
        'InterCluster': pd.Series(InterCluster)
    })
    
    return peakev

def declustpeaks(data, minpeakdistance=10, minrundistance=7, qt=None):
     pks, _ = find_peaks(data, distance=minpeakdistance, height=qt)
     peakev = declust(data, threshold=qt, minrundistance=minrundistance)
     
     intcl = [True] + peakev['InterCluster'].to_list()
     intcl.pop()
     
    
     peakex = pd.DataFrame({
          'Qs': peakev['thExceedances'],
          'Istart': intcl,
          'clusters': peakev['clusters'],
          'IsClustermax': peakev['isClusterMax'],
          'exceedances': peakev['exceedanceTimes']
     })
     
     evmax = peakex[peakex['IsClustermax']==True]
     ziz = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.max() - x.min() + 1).reset_index()
     st = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.min()).reset_index()
     end = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.max()).reset_index()    
     evmax['dur']= pd.Series(ziz['exceedances'].values,index=evmax.index)
     evmax['durx']= pd.Series(peakev['sizes'].dropna().values,index=evmax.index)
     
     peakdt = pd.DataFrame({
          'Q': evmax['Qs'],
          'max': evmax['exceedances'],
          'cluster': evmax['clusters']
     })
     peakdt['start']=pd.Series(st['exceedances'].values,index=peakdt.index)
     peakdt['end']=pd.Series(end['exceedances'].values,index=peakdt.index)
     peakdt['dur']=pd.Series(ziz['exceedances'].values,index=peakdt.index)
     peakdt = peakdt.sort_values(by=['Q'], ascending=False)
     return peakdt


def empdis(x, nyr):
    ts = list(range(1, len(x) + 1))
    ts = pd.DataFrame(ts, columns=['ts']).sort_values(by=0,axis=1).reset_index(drop=True)
    x = pd.DataFrame(x, columns=['x']).sort_values(by=0,axis=1).reset_index(drop=True)
    
    epyp = len(x) / nyr
    rank = x.rank(axis=0, method='min', na_option='keep')
    hazen = (rank - 0.5) / len(x)
    cunnane = (rank - 0.4) / (len(x) + 0.2)
    gumbel = -np.log(-np.log(hazen))
    n = len(x)
    rpc = (1 / (1 - np.arange(1, n + 1) / (n + 1))) / 12
    nasmp = x.apply(lambda col: col.count(), axis=0)
    epdp = rank / np.repeat(nasmp+1,rank.shape[1])
    empip = pd.DataFrame({
        'emp.RP': (1 / (epyp * (1 - epdp))).to_numpy().flatten(),
        'haz.RP': (1 / (epyp * (1 - hazen))).to_numpy().flatten(),
        'cun.RP': (1 / (epyp * (1 - cunnane))).to_numpy().flatten(),
        'gumbel': (gumbel).to_numpy().flatten(),
        'emp.f': (epdp).to_numpy().flatten(),
        'emp.hazen': (hazen).to_numpy().flatten(),
        'emp.cunnane': (cunnane).to_numpy().flatten(),
        'Q': (x['x']).to_numpy().flatten(),
        'timestamp': (ts['ts']).to_numpy().flatten()
    })
    return empip

def empdisl(x, nyr):
    # Convert the input to a pandas DataFrame and sort it in decreasing order
    x = pd.DataFrame(np.sort(x, axis=0)[::-1])
    
    # Calculate the event per year (epyp)
    epyp = len(x) / nyr
    
    # Calculate rank
    rank = x.rank(axis=0, method='min', na_option='keep', ascending=False)
    
    # Calculate Hazen
    hazen = (rank - 0.5) / len(x)
    
    # Calculate Gumbel
    gumbel = -np.log(-np.log(hazen))
    
    # Calculate number of non-missing values (nasmp)
    nasmp = np.sum(~np.isnan(x.values), axis=0)
    
    # Calculate Empirical Distribution Probability (epdp)
    epdp = rank / (nasmp + 1)
    
    # Create the empip DataFrame with calculated columns
    empip = pd.DataFrame({
        "emp.RP": 1 / (epyp * (1 - epdp)).to_numpy().flatten(),
        "haz.RP": 1 / (epyp * (1 - hazen)).to_numpy().flatten(),
        "gumbel": gumbel.to_numpy().flatten(),
        "emp.f": epdp.to_numpy().flatten(),
        "emp.hazen": hazen.to_numpy().flatten(),
        "Q": x.to_numpy().flatten() 
    })
    
    return empip
