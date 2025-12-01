import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import scipy.io
import os
from eva_functions import tsEvaNonStationary
from eva_functions import tsEvaPlotSeriesTrendStdDevFromAnalysisObj
from eva_functions import tsEvaPlotGEVImageScFromAnalysisObj
from eva_functions import tsEvaPlotGPDImageScFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGPDFromAnalysisObj
from eva_functions import tsEvaPlotTransfToStatFromAnalysisObj
from eva_functions import tsEvaPlotGEV3DFromAnalysisObj

def datetime_to_datenum(dt):
    # MATLAB datenum = Python ordinal + fractional day + 366
    ord_num = dt.toordinal()
    frac_day = (dt - datetime(dt.year, dt.month, dt.day)).total_seconds() / 86400
    return ord_num + frac_day + 366

# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeriesHebrides.csv"
#data = pd.read_csv('/home/quadrani/tmp/Lorenzo/PytsEva-master-dev/test/data/timeAndSeriesHebrides.csv', header=None)
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
extremesRange = [0.2, 1.2]
seasonalExtrRange = [0.1, 1.1]
seriesDescr = 'Hebrides'

timeWindow = 365.25 * 6  # 6 years
minPeakDistanceInDays = 3

minTS = np.min(timeAndSeries[:, 0])
maxTS = np.max(timeAndSeries[:, 0])
axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 28
titleFontSize = 30

# Preparing xticks
years = np.arange(1980, 2016, 2)
months = np.ones_like(years)
days = np.ones_like(years)
dtns = np.column_stack((years, months, days))
dts = [datetime(int(y), int(m), int(d)) for y, m, d in zip(years, months, days)]
tickTmStmp = [datetime_to_datenum(dt) for dt in dts]

wr = np.linspace(min(extremesRange), max(extremesRange), 1501)

print('trend only statistics (transformation + eva + backtransformation)')
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trend', minPeakDistanceInDays=minPeakDistanceInDays)

print('  plotting the series')
hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel='Lvl (m)', title=seriesDescr, titleFontSize=titleFontSize, dateformat='%y', xtick=tickTmStmp)
# print('  saving the series plot')
# plt.savefig('seriesTrendOnly_py.png')
# plt.show()

# Uncomment the following lines if needed
#print('  plotting and saving the 3D GEV graph')
#hndl = tsEvaPlotGEV3DFromAnalysisObj(wr, nonStatEvaParams, statTransfData, xlabel='Lvl (m)', axisfontsize=axisFontSize3d)
#plt.title('GEV 3D', fontsize=titleFontSize)
#plt.savefig('GEV3DTrendOnly.png')

print('  plotting and saving the 2D GEV graph')

hndl = tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp)
plt.savefig('GEV2DTrendOnly_py.png')
plt.title('GEV', fontsize=titleFontSize)
plt.show()

print('  plotting and saving the 2D GPD graph')
hndl = tsEvaPlotGPDImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp)
plt.title('GPD', fontsize=titleFontSize)
plt.savefig('GPD2DTrendOnly_py.png')
plt.show()

# Computing and plotting the return levels for a given time
timeIndex = 1000
timeStamps = statTransfData.timeStamps
print(f'  plotting return levels for time {datetime.fromtimestamp(timeStamps[timeIndex])}')
print('  ... for GEV the sample is small and the confidence interval is broad')
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, timeIndex, ylim=[0.5, 1.5])
plt.savefig('GEV_ReturnLevels_py.png')
plt.show()
hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(nonStatEvaParams, timeIndex, ylim=[0.5, 1.5])
plt.savefig('GPD_ReturnLevels_py.png')
plt.show()


print('plotting and saving stationary series')
hndl = tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, dateformat='%y', xtick=tickTmStmp)
plt.savefig('statSeriesTrendOnly_py.png')
plt.show()

print('seasonal statistics')
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='seasonal', minPeakDistanceInDays=minPeakDistanceInDays)

wr = np.linspace(min(seasonalExtrRange), max(seasonalExtrRange), 1501)


print('  plotting a slice of data ')
slice = [1988, 1993]
plotTitle = '1988-1993'
print('    plotting the series')
hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData,ylabel='Lvl (m)', dateformat='%y', title=plotTitle, minyear=slice[0], maxyear=slice[1])
print('    saving the series plot')
plt.savefig('seriesSeasonal_py.png')
plt.show()

print('plotting and saving stationary series')
hndl = tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, dateformat='%y', minyear=slice[0], maxyear=slice[1])
plt.savefig('statSeriesTrendOnly_2_py.png')
plt.show()

print('    plotting and saving the 3D GEV graph')
hndl = tsEvaPlotGEV3DFromAnalysisObj(wr, nonStatEvaParams, statTransfData, xlabel='Lvl (m)', dateformat='%y', minyear=slice[0], maxyear=slice[1], axisfontsize=axisFontSize3d)
plt.title(f'GEV 3D, {plotTitle}', fontsize=titleFontSize)
plt.savefig('GEV3DSeasonal_py.png')
plt.show()

print('    plotting and saving the 2D GEV graph')

hndl = tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', minyear=slice[0], maxyear=slice[1], dateformat='%y')
plt.title(f'GEV {plotTitle}', fontsize=titleFontSize)
plt.savefig('GEV2DSeasonal_py.png')
plt.show()
print('    plotting and saving the 2D GPD graph')
hndl = tsEvaPlotGPDImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', minyear=slice[0], maxyear=slice[1], dateformat='%y')
plt.title(f'GPD {plotTitle}', fontsize=titleFontSize)
plt.savefig('GPD2DSeasonal_py.png', format='png')
plt.show()
