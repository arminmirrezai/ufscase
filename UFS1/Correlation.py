import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

runAll = False

columns = ["keyword 1", "keyword 2", "correlation", "lag"]
correlations = pd.DataFrame(columns = columns)

path = "C:/Users/Stagiair/Documents/ufscase-master/Data/"
allData = pd.read_csv(path + 'data_UFS2.csv', delimiter = ";")
keywords = allData.groupby(['keyword','countryCode']).size().reset_index()[['keyword', 'countryCode']].to_dict('r')
keywordCombinations = list(combinations(keywords, 2))

for keywords in keywordCombinations:
    print(keywords)
    ##############################################################################
    # Get right data and calculate correlation
    ##############################################################################
    keyword1 = keywords[0]['keyword']
    keyword2 = keywords[1]['keyword']
    
    data1 = allData[(allData['keyword'] == keyword1) & (allData['countryCode'] == keywords[0]['countryCode'])]['interest'].reset_index(drop = True).rename({ 'interest': 'interest1' })
    data2 = allData[(allData['keyword'] == keyword2) & (allData['countryCode'] == keywords[1]['countryCode'])]['interest'].reset_index(drop = True).rename({ 'interest': 'interest2' })
    data = pd.concat([data1, data2], axis=1)
    
    if runAll:
        correlation = data.corr().iloc[0, 1]
    
    ##############################################################################
    # Plot moving window with the median of each window 
    ##############################################################################
    medianWindow = 25
    
    if runAll:
        f, ax = plt.subplots(figsize = (7, 3))
        data.rolling(window = medianWindow, center = True).median().plot(ax = ax)
        ax.set(xlabel = 'Time', ylabel = 'Correlation')
        ax.set(title = f"Overall correlation = {np.round(correlation,2)}");
    
    ##############################################################################
    # Moving window correlation. 
    ##############################################################################
    windowSize = 30
    
    if runAll:
        rollingCorrelation = data1.rolling(window = windowSize, center = True).corr(data2)
        f, ax = plt.subplots(2, 1, figsize = (14, 6), sharex = True)
        data.rolling(window = medianWindow, center = True).median().plot(ax = ax[0])
        ax[0].set(xlabel = 'Frame', ylabel = 'Moving window median')
        rollingCorrelation.plot(ax=ax[1])
        ax[1].set(xlabel = 'Frame', ylabel = 'Correlation')
        plt.suptitle("Moving window median and rolling window correlation")
    
    ##############################################################################
    # Correlations with different lags
    ##############################################################################
    shiftMaximum = 15
    
    def laggedCrossCorrelation(x, y, lag = 0): return x.corr(y.shift(lag))
    
    laggedCorr = [laggedCrossCorrelation(data1, data2, lag) for lag in range(-shiftMaximum, shiftMaximum + 1)]
    indexMaxCorr = np.argmax(laggedCorr)
    offset = shiftMaximum + 1 - indexMaxCorr
    correlations = correlations.append({ "keyword 1": keyword1, "keyword 2": keyword2, "correlation": laggedCorr[indexMaxCorr], "lag": offset }, ignore_index = True)
    
    if runAll:
        f, ax = plt.subplots(figsize = (14,3))
        ax.plot(laggedCorr)
        ax.axvline(shiftMaximum + 1, color = 'k', linestyle = '--', label = 'Center')
        ax.axvline(indexMaxCorr, color = 'r', linestyle = '--', label = 'Peak synchrony')
        ax.set(title = f'Offset = {offset} frames\n{keyword1} leads <> {keyword2} leads', xlim = [0, 2 * shiftMaximum + 1], xlabel = 'Offset', ylabel = 'Correlation')
        ax.set_xticks([0, shiftMaximum / 2, shiftMaximum + 1, 3 * shiftMaximum / 2 + 1, 2 * shiftMaximum + 1])
        ax.set_xticklabels([-shiftMaximum, -shiftMaximum / 2, 0, shiftMaximum / 2, shiftMaximum]);
        plt.legend()
    
    ##############################################################################
    # Windowed Time Lagged Cross Correlation. Data splitted into frames
    ##############################################################################
    if runAll:
        nSplits = 5
        
        samplesPerSplit = len(data) / nSplits
        allLaggedCorr = []
        
        for split in range(nSplits):
            partOfData1 = data1.loc[split * samplesPerSplit : (split + 1) * samplesPerSplit]
            partOfData2 = data2.loc[split * samplesPerSplit : (split + 1) * samplesPerSplit]
            laggedCorr = [laggedCrossCorrelation(partOfData1, partOfData2, lag) for lag in range(-shiftMaximum, shiftMaximum + 1)]
            allLaggedCorr.append(laggedCorr)
        allLaggedCorr = pd.DataFrame(allLaggedCorr)
        
        f, ax = plt.subplots(figsize = (10, 5))
        sns.heatmap(allLaggedCorr, cmap = 'RdBu_r', ax = ax)
        ax.set(title = f'Windowed Time Lagged Cross Correlation', xlim = [0, 2 * shiftMaximum + 1], xlabel = 'Offset',ylabel = 'Split number')
        ax.set_xticks([0, shiftMaximum / 2, shiftMaximum + 1, 3 * shiftMaximum / 2 + 1, 2 * shiftMaximum + 1])
        ax.set_xticklabels([-shiftMaximum, -shiftMaximum / 2, 0, shiftMaximum / 2, shiftMaximum]);
    
    ##############################################################################
    # Windowed Time Lagged Cross Correlation. Smooth frames
    ##############################################################################
    if runAll:
        stepSize = 10
        
        start = 0
        end = start + 2 * shiftMaximum
        allLaggedCorr = []
        while end < len(data):
            partOfData1 = data1.iloc[start:end]
            partOfData2 = data2.iloc[start:end]
            laggedCorr = [laggedCrossCorrelation(partOfData1, partOfData2, lag) for lag in range(-shiftMaximum, shiftMaximum + 1)]
            allLaggedCorr.append(laggedCorr)
            start = start + stepSize
            end = end + stepSize
        allLaggedCorr = pd.DataFrame(allLaggedCorr)
        
        f, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(allLaggedCorr, cmap = 'RdBu_r', ax = ax)
        ax.set(title = f'Rolling Windowed Time Lagged Cross Correlation',xlim = [0, 2 * shiftMaximum], xlabel = 'Offset',ylabel = 'Split number')
        ax.set_xticks([0, shiftMaximum / 2, shiftMaximum + 1, 3 * shiftMaximum / 2 + 1, 2 * shiftMaximum + 1])
        ax.set_xticklabels([-shiftMaximum, -shiftMaximum / 2, 0, shiftMaximum / 2, shiftMaximum]);
        
importantCorrelations = correlations.reindex(correlations['correlation'].abs().sort_values(ascending=False).index)
importantCorrelations = importantCorrelations[importantCorrelations['correlation'] >= 0.85]
print(importantCorrelations)