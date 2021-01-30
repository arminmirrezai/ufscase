#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:27:42 2021

@author: safouane
"""
import os
import pandas as pd
import ApiExtract
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def testSeasonality(df, keywords):
    
    noSeasonality = 0
    for keyword in keywords:

        product = readData(df, keyword)
                  
        seasonalComponent = STL(product, seasonal=13).fit().seasonal
        """it is advisable to isolate the trend before embarking on test for 
        presence of seasonal effect in a series."""
        
        pValue = wilcoxon(seasonalComponent[104:156], seasonalComponent[156:208], zero_method='zsplit')[1]
        """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        The Wilcoxon signed-rank test tests the null hypothesis that two 
        related paired samples come from the same distribution. In particular, 
        it tests whether the distribution of the differences x - y is symmetric 
        about zero. It is a non-parametric version of the paired T-test."""
        
        if pValue < 0.01:
            plt.figure(figsize = (15,1))
            plt.plot(seasonalComponent, color = 'red')
            plt.title(keyword)
            plt.show()
            print("No seasonality for", keyword, pValue)
            noSeasonality += 1
        
    print("\n#of time series without seasonality:",noSeasonality)
    
def testStationarity(df, keywords):

    for keyword in keywords:
        
        product = readData(df, keyword)

        stl = STL(product, seasonal=13)
        res = stl.fit()
        season = res.seasonal
        result = adfuller(season)
        
        if result[1] > 0.05:
            print(keyword, result[1])

def readData(df, keyword):

    data = df[df.keyword == keyword][['interest', 'startDate']]
    data = data.interest.rename(index = data.startDate)

    # startYear = data.index[0].year
    # endYear = data.index[len(data)-1].year + 1

    # splitThreshold = int(len(data)*(1-1/(endYear - startYear)))

    return data#, data[:splitThreshold], data[splitThreshold+1:]

def main():

    country = 'NL'
    startYear = 2016
    endYear = 2021
    df = ApiExtract.extract(range(startYear, endYear), country)
    keywords = df.keyword.unique() #['zwezerik']

    #testStationarity(df, keywords)
    testSeasonality(df, keywords)



if __name__ == "__main__":
    main()