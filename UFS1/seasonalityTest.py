#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:27:42 2021

@author: safouane
"""
import os
import pandas as pd

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def testSeasonality(df, keywords):
    
    noSeasonality = 0
    for keyword in keywords:
        
        data = df[df.keyword == keyword]
        product = data.interest.rename(index = data.date)
        product.index = pd.to_datetime(product.index)
        product = pd.Series(product, index = pd.date_range(product.index[0], periods=len(product), freq='W'), name = keyword)
        
        
        #seasonalComponent = STL(product, seasonal=13).fit().seasonal
        """it is advisable to isolate the trend before embarking on test for 
        presence of seasonal effect in a series."""
        
        
        pValue = wilcoxon(product, zero_method='wilcox')[1]
        """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        The Wilcoxon signed-rank test tests the null hypothesis that two 
        related paired samples come from the same distribution. In particular, 
        it tests whether the distribution of the differences x - y is symmetric 
        about zero. It is a non-parametric version of the paired T-test."""
        
        if pValue > 0.01:
            plt.figure(figsize = (15,1))
            plt.plot(product, color = 'red')
            plt.title(keyword)
            plt.show()
            print("No seasonality for", keyword, pValue)
            noSeasonality += 1
        
    print("\n#of time series without seasonality:",noSeasonality)
    
def testStationarity(df, keywords):

    for keyword in keywords:
        
        data = df[df.keyword == keyword]
        product = data.interest.rename(index = data.date)
        product.index = pd.to_datetime(product.index)
        
        product = pd.Series(product, index = pd.date_range(product.index[0], periods=len(product), freq='W'), name = keyword)

        stl = STL(product, seasonal=13)
        res = stl.fit()
        season = res.seasonal
        result = adfuller(season)
        
        if result[1] > 0.05:
            print(keyword, result[1])

def getKeywords(df):

    keywords = [df.keyword[0]]    
    for k in range(1,len(df)):  
        if df.keyword[k] != df.keyword[k-1]:
            keywords.append(df.keyword[k])

    return keywords

def main():
    
    country = 'DE'
    
    os.chdir("/Users/safouane/Desktop/Data")
    df = pd.read_csv('data_UFS.csv')
    df = df[df.countryCode == country]
    
    keywords = getKeywords(df)
    
    #testStationarity(df, keywords)
    testSeasonality(df, keywords)

if __name__ == "__main__":
    main()