#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:27:42 2021

@author: safouane
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np

def determineSeason(keyword, seasonRange):
    for i in range(len(seasonRange)):
        seasonRange['Start'][i] = seasonRange['Start'][i].month
        seasonRange['End'][i] = seasonRange['End'][i].month

    if seasonRange['Start'].is_unique and seasonRange['End'].is_unique:
        print(keyword, "has clear seasonality. ===>", seasonRange['Start'] ,seasonRange['End'])

def filterDates(dates):

    startDates = [dates[0].date()]
    endDates = []
    for i in range(1,len(dates)):
        if dates[i] - dates[i-1] != pd.to_timedelta('7 days'):
            endDates.append(dates[i-1].date())
            startDates.append(dates[i].date())
    endDates.append(dates[len(dates)-1].date())
    seasonRange = pd.DataFrame(zip(startDates, endDates),columns=["Start", "End"])

    return seasonRange

def popularPeriods(keyword, smoothedProbs):
    
    states = smoothedProbs.round() #round state probabilities to zero and one
    popularStateDates = states.where(states == 0).dropna().index #gather popular state dates

    seasonRange = filterDates(popularStateDates)

    #determineSeason(keyword, seasonRange)
    
    return seasonRange

def markovSwitching(LTSDecomposition):
    
    season = LTSDecomposition.seasonal
    
    mod_kns = sm.tsa.MarkovRegression(season, k_regimes=2, trend='nc', switching_variance=True)
    res_kns = mod_kns.fit()
    probabilities = res_kns.smoothed_marginal_probabilities[0]
    """
    (np.ones(len(probabilities))-probabilities).plot()#Fast correction for states
    plt.title('Smoothed probability of recession')
    plt.show()
    """
    return probabilities

def singleDecomp(df, keyword):
        
    dataFrame = df[df['keyword'].str.contains(keyword)]
    apfelstrudel = dataFrame['interest'].tolist()
    
    apfelstrudel = pd.Series(apfelstrudel, index = pd.date_range('1-1-2017', periods=len(apfelstrudel), freq='W'), name = keyword)
    
    stl = STL(apfelstrudel, seasonal=13)
    decomposition = stl.fit()

    #decomposition.plot()
    #plt.show()

    return decomposition

def checkAllKeywords(df, keywords):
    
    for keyword in keywords:
        check_a_Keyword(df, keyword)
    
def check_a_Keyword(df, keyword):
    
    singleDecomposition = singleDecomp(df, keyword)
    smoothedProbs = markovSwitching(singleDecomposition)#Probability of being in low popularity season!
    
    seasonRange = popularPeriods(keyword, smoothedProbs)

    print(seasonRange)
    
def nonStatKeywords(df, keywords):

    for keyword in keywords:
        
        dataFrame = df[df['keyword'].str.contains(keyword)]
        product = dataFrame['interest'].tolist()
        product = pd.Series(product, index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)

        stl = STL(product, seasonal=13)
        res = stl.fit()
        season = res.seasonal
        result = adfuller(season)
        
        if result[1] > 0.05:
            print(keyword, result[1])

def getKeywords(df):

    keywords = []    
    for k in range(len(df)-1):  
        if df.iloc[k, 5] != df.iloc[k+1, 5]:
            keywords.append(df.iloc[k, 5])

    return keywords

def main():
    
    #os.chdir("/Users/safouane/Desktop/seminar/data")
    #df = pd.read_csv('case_study_trends_data_20210107.csv')
    df = pd.read_csv(r'Data\case_study_trends_data_20210107.csv')
  
    #keywords = getKeywords(df)
    #nonStatKeywords(df, keywords)

    keyword = 'rotkohl'
    check_a_Keyword(df, keyword)
    
    #checkAllKeywords(df, keywords)

if __name__ == "__main__":
    register_matplotlib_converters()
    sns.set_style('darkgrid')
    plt.rc('figure',figsize=(16,12))
    plt.rc('font',size=13)
    main()