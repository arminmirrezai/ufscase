
import os
import ApiExtract
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def multiPlot(trainSets, testSets, forecasts):

    fig, ax = plt.subplots(len(testSets.columns), 1)
    for i in range(len(testSets.columns)):
        ax[i].plot(trainSets[trainSets.columns[i]], color = 'black')
        ax[i].plot(testSets[testSets.columns[i]], color = 'green')
        ax[i].plot(forecasts[forecasts.columns[i]], 'r--')

    plt.show()

def runKeywords(df, keywords):
    trainSets = pd.DataFrame(columns=keywords) 
    testSets = pd.DataFrame(columns=keywords)
    forecasts = pd.DataFrame(columns = keywords)
    for keyword in keywords:
        (trainData, testData) = readData(df, keyword)
        trainSets[keyword] = trainData
        testSets[keyword] = testData
        forecasts[keyword] = autoArima(trainData, testData)[1]

    return trainSets, testSets, forecasts

def singlePlot(trainData, testData, arimaForecast, SARIMA):

    print("the lowest value is", min(arimaForecast))

    plt.figure()
    plt.plot(trainData, color = 'black', label = "Training data")
    plt.plot(testData, color = 'green', label = "Testing data")
    plt.plot(arimaForecast, 'r-.', label = "Prediction")
    plt.legend(loc = 'upper left')
    plt.show()

    SARIMA.plot_diagnostics()
    plt.show()

def manualArima(trainData, testData, order, seasonal_order):
    
    SARIMAestimation = ARIMA(trainData ,order=order, seasonal_order=seasonal_order).fit()
    forecast = SARIMAestimation.predict(start=testData.index[0], end=testData.index[len(testData.index)-1])
    arimaForecast = pd.Series(forecast, index = testData.index)

    return SARIMAestimation, arimaForecast

def autoArima(trainData, testData):
    
    SARIMA = pm.auto_arima(trainData, start_p=1, start_q=1,test='adf',max_p=3, max_q=3, m=len(testData) ,start_P=0, seasonal=True,d=None, D=1, trace=True,
                         error_action='ignore',  suppress_warnings=True, stepwise=True, trend = 'ct')
    forecast = SARIMA.predict(n_periods = len(testData))
    arimaForecast = pd.Series(forecast, index = testData.index)

    return SARIMA, arimaForecast

def readData(df, keyword):

    data = df[df.keyword == keyword][['interest', 'startDate']]
    data = data.interest.rename(index = data.startDate)

    startYear = data.index[0].year
    endYear = data.index[len(data)-1].year + 1

    splitThreshold = int(len(data)*(1-1/(endYear - startYear)))

    return data[:splitThreshold], data[splitThreshold+1:]

def main():

    ################ VARIABLES:
    country = 'NL'
    startYear = 2016
    endYear = 2021
    keyword = 'aioli'
    
    ################ MAIN CODE:
    df = ApiExtract.extract(range(startYear, endYear), country)
    keywords = df.keyword.unique()

    (trainData, testData) = readData(df, keyword)
    #(SARIMA, autoForecastData) = autoArima(trainData, testData)

    (SARIMAestimation, manualForecacastData) = manualArima(trainData, testData, (0,0,0), (1,1,0,52))

    #(trainSets, testSets, arimaModels) = runKeywords(df, keywords[:10])

    ################ OUTPUT AND PLOTTING:
    #multiPlot(trainSets, testSets, arimaModels)
    
    #singlePlot(trainData, testData, autoForecastData, SARIMA)
    
    singlePlot(trainData, testData, manualForecacastData, SARIMAestimation)

if __name__ == "__main__":
    main()
