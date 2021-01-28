
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
        ax[i].plot(forecasts[forecasts.columns[i]], color = 'red')

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

    plt.figure()
    plt.plot(trainData, color = 'black', label = "Training data")
    plt.plot(testData, color = 'green', label = "Testing data")
    plt.plot(arimaForecast, color = 'red', label = "Prediction")
    plt.legend(loc = 'upper left')
    plt.show()

    SARIMA.plot_diagnostics()
    plt.show()

def manualArima(trainData, testData, order, seasonal_order):

    SARIMA = ARIMA(trainData ,order=order, seasonal_order=seasonal_order, trend='ct').fit()
    forecast = SARIMA.predict(start=testData.index[0], end=testData.index[len(testData.index)-1])
    arimaForecast = pd.Series(forecast, index = testData.index)

    return SARIMA, arimaForecast

def autoArima(trainData, testData):
    
    SARIMA = pm.auto_arima(trainData, start_p=1, start_q=1,test='adf',max_p=3, max_q=3, m=len(testData) ,start_P=0, seasonal=True,d=None, D=1, trace=True,
                         error_action='ignore',  suppress_warnings=True, stepwise=True)
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
    country = 'DE'
    startYear = 2016
    endYear = 2021
    keyword = 'apfelstrudel'
    keywords = pd.read_csv(r'/Users/safouane/Desktop/Data/KeyWords/NL_EN.txt').NL    
    
    ################ MAIN CODE:
    df = ApiExtract.extract(range(startYear, endYear), country)
    
    (trainData, testData) = readData(df, keyword)
    (SARIMA, autoForecastData) = autoArima(trainData, testData)

    #(trainSets, testSets, arimaModels) = runKeywords(df, keywords[:10])
    #(SARIMA, manualForecacastData) = manualArima(trainData, testData, (0,1,2), (1,1,0,52))

    ################ OUTPUT AND PLOTTING:
    #multiPlot(trainSets, testSets, arimaModels)
    
    singlePlot(trainData, testData, autoForecastData, SARIMA)
    
    #singlePlot(trainData, testData, manualForecacastData, SARIMA)

if __name__ == "__main__":
    main()
