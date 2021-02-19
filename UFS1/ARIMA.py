import ApiExtract
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from ApiExtract import extract
import Description

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
    plt.plot(arimaForecast, 'r:', label = "Prediction")
    plt.legend(loc = 'upper left')
    plt.show()

    SARIMA.plot_diagnostics()
    plt.show()

def manualArima(trainData, testData, order, seasonal_order):
    
    SARIMA = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order).fit(trainData)
    forecast = SARIMA.predict(len(testData))
    arimaForecast = pd.Series(forecast, index = testData.index)

    return SARIMA, arimaForecast

def autoArima(trainData, testData, seasonality, trend, d):
    
    SARIMA = pm.auto_arima(trainData, start_p=1, start_q=1,test='adf',max_p=3, max_q=3, m=len(testData) ,start_P=0, seasonal=seasonality,d=d, D=None, trace=True,
                         error_action='ignore',  suppress_warnings=True, stepwise=True, trend = trend)
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
    country = 'ES'
    startYear = 2016
    endYear = 2021
    keyword = 'guarnicion'
    seasonality = True
    df = extract(range(startYear, endYear), country)
    dd = Description.Data(df)
    (stationary, det_trend) = (dd.statistics.stationary(keyword))
    d = 0 if stationary else 1
    trend = "ct" if det_trend else "c"

    ################ MAIN CODE:
    df = ApiExtract.extract(range(startYear, endYear), country)
    keywords = df.keyword.unique()

    (trainData, testData) = readData(df, keyword)
    
    #(trainSets, testSets, arimaModels) = runKeywords(df, keywords[:10])
    (SARIMA, autoForecastData) = autoArima(trainData, testData, seasonality, trend, d)

    #(SARIMA, manualForecacastData) = manualArima(trainData, testData, (0,1,1), (1,1,0,52))

    ################ OUTPUT AND PLOTTING:
    #multiPlot(trainSets, testSets, arimaModels)
    
    singlePlot(trainData, testData, autoForecastData, SARIMA)
    
    #singlePlot(trainData, testData, manualForecacastData, SARIMA)

if __name__ == "__main__":
    main()