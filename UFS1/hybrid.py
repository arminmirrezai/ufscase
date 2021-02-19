from re import L, T
import ApiExtract
import pandas as pd
import numpy as np
import pmdarima as pm
import itertools
import matplotlib.pyplot as plt
from operator import index
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import time
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from ApiExtract import extract
import Description
import sys

def plot_hybrid(trainData, testData, sarima_forecast, lstm_forecast):

    plt.figure()
    plt.title("SARIMA model")
    plt.plot(trainData, color = 'black', label = "Training data")
    plt.plot(testData, color = 'green', label = "Testing data")
    plt.plot(sarima_forecast, 'r:', label = "Prediction")
    plt.legend(loc = 'upper left')
    plt.show()

    resids = testData - sarima_forecast
    stats.probplot(resids, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot for SARIMA residuals")
    plt.show()

    plt.figure()
    plt.title("lstm predicted residuals")
    plt.plot(resids, color='green', label='test residuals')
    plt.plot(lstm_forecast, 'r:', label='predicted residuals')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.title("Hybrid SARIMA/LSTM model")
    plt.plot(trainData, color = 'black', label = "Training data")
    plt.plot(testData, color = 'green', label = "Testing data")
    plt.plot(sarima_forecast + lstm_forecast, 'r:', label = "ARIMA LSTM prediction")
    plt.legend(loc = 'upper left')
    plt.show()

    residuals = testData - sarima_forecast - lstm_forecast
    stats.probplot(np.array([resid[0] for resid in residuals]), dist="norm", plot=plt)
    plt.title("Normal Q-Q plot for hybrid residuals")
    plt.show()

def getForecasts(sarima, optimal_params, test_data, params_lstm):

    horizon = len(test_data.index)
    sarima_forecast = sarima.predict(horizon)
    sarima_forecast = pd.Series(sarima_forecast, index=test_data.index)
    
    train_resids = sarima.resid().reshape(-1,1)
    test_resids = np.array(test_data - sarima_forecast).reshape(-1,1)

    # plt.figure()
    # plt.plot(train_resids)
    # plt.show()

    lstm_forecast = lstm(optimal_params, train_resids, test_resids, 0)[1]
    lstm_forecast = pd.Series(lstm_forecast, index=test_data.index)

    arima_performance = calculate_performance(test_data, sarima_forecast)
    lstm_performance = calculate_performance(test_data, sarima_forecast + lstm_forecast)
    
    if arima_performance[1] < lstm_performance[1]:
        print("Hybrid model did not perform well, performing a grid search once again:")
        optimal_params =  runLstm(train_resids, test_resids, params_lstm)
        sarima_forecast, lstm_forecast = getForecasts(sarima, optimal_params, test_data, params_lstm)
    else:
        print(f"The mean absolute error goes from {arima_performance[1]} to {lstm_performance[1]} after fitting the arima residuals using LSTM!")

    return sarima_forecast, lstm_forecast

def calculate_performance(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return round(mse, 3), round(mae, 3), round(rmse, 3), round(msle, 3), round(mape, 3)

def lstm(params, train_resids, test_resids, teller):

    if teller == 0:
        print("Fitting a hybrid model using the best parameter combination .....")
    else:
        print(f"Computing performance for parameter combination {teller}")

    look_back = params[0] 
    output_nodes = params[1]
    nb_epoch = params[2]    #number of times the algorithm will work through the entire training set.
    batch_size = params[3]  #number of samples to work through.
    hidden_nodes = int(2*(look_back+output_nodes)/3)

    # scaler = MinMaxScaler(feature_range=(-1, 1)) #Rescale the training residuals
    # train_resids = scaler.fit_transform(train_resids)
    # test_resids = scaler.fit_transform(test_resids)

    generator = TimeseriesGenerator(train_resids, train_resids, length=look_back, batch_size=batch_size)

    model = Sequential()
    model.add(LSTM(hidden_nodes, activation='tanh', recurrent_activation='sigmoid'))
    model.add(Dense(output_nodes))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(generator, epochs=nb_epoch, verbose=0)
    
    lstm_prediction = []
    first_eval_batch = train_resids[-look_back:]
    current_batch = first_eval_batch.reshape((1, look_back, 1))
    for i in range(int(len(test_resids) / output_nodes)):
        pred = model.predict(current_batch)[0]
        for p in pred: lstm_prediction.append(np.array([p]))
        current_batch = current_batch[:,output_nodes:,:]
        for p in pred: current_batch = np.append(current_batch, [[np.array([p])]], axis=1)
    
    # lstm_prediction = list(scaler.inverse_transform(lstm_prediction))

    mse, mae, rmse, msle, mape = calculate_performance(test_resids, lstm_prediction)

    info = list(params) + [mse, mae, rmse, msle, mape]

    return info, lstm_prediction

def runLstm(train_resids, test_resids, params):

    param_combinations = list(itertools.product(*params))
    print("number of combinations for grid search:", len(param_combinations))
    
    pool = mp.Pool(mp.cpu_count())
    performance = pool.starmap(lstm, [(combination, train_resids, test_resids, param_combinations.index(combination)+1) for combination in param_combinations])
    performance = [p[0] for p in performance]

    pool.close()

    performance_df = pd.DataFrame(performance, columns = ['look back', 'output nodes', 'epochs', 'batch size', 'MSE', 'MAE', 'RMSE', 'MSLE', 'MAPE'])
    print(performance_df)

    optimal_params = performance_df.iloc[performance_df.RMSE.argmin(), :4]
    print("The best parameter combination is:\n", optimal_params)
    print("RMSE:", performance_df.iloc[performance_df.RMSE.argmin(),6])

    return np.array(optimal_params).astype(int)

def runSarima(train_data, order, seasonal_order, trend):

    m = 52 #Frequenty of the data

    sarima = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order, trend = trend).fit(train_data)
    # sarima.plot_diagnostics()
    # plt.show()
    residuals = sarima.resid().reshape(-1,1)

    train_resids = residuals[:len(residuals)-m]
    test_resids = residuals[len(residuals)-m:] #Will be used for the performance measures

    return sarima, train_resids, test_resids

def readData(df, keyword):

    data = df[df.keyword == keyword][['interest', 'startDate']]
    data = data.interest.rename(index = data.startDate)

    startYear = data.index[0].year
    endYear = data.index[len(data)-1].year + 1

    splitThreshold = int(len(data)*(1-1/(endYear - startYear)))

    return data[:splitThreshold], data[splitThreshold+1:]

def main():
    ################ VARIABLES:
    start_year = 2016
    end_year = 2021
    country = 'ES'
    keyword = 'guarnicion'
    order = (1,0,0)
    seasonal_order = (1,0,1,52)
    df = extract(range(start_year, end_year), country)
    dd = Description.Data(df)
    (stationary, det_trend) = (dd.statistics.stationary(keyword))
    d = 0 if stationary else 1
    trend = "ct" if det_trend else "c"
    params_lstm = [[52, 65, 78, 91], [1], [400], [104, 156, 208]]
    
    ################ MAIN CODE:
    df = ApiExtract.extract(range(start_year, end_year), country)
    keywords = df.keyword.unique()

    # for product in keywords[93:110]:
    #     data = df[df.keyword == product][['interest', 'startDate']]
    #     data = data.interest.rename(index = data.startDate)
    #     plt.figure()
    #     plt.plot(data)
    #     plt.title(product)
    #     plt.show()

    train_data, test_data = readData(df, keyword)

    for p in params_lstm[1]: 
        if (len(test_data) % p != 0): sys.exit("The length of the test data is not a multiple of the output nodes")

    sarima, train_resids, test_resids = runSarima(train_data, order, seasonal_order, trend)
    optimal_params =  runLstm(train_resids, test_resids, params_lstm) #[65, 1, 400, 208]
    sarima_forecast, lstm_forecast = getForecasts(sarima, optimal_params, test_data, params_lstm)

    ################ OUTPUT AND PLOTTING:
    plot_hybrid(train_data, test_data, sarima_forecast, lstm_forecast)

if __name__ == "__main__":
    main()
