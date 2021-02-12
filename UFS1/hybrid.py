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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import time
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

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

def getForecasts(sarima, optimal_params, test_data):

    horizon = len(test_data.index)
    sarima_forecast = sarima.predict(horizon)
    sarima_forecast = pd.Series(sarima_forecast, index=test_data.index)
    
    train_resids = sarima.resid().reshape(-1,1)
    test_resids = np.array(test_data - sarima_forecast).reshape(-1,1)

    lstm_forecast = lstm(optimal_params, train_resids, test_resids)[1]
    lstm_forecast = pd.Series(lstm_forecast, index=test_data.index)

    return sarima_forecast, lstm_forecast

def calculate_performance(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return round(mse, 3), round(mae, 3), round(rmse, 3)

def lstm(params, train_resids, test_resids):

    look_back = params[0]
    hidden_nodes = params[1]
    output_nodes = params[2]
    nb_epoch = params[3]    #number of times the algorithm will work through the entire training set.
    batch_size = params[4]  #number of samples to work through.

    # scaler = MinMaxScaler(feature_range=(-1, 1)) #Rescale the training residuals
    # train_resids = scaler.fit_transform(train_resids)
    # test_resids = scaler.fit_transform(test_resids)

    generator = TimeseriesGenerator(train_resids, train_resids, length=look_back, batch_size=batch_size, shuffle = True)

    model = Sequential()
    model.add(LSTM(hidden_nodes, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(Dense(output_nodes))
    model.compile(optimizer='adam', loss='mean_squared_error')
    start_time = time.time()
    model.fit(generator, epochs=nb_epoch, verbose=0)
    print(f"Time passed fitting with parameters {params}: {time.time()-start_time} seconds")
    
    lstm_prediction = []
    first_eval_batch = train_resids[-look_back:]
    current_batch = first_eval_batch.reshape((1, look_back, 1))
    for i in range(len(test_resids)):
        pred = model.predict(current_batch)[0]
        lstm_prediction.append(pred)
        current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)
    
    # lstm_prediction = list(scaler.inverse_transform(lstm_prediction))

    mse, mae, rmse = calculate_performance(test_resids, lstm_prediction)

    info = list(params) + [mse, mae, rmse]

    return info, lstm_prediction

def runLstm(train_resids, test_resids, params):

    param_combinations = list(itertools.product(*params))
    print("number of combinations for grid search:", len(param_combinations))
    
    pool = mp.Pool(mp.cpu_count())
    performance = pool.starmap(lstm, [(combination, train_resids, test_resids) for combination in param_combinations])
    performance = [p[0] for p in performance]
    pool.close()

    # performance = []
    # for i in range(len(param_combinations)):
    #     print("\nComputing prestation for combination", i+1, "...................................\n")
    #     performance.append(lstm(param_combinations[i], train_resids, test_resids)[0])
    
    performance_df = pd.DataFrame(performance, columns = ['look back', 'hidden nodes', 'output nodes', 'epochs', 'batch size', 'MSE', 'MAE', 'RMSE'])
    print(performance_df)

    optimal_params = performance_df.iloc[performance_df.RMSE.argmin(), :5]
    print("The best parameter combination is:\n", optimal_params)
    print("RMSE:", performance_df.iloc[performance_df.RMSE.argmin(),7])

    return np.array(optimal_params).astype(int)

def runSarima(train_data, order, seasonal_order):

    m = 52 #Frequenty of the data

    sarima = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order, trend = 'ct').fit(train_data)
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
    country = 'DE'
    start_year = 2016
    end_year = 2021
    keyword = 'apfelstrudel'
    order = (0,1,1)
    seasonal_order = (1,1,0,52)
    params_lstm = [[52], [10, 18, 20, 26, 32, 50], [1], [400, 500], [26, 30, 40, 52, 65]]
    
    ################ MAIN CODE:
    df = ApiExtract.extract(range(start_year, end_year), country)
    keywords = df.keyword.unique()

    train_data, test_data = readData(df, keyword)
    
    sarima, train_resids, test_resids = runSarima(train_data, order, seasonal_order)
    optimal_params = runLstm(train_resids, test_resids, params_lstm)
    sarima_forecast, lstm_forecast = getForecasts(sarima, optimal_params, test_data)

    ################ OUTPUT AND PLOTTING:
    plot_hybrid(train_data, test_data, sarima_forecast, lstm_forecast)

if __name__ == "__main__":
    main()
