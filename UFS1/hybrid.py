import os
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from scipy import stats
import ApiExtract
from Models import Arima, Lstm
import sys
import DataUtil

def plot_hybrid(trainData, testData, sarima_forecast, lstm_forecast, residuals):

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
    plt.plot(pd.DataFrame(residuals, index=trainData.index), color='black', label='train residuals')
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


def getForecasts(train_resids, optimal_params, test_data):
    sarima_forecast = dd.predict()
    test_resids = np.array(test_data - sarima_forecast).reshape(-1,1)

    lstm_forecast = lstm(optimal_params, train_resids, test_resids, 0)[1]
    lstm_forecast = pd.Series(lstm_forecast, index=test_data.index)

    arima_performance = round(mae(test_data, sarima_forecast), 3)
    lstm_performance = round(mae(test_data, sarima_forecast + lstm_forecast), 3)

    print(f"\nThe mean absolute error goes from {arima_performance} to {lstm_performance} after fitting the arima residuals using LSTM")

    return sarima_forecast, lstm_forecast


def lstm(params, train_resids, test_resids, teller):
    if teller == 0: print("Fitting a hybrid model using the best parameter combination .....")
    else: print(f"Computing performance for parameter combination {teller}")

    [look_back, output_nodes, nb_epoch, batch_size] = [elt for elt in params]
    m = Lstm(train_resids, test_resids, look_back,  output_nodes, nb_epoch, batch_size)
    m.fit()
    lstm_prediction = m.predict()
    info = list(params) + [m.mse(), m.mae(),m.rmse()]

    return info, lstm_prediction


def gridSearch(residuals):
    m = 52 if (dd.time_series(keyword).index[1].month - dd.time_series(keyword).index[0].month) == 0 else 12

    train_resids = residuals[:len(residuals)-int(m/2)]
    test_resids = residuals[len(residuals)-int(m/2):]

    param_combinations = list(itertools.product(*params_lstm))
    print("number of combinations for grid search:", len(param_combinations))

    pool = mp.Pool(2)
    performance = pool.starmap(lstm, [(combination, train_resids, test_resids, param_combinations.index(combination)+1) for combination in param_combinations])
    performance = [p[0] for p in performance]
    pool.close()

    performance_df = pd.DataFrame(performance, columns = ['look back', 'output nodes', 'epochs', 'batch size', 'MSE', 'MAE', 'RMSE'])
    print(performance_df)

    optimal_params = performance_df.iloc[performance_df.RMSE.argmin(), :4]
    print("The best parameter combination is:\n", optimal_params)
    print("RMSE:", performance_df.iloc[performance_df.RMSE.argmin(),6])

    return np.array(optimal_params).astype(int)


def main():
    train_data = dd.time_series(keyword, True)
    test_data = dd.time_series(keyword, False)

    for p in params_lstm[1]: 
        if (len(test_data) % p != 0): sys.exit("The length of the test data is not a multiple of the output nodes")

    residuals = np.array(dd.residuals).reshape(-1,1)

    optimal_params = gridSearch(residuals)
    sarima_forecast, lstm_forecast = getForecasts(residuals, optimal_params, test_data)

    lstm_forecast.to_csv('/Users/safouane/Desktop/cluster_forecasts/'+keyword+'.csv')

    plot_hybrid(train_data, test_data, sarima_forecast, lstm_forecast, residuals)


def cluster_df():
    cluster_mean = pd.read_csv('/Users/safouane/Desktop/cluster_means.csv')[keyword]
#     cluster_mean = pd.read_csv('C:/Users/Stagiair/Documents/Seminar/cluster_means.csv')[keyword]
    df = ApiExtract.extract(range(start_year, end_year), country)[:len(cluster_mean.index)]
    df['interest'] = cluster_mean
    df['keyword'] = keyword
    df['category'] = 'cluster'

    return df

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    start_year = 2016
    end_year = 2021
    country = 'ES'
    keyword = 'Cluster7'
    params_lstm = [[52], [1], [600, 800, 1000], [104, 208]]

    df = DataUtil.get_cluster_means('/Users/safouane/Desktop/7clusters.csv', 'k_medoids', 'euclidean') if 'Cluster' in keyword else ApiExtract.extract(range(start_year, end_year), country)
#     df = DataUtil.get_cluster_means('C:/Users/Stagiair/Documents/Seminar/7clusters.csv', 'k_medoids', 'euclidean') if 'Cluster' in keyword else ApiExtract.extract(range(start_year, end_year), country)
    dd = Arima(df, 1)
    dd.get_covid_policy()
    dd.fit(keyword)

    main()
