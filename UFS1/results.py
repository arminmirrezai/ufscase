from sklearn.metrics import mean_squared_error, mean_absolute_error
import DataUtil
from Models import Arima
import ApiExtract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

start_year = 2016
end_year = 2022
horizon = 33
method = 'k_medoids' #hierarchical_ward
distance = 'euclidean'

lstm_df = pd.read_csv('/Users/safouane/Desktop/predictions_'+method+'_'+distance+'.csv')
#df = DataUtil.get_cluster_means('/Users/safouane/Desktop/clusters_kmed_7.csv', method, distance)
df_kwC = DataUtil.get_cluster_means('/Users/safouane/Desktop', method, distance)


df_nl = ApiExtract.extract(range(start_year, end_year), 'NL')
df_de = ApiExtract.extract(range(start_year, end_year), 'DE')
df_es = ApiExtract.extract(range(start_year, end_year), 'ES')
data = pd.concat([df_nl, df_de, df_es])

test_dates = pd.date_range(start='2020-07-12', periods=33, freq='W')
dates = pd.date_range(start='2016-02-28', periods=261, freq='W')
x=DataUtil.get_corona_policy(dates, 'All')
x_test=x[x.index >= test_dates[0]]
model = Arima(df_kwC, 1.0)
model.get_covid_policy()

for keyword in df_kwC.keyword.unique():

    train_data = model.time_series(keyword, True)
    
    keywords_clusters = eval(df_kwC[df_kwC.keyword==keyword]['cluster keywords'][0])
    cluster_mean = np.mean([data[(data.keyword == prod[0]) & (data.country == prod[1]) & (data.startDate >= test_dates[0])]['interest'] for prod in keywords_clusters], 0)
   
    model.fit(keyword)
    arima_prediction = model.model.predict(len(test_dates), x_test, index=test_dates)

    lstm_prediction = lstm_df[keyword]
    hybrid_prediction = arima_prediction + array(lstm_prediction[0])
    
    mae = mean_absolute_error(cluster_mean, hybrid_prediction)
    mse = mean_squared_error(cluster_mean, hybrid_prediction)

    print(f"{keyword}: mae = {mae} and mse = {mse}")

    cluster_mean = pd.DataFrame(array(cluster_mean), index = test_dates)
    hybrid_prediction = pd.DataFrame(array(hybrid_prediction), index = test_dates)

    plt.figure()
    for prod in keywords_clusters:
        plt.plot(pd.DataFrame(array(data[(data.keyword == prod[0]) & (data.country == prod[1]) & (data.startDate < test_dates[0])]['interest']), index=train_data.index), color='black')
    plt.plot(train_data, color='red', label='Train data')
    plt.plot(cluster_mean, color='green', label='Test data')
    plt.plot(hybrid_prediction,':r', label='Hybrid prediction')
    plt.legend(loc = 'upper left')
    plt.show()