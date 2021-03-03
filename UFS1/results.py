from sklearn.metrics import mean_squared_error, mean_absolute_error
import DataUtil
from Models import Arima
import ApiExtract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

# a=
# a['1'][0][0/1]

start_year = 2016
end_year = 2022
horizon = 33
method = 'k_medoids'
distance = 'euclidean'

lstm_df = pd.read_csv('/Users/safouane/Desktop/predictions_'+method+'_'+distance+'.csv')
df = DataUtil.get_cluster_means('/Users/safouane/Desktop/7clusters.csv', method, distance)
df_kwC = pd.read_csv('/Users/safouane/Desktop/7clusters.csv')
keywords_clusters = eval(df_kwC[df_kwC[df_kwC.columns[0]] == method][distance][0])['keywords'][0]

df_nl = ApiExtract.extract(range(start_year, end_year), 'NL')
df_de = ApiExtract.extract(range(start_year, end_year), 'DE')
df_es = ApiExtract.extract(range(start_year, end_year), 'ES')
data = pd.concat([df_nl, df_de, df_es])

test_dates = pd.date_range(start='2020-07-12', periods=33, freq='W')
dates = pd.date_range(start='2016-02-28', periods=261, freq='W')
x=DataUtil.get_corona_policy(dates, 'All')
x_test=x[x.index >= test_dates[0]]
model = Arima(df, 1.0)
model.get_covid_policy()

for i in range(1,8):
    keyword = 'Cluster'+str(i)
    model.fit(keyword)
    
    train_data = model.time_series(keyword, True)
    
    cluster_mean = np.mean([data[(data.keyword == prod[0]) & (data.country == prod[1]) & (data.startDate >= test_dates[0])]
                       ['interest'] for prod in keywords_clusters[str(i)]], 0)

    model.fit('Cluster'+str(i))
    arima_prediction = model.model.predict(len(test_dates), x_test, index=test_dates)

    lstm_prediction = lstm_df[keyword]
    lstm_prediction = pd.DataFrame(np.array(lstm_prediction), index = lstm_df.date)
    hybrid_prediction = np.array(arima_prediction) + np.array(lstm_prediction)

    mae = mean_absolute_error(cluster_mean, hybrid_prediction)
    mse = mean_squared_error(cluster_mean, hybrid_prediction)

    print(f"{keyword}: mae = {mae} and mse = {mse}")

    # plt.figure()
    # plt.plot(train_data, color='black', label='Train data')
    # plt.plot(cluster_mean, color='green', label='Test data')
    # plt.plot(hybrid_prediction, color=':r', label='Hybrid prediction')
    # plt.show()