from typing import Type
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import ARIMA
from Description import Data
from Decompositions import Decompose
from scipy.stats.distributions import chi2
from statsmodels.stats.diagnostic import het_arch
from csv import reader
from DataUtil import get_corona_policy, getPath, saveResult
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os

class Arima:

    model: Type[ARIMA]
    df_train: Type[pd.DataFrame]
    df_test: Type[pd.DataFrame]

    def __init__(self, df, train_percentage=0.87):
        if train_percentage != 1.0:
            self.df_train = df[df.startDate <= df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
            self.df_test = df[df.startDate > df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        else:
            self.df_train = df           
            self.df_test = pd.Series(np.ones(25), index=pd.date_range(start='2020-07-12', end='2020-12-27', freq='W'))
        self.train_percentage=train_percentage
        self.x_train, self.x_test = None, None
        self.dd = Data(self.df_train)
        self.dmp = Decompose(self.df_train)
        self.model = pm.ARIMA
        self.test = self.Tests(self)
        self.stats = ""
        self.kw = self.df_train.keyword.unique()[0]

    @property
    def residuals(self):
        return self.model.resid()

    @property
    def aic(self):
        return self.model.aic()

    @property
    def log_likelihood(self):
        return len(self.model.params()) - self.aic/2

    def time_series(self, keyword, train=True) -> pd.Series:
        if train:
            ts = self.df_train[self.df_train.keyword == keyword]['interest']
            ts.index = self.df_train.startDate.unique()
        else:
            if self.train_percentage != 1:
                ts = self.df_test[self.df_test.keyword == keyword]['interest']
                ts.index = self.df_test.startDate.unique()
            else: ts = self.df_test
        return ts

    def get_dummies(self, keyword, benchmark=3) -> pd.Series:
        self.dmp.decompose_robustSTL(keyword)
        scores = self.dmp.outlier_score()
        dummies = pd.Series(0, index=scores.index)
        dummies[scores.index[scores > benchmark]] = 1
        return dummies

    def get_covid_policy(self):
        if not self.df_test.empty:
            dates = self.time_series(self.kw).append(self.time_series(self.kw, False)).index
        else:
            dates = self.time_series(self.kw).index
        x = get_corona_policy(dates, self.df_train.country.unique()[0])
        self.x_train = x[x.index <= self.df_train.startDate.unique()[-1]]
        self.x_test = x[x.index > self.df_train.startDate.unique()[-1]]
        return self.x_train, self.x_test

    def predict(self):
        ts = self.time_series(self.kw, train=False)
        n_periods = len(ts.index)
        # dummies = np.zeros_like(ts).reshape(-1, 1)
        return pd.Series(self.model.predict(n_periods, self.x_test), index=ts.index)

    def plot_predict(self):
        ts = self.time_series(self.kw).append(self.time_series(self.kw, False))
        plt.figure()
        plt.plot(ts, color='black', label=str(self.kw))
        plt.plot(self.predict(), color='red', label='Prediction')
        plt.legend(loc='upper left')
        plt.show()

    def fit(self, keyword, method='lbfgs', robust=False, save=False):
        """
        Fit the best arima or sarima model for the keyword
        :param robust: robust estimation by using dummies for outliers
        :param save: save parameter results
        :param method: Default Nelder-Mead based on speed
        :param keyword: keyword
        :return: fitted model
        """
        if robust:
            raise NotImplementedError('Robust not implemented yet')
        self.kw = keyword
        stationary, has_trend = self.dd.statistics.stationary(keyword)
        ts = self.time_series(keyword)
        x = self.get_dummies(keyword) if robust else None
        if stationary:
            self._model(ts, x, stationary, 'ct' if has_trend else 'c', 0, method)
        else:
            stationary, has_trend = self.dd.statistics.stationary(keyword, first_difference=True)
            self._model(ts, x, stationary, 'ct' if has_trend else 'c', 1, method)

        self._write_stats(outliers=x)
        if save:
            self.save_stats()
        return self.model

    def _model(self, ts, dummies, stationary, trend, diff, method='lbfgs'):
        # TODO implement robust and corona variable
        exog = np.array(dummies).reshape(-1, 1) if dummies is not None else None
        exog = self.x_train
        years = ts.index[-1].year - ts.index[0].year + 1
        periods = 52 if (ts.index[2].month - ts.index[0].month) in {0, 1} else 12
        hyper_params = self.get_hyperparams()
        if len(hyper_params) == 0:
            sarimax = pm.auto_arima(y=ts, X=exog, seasonal=True, stationary=stationary, d=diff, max_p=10,
                                    method=method, trend=trend, with_intercept=True, max_order=None,
                                    max_P=int(years/2), D=pm.arima.nsdiffs(ts, periods), m=periods,
                                    stepwise=True, maxiter=45, sarimax_kwargs={'cov_type': None})
        else:
            sarimax = pm.ARIMA(order=eval(hyper_params['Order']), seasonal_order=eval(hyper_params['Seasonal order']),
                               method=method, maxiter=45, trend=hyper_params['Trend']).fit(y=ts, X=exog)
        self.model = sarimax

    def get_hyperparams(self):
        """
        Get the hyperparameteres for the specific file if already ran (only for one period for now)
        :return: result of hyperparams
        """
        if 'method' in self.df_train.keys():
            method = self.df_train['method'].unique()[0]
            distance = self.df_train['distance'].unique()[0]
            folder_name = f"Models/Final Sarimax/{method}/{distance}"
        else:
            folder_name = f"Models/Final Sarimax/{self.df_train.country.unique()[0]}"
        path = getPath(self.kw,  folder_name)
        res = dict()
        if os.path.exists(path):
            with open(path, 'r') as file:
                lines = reader(file, delimiter='\n')
                for line in lines:
                    stat = line[0].split(': ')
                    res[stat[0]] = stat[1]
        return res

    def save_stats(self):
        """
        Save stats in the order file
        """
        if 'method' in self.df_train.keys():
            method = self.df_train['method'].unique()[0]
            distance = self.df_train['distance'].unique()[0]
            folder_name = f"Models/Final Sarimax/{method}/{distance}"
        else:
            folder_name = f"Models/Final Sarimax/{self.df_train.country.unique()[0]}"
        saveResult(self.kw, folder_name, txt=self.stats)

    def _write_stats(self, outliers):
        """
        Write the stats of the model
        :return:
        """
        self.stats = ""
        if outliers is not None:
            self.stats += "Outliers: " + str(sum(outliers)) + "\n"
        self.stats += "Trend: " + self.model.trend + "\n"
        self.stats += "Order: " + str(self.model.order) + "\n"
        self.stats += "Seasonal order: " + str(self.model.seasonal_order) + "\n"
        self.stats += "SSR: " + str(sum(np.square(self.residuals))) + "\n"
        self.stats += "AIC: " + str(self.aic) + "\n"

    class Tests:

        def __init__(self, arima):
            self.model = arima.model

        def heteroskedastic(self, significance=0.05) -> bool:
            """
            Goldfeldt Quandt test heteroskedasticity
            :param significance: significance level
            :return: True or False
            """
            return self.model.arima_res_.test_heteroskedasticity(method=None, alternative='two-sided')[0, 1] \
                < significance

        def conditional_heteroskedastic(self, significance=0.05) -> bool:
            """
            Engles LM test H0 homoskedastic
            :param significance: significance level
            :return: True or False of CH
            """
            return het_arch(self.model.resid, ddof=sum(self.model.order))[2] < significance

        @staticmethod
        def llr_test(model1: pm.ARIMA, model2: pm.ARIMA, significance=0.05) -> bool:
            """
            Likelihood ratio test
            :param model1: H0 model
            :param model2: HA model
            :param significance: significance level
            :return: H0 result test
            """
            k1 = len(model1.params())
            k2 = len(model2.params())
            lr = 2 * (k1 - k2) + model2.aic() - model1.aic()
            return chi2.sf(lr, k2 - k1) > significance


class Lstm:

    def __init__(self, train_resids, test_resids, look_back, output_nodes, nb_epoch, batch_size):
        self.train_resids = train_resids
        self.test_resids = test_resids
        self.look_back = look_back
        self.output_nodes = output_nodes
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.hidden_nodes = int(2 * (look_back + output_nodes) / 3)
        self.model = Sequential()

    def mse(self):
        return mean_squared_error(self.test_resids, self.predict())

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return mean_absolute_error(self.test_resids, self.predict())

    def time_series_generator(self):
        return TimeseriesGenerator(self.train_resids, self.train_resids, length=self.look_back, batch_size=self.batch_size)

    def fit(self):
        self.model.add(LSTM(self.hidden_nodes, activation='tanh',
                       recurrent_activation='sigmoid'))
        self.model.add(Dense(self.output_nodes))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model.fit(self.time_series_generator(), epochs=self.nb_epoch, verbose=0)

    def predict(self):
        prediction = []
        first_eval_batch = self.train_resids[-self.look_back:]
        current_batch = first_eval_batch.reshape((1, self.look_back, 1))

        for _ in range(int(len(self.test_resids) / self.output_nodes)):
            pred = self.model.predict(current_batch)[0]
            current_batch = current_batch[:, self.output_nodes:, :]
            for p in pred:
                prediction.append(np.array([p]))
                current_batch = np.append(current_batch, [[np.array([p])]], axis=1)
        return prediction

