from typing import Type
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from pmdarima import ARIMA
from Description import Data
from Decompositions import Decompose
from scipy.stats.distributions import chi2
from statsmodels.stats.diagnostic import het_arch
from DataUtil import *
from csv import reader

class Arima:

    model: Type[ARIMA]
    df_train: Type[pd.DataFrame]
    df_test: Type[pd.DataFrame]

    def __init__(self, df, train_percentage=1.0):
        if train_percentage != 1.0:
            self.df_train = df[df.startDate <= df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
            self.df_test = df[df.startDate > df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        else:
            self.df_train = df
            self.df_test = pd.DataFrame
        self.dd = Data(self.df_train)
        self.dmp = Decompose(self.df_train)
        self.model = pm.ARIMA
        self.test = self.Tests(self)
        self.stats = ""
        self.kw = ""

    @property
    def residuals(self):
        return self.model.resid()

    @property
    def aic(self):
        return self.model.aic()

    @property
    def log_likelihood(self):
        return len(self.model.params()) - self.aic/2

    def save_stats(self, path: str):
        with open(path, 'w') as file:
            file.write(self.stats)

    def time_series(self, keyword, train=True) -> pd.Series:
        if train:
            ts = self.df_train[self.df_train.keyword == keyword]['interest']
            ts.index = self.df_train.startDate.unique()
        else:
            ts = self.df_test[self.df_test.keyword == keyword]['interest']
            ts.index = self.df_test.startDate.unique()
        return ts

    def get_dummies(self, keyword, benchmark=3) -> pd.Series:
        self.dmp.decompose_robustSTL(keyword)
        scores = self.dmp.outlier_score()
        dummies = pd.Series(0, index=scores.index)
        dummies[scores.index[scores > benchmark]] = 1
        return dummies

    def predict(self):
        ts = self.time_series(self.kw, train=False)
        n_periods = len(ts.index)
        dummies = np.zeros_like(ts).reshape(-1, 1)
        return pd.Series(self.model.predict(n_periods, dummies), index=ts.index)

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
        exog = np.array(dummies).reshape(-1, 1) if dummies is not None else None
        years = ts.index[-1].year - ts.index[0].year + 1
        periods = 52 if (ts.index[1].month - ts.index[0].month) == 0 else 12
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
        path = getPath(self.kw,  f"Models/Sarimax/{self.df_train.country.unique()[0]}")
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
        folder_name = f"Models/Sarimax/{self.df_train.country.unique()[0]}"
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





# def garch_model(self):
    #     if self.test.conditional_heteroskedastic():
    #         p, o, q = self.model.order
    #         if p > 0 or o > 0:
    #             am = arch_model(self.residuals, p=p, o=o, q=q)
    #             self.model = am.fit(update_freq=0)
    #     else:  # errors are homoskedastic
    #         pass