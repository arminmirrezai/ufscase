import numpy as np
import pmdarima as pm
from Description import Data
from scipy.stats.distributions import chi2
from statsmodels.stats.diagnostic import het_arch


class Arima:

    def __init__(self, df):
        self.df = df
        self.dd = Data(df)
        self.model = pm.ARIMA
        self.test = self.Tests(self)
        self.stats = ""

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

    def time_series(self, keyword):
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()
        return ts

    def fit(self, keyword, method='nm'):
        """
        Fit the best arima or sarima model for the keyword
        :param method: Default Nelder-Mead based on speed
        :param keyword: keyword
        :return: fitted model
        """
        stationary, has_trend = self.dd.statistics.stationary(keyword)
        if stationary:
            arima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=0, method=method,
                                  trend='ct' if has_trend else 'c', with_intercept=True, max_order=None)
            sarima = pm.auto_arima(self.time_series(keyword), seasonal=True, stationary=stationary, d=0, method=method,
                                   trend='ct' if has_trend else 'c', with_intercept=True, max_order=None)
        else:
            stationary, has_trend = self.dd.statistics.stationary(keyword, first_difference=True)
            arima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=1, method=method,
                                  trend='ct' if has_trend else 'c', with_intercept=True, max_order=None)
            sarima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=1, method=method,
                                   trend='ct' if has_trend else 'c', with_intercept=True, max_order=None)
        if arima.order == sarima.order:
            self.model = arima
        else:
            self.model = arima if self.llr_test(arima, sarima) else sarima
        self._write_stats(keyword)
        return self.model

    def garch_model(self):
        pass

    def _write_stats(self, keyword):
        """
        Write the stats of the model
        :param keyword:
        :return:
        """
        self.stats += "\n" + keyword + ":\n"
        self.stats += "Order: " + str(self.model.order) + "\n"
        self.stats += "SSR: " + str(sum(np.square(self.residuals)))
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
            lr = 2 * (k1 - k2) + model2.aic - model1.aic
            return chi2.sf(lr, k2 - k1) > significance