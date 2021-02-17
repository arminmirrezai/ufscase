import numpy as np
import pandas as pd
import pmdarima as pm
from Description import Data
from Decompositions import Decompose
from scipy.stats.distributions import chi2
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model


class Arima:

    def __init__(self, df):
        self.df = df
        self.dd = Data(df)
        self.dmp = Decompose(df)
        self.model = pm.ARIMA
        self.test = self.Tests(self)
        self.dummy = None
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

    def get_dummies(self, keyword, benchmark=2.5):
        self.dmp.decompose_robustSTL(keyword)
        scores = self.dmp.outlier_score()
        dummies = pd.Series(0, index=scores.index)
        dummies[scores.index[scores > benchmark]] = 1
        self.stats += "Outliers: " + str(int(len(dummies))) + "\n"
        return dummies

    def fit(self, keyword, method='nm'):
        """
        Fit the best arima or sarima model for the keyword
        :param method: Default Nelder-Mead based on speed
        :param keyword: keyword
        :return: fitted model
        """
        stationary, has_trend = self.dd.statistics.stationary(keyword)
        if stationary:
            self._fit(self.time_series(keyword), self.get_dummies(), stationary, 'ct' if has_trend else 'c', 0, method)
        else:
            stationary, has_trend = self.dd.statistics.stationary(keyword, first_difference=True)
            self._fit(self.time_series(keyword), self.get_dummies(), stationary, 'ct' if has_trend else 'c', 1, method)

        self._write_stats(keyword)
        return self.model

    def _fit(self, ts, dummies, stationary, trend, diff, method='nm'):
        exog = np.array(dummies).reshape(-1, 1)
        arima = pm.auto_arima(y=ts, X=exog, seasonal=False, stationary=stationary, d=diff, method=method,
                              trend=trend, with_intercept=True, max_order=None, max_P=len(ts.index))
        sarima = pm.auto_arima(y=ts, X=exog, seasonal=True, stationary=stationary, d=diff, method=method,
                               trend=trend, with_intercept=True, max_order=None, max_P=len(ts.index))
        self.model = arima if self.test.llr_test(arima, sarima) else sarima

    def garch_model(self):
        if self.test.conditional_heteroskedastic():
            p, o, q = self.model.order
            if p > 0 or o > 0:
                am = arch_model(self.residuals, p=p, o=o, q=q)
                self.model = am.fit(update_freq=0)
        else:  # errors are homoskedastic
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
            lr = 2 * (k1 - k2) + model2.aic() - model1.aic()
            return chi2.sf(lr, k2 - k1) > significance
