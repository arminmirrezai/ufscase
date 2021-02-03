import statsmodels.api as sm
from statsmodels.tsa import seasonal
from Description import Data
from scipy import stats
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Decompose:

    def __init__(self, df):
        self.df = df
        self.decomposition = None
        self.stats = Data(df).statistics

    @property
    def trend(self):
        return self.decomposition.trend

    @property
    def seasonal(self):
        return self.decomposition.seasonal

    @property
    def remainder(self):
        return self.decomposition.resid

    def time_series(self, keyword):
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()
        return ts

    def time_series_box_cox(self, keyword):
        ts_bc, _ = stats.boxcox(self.time_series(keyword))
        ts_bc = pd.Series(ts_bc)
        ts_bc.index = self.df.startDate.unique()
        return ts_bc

    def decompose_ma(self, keyword):
        """
        Decomposition by moving average design
        :param keyword: keyword to be used
        """
        self.decomposition = sm.tsa.seasonal_decompose(self.time_series(keyword), model='additive')

    def decompose_stl(self, keyword, robust=False):
        """
        Decomposition by STL LOESS
        :param robust: robust estimation orr not
        :param keyword: keyword to be used
        """
        ts = self.time_series(keyword)
        ts_bc = self.time_series_box_cox(keyword)
        decomp_add = seasonal.STL(ts, robust=robust).fit()
        decomp_mult = seasonal.STL(ts_bc, robust=robust).fit()
        if stats.jarque_bera(decomp_add.resid).pvalue > stats.jarque_bera(decomp_mult.resid).pvalue:
            self.decomposition = decomp_add
        else:
            self.decomposition = decomp_mult

    def outlier_score(self):
        """
        Output outlier scores based on 1.5 above first and third quantile
        :return:
        """
        quants = np.quantile(self.remainder, [0.25, 0.75])
        iqr = np.diff(quants)
        lb, ub = quants + 1.5 * iqr * [-1, 1]
        scores = np.zeros_like(self.remainder)
        for i in range(len(scores)):
            error = self.remainder[i]
            if error < lb:
                scores[i] = (lb - error) / iqr
            elif error > ub:
                scores[i] = (error - ub) / iqr
        scores = pd.Series(scores)
        scores.index = self.df.startDate.unique()
        return scores

    def trend_F(self):
        """"
        F-measure for trend
        """
        return max(0, 1 - np.var(self.decomposition.resid)/np.var(self.decomposition.trend + self.decomposition.resid))

    def seasonality_F(self):
        """
        F-measure for seasonality
        :return:
        """
        return max(0, 1 - np.var(self.decomposition.resid)/np.var(self.decomposition.seasonal + self.decomposition.resid))


