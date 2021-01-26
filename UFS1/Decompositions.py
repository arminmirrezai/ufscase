import statsmodels.api as sm
from statsmodels.tsa import seasonal
from Description import Data
import numpy as np


class Decompose:

    def __init__(self, df):
        self.df = df
        self.ts_decomp = None
        self.stats = Data(df).statistics

    @property
    def trend(self):
        return self.ts_decomp.trend

    @property
    def seasonal(self):
        return self.ts_decomp.seasonal

    @property
    def remainder(self):
        return self.ts_decomp.resid

    def decompose_ma(self, keyword):
        """
        Decomposition by moving average design
        :param keyword: keyword to be used
        """
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()

        if self.stats.stationary(keyword):
            self.ts_decomp = sm.tsa.seasonal_decompose(ts, model='additive')
        elif self.stats.stationary(keyword, first_difference=True):
            self.ts_decomp = sm.tsa.seasonal_decompose(ts, model='multiplicative')
        else:
            print(f"No decomposition possible for {keyword}")

    def decompose_stl(self, keyword):
        """
        Decomposition by STL LOESS
        :param keyword: keyword to be used
        """
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()
        self.ts_decomp = seasonal.STL(ts)

    def trend_F(self):
        """"
        F-measure for trend
        """
        return max(0, 1 - np.var(self.ts_decomp.resid)/np.var(self.ts_decomp.trend + self.ts_decomp.resid))

    def seasonality_F(self):
        """
        F-measure for seasonality
        :return:
        """
        return max(0, 1 - np.var(self.ts_decomp.resid)/np.var(self.ts_decomp.seasonal + self.ts_decomp.resid))


