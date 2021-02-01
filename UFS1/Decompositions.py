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
        self.ts_decomp = sm.tsa.seasonal_decompose(self.time_series(keyword), model='additive')

    def decompose_stl(self, keyword):
        """
        Decomposition by STL LOESS
        :param keyword: keyword to be used
        """
        ts = self.time_series(keyword)
        ts_bc = self.time_series_box_cox(keyword)
        decomp_add = seasonal.STL(ts).fit()
        decomp_mult = seasonal.STL(ts_bc).fit()
        if stats.jarque_bera(decomp_add.resid).pvalue > stats.jarque_bera(decomp_mult.resid).pvalue:
            self.ts_decomp = decomp_add
        else:
            self.ts_decomp = decomp_mult

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


