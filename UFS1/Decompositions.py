import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa import seasonal
from Description import Data
from scipy import stats
from lib2 import RobustSTL as RB
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

    def plot(self):
        """
        Plot the decomposition results
        """
        if type(self.decomposition) == pd.DataFrame:
            for i in range(4):
                plt.subplot(4, 1, (i + 1))
                if i == 3:
                    plt.scatter(self.decomposition.index, self.decomposition.iloc[:, i])
                else:
                    plt.plot(self.decomposition.iloc[:, i], color='blue')
                plt.title(self.decomposition.columns[i])
            plt.tight_layout()
        else:
            self.decomposition.plot()
        plt.show()

    def periods(self):
        dates = pd.DatetimeIndex([self.df.startDate.unique()[0], self.df.startDate.unique()[1]]).date
        return 12 if (dates[1] - dates[0]).days > 7 else 52

    def time_series(self, keyword):
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()
        return ts

    def time_series_box_cox(self, keyword):
        try:
            ts_bc, _ = stats.boxcox(self.time_series(keyword))
            ts_bc = pd.Series(ts_bc)
            ts_bc.index = self.df.startDate.unique()
            return ts_bc
        except ValueError:
            print(f"Sparsity of {keyword} is to large")
            return self.time_series(keyword)

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
        return self.decomposition

    def decompose_robustSTL(self, keyword):
        """
        Robust STL method based with smoother seasonality
        :param keyword: keyword to be used
        """
        result = RB.RobustSTL(self.df[self.df.keyword == keyword].interest, self.periods(), H=13)
        self.decomposition = pd.DataFrame(list(map(list, zip(*result))), index=self.df.startDate.unique(),
                                          columns=['observed', 'trend', 'seasonal', 'resid'])
        return self.decomposition

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


