import matplotlib.pyplot as plt
from pathlib import Path
from ApiExtract import createDir
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class Data:

    def __init__(self, df):
        self.df = df
        self.statistics = self.Statistics(self.df)
        self.visualize = self.Visualize(self.df)

    class Visualize:

        def __init__(self, df):
            self.df = df

        def figure(self, keyword, save=False):
            self.df[self.df.keyword == keyword].plot(x='startDate', y='interest', label=keyword)
            plt.ylabel('Relative search volume')
            plt.xlabel('Weeks')
            if save:
                self._save_graph(keyword)
            plt.show()

        def hist_sparsity(self, save=False):
            sparsity = [1 - (np.count_nonzero(self.df[self.df.keyword == keyword]['interest']))/self.df.startDate.nunique()
                        for keyword in self.df.keyword.unique()]
            plt.hist(sparsity, bins=np.arange(10)/10, label='Sparsity of time series')
            plt.xlabel('% of sparsity')
            plt.ylabel('Number of time series')
            plt.legend()
            if save:
                self._save_graph('sparsity')
            plt.show()

        def _save_graph(self, file_name):
            folder_path = Path.cwd().absolute().parents[0].as_posix() + "/Data/Graphs/" + self.df.country.unique()[0]
            createDir(folder_path)
            plt.savefig(folder_path + "/" + file_name + ".png")

    class Statistics:

        def __init__(self, df):
            self.df = df

        def sparsity(self, keyword):
            return 1 - np.count_nonzero(self.df[self.df.keyword == keyword]['interest']) / self.df.startDate.nunique()

        def stationary(self, keyword, first_difference=False, significance='5%'):
            if significance not in {'1%', '5%', '10%'}:
                raise ValueError("Significance leven can only be 1,5 or 10 %")
            ts = self.df[self.df.keyword == keyword]['interest']
            if first_difference:
                ts = (ts - ts.shift()).dropna()
            adf_test = adfuller(ts, autolag='AIC')
            kpss_test = kpss(ts, regression='c', nlags='auto')
            h0_adf = abs(adf_test[0]) < abs(adf_test[4][significance])
            h0_kpss = abs(kpss_test[0]) > abs(kpss_test[3][significance])
            if h0_kpss and not h0_adf:  # stationary
                return True
            elif not h0_kpss and h0_adf:  # non stationary
                return False
            elif h0_kpss and h0_adf:  # de trend series
                return False
            elif not h0_kpss and not h0_adf:  # difference series
                return False










