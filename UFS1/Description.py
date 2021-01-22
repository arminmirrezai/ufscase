import matplotlib.pyplot as plt
from pathlib import Path
from ApiExtract import createDir
import numpy as np
from statsmodels.tsa.stattools import adfuller


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

        def stationary(self, keyword):
            p_bic = adfuller(self.df[self.df.keyword == keyword]['interest'], autolag='BIC')[1]
            p_aic = adfuller(self.df[self.df.keyword == keyword]['interest'], autolag='AIC')[1]
            min_p = min(p_bic, p_aic)
            return min_p < 0.025, min_p


        




