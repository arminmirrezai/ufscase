import matplotlib.pyplot as plt
from pathlib import Path
from ApiExtract import createDir
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings("ignore")


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

        def figure_residuals(self):
            pass

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

        def sparsity(self, keyword, country=''):
            if country != '':
                return 1 - np.count_nonzero(self.df[(self.df.keyword == keyword) &
                                            (self.df.country == country)]['interest']) / self.df.startDate.nunique()
            else:
                return 1 - np.count_nonzero(self.df[self.df.keyword == keyword]['interest']) / self.df.startDate.nunique()

        def stationary(self, keyword, first_difference=False, significance='5%'):
            """
            Stationarity test strategy implemented
            :param keyword: keyword
            :param first_difference: True if you want to take first difference
            :param significance: could be 1, 5 or 10 %
            :return: Stationary or not, with deterministic trend or not
            """
            if significance not in {'1%', '5%', '10%'}:
                raise ValueError("Significance level can only be 1,5 or 10 %")
            ts = self.df[self.df.keyword == keyword]['interest']
            if first_difference: ts = ts.diff(1).dropna()

            adf_test = adfuller(ts, regression='ct', autolag='AIC', regresults=True)
            has_unit_root = adf_test[0] > adf_test[2][significance]  # negative one sided
            if has_unit_root:
                has_det_trend = ttest_1samp(ts.diff(1).dropna(), 0).pvalue < int(significance[:-1]) / 100
                if not has_det_trend:
                    adf_test = adfuller(ts, regression='c', autolag='AIC', regresults=True)
                    has_unit_root = adf_test[0] > adf_test[2][significance]  # negative one sided
                return has_unit_root, has_det_trend
            else:
                reg = adf_test[3].resols
                r = np.append(np.zeros_like(reg.params[:-1]), 1)
                has_det_trend = reg.t_test(r).pvalue.item(0) < int(significance[:-1]) / 100

                kpss_test = kpss(ts, regression=('ct' if has_det_trend else 'c'))
                return kpss_test[0] < kpss_test[3][significance], has_det_trend












