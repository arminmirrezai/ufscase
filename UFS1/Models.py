import numpy as np
import pmdarima as pm
from Description import Data
from scipy.stats.distributions import chi2


class Arima:

    def __init__(self, df):
        self.df = df
        self.dd = Data(df)
        self.model = pm.ARIMA
        self.stats = ""

    @property
    def residuals(self):
        return self.model.resid() if self.model is not None else 0

    @property
    def aic(self):
        return self.model.aic

    @property
    def log_likelihood(self):
        return len(self.model.params()) - self.aic/2

    def time_series(self, keyword):
        ts = self.df[self.df.keyword == keyword]['interest']
        ts.index = self.df.startDate.unique()
        return ts

    def fit(self, keyword):
        """
        Fit the best arima or sarima model for the keyword
        :param keyword: keyword
        :return: fitted model
        """
        stationary, has_trend = self.dd.statistics.stationary(keyword)
        if stationary:
            arima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=0,
                                  trend='ct' if has_trend else 'c', with_intercept=True)
            sarima = pm.auto_arima(self.time_series(keyword), seasonal=True, stationary=stationary, d=0,
                                   trend='ct' if has_trend else 'c', with_intercept=True)
        else:
            stationary, has_trend = self.dd.statistics.stationary(keyword, first_difference=True)
            arima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=1,
                                  trend='ct' if has_trend else 'c', with_intercept=True)
            sarima = pm.auto_arima(self.time_series(keyword), seasonal=False, stationary=stationary, d=1,
                                   trend='ct' if has_trend else 'c', with_intercept=True)
        if arima.order == sarima.order:
            self.model = arima
        else:
            self.model = arima if self.llr_test(arima, sarima) else sarima
        self._write_stats(keyword)
        return self.model

    def _write_stats(self, keyword):
        """
        Write the stats of the model
        :param keyword:
        :return:
        """
        self.stats += keyword + ":\n"
        self.stats += "Order: " + str(self.model.order) + "\n"
        self.stats += "SSR: " + str(sum(np.square(self.residuals)))
        self.stats += "AIC: " + str(self.aic)

    @staticmethod
    def llr_test(model1: pm.ARIMA, model2: pm.ARIMA, significance=0.05):
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

#
#
# def multiPlot(trainSets, testSets, forecasts):
#
#     fig, ax = plt.subplots(len(testSets.columns), 1)
#     for i in range(len(testSets.columns)):
#         ax[i].plot(trainSets[trainSets.columns[i]], color = 'black')
#         ax[i].plot(testSets[testSets.columns[i]], color = 'green')
#         ax[i].plot(forecasts[forecasts.columns[i]], 'r--')
#
#     plt.show()
#
#
#
#
#
# def runKeywords(df, keywords):
#     trainSets = pd.DataFrame(columns=keywords)
#     testSets = pd.DataFrame(columns=keywords)
#     forecasts = pd.DataFrame(columns = keywords)
#     for keyword in keywords:
#         (trainData, testData) = readData(df, keyword)
#         trainSets[keyword] = trainData
#         testSets[keyword] = testData
#         forecasts[keyword] = autoArima(trainData, testData)[1]
#
#     return trainSets, testSets, forecasts
#
# def singlePlot(trainData, testData, arimaForecast, SARIMA):
#
#     print("the lowest value is", min(arimaForecast))
#
#     plt.figure()
#     plt.plot(trainData, color = 'black', label = "Training data")
#     plt.plot(testData, color = 'green', label = "Testing data")
#     plt.plot(arimaForecast, 'r-.', label = "Prediction")
#     plt.legend(loc = 'upper left')
#     plt.show()
#
#     SARIMA.plot_diagnostics()
#     plt.show()
#
# def manualArima(trainData, testData, order, seasonal_order):
#
#     SARIMAestimation = ARIMA(trainData ,order=order, seasonal_order=seasonal_order).fit()
#     forecast = SARIMAestimation.predict(start=testData.index[0], end=testData.index[len(testData.index)-1])
#     arimaForecast = pd.Series(forecast, index = testData.index)
#
#     return SARIMAestimation, arimaForecast
#
# def autoArima(trainData, testData):
#
#     SARIMA = pm.auto_arima(trainData, start_p=1, start_q=1,test='adf',max_p=3, max_q=3, m=len(testData) ,start_P=0, seasonal=True,d=None, D=1, trace=True,
#                          error_action='ignore',  suppress_warnings=True, stepwise=True, trend = 'ct')
#     forecast = SARIMA.predict(n_periods = len(testData))
#     arimaForecast = pd.Series(forecast, index = testData.index)
#
#     return SARIMA, arimaForecast
#
