import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# STL
register_matplotlib_converters()
sns.set_style('darkgrid')

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)

df = pd.read_csv(r'C:\Users\Naam\Desktop\seminar case study\ufscode\Data\case_study_trends_data_20210107.csv')
ackerbohne = df.iloc[1:209, 1].tolist()

ackerbohne = pd.Series(ackerbohne, index = pd.date_range('1-1-2017', periods=len(ackerbohne), freq='W'), name = 'Ackerbohne')

stl = STL(ackerbohne, seasonal=13)
res = stl.fit()
fig = res.plot()
plt.show()

# Triple exponentional soothing
fit1 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
# fit2 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
# fit4 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
# results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
results["Additive Dam"]   = [fit3.params[p] for p in params] + [fit3.sse]
# results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]

ax = ackerbohne.plot(figsize=(10,6), marker='o', color='black', title="Forecasts from Holt-Winters' multiplicative method" )
ax.set_ylabel("Interest in Ackerbohne")
ax.set_xlabel("Year")
fit1.fittedvalues.plot(ax=ax, style='--', color='red')
fit2.fittedvalues.plot(ax=ax, style='--', color='green')

plt.show()

fit1.forecast(8).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)
fit2.forecast(8).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)

plt.show()
print("Forecasting Ackerbohne interest in DE using Holt-Winters method with both additive and multiplicative seasonality.")

fit1 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='add', initialization_method="estimated").fit()
# fit2 = ExponentialSmoothing(ackerbohne, seasonal_periods=4, trend='add', seasonal='mul', initialization_method="estimated").fit()

df = pd.DataFrame(np.c_[ackerbohne, fit1.level, fit1.trend, fit1.season, fit1.fittedvalues],
                  columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],index=ackerbohne.index)
df.append(fit1.forecast(8).rename(r'$\hat{y}_t$').to_frame(), sort=True)

# df = pd.DataFrame(np.c_[ackerbohne, fit2.level, fit2.trend, fit2.season, fit2.fittedvalues],
#                   columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],index=ackerbohne.index)
# df.append(fit2.forecast(8).rename(r'$\hat{y}_t$').to_frame(), sort=True)

states1 = pd.DataFrame(np.c_[fit1.level, fit1.trend, fit1.season], columns=['level','slope','seasonal'], index=ackerbohne.index)
# states2 = pd.DataFrame(np.c_[fit2.level, fit2.trend, fit2.season], columns=['level','slope','seasonal'], index=ackerbohne.index)
fig, [[ax1, ax4],[ax2, ax5], [ax3, ax6]] = plt.subplots(3, 2, figsize=(12,8))
states1[['level']].plot(ax=ax1)
states1[['slope']].plot(ax=ax2)
states1[['seasonal']].plot(ax=ax3)
states2[['level']].plot(ax=ax4)
states2[['slope']].plot(ax=ax5)
states2[['seasonal']].plot(ax=ax6)
plt.show()


