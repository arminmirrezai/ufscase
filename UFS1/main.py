from ApiExtract import extract
from Decompositions import Decompose
from scipy import stats
import Description
import Models
import numpy as np
import time

df = extract(range(2015, 2021), 'NL')
dd = Description.Data(df)
dmp = Decompose(df)
arima = Models.Arima(df)


def run_kws(kws, method):
    for kw in kws:
        arima.fit(kw, method)


methods = ['newton', 'nm', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg', 'basishopping']
random_kws = df.keyword.unique()[np.random.randint(0, df.keyword.nunique(), 10)]

# speed test for arima fitting
for method in methods:
    try:
        start_time = time.time()
        run_kws(random_kws, method)
        print("Method: " + method)
        print("run time: " + str(time.time() - start_time))
    except:
        print(f"Method {method} did not converge")
        pass
