from ApiExtract import extract
from Decompositions import Decompose
import Description
import time
import Models
from sklearn.metrics import mean_squared_error
import os
import errno
import signal
from functools import wraps

class TimeoutError(Exception):
    pass


def timeout(seconds=30, error_messag=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_messag)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                res = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return res

        return wraps(func)(wrapper)
    return decorator


df = extract(range(2016, 2021), 'ES')
dd = Description.Data(df)
dmp = Decompose(df)

seconds = 180
methods = ['nm', 'bfgs', 'lbfgs']
model = Models.Arima(df, 0.8)


@timeout(seconds, os.strerror(errno.ETIMEDOUT))
def fit(kw, method):
    start_time = time.time()
    model.fit(kw, method)
    error = mean_squared_error(model.time_series(kw, False), model.predict())
    print(f"Method: {method}, MSE: {error}, AIC: {model.aic}, Time: {time.time() - start_time} seconds")


for method in methods:
    try:
        fit('jamon', method)
    except TimeoutError:
        print(f"Method {method} took longer than {seconds} seconds")
    except BaseException as e:
        print(e)
        print(f"Method {method} failed!")




