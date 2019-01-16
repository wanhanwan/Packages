import pandas as pd
import numpy as np
from functools import partial


def df_rolling2(dataframe, window, apply_func, *args, **kwargs):
    arr = dataframe.sort_index().values
    temp = np.moveaxis(_np_rolling_window(arr.T, window),0,-1)
    rslt = np.array([apply_func(x, *args, **kwargs) for x in temp])
    return pd.DataFrame(rslt, index=dataframe.index[-len(rslt):], columns=dataframe.columns[:rslt.shape[1]])


def df_rolling(dataframe, window, apply_func, *args, **kwargs):
    arr = dataframe.sort_index().values
    appfun = partial(apply_func, *args, **kwargs)
    rslt = np.apply_along_axis(appfun, axis=1, arr=np.moveaxis(_np_rolling_window(arr.T, window),0,-1))
    return pd.DataFrame(rslt, index=dataframe.index[-len(rslt):], columns=dataframe.columns[:rslt.shape[1]])


def _np_rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


class RollingResultWrapper(object):
    def __init__(self, func):
        self.rollingfunc = func
        self.rlist = []

    def reset(self):
        self.rlist = []

    def __call__(self, *args, **kwargs):
        """pandas.DataFrame.rolling Wrapper
        当func返回多个值的时候，自动返回一个DataFrame
        """
        rslt = self.rollingfunc(*args, **kwargs)
        self.rlist.append(rslt)
        return 1

    @property
    def df(self):
        return pd.DataFrame(self.rlist)


def expweighted(half_window, arr_len=None, arr=None, scale=False):
    if arr is None:
        arr = np.ones(arr_len)
    w = 0.5 ** (np.arange(len(arr)) / half_window)
    w = w[::-1]
    if scale:
        w /= w.sum()
    if arr is None:
        return w
    return arr * w


def save_excel(data_dict, file_name, date_format=None,
               datetime_format=None):
    with pd.ExcelWriter(file_name,
                        date_format=date_format,
                        datetime_format=datetime_format) as f:
        for sheet_name, df in data_dict.items():
            df.to_excel(f, sheet_name=sheet_name)
