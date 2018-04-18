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



