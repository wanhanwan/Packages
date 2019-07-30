import pandas as pd
import numpy as np


def df_rolling2(df, window, apply_func, raw=False, *args, **kwargs):
    arr = df.values
    temp = np.moveaxis(np_rolling_window(arr.T, window),0,-1)
    rslt = np.asarray([apply_func(x, *args, **kwargs) for x in temp], dtype='float')
    if raw:
        return rslt
    if rslt.ndim == 1:
        return pd.Series(rslt, index=df.index[-len(rslt):])
    return pd.DataFrame(rslt, index=df.index[-len(rslt):])


def np_rolling_window(array, window):
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
    w = 0.5 ** (np.arange(len(arr)) / half_window)[::-1]
    # w = w[::-1]
    if scale:
        w /= w.sum()
    if arr is None:
        return w
    return arr * w


def return2nav(ret_data, start_point=None):
    """收益转净值"""
    nav = (1.0 + ret_data).cumprod()
    if start_point is None:
        start_point = ret_data.index.min() - pd.Timedelta(1, unit='D')
    nav.loc[start_point, :] = 1.0
    nav.sort_index(inplace=True)
    nav.fillna(method='ffill', inplace=True)
    return nav


def calc_cov(mat1, mat2=None, weight=None, use_ewavg=True):
    """计算加权协方差矩阵
    Parameters:
    ----------
    mat1: 2D-Array
        每一列是一个列向量
    mat2: 2D-Array
        同上
    weight: 1D-Array
        权重向量
    use_ewavg: bool
        是否使用加权历史均值计算协方差
    """
    if mat2 is None:
        mat2 = mat1
    if weight is None or not use_ewavg:
        weight = np.ones(mat1.shape[0], dtype='float64') / mat1.shape[0]
    mat1_demean = mat1 - np.average(mat1, axis=0, weights=weight)[None, :]
    mat2_demean = mat2 - np.average(mat2, axis=0, weights=weight)[None, :]
    mat2_demean *= weight[:, None]
    m = np.dot(mat1_demean.T, mat2_demean)
    return m


def calc_corr(mat1, mat2=None, weight=None, use_ewavg=True):
    """计算加权相关系数"""
    cov = calc_cov(mat1, mat2, weight, use_ewavg)
    std = np.sqrt(np.diag(cov))
    corr = cov / std[:, None] / std[None, :]
    return corr
