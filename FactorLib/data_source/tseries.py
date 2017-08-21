from empyrical.stats import cum_returns_final
from .trade_calendar import _to_offset, tc
import pandas as pd
import numpy as np


def _convert_to_nav(ret, ret_freq=None):
    """
    从收益率转成净值数据
    :param ret_freq: ret的时间频率。若为None,默认时间频率是'1d'
    :param ret: series or dataframe with datetimeindex.当ret为dataframe时，每一列代表一组收益率序列。
    :return nav: 净值
    """
    if ret_freq is None:
        ret_freq = '1d'

    last_day = tc.tradeDayOffset(ret.index.min(), -1, freq=ret_freq, retstr=None)
    ret.loc[last_day] = 0
    nav = (ret.sort_index() + 1).cumprod()
    return nav


def _flaten(ret):
    return ret.unstack()


def _is_nan(p):
    try:
        return np.isnan(p).all()
    except:
        return np.isnan(p)


def resample_returns(ret, convert_to, ret_freq=None):
    """
    resample收益率序列。

    :param ret: series or dataframe. 收益率序列

    :param convert_to: 目标频率。支持交易日频率和日历日频率

    :param ret_freq: ret的频率

    :return: p

    """
    is_mul = False
    if isinstance(ret.index, pd.MultiIndex):
        ret = _flaten(ret)
        is_mul = True
    offset = _to_offset(convert_to)
    p = ret.resample(rule=offset, closed='right', label='right').agg(cum_returns_final)
    if is_mul:
        p = p.stack()
    if offset.onOffset(ret.index[-1]) and _is_nan(p.iloc[-1]):
        p = p.iloc[:-1]
    return p


def resample_func(data, convert_to, func):
    """
    应用任意一个重采样函数。

    :param data: dataframe or series

    :param convert_to: 目标频率。支持交易日频率和日历日频率

    :param func: 重采样函数，字符串只应用于pandas.patch函数

    :return: p

    """
    is_mul = False
    offset = _to_offset(convert_to)
    if isinstance(data.index, pd.MultiIndex):
        if 'IDs' in data.index.names:
            data = data.reset_index(level='IDs')
        else:
            data = data.reset_index(level=1)
        is_mul = True
        p = data.groupby('IDs', group_keys=False).resample(
            rule=offset, closed='right', label='right').agg(func)
    elif isinstance(data.index, pd.DatetimeIndex):
        p = data.resample(rule=_to_offset(convert_to), closed='right', label='right').agg(func)
    else:
        raise ValueError("The index of data must be MultiIndex or DatetimeIndex")
    if is_mul:
        p = p.swaplevel().sort_index()
    if len(func) > 1 and isinstance(func, dict):
        p = p.swaplevel().unstack()
    if offset.onOffset(data.index[-1]) and _is_nan(p.iloc[-1]):
        p = p.iloc[:-1]
    return p
