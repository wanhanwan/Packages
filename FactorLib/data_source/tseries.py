# coding: utf-8
from empyrical.stats import cum_returns_final
from .trade_calendar import _to_offset, tc
from fastcache import clru_cache
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
            raise KeyError("No index column named IDs")
        is_mul = True
        p = data.groupby('IDs', group_keys=False).resample(
            rule=offset, closed='right', label='right').agg(func)
    elif isinstance(data.index, pd.DatetimeIndex):
        p = data.resample(rule=_to_offset(convert_to), closed='right', label='right').agg(func)
    else:
        raise ValueError("The index of data must be MultiIndex or DatetimeIndex")
    if is_mul:
        p = p.set_index('IDs', append=True).sort_index()
    if len(func) > 1 and isinstance(func, dict):
        p = p.swaplevel().unstack()
    if offset.onOffset(data.index[-1]) and _is_nan(p.iloc[-1]):
        p = p.iloc[:-1]
    # p.index.freq = offset
    return p


def date_shift(dataframe, shift):
    """
    数据框的时间漂移。

    当数据框的索引是MultiIndex时，函数会先把数据框进行unstack再进行shift
    :param dataframe: 数据框

    :param shift: 漂移步长

    :return: dataframe

    """

    mul = False
    if isinstance(dataframe.index, pd.MultiIndex):
        dataframe = dataframe.unstack()
        mul = True
    dataframe = dataframe.shift(shift)
    if mul:
        return dataframe.stack()
    else:
        return dataframe


@clru_cache()
def _tradeDayOffset(x, shift, freq):
    return tc.tradeDayOffset(x, shift, freq=freq, retstr=None)


def move_dtindex(dataframe, shift, freq):

    """
    数据框的时间索引漂移，只改变索引，数据并不发生漂移
    :param dataframe: 数据框

    :param shift: 漂移步长

    :param freq: 漂移频率

    :return: new_frame

    """
    is_ser = False
    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame()
        is_ser = True
    if isinstance(dataframe.index, pd.MultiIndex):
        level_number = dataframe.index._get_level_number('date')
        dates_shift = [_tradeDayOffset(x, shift, freq=freq) for x in dataframe.index.levels[level_number]]
        new_frame = pd.DataFrame(dataframe.values, index=dataframe.index.set_levels(dates_shift, level='date'),
                                 columns=dataframe.columns)
    else:
        dates_shift = [_tradeDayOffset(x, shift, freq=freq) for x in dataframe.index]
        new_frame = pd.DataFrame(dataframe.values, index=pd.DatetimeIndex(dates_shift, name='date'),
                                 columns=dataframe.columns)
    if is_ser:
        new_frame = new_frame.iloc[:, 0]
    return new_frame


def reindex_date_of_multiindex(series, dates, method='ffill'):
    """对一个具有多重索引的DataFrame进行日期重索引."""
    if isinstance(series, pd.Series):
        df = series.unstack()
        new_df = df.reindex(dates, method=method).stack()
    else:
        raise NotImplementedError("reindexing a dataframe is not implemented.")
    return new_df
