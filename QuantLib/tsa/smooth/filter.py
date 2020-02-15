#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# filter.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2019/11/25 下午1:04:04
"""滤波器"""
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
pd.options.mode.use_inf_as_na = True


def llt_filter(ts, d=6, alpha=None):
    """LLT 低延迟二阶低通滤波器
    
    Parameters:
    -----------
    ts: Series
        时间序列
    d: int 
        滤波器参数
    alpha: float
        滤波器参数，这里取:2/(1+d)
    """
    if alpha is None:
        alpha = 2 / (1 + d)
    ts_v = np.concatenate(([ts.iat[0]]*50, ts.values))
    llt = np.zeros(len(ts_v))
    for t, pt in enumerate(ts_v):
        if t == 0:
            llt_t = (alpha - alpha ** 2 / 4) * pt
            llt[t] = llt_t
        if t == 1:
            llt_t = (alpha - alpha ** 2 / 4) * pt + (alpha ** 2 / 2) * ts_v[t-1]
            llt[t] = llt_t
        if t == 2:
            llt_t = ((alpha - alpha ** 2 / 4) * pt + (alpha ** 2 / 2) * ts_v[t - 1] -
                     (alpha - 3 * alpha ** 2 / 4) * ts_v[t-2])
            llt[t] = llt_t
        else:
            llt_t = ((alpha - alpha ** 2 / 4) * pt + (alpha ** 2 / 2) * ts_v[t - 1] -
                     (alpha - 3 * alpha ** 2 / 4) * ts_v[t-2] + 2 * (1 - alpha) * llt[t-1] -
                     (1 - alpha) ** 2 * llt[t-2])
            llt[t] = llt_t
    llt = pd.Series(llt[50:], index=ts.index)
    # llt.iloc[:36] = np.nan
    return llt


def hp_filter(ts, min_periods=10, freq='M', lamb=None):
    """
    HP滤波

    Parameters:
    -----------
    ts: Series
    window: int
        滚动时间窗口
    alpha: float:
        hp滤波参数
    """
    if lamb is None:
        lamb = {
            'M' : 129600,
            'Q' : 1600,
            'Y' : 6.25,
            'D' : 1600*60**4
        }[freq]

    def _get_last_one(sub_ts, l):
        sub_cy, sub_trend = hpfilter(sub_ts, l)
        return sub_cy[-1]

    ts_nonna = ts.dropna()
    cycle, trend = hpfilter(ts_nonna, lamb)
    roll_trend = ts_nonna.expanding(
        min_periods=min_periods).agg(_get_last_one, l=lamb)
    roll_trend.iloc[:min_periods] = cycle
    return roll_trend.reindex(ts.index)
