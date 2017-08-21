"""加载财务数据的工具函数"""
import pandas as pd
import numpy as np
from utils.tool_funcs import RollBackNPeriod, get_all_report_periods
from utils.datetime_func import DateStr2Datetime, Datetime2DateStr


def get_single_quater_data(raw_data):
    '''把数据转变成单季度数据, 多用在利润表和现金流量表
    raw_data代表财务数据，以报告期为索引
    '''
    all_periods = raw_data.index.get_level_values(0).unique()
    l = []
    for period in all_periods:
        lq = RollBackNPeriod(Datetime2DateStr(period), 1)
        if period.month == 3:
            l.append(raw_data.loc[[period]])
        elif DateStr2Datetime(lq) in all_periods:
            data_lq = raw_data.loc[period] - raw_data.loc[DateStr2Datetime(lq)]
            data_lq.index = pd.MultiIndex.from_product([[period], data_lq.index], names=['date', 'IDs'])
            l.append(data_lq)
        else:
            pass
    new_data = pd.concat(l)
    return new_data


def get_last_nyear_report(raw_data, back=1):
    '''过去n年报数据'''
    all_periods = raw_data.index.get_level_values(0).unique()
    l = []
    for period in all_periods:
        ly = DateStr2Datetime(str(period.year - back) + '1231')
        if ly in all_periods:
            data_ly = raw_data.loc[[ly]].copy()
            data_ly.index = pd.MultiIndex.from_product([[period], data_ly.index.get_level_values(1)],
                                                       names=data_ly.index.names)
            l.append(data_ly)
        else:
            pass
    new_data = pd.concat(l)
    return new_data


def get_latest_year_report(raw_data, keep_dates=True):
    '''最近的一份年报
    keep_dates: 只保留raw_data中的日期
    '''
    if keep_dates:
        all_periods = raw_data.index.get_level_values(0).unique()
    else:
        min_period = raw_data.index.get_level_values(0).min()
        max_period = raw_data.index.get_level_values(0).max()
        all_periods = get_all_report_periods(min_period, max_period)
    l=[]
    for period in all_periods:
        if period.month != 12:
            ly = DateStr2Datetime(str(period.year - 1) + '1231')
        else:
            ly = period
        if ly in all_periods:
            data_ly = raw_data.loc[[ly]].copy()
            data_ly.index = pd.MultiIndex.from_product([[period], data_ly.index.get_level_values(1)],
                                                       names=data_ly.index.names)
            l.append(data_ly)
        else:
            pass
    new_data = pd.concat(l)
    return new_data


def get_nyears_back(raw_data, back=1):
    """N年前同期数据"""
    all_periods = raw_data.index.get_level_values(0).unique()
    l=[]
    for period in all_periods:
        ly = pd.datetime(period.year-back, period.month, period.day)
        if ly in all_periods:
            data_ly = raw_data.loc[[ly]].copy()
            data_ly.index = pd.MultiIndex.from_product([[period], data_ly.index.get_level_values(1)],
                                                       names=data_ly.index.names)
            l.append(data_ly)
        else:
            pass
    new_data = pd.concat(l)
    return new_data


def get_ttm_data(raw_data):
    '''把数据转换成TTM数据'''
    data_last_year = get_last_nyear_report(raw_data, 1)   # 上年年报
    data_ly = shift_period(raw_data, 4)
    return raw_data + data_last_year - data_ly


def shift_period(raw_data, shift_periods, keep_data_available=False):
    '''
    财务数据报告期前移/后移
    raw_data代表财务数据，以报告期为索引
    keep_data_available:保证数据的可获得性，一季度的上一季度是上一年的三季度
    '''
    if keep_data_available:
        l = []
        all_periods = raw_data.index.get_level_values(0).unique()
        for period in all_periods:
            # 保证数据的可获得性，一季度的上一季度是上一年的三季度
            if period.month == 3 and shift_periods == 1:
                target_period = DateStr2Datetime(
                    RollBackNPeriod(Datetime2DateStr(period), shift_periods+1))
            else:
                target_period = DateStr2Datetime(
                    RollBackNPeriod(Datetime2DateStr(period), shift_periods))
            if target_period in all_periods:
                target_data = raw_data.loc[[target_period]].copy()
                target_data.index = pd.MultiIndex.from_product([[period], target_data.index.get_level_values(1)],
                                                               names=target_data.index.names)
                l.append(target_data)
        return pd.concat(l)
    min_period = raw_data.index.get_level_values(0).min()
    max_period = raw_data.index.get_level_values(0).max()
    period_index = get_all_report_periods(min_period, max_period)
    data_unstack = raw_data.unstack().sort_index().reindex(period_index).shift(shift_periods)
    return data_unstack.stack()


def _inc_rate(data):
    if np.nan not in data and data.iloc[1] != 0:
        return (data.iloc[0] - data.iloc[1]) / abs(data.iloc[1])
    else:
        return np.nan


def yoy_tb(raw_data, quater=False):
    '''同比增长率
    quater: 是否拆成单季度同比
    '''
    if quater:
        raw_data = get_single_quater_data(raw_data)
    data_lq = shift_period(raw_data, 4)
    data = pd.concat([raw_data, data_lq], axis=1)
    inc = data.apply(_inc_rate, axis=1).to_frame()
    inc.columns = ['inc_tb']
    return inc


def yoy_hb(raw_data):
    '''环比增长率
    raw_data应为当年的季度累计值
    '''
    raw_data = get_single_quater_data(raw_data)
    data_lq = shift_period(raw_data, 1)
    data = pd.concat([raw_data, data_lq], axis=1)
    inc = data.apply(_inc_rate, axis=1).to_frame()
    inc.columns = ['inc_hb']
    return inc


def inc_tb(raw_data, quater=False):
    '''同比变化绝对量
    quater: 是否拆成单季度同比
    '''
    if quater:
        raw_data = get_single_quater_data(raw_data)
    data_lq = shift_period(raw_data, 4)
    data = pd.concat([raw_data, data_lq], axis=1)
    data['inc_tb'] = data.iloc[:, 0] - data.iloc[:, 1]
    return data[['inc_tb']]


def inc_hb(raw_data):
    '''环比变化绝对量
    raw_data 应为当年的季度累计值
    '''
    raw_data = get_single_quater_data(raw_data)
    data_lq = shift_period(raw_data, 1)
    data = pd.concat([raw_data, data_lq], axis=1)
    data['inc_hb'] = data.iloc[:, 0] - data.iloc[:, 1]
    return data[['inc_hb']]
