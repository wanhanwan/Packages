# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:51:14 2018

@author: ws
"""

import pandas as pd


def get_date_fy1(date):
    """可能的最近预测报告期
    当日期在4月30号之前, fy1可能是去年年报；
    当日期是在4月30号之后,fy1可能是当年年报
    """
    year = date // 10000
    md = date % 10000
    if md < 430:
        return (year - 1) * 10000 + 1231
    return year * 10000 + 1231


def get_date_fy2(date):
    """可能的最近预测报告期的下一年
    当日期在4月30号之前, fy2可能是当年年报
    当日期是在4月30号之后, fy2可能是下一年年报
    """
    year = date // 10000
    md = date % 10000
    if md < 430:
        return year * 10000 + 1231
    return (year + 1) * 10000 + 1231


def get_date_fy3(date):
    """可能的最近预测报告期的下一年
    当日期在4月30号之前, fy2可能是当年年报
    当日期是在4月30号之后, fy2可能是下一年年报
    """
    year = date // 10000
    md = date % 10000
    if md < 430:
        return (year + 1) * 10000 + 1231
    return (year + 2) * 10000 + 1231


def load_fy1(raw_data, field_name, dates, ids=None, stat_type=90):
    """
    最近预测报告期的数据, 由于一致预期的预测报告期多
    为年报, 所以最开始要筛选出quarter是4的数据
    """
    if ids is not None:
        raw_data = raw_data.query(
            "IDs in @ids & quarter==4 & year_type==1.0 & stat_type==@stat_type")
    else:
        raw_data = raw_data.query(
            "quarter==4 & year_type==1.0 & stat_type==@stat_type")

    r = []
    for date in dates:
        available_report_date = get_date_fy1(date)
        data = raw_data.query("ann_dt <= @date & date >= @available_report_date")
        tmp = data.groupby('IDs')[field_name].last()
        tmp.index = pd.MultiIndex.from_product(
            [[date], tmp.index],
            names=['date', 'IDs'])
        r.append(tmp)
    return pd.concat(r)


def load_fy2(raw_data, field_name, dates, ids=None, stat_type=90):
    """
    最近预测报告期的数据, 由于一致预期的预测报告期多
    为年报, 所以最开始要筛选出quarter是4的数据
    """
    if ids is not None:
        raw_data = raw_data.query(
            "IDs in @ids & quarter==4 & year_type==2.0 & stat_type==@stat_type")
    else:
        raw_data = raw_data.query(
            "quarter==4 & year_type==2.0 & stat_type==@stat_type")

    r = []
    for date in dates:
        available_report_date = get_date_fy2(date)
        data = raw_data.query("ann_dt <= @date & date >= @available_report_date")
        tmp = data.groupby('IDs')[field_name].last()
        tmp.index = pd.MultiIndex.from_product(
            [[date], tmp.index],
            names=['date', 'IDs'])
        r.append(tmp)
    return pd.concat(r)


def load_fy3(raw_data, field_name, dates, ids=None, stat_type=90):
    """
    最近预测报告期的数据, 由于一致预期的预测报告期多
    为年报, 所以最开始要筛选出quarter是4的数据
    """
    if ids is not None:
        raw_data = raw_data.query(
            "IDs in @ids & quarter==4 & year_type==3.0 & stat_type==@stat_type")
    else:
        raw_data = raw_data.query(
            "quarter==4 & year_type==3.0 & stat_type==@stat_type")

    r = []
    for date in dates:
        available_report_date = get_date_fy3(date)
        data = raw_data.query("ann_dt <= @date & date >= @available_report_date")
        tmp = data.groupby('IDs')[field_name].last()
        tmp.index = pd.MultiIndex.from_product(
            [[date], tmp.index],
            names=['date', 'IDs'])
        r.append(tmp)
    return pd.concat(r)


def load_spec_year(raw_data, field_name, spec_year, dates, ids=None, stat_type=90):
    """
    最新预测指定年份的一致预期数据
    """
    if ids is not None:
        raw_data = raw_data.query(
            "IDs in @ids & quarter==4 & year==@spec_year & stat_type==@stat_type")
    else:
        raw_data = raw_data.query(
            "quarter==4 & year==@spec_year & stat_type==@stat_type")

    r = []
    for date in dates:
        data = raw_data.query("ann_dt <= @date")
        if not data.empty:
            tmp = data.groupby('IDs')[field_name].last()
            tmp.index = pd.MultiIndex.from_product(
                [[date], tmp.index],
                names=['date', 'IDs'])
            r.append(tmp)
    return pd.concat(r)


d = pd.read_hdf(r"D:\data\finance\ashareconsensusdata\net_profit_avg.h5", "data")
dd3 = load_spec_year(d, 'net_profit_avg', 2018, dates=[20180202, 20180209])