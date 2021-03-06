# coding: utf-8
from functools import wraps
from fastcache import clru_cache
from collections import Iterable
from datetime import datetime as pdDateTime
from FactorLib.data_source.trade_calendar import tc
from xlrd.xldate import xldate_as_datetime

import pandas as pd
import numpy as np

# 日期字符串(20120202)转成datetime（timestamp），如果不是日期字符串，则返回None
def DateStr2Datetime(date_str):
    try:
        return pdDateTime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))
    except:
        return None

# datetime（timestamp)转成日期字符串(20120202)
def Datetime2DateStr(date):
    Year = date.year
    Month = date.month
    Day = date.day
    if Month<10:
        Month = '0'+str(Month)
    else:
        Month = str(Month)
    if Day<10:
        Day = '0'+str(Day)
    else:
        Day = str(Day)
    return str(Year)+Month+Day


# datetime(timestamp)转成数字日期(20120202)
def Datetime2IntDate(date):
    Year = date.year
    Month = date.month
    Day = date.day
    return Year * 10000 + Month * 100 + Day


@clru_cache()
def IntDate2Datetime(date: int):
    return pd.to_datetime(str(date))


# 日期字符串(20120202)转成datetime（timestamp），如果不是日期字符串，则返回None
def DateStr2Datetime(date_str):
    try:
        return pdDateTime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))
    except:
        return None


# matlab格式的日期
def Datetime2MatlabDatetime(dates):
    if isinstance(dates, Iterable):
        return ((np.array(dates, dtype='datetime64') - np.datetime64('1970-01-01T00:00:00')) /
                np.timedelta64(1, 'D')).astype('int32')
    else:
        return int((np.datetime64(dates) - np.datetime64('1970-01-01T00:00:00')) /
                   np.timedelta64(1, 'D'))


# matlab格式转datetime
def MatlabDatetime2Datetime(dates):
    return pd.to_datetime(dates, unit='D')


# excel数字格式转成datetime
def ExcelDatetime2Datetime(dates):
    datetimes = [xldate_as_datetime(x, 0) for x in dates]
    return pd.to_datetime(datetimes)


def DateRange2Dates(func):
    """
    函数装饰器。

    把func中的时间参数(start_date, end_date, dates)都转成dates。

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = kwargs.get('start_date')
        end = kwargs.get('end_date')
        dates = kwargs.get('dates')
        d = tc.get_trade_days(start, end, retstr=None)
        if dates is not None:
            try:
                dates = pd.DatetimeIndex(dates)
            except:
                dates = pd.DatetimeIndex([dates])
            dates = list(dates[np.in1d(dates.date, d.date)])
            dates.sort()
        else:
            dates = d
        kwargs = {k:v for k,v in kwargs.items() if k not in ['start_date','end_date','dates']}
        return func(*args, dates=dates, **kwargs)
    return wrapper


# ******************财务相关**************************
def GetDefaultAnnouncementDate(report_periods, adjusted=False):
    """
    根据财报披露的时间规则找到公告日期
    
    Parameters:
    -----------
    report_periods: str or list of str
        报告期  
    adjusted: bool
        调整前的财报还是调整后的财报。
    
    Returns:
        DatetimeIndex
    """
    from pandas.tseries.offsets import MonthEnd
    periods = pd.to_datetime(report_periods)
    quarters = periods.quarter
    if isinstance(quarters, int):
        quarters = 1 if quarters==3 else quarters
    else:
        quarters = np.where(quarters==3, 1, quarters)
    announcements = periods + MonthEnd(1) * quarters
    if adjusted:
        announcements = announcements + MonthEnd(12)
    return announcements

    
    
    
    
