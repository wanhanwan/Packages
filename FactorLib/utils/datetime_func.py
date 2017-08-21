from pandas import datetime as pdDateTime
from data_source.trade_calendar import tc
import pandas as pd


# 获取某个日期序列的每月最后一天序列
def getMonthLastDay(dates, step=1):
    dates.sort()
    MonthLastDay = []
    MonthLastDay.append(dates[0])
    s = 0
    for iDate in dates:
        if (iDate[:6]==MonthLastDay[-1][:6]):
            MonthLastDay[-1] = iDate
        else:
            s += 1
            if s >= step:
                MonthLastDay.append(iDate)
                s = 0
            else:
                MonthLastDay[-1] = iDate
    return MonthLastDay
# 获取某个日期序列的每月第一天序列
def getMonthFirstDay(dates, step=1):
    dates.sort()
    MonthFirstDay = []
    MonthFirstDay.append(dates[0])
    s = 0
    for iDate in dates:
        if (iDate[:6]!=MonthFirstDay[-1][:6]):
            s += 1
            if s >= step:
                MonthFirstDay.append(iDate)
                s = 0
    return MonthFirstDay
# 获取某个日期序列的每周最后一天序列
def getWeekLastDay(dates, step=1):
    dates.sort()
    WeekLastDay = []
    WeekLastDay.append(dates[0])
    s = 0
    for iDate in dates:
        iDateTime = DateStr2Datetime(iDate)
        iDateWeekDay = iDateTime.weekday()
        tempDateTime = DateStr2Datetime(WeekLastDay[-1])
        tempWeekDay = tempDateTime.weekday()
        if (iDateTime-tempDateTime).days != (iDateWeekDay-tempWeekDay):
            s += 1
            if s >= step:
                WeekLastDay.append(iDate)
                s = 0
            else:
                WeekLastDay[-1] = iDate
        else:
            WeekLastDay[-1] = iDate
    return WeekLastDay
# 获取某个日期序列的每周第一天序列
def getWeekFirstDay(dates, step=1):
    dates.sort()
    WeekFirstDay = []
    WeekFirstDay.append(dates[0])
    s = 0
    for iDate in dates:
        iDateTime = DateStr2Datetime(iDate)
        iDateWeekDay = iDateTime.weekday()
        tempDateTime = DateStr2Datetime(WeekFirstDay[-1])
        tempWeekDay = tempDateTime.weekday()
        if (iDateTime-tempDateTime).days != (iDateWeekDay-tempWeekDay):
            s += 1
            if s >= step:
                WeekFirstDay.append(iDate)
                s = 0
    return WeekFirstDay
# 获取某个日期序列的每年最后一天序列
def getYearLastDay(dates, step=1):
    dates.sort()
    YearLastDay = []
    YearLastDay.append(dates[0])
    s = 0
    for iDate in dates:
        if (iDate[:4]==YearLastDay[-1][:4]):
            YearLastDay[-1] = iDate
        else:
            s += 1
            if s >= step:
                YearLastDay.append(iDate)
                s = 0
            else:
                YearLastDay[-1] = iDate
    return YearLastDay
# 获取某个日期序列的每年第一天序列
def getYearFirstDay(dates, step=1):
    dates.sort()
    YearFirstDay = []
    YearFirstDay.append(dates[0])
    s = 0
    for iDate in dates:
        if (iDate[:4]!=YearFirstDay[-1][:4]):
            s += 1
            if s >= step:
                YearFirstDay.append(iDate)
                s = 0
    return YearFirstDay

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


# 日期字符串(20120202)转成datetime（timestamp），如果不是日期字符串，则返回None
def DateStr2Datetime(date_str):
    try:
        return pdDateTime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))
    except:
        return None


def GetDatetimeLastDayOfMonth(idx_datetime):
    s = pd.Series(idx_datetime, index=idx_datetime)
    lastdays = s.groupby(pd.TimeGrouper(freq='M')).max()
    return pd.DatetimeIndex(lastdays, name='date').dropna().drop_duplicates()


def GetDatetimeLastDayOfWeek(idx_datetime):
    s = pd.Series(idx_datetime, index=idx_datetime)
    lastdays = s.groupby(pd.TimeGrouper(freq='W')).max().dropna()
    return pd.DatetimeIndex(lastdays, name='date').drop_duplicates()


def GetDatetimeLastDayOfYear(idx_datetime):
    s = pd.Series(idx_datetime, index=idx_datetime)
    lastdays = s.groupby(pd.TimeGrouper(freq='A')).max()
    return pd.DatetimeIndex(lastdays, name='date').dropna().drop_duplicates()


def GetDatetimeLastDay(idx_datetime, freq, use_trading=True):
    s = pd.Series(idx_datetime, index=idx_datetime)
    if freq.endswith('D') and use_trading:
        n = int(freq[0])
        lastdays = s.groupby(pd.TimeGrouper(tc.chn_trade_calendar*n)).max()
        return pd.DatetimeIndex(lastdays, name='date')
    lastdays = s.groupby(pd.TimeGrouper(freq=freq)).max()
    return pd.DatetimeIndex(lastdays, name='date').dropna().drop_duplicates()
