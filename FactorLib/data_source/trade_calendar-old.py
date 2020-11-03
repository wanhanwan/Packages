# coding: utf-8
import os
import warnings
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import (DateOffset,
                                    CustomBusinessDay,
                                    CustomBusinessMonthEnd,
                                    Week,
                                    QuarterEnd,
                                    YearEnd,
                                    # as_timestamp,
                                    )
from pandas._libs.tslibs.offsets import apply_wraps
from functools import wraps
from datetime import time, timedelta
from fastcache import clru_cache
pd.offsets.CustomBusinessMonthEnd


as_timestamp = pd.Timestamp
_default_min_date = as_timestamp('20000101')
_default_max_date = as_timestamp('20201231')


def _read_holidays():
    root_dir = os.path.abspath(__file__+'/../..')
    csv_path = os.path.join(root_dir, 'resource/trade_dates.csv')
    allTradeDays = pd.read_csv(csv_path, index_col=0, header=None, squeeze=True, dtype='str').values.tolist()
    allTradeDays_idx = pd.DatetimeIndex(allTradeDays)
    all_dates = pd.date_range(min(allTradeDays), max(allTradeDays))
    holidays = [x for x in all_dates if (x not in allTradeDays_idx) and (x.weekday() not in [5, 6])]
    return holidays
chn_holidays = _read_holidays()


class CustomBusinessWeekEnd(DateOffset):
    _cacheable = False
    _prefix = 'CBWE'
    _attributes = frozenset(['calendar', 'holidays'])

    def __init__(self, n=1, normalize=False, weekmask='Mon Tue Wed Thu Fri',
                 holidays=None, calendar=None, **kwds):
        self.n = n
        object.__setattr__(self, "normalized", normalize)
        self.kwds.update(kwds)
        object.__setattr__(self, "offset", kwds.get('offset', timedelta(0)))
        object.__setattr__(self, "cbday", CustomBusinessDay(n=1, normalize=normalize, weekmask=weekmask, holidays=holidays,
                                                            calendar=calendar, offset=self.offset))
        object.__setattr__(self, "calendar", self.cbday.calendar)
        object.__setattr__(self, "holidays", holidays)
        object.__setattr__(self, "w_offset", Week(weekday=4))

    def to_string(self):
        return f"{self.n}w"

    @apply_wraps
    def apply(self, other):
        n = self.n
        result = other
        if n == 0:
            n = 1
        if n > 0:
            while result <= other:
                next_fri = other + n * self.w_offset
                result = self.cbday.rollback(next_fri)
                n += 1
        else:
            while result >= other:
                last_fri = other + n * self.w_offset
                result = self.cbday.rollback(last_fri)
                n -= 1
        return result

    def onOffset(self, dt):
        if self.normalize and not _is_normalized(dt):
            return False
        if not self.cbday.onOffset(dt):
            return False
        return (dt + self.cbday).week != dt.week


class CustomBusinessQuaterEnd(DateOffset):
    _cacheable = False
    _prefix = 'CBQE'
    _attributes = frozenset(
        {'holidays', 'calendar'})

    def __init__(self, n=1, normalize=False, weekmask='Mon Tue Wed Thu Fri',
                 holidays=None, calendar=None, **kwds):
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "normalized", normalize)
        self.kwds.update(kwds)
        object.__setattr__(self, "offset", kwds.get('offset', timedelta(0)))
        object.__setattr__(self, "startingMonth", kwds.get('startingMonth', 3))
        object.__setattr__(self, "cbday", CustomBusinessDay(n=1, normalize=normalize, weekmask=weekmask, holidays=holidays,
                                                            calendar=calendar))
        object.__setattr__(self, "calendar", self.cbday.calendar)
        object.__setattr__(self, "holidays", holidays)
        object.__setattr__(self, "q_offset", QuarterEnd(1))

    def to_string(self):
        return f"{self.n}q"

    @apply_wraps
    def apply(self, other):
        n = self.n
        cur_qend = self.q_offset.rollforward(other)
        cur_cqend = self.cbday.rollback(cur_qend)

        if n == 0 and other != cur_cqend:
            n += 1
        if other < cur_cqend and n >= 1:
            n -= 1
        if other > cur_cqend and n <= -1:
            n += 1

        new = cur_qend + n * self.q_offset
        result = self.cbday.rollback(new)
        return result

    def onOffset(self, dt):
        if self.normalize and not _is_normalized(dt):
            return False
        if not self.cbday.onOffset(dt):
            return False
        return (dt + self.cbday).quarter != dt.quarter


class CustomBusinessYearEnd(DateOffset):
    _cacheable = False
    _prefix = 'CBYE'
    _default_month = 12

    def __init__(self, n=1, normalize=False, weekmask='Mon Tue Wed Thu Fri',
                 holidays=None, calendar=None, **kwds):
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "normalized", normalize)
        self.kwds.update(kwds)
        object.__setattr__(self, "offset", timedelta(0))
        object.__setattr__(self, "month", self._default_month)
        try:
            kwds.pop('month')
        except Exception as e:
            pass
        object.__setattr__(self, "cbday", CustomBusinessDay(n=1, normalize=normalize, weekmask=weekmask, holidays=holidays,
                                                            calendar=calendar, **kwds))
        self.kwds['calendar'] = self.cbday.calendar
        object.__setattr__(self, "y_offset", YearEnd(1))

    def to_string(self):
        return f"{self.n}y"

    @apply_wraps
    def apply(self, other):
        n = self.n
        cur_yend = self.y_offset.rollforward(other)
        cur_cyend = self.cbday.rollback(cur_yend)

        if n == 0 and other != cur_cyend:
            n += 1
        if other < cur_cyend and n >= 1:
            n -= 1
        if other > cur_cyend and n <= -1:
            n += 1

        new = cur_yend + n * self.y_offset
        result = self.cbday.rollback(new)
        return result

    def onOffset(self, dt):
        if self.normalize and not _is_normalized(dt):
            return False
        if not self.cbday.onOffset(dt):
            return False
        return (dt + self.cbday).year != dt.year


class MyTradeDays(CustomBusinessDay):
    def to_string(self):
        return f"{self.n}d"


class MyTradeMonths(CustomBusinessMonthEnd):
    def to_string(self):
        return f"{self.n}m"

traderule_alias_mapping = {
    'd': MyTradeDays(holidays=chn_holidays),
    'w': CustomBusinessWeekEnd(holidays=chn_holidays),
    'm': MyTradeMonths(holidays=chn_holidays),
    'q': CustomBusinessQuaterEnd(holidays=chn_holidays),
    'y': CustomBusinessYearEnd(holidays=chn_holidays)
}


def _to_offset(freq):
    if isinstance(freq, DateOffset):
        return freq
    if freq[-1] in traderule_alias_mapping:
        return traderule_alias_mapping.get(freq[-1]) * int(freq[:-1])
    else:
        return to_offset(freq)


def _validate_date_range(start, end):
    start = _default_min_date if start is None else as_timestamp(start)
    end = _default_max_date if end is None else as_timestamp(end)
    return max(start, _default_min_date), min(end, _default_max_date)


def handle_retstr(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if 'retstr' not in kwargs:
            retstr = '%Y%m%d'
        elif kwargs['retstr'] is None:
            return func(self, *args, **kwargs)
        else:
            retstr = kwargs['retstr']
        result = func(self, *args, **kwargs).strftime(retstr)
        try:
            return result.tolist()
        except:
            return result
    return wrapper


class trade_calendar(object):

    @handle_retstr
    def get_trade_days(self, start_date=None, end_date=None, freq='1d',
                       reverse=False, begin_date=False, **kwargs):
        """
        获得日期序列，支持日历日和交易日。

        freq支持自定义格式和Pandas自带的DateOffset格式。当freq是字符串时，它由两部分构成。第一个字符为数字
        表示间隔数目,后面字母表示频率,目前交易日频率包括d、w、m、q、y(日、周、月、季、年)。

        日期序列的范围是2000-01-01至2020-12-31
        """
        start_date, end_date = _validate_date_range(start_date, end_date)
        offset = _to_offset(freq)
        if reverse:
            n = int(freq[:-1])
            offset_unit = _to_offset('1%s'%freq[-1])
            raw = pd.date_range(start_date, end_date, freq=offset_unit)
            result =  raw[::-n].sort_values()
        else:
            result = pd.date_range(start_date, end_date, freq=offset)
        if begin_date and freq[-1]!='d':
            tmp = [self.tradeDayOffset(x, -1, freq='1'+freq[-1]) for x in result]
            result = pd.DatetimeIndex([self.tradeDayOffset(x, 1, retstr=None) for x in tmp])
        return result

    @handle_retstr
    def get_trade_time(self, start_time, end_time, freq='1H', **kwargs):
        """
        获得交易时间序列，支持秒、分钟、小时
        时间格式: "YYYYMMDD HH:MM:SS"
        """
        from datetime import time
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        trade_days = self.get_trade_days(start_time.strftime("%Y%m%d"),
                                         end_time.strftime("%Y%m%d"),
                                         retstr=None)
        raw = pd.date_range(start_time, end_time, freq=freq)
        raw = raw[pd.DatetimeIndex(raw.date).isin(trade_days)]
        time_of_raw = raw.time
        raw = raw[((time_of_raw > time(9, 30)) & (time_of_raw <= time(11, 30))) |
                  ((time_of_raw > time(13, 0)) & (time_of_raw <= time(15, 0)))]
        return pd.DatetimeIndex(raw)

    @handle_retstr
    def tradeDayOffset(self, today, n, freq='1d', incl_on_offset_today=False, **kwargs):
        """
        日期漂移

        若参数n为正，返回以today为起始日向前推第n个交易日，反之亦然。
        若n为零，返回以today为起点，向后推1个freq的交易日。

        注意:
            若incl_on_offset_today=True，today on offset时，漂移的起点是today，today not
            offset时，漂移的起点是today +- offset
            若incl_on_offset_today=False，日期漂移的起点是today +- offset。

        例如：
            2017-08-18是交易日, 2017-08-20不是交易日，则：

            tradeDayOffset('2017-08-18', 1, freq='1d', incl_on_offset_today=False) -> 2017-08-21

            tradeDayOffset('2017-08-18', 1, freq='1d', incl_on_offset_today=True) -> 2017-08-18

            tradeDayOffset('2017-08-18', 2, freq='1d', incl_on_offset_today=True) -> 2017-08-21

            tradeDayOffset('2017-08-18', -1, freq='1d', incl_on_offset_today=True) -> 2017-08-18

            tradeDayOffset('2017-08-18', -2, freq='1d', incl_on_offset_today=True) -> 2017-08-17

            tradeDayOffset('2017-08-18', 0, freq='1d', incl_on_offset_today=False) -> 2017-08-18

            tradeDayOffset('2017-08-20', 0, freq='1d', incl_on_offset_today=True) -> 2017-08-18
        """

        today = as_timestamp(today)
        if not self.is_trade_day(today):
            today = self.get_latest_trade_days(today, retstr=None)
            incl_on_offset_today = False
        if n == 0:
            if int(freq[0]) > 1:
                warnings.warn('invalid step length of freq. It must be 1 when n=0')
                n = -1 * (int(freq[0]))
                freq = "1%s" % freq[1]
            else:
                return traderule_alias_mapping[freq[1]].rollback(today)
        offset = _to_offset(freq)
        if incl_on_offset_today and offset.onOffset(today):
            n = int((abs(n) - 1) * (n / abs(n)))
        return today + n * offset

    @staticmethod
    def is_trade_day(day, freq='1d'):
        """
        交易日判断。
        """
        time_stamp = as_timestamp(day)
        return _to_offset(freq).onOffset(time_stamp)

    @clru_cache()
    def is_last_day_of_month(self, day):
        timestamp = as_timestamp(day)
        if not self.is_trade_day(timestamp):
            return False
        next_day = self.tradeDayOffset(timestamp, 1, retstr=None)
        return timestamp.month < next_day.month or timestamp.year < next_day.year

    @clru_cache()
    def is_first_day_of_month(self, day):
        timestamp = as_timestamp(day)
        if not self.is_trade_day(timestamp):
            return False
        last_day = self.tradeDayOffset(timestamp, -1, retstr=None)
        return timestamp.month > last_day.month or timestamp.year > last_day.year

    @handle_retstr
    def get_latest_trade_days(self, days, **kwargs):
        """
        遍历days中的每个元素，返回距离每个元素最近的交易日。
        """
        from pandas.core.dtypes.inference import is_list_like
        if is_list_like(days):
            timeindex = pd.DatetimeIndex(days)
            return pd.DatetimeIndex([traderule_alias_mapping['d'].rollback(x) for x in timeindex])
        else:
            return traderule_alias_mapping['d'].rollback(as_timestamp(days))

    @staticmethod
    def is_trading_time(date_time):
        """
        交易时间判断
        """
        is_tradingdate = trade_calendar.is_trade_day(date_time.date())
        is_tradingtime = time(9, 25, 0) < date_time.time() < time(15, 0, 0)
        return is_tradingdate and is_tradingtime

    def days_between(self, start, end):
        """
        两个日期之间间隔的交易日数量
        """
        days = self.get_trade_days(start, end, retstr=None)
        return len(days) - 1 if days[0]==as_timestamp(start) else len(days)


tc = trade_calendar()


if __name__ == '__main__':
    d = tc.get_trade_time('20100101', '20100105', freq='1H')
    print(d)
