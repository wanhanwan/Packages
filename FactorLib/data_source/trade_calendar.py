# encoding: utf-8
# author: wanhanwan
import os
import six
import warnings
import numpy as np
import pandas as pd
from datetime import time
from functools import wraps
from collections import Iterable
from PkgConstVars import PACKAGE_PATH

from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

try:
    from pandas._libs.tslibs import to_offset
except:
    from pandas.tseries.frequencies import to_offset

_default_min_date = pd.Timestamp('20000101')
_default_max_date = pd.Timestamp('20201231')
CustomBusinessDay = pd.offsets.CustomBusinessDay


def is_non_string_iterable(arg):
    return (
        isinstance(arg, Iterable)
        and not isinstance(arg, six.string_types)
    )

def _read_holidays():
    csv_path = os.path.join(PACKAGE_PATH,'FactorLib','resource/trade_dates.csv')
    allTradeDays = pd.to_datetime(
        pd.read_csv(csv_path, index_col=0, header=None, squeeze=True, dtype='str').tolist(),
        format='%Y%m%d'
    )
    holidays = (
        pd
        .date_range(min(allTradeDays), max(allTradeDays))
        .difference(allTradeDays)
    )
    return holidays

CHN_A_Calendar = np.busdaycalendar(
    holidays=_read_holidays().to_numpy(dtype='datetime64[D]')
)
bday_chn_ashare = pd.offsets.CustomBusinessDay(calendar=CHN_A_Calendar)
bmonthbegin_chn_ashare = pd.offsets.CustomBusinessMonthBegin(calendar=CHN_A_Calendar)
bmonthend_chn_ashare = pd.offsets.CustomBusinessMonthEnd(calendar=CHN_A_Calendar)


def _validate_date_range(start, end):
    if start:
        start = max(pd.Timestamp(start), _default_min_date)
    if end:
        end = min(pd.Timestamp(end), _default_max_date)
    return start, end


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
                       periods=None, **kwargs):
        """
        获得两个日期之间的交易日序列。

        日期序列的范围是2000-01-01至2020-12-31

        Parameters:
        ----------
        start_date: str or date object
            起始日期
        end_date: str or date object
            结束日期
        freq: str or pandas offsets
            日期频率。
        periods: int
            日期数量
        """
        start_date, end_date = _validate_date_range(start_date, end_date)

        if periods:
            periods2 = periods * 2
            if start_date is None and end_date is None:
                start_date = _default_min_date
        else:
            periods2 = None
            start_date = start_date or _default_min_date
            end_date = end_date or _default_max_date

        natural_days = pd.date_range(
            start_date, end_date, freq=freq.upper(), periods=periods2
        )
        if freq.endswith(('S', 'd', 'D')):
            trade_days = (natural_days - bday_chn_ashare + bday_chn_ashare).unique()
        else:
            trade_days = (natural_days + bday_chn_ashare - bday_chn_ashare).unique()
        trade_days = trade_days[trade_days.slice_indexer(start_date, end_date)]

        if periods and start_date:
            return trade_days[:periods]
        elif periods and end_date:
            return trade_days[-periods:]
        else:
            return trade_days

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
        if n == 0:
            raise ValueError("absolute value of parameter 'n' must be positive!")

        if is_non_string_iterable(today):
            days = pd.to_datetime(today)
        else:
            days = pd.DatetimeIndex([pd.to_datetime(today)])

        raw_days = days.copy()

        move_forward = n > 0
        begin_offset = freq.endswith('S')
        offset = to_offset(freq.upper())

        # 先把日期归位到offset
        if begin_offset:
            days = days + offset - offset
            bdays = days - bday_chn_ashare + bday_chn_ashare
        else:
            days = days - offset + offset
            bdays = days + bday_chn_ashare - bday_chn_ashare

        if move_forward:
            if incl_on_offset_today:
                td = np.where(raw_days <= bdays, n-1, n)
            else:
                td = np.where(raw_days < bdays, n-1, n)
        else:
            if incl_on_offset_today:
                td = np.where(raw_days < bdays, n, n+1)
            else:
                td = np.where(raw_days <= bdays, n, n+1)

        if freq.endswith('d'):
            days = pd.DatetimeIndex(np.where(td != 0, days+bday_chn_ashare*td, days).astype('datetime64[D]'))
        else:
            days = pd.DatetimeIndex(np.where(td != 0, days+offset*td, days).astype('datetime64[D]'))

        if not begin_offset:
            days = days + bday_chn_ashare - bday_chn_ashare
        else:
            days = days - bday_chn_ashare + bday_chn_ashare

        if not is_non_string_iterable(today):
            days = days[0]

        return days

    @staticmethod
    def is_trade_day(days):
        """
        交易日判断。
        """
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])
        is_bdays = np.is_busday(
            datetimes.to_numpy(dtype='datetime64[D]'), busdaycal=CHN_A_Calendar
        )
        if not is_non_string_iterable(days):
            return is_bdays[0]
        return is_bdays

    def is_first_trade_day(self, days, freq='m'):
        """
        是否是某个频率下的第一个交易日
        :param days: str or list
            待检验日期
        :param freq: str
            频率，'m'代表月份、'w'代表周、'q'代表季度、'y'代表年份
        :return: bool
        """
        field = {
            'm': 'month',
            'w': 'week',
            'q': 'quarter',
            'y': 'year'
        }[freq]
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])
        is_trade_days = self.is_trade_day(datetimes)
        last_trade_days = datetimes - bday_chn_ashare
        result = np.logical_and(
            is_trade_days,
            getattr(datetimes, field) != getattr(last_trade_days, field)
        )

        if not is_non_string_iterable(days):
            return result[0]
        return result

    def is_last_trade_day(self, days, freq='m'):
        """
        是否是某个频率下的最后一个个交易日
        :param days: str or list
            待检验日期
        :param freq: str
            频率，'m'代表月份、'w'代表周、'q'代表季度、'y'代表年份
        :return: bool
        """
        field = {
            'm': 'month',
            'w': 'week',
            'q': 'quarter',
            'y': 'year'
        }[freq]
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])
        is_trade_days = self.is_trade_day(datetimes)
        next_trade_days = datetimes + bday_chn_ashare
        result = np.logical_and(
            is_trade_days,
            getattr(datetimes, field) != getattr(next_trade_days, field)
        )
        if not is_non_string_iterable(days):
            return result[0]
        return result


    def is_last_day_of_month(self, days):
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])

        is_trade_days = self.is_trade_day(datetimes)
        next_days = datetimes + bday_chn_ashare

        result = np.logical_and(is_trade_days, next_days.month != datetimes.month)

        if not is_non_string_iterable(days):
            return result[0]
        else:
            return result

    def is_first_day_of_month(self, days):
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])

        is_trade_days = self.is_trade_day(datetimes)
        next_days = datetimes - bday_chn_ashare

        result = np.logical_and(is_trade_days, next_days.month != datetimes.month)

        if not is_non_string_iterable(days):
            return result[0]
        else:
            return result

    @handle_retstr
    def get_latest_trade_days(self, days, **kwargs):
        """
        遍历days中的每个元素，返回距离每个元素最近的交易日。
        """
        if is_non_string_iterable(days):
            datetimes = pd.to_datetime(days)
        else:
            datetimes = pd.DatetimeIndex([days])
        dt = datetimes + bday_chn_ashare - bday_chn_ashare
        if not is_non_string_iterable(days):
            return dt[0]
        return dt

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
        return len(days) - 1 if days[0]==pd.Timestamp(start) else len(days)


tc = trade_calendar()


if __name__ == '__main__':
    d = tc.get_trade_days('20201001', '20201031')
    # d = tc.tradeDayOffset(['20200529'], -60, freq='1d', incl_on_offset_today=True)
    # d = tc.is_trade_day('20100101')
    # d = tc.is_last_day_of_month('20100129')
    print(d)
