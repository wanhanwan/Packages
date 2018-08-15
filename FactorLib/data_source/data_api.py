# coding: utf-8
import numpy as np
from .base_data_source_h5 import sec, tc, data_source
from .wind_financial_data_api.data_api import load_dividends

# 返回历史A股成分股
def get_history_ashare(date):
    return sec.get_history_ashare(date)

#返回交易日期
def get_trade_days(start_date=None,end_date=None,freq='1d'):
    return tc.get_trade_days(start_date, end_date, freq)

def trade_day_offset(today, n, freq='1d'):
    return tc.tradeDayOffset(today,n,freq)


def dividend_yield(ids=None, start_date=None, end_date=None, dates=None,
                   idx=None, report_year=None):
    """股息率"""
    cash_div = load_dividends(ids, dates, start_date, end_date, idx, report_year)
    close = data_source.load_factor('close', '/stocks/', ids=ids, start_date=start_date,
                                    end_date=end_date, dates=dates, idx=idx)
    ratio = cash_div['dividend'].div(close['close'], axis='index', fill_value=0.0)
    ratio = ratio[~np.isinf(ratio)]
    return ratio.to_frame('dividend_yield')
