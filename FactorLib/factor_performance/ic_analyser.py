from const import DATEMULTIPLIER
from scipy import stats
from utils.datetime_func import DateStr2Datetime
import pandas as pd
import numpy as np

class IC_Calculator(object):
    def __init__(self):
        self.factor = None
        self.stock_returns = None
        self.ic_analyzer = None

    def set_factor(self,factor):
        self.factor = factor

    def set_stock_returns(self, env):
        freq = env._config.freq
        window_len = DATEMULTIPLIER[freq[1]] * int(freq[0])
        dates = np.array(list(map(
            lambda x: env._trade_calendar.tradeDayOffset(x, window_len, incl_on_offset_today=True), env._factor_group_info_dates)))
        # 计算未来20天的收益分成两部分计算，第一部分是return_@nd中已经包含的日期，第二部分是为包含的日期
        max_date = env._h5DB.get_date_range('return_%dd' % window_len, '/stock_momentum/')[1]
        factor_dates_included = [x for x in env._factor_group_info_dates if x <= DateStr2Datetime(max_date)]
        dates_included = dates[dates<=max_date]
        stock_returns = env._data_source.load_factor('return_%dd' % window_len, '/stock_momentum/', dates=dates_included.tolist())
        stock_returns.index = stock_returns.index.set_levels(factor_dates_included, level=0)
        
        if max(dates) > max_date:
            close_price_max_date = env._data_source.h5DB.get_date_range('adj_close', '/stocks/')[1]
            excess_dates = dates[dates>max_date]
            close_price_dates = np.where(excess_dates>close_price_max_date, close_price_max_date, excess_dates)
            close_price = env._data_source.get_history_price(None, dates=close_price_dates.tolist(), adjust=True)
            close_price = close_price.unstack().reindex(pd.DatetimeIndex(excess_dates), method='pad').stack()
            
            excess_dates_factor = env._factor_group_info_dates[dates>max_date]
            close_price_pre = env._data_source.get_history_price(None, dates=excess_dates_factor, adjust=True)
            close_price.index = close_price.index.set_levels(excess_dates_factor, level=0)

            d = pd.concat([close_price, close_price_pre], axis=1)
            stock_returns_2 = (d.iloc[:, 0] / d.iloc[:, 1] - 1).to_frame()
            stock_returns_2.columns = stock_returns.columns
            stock_returns_2.index.names = ['date', 'IDs']
        else:
            stock_returns_2 = pd.DataFrame()
        self.stock_returns = stock_returns.append(stock_returns_2).sort_index()

    def calculate(self, freq='1m', method='spearman'):

        def corr(data, method='spearman'):
            return data.corr(method=method).iat[0,1]

        common = pd.concat([self.factor.data, self.stock_returns],axis=1, join='inner')
        ic_series = common.groupby(level=0).apply(corr, method=method)
        count = common.groupby(level=0)[self.factor.name].count()
        return ic_series, count


class IC_Analyzer(object):
    def __init__(self):
        self.ic_series = None
        self.hold_nums = None

    def set_ic_series(self, series, hold_numbers):
        self.ic_series = series
        self.hold_nums = hold_numbers

    def ma(self, window):
        return self.ic_series.rolling(window).mean()

    def update_info(self, new_values, new_hold_nums):
        try:
            old = self.ic_series[~self.ic_series.index.isin(new_values.index)]
        except:
            old = pd.Series()
        try:
            old_hold_nums = self.hold_nums[~self.hold_nums.index.isin(new_hold_nums.index)]
        except:
            old_hold_nums = pd.Series()
        self.ic_series = old.append(new_values).sort_index()
        self.hold_nums = old_hold_nums.append(new_hold_nums).sort_index()
    
    def get_state(self):
        return {'ic_series':self.ic_series, 'hold_nums': self.hold_nums}

    def set_state(self, ic_series):
        self.set_ic_series(ic_series['ic_series'], ic_series['hold_nums'])

    def to_frame(self):
        return self.ic_series.to_frame().rename(columns={0: 'ic'})
    
    def describe(self):
        _dict = {
            'mean': self.ic_series.mean(),
            'std': self.ic_series.std(),
            'minimun': self.ic_series.min(),
            'maximum': self.ic_series.max(),
            't_stats': stats.ttest_1samp(self.ic_series, 0)[0],
            'average_stocks': self.hold_nums.mean(),
            'periods': len(self.ic_series)
        }
        return pd.Series(_dict)