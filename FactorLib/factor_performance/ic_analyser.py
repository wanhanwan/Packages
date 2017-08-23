from scipy import stats
from ..utils.datetime_func import DateStr2Datetime
from QuantLib.factor_validation import cal_ic
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
        pass

    def calculate(self, freq, method='spearman'):
        ic_series, count = cal_ic(self.factor.data, window=freq, factor_name=self.factor.name, rank=method=='pearson',
                                  retstocknums=True, stock_validation='typical')
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