# coding: utf-8
from FactorLib.data_source.base_data_source_h5 import sec, ncdb, tc, data_source
from fastcache import clru_cache
from QuantLib.stockFilter import _intersection, _difference, _union
import re
import os
import pandas as pd


class StockUniverse(object):
    def __init__(self, base_universe, other=None, algorithm=None):
        self.base = base_universe
        self.other = other
        self.algorithm = algorithm

    def __add__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _union)
        else:
            return self.__class__(self, other, _union)

    def __sub__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _difference)
        else:
            return self.__class__(self, other, _difference)

    def __mul__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _intersection)
        else:
            return self.__class__(self, other, _intersection)

    # @clru_cache()
    def get(self, start_date=None, end_date=None, dates=None):
        if self.algorithm is None:
            return sec.get_index_members(self.base, dates, start_date, end_date)
        else:
            base = self.base.get(start_date, end_date, dates)
            other = self.other.get(start_date, end_date, dates)
            return self.algorithm(base, other)


class StockDummy(object):
    """行业哑变量封装接口
    所有的数据放在ncdb数据目录中的dummy文件夹
    """
    data_dir = ncdb.data_path + '/dummy'
    all_dummies = [x[:-3] for x in os.listdir(data_dir)]

    def __init__(self, name):
        self.name = name

    def check_existence(self, name):
        return name in self.all_dummies

    def get(self, start_date=None, end_date=None, dates=None):
        """获取哑变量数据"""
        if dates is None:
            dates = tc.get_trade_days(start_date=start_date, end_date=end_date)
        dummy = ncdb.load_as_dummy(self.name, '/dummy/', dates=dates)
        return dummy

    def get_stocks_of_single_field(self, field_name, start_date=None, end_date=None,
                                   dates=None):
        """获取某一个板块的成分股"""
        if field_name not in self.all_fields:
            raise KeyError
        dummy = self.get(start_date, end_date, dates)
        return dummy.loc[dummy[field_name] == 1, [field_name]]

    def get_return(self, start_date=None, end_date=None, dates=None):
        """每个板块市值加权的日收益率"""
        dummy = self.get(start_date, end_date, dates)
        mv = data_source.load_factor('float_mkt_value', '/stocks/', start_date=start_date,
                                     end_date=end_date, dates=dates) ** 0.5
        mv = mv.reindex(dummy.index, copy=False)
        r = data_source.load_factor('daily_returns', '/stocks/', start_date=start_date,
                                    end_date=end_date, dates=dates)
        r = r.reindex(dummy.index, copy=False)
        r_mv = r.values * mv.values
        dummy_ret = pd.DataFrame(r_mv * dummy.values, index=dummy.index, columns=dummy.columns).groupby('date').sum()
        dummy_weight = pd.DataFrame(mv.values * dummy.values, index=dummy.index, columns=dummy.columns).groupby('date').sum()
        return dummy_ret / dummy_weight

    @property
    def all_fields(self):
        return ncdb.load_file_attr(self.name, '/dummy/', 'indu_names')


def from_formula(formula):
    """把字符串公式转换成StockUniverse

    Examples :
    -------------------
    >>>from_formula('000300 + 000905')
    """
    def match(matched):
        if matched.group(0).replace(" ", ""):
            return "StockUniverse('%s')" % matched.group(0)
        else:
            return ""

    p = re.compile('[^\(\)\+\-\*]*')
    s = re.sub(p, match, formula).replace(' ', '')
    return eval(s)


if __name__ == '__main__':
    d = StockDummy('user_dummy_class_1')
    r = d.get_return(start_date='20180201', end_date='20180228')
    print(r)
