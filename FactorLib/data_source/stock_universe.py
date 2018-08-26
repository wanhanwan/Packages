# coding: utf-8
from FactorLib.data_source.base_data_source_h5 import sec, data_source
from QuantLib.stockFilter import _intersection, _difference, _union
from warnings import warn
import re
import os
import pandas as pd


class StockUniverse(object):
    def __init__(self, base_universe, other=None, algorithm=None):
        self.base = base_universe
        self.other = other
        self.algorithm = algorithm

    def __repr__(self):
        if self.algorithm is None:
            return self.base

        if self.algorithm is _intersection:
            operator = '*'
        elif self.algorithm is _difference:
            operator = '-'
        else:
            operator = '+'
        return "Universe(%s %s %s)"%(self.base, operator, self.other)

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
            try:
                base = self.base.get(start_date, end_date, dates)
            except KeyError as e:
                warn("%s not exist or dates out of range!"%self.base, UserWarning)
                return self.other.get(start_date, end_date, dates)
            try:
                other = self.other.get(start_date, end_date, dates)
            except KeyError as e:
                warn("%s not exist or dates out of range!" % self.other, UserWarning)
                return base
            return self.algorithm(base, other)


class StockDummy(object):
    """行业哑变量封装接口
    所有的数据放在ncdb数据目录中的dummy文件夹
    """
    data_dir = data_source.h5DB.data_path + '/dummy'
    all_dummies = [x[:-3] for x in os.listdir(data_dir)]

    def __init__(self, name='cs_level_1'):
        self.name = name

    def check_existence(self, name):
        return name in self.all_dummies

    def get(self, start_date=None, end_date=None, dates=None, universe=None, idx=None,
            drop_first=False):
        """获取哑变量数据"""
        # dates = tc.get_trade_days(start_date, end_date) if dates is None else dates
        dummy = data_source.sector.get_industry_dummy(ids=universe, start_date=start_date,
                                                      end_date=end_date, dates=dates,
                                                      idx=idx, industry=self.name,
                                                      drop_first=drop_first)
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
        return pd.read_hdf(os.path.join(self.data_dir, self.name+'.h5'), "mapping")


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
    d = StockDummy('cs_level_2')
    u = StockUniverse('CI005018')
    r = d.get(start_date='20180201', end_date='20180228', universe=u)
    print(r)
