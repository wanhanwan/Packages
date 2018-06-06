"""Python调用天软的封装"""
import sys
sys.path.append(r"D:\programs\Analyse.NET")

import pandas as pd
import TSLPy3 as tsl
import os
from FactorLib.utils.tool_funcs import tradecode_to_tslcode, tslcode_to_tradecode
from FactorLib.utils.datetime_func import DateRange2Dates
from FactorLib.data_source.base_data_source_h5 import sec
from FactorLib.utils.TSDataParser import *
from functools import reduce, partial


_ashare = "'上证A股;深证A股;创业板;中小企业板;暂停上市;终止上市'"
_condition = 'firstday()<=getsysparam(pn_date())'


def _gstr_from_func(func_name, func_args):
    func_str = "data := {func_name}({args}); return data;".format(func_name=func_name, args=",".join(func_args))
    return func_str


def CsQuery(field_dict, end_date, bk_name=_ashare, stock_list=None, condition="1"):
    """对天软Query函数的封装
    Parameters:
    ===========
    field_dict:
    """
    if stock_list is None:
        stock_list = "''"
    else:
        stock_list = "'%s'" % ";".join(map(tradecode_to_tslcode, stock_list))
    encode_date = tsl.EncodeDate(end_date.year, end_date.month, end_date.day)
    func_name = "Query"
    func_args = [bk_name, stock_list, condition, "''"] + list(reduce(lambda x, y: x+y, field_dict.items()))
    script_str = _gstr_from_func(func_name, func_args)
    data = tsl.RemoteExecute(script_str, {'CurrentDate': encode_date})
    df = parse2DArray(data, column_decode=['IDs'])
    df['IDs'] = df['IDs'].apply(tslcode_to_tradecode)
    df['date'] = end_date
    return df.set_index(['date', 'IDs'])


@DateRange2Dates
def PanelQuery(field_dict, start_date=None, end_date=None, dates=None,
               bk_name=_ashare, stock_list=None, condition="1"):
    """对天软Query函数的封装
    Parameters:
    ===========
    field_dict:
    """
    field_dict.update({"'IDs'": 'DefaultStockID()'})
    data = []
    for date in dates:
        idata = CsQuery(field_dict, date, bk_name=bk_name, stock_list=stock_list, condition=condition)
        data.append(idata)
    return pd.concat(data).sort_index()


def partialCsQueryFunc(*args, **kwargs):
    """CsQuery的偏函数"""
    return partial(CsQuery, *args, **kwargs)


def _read_factors():
    file_pth = os.path.abspath(os.path.dirname(__file__)+'/..')
    file_pth = os.path.join(file_pth, 'resource', 'tsl_tableinfo.xlsx')
    factorIDs = pd.read_excel(file_pth, index_col=0, header=0)
    return factorIDs


class TableInfo(object):
    factorIDs = _read_factors()

    def factor_id(self, factor_name):
        return self.factorIDs.at[factor_name, 'ID']

    def factor_engname(self, factor_name):
        return self.factorIDs.at[factor_name, 'Eng_Name']


class TSLDBOnline(object):
    table_info = TableInfo()

    def _wrap_loaddata(self, field_name, func_name, start=None, end=None, dates=None, ids=None):
        factor_name = self.table_info.factor_engname(field_name)
        field_dict = {"'%s'" % factor_name: func_name}
        return PanelQuery(field_dict, start_date=start, end_date=end, dates=dates, stock_list=ids)

    def load_ttm(self, field_name, start=None, end=None, dates=None, ids=None):
        """ttm数据"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_ttm(%s)' % factor_id
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_sq(self, field_name, start=None, end=None, dates=None, ids=None):
        """最新单季度"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_sq(%s)' % factor_id
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_latest_year(self, field_name, n=0, start=None, end=None, dates=None, ids=None):
        """最新年报"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_latest_year(%s, %s)' % (factor_id, n)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_last_nyear(self, field_name, n, start=None, end=None, dates=None, ids=None):
        """N年之前的财务数据"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_last_nyear(%s, %s)' % (factor_id, n)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_last_nyear_ttm(self, field_name, n, start=None, end=None, dates=None, ids=None):
        """N年之前TTM数据"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_last_nyear_ttm(%s, %s)' % (factor_id, n)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_last_nyear_sq(self, field_name, n, start=None, end=None, dates=None, ids=None):
        """N年之前单季度数据"""
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'load_last_nyear_sq(%s, %s)' % (factor_id, n)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def inc_rate_hb(self, field_name, stat_type=0, start=None, end=None, dates=None, ids=None):
        """环比增长率
        stat_type:
            0: 最新报表(累计)
            1：最新单季度
            2: ttm
        """
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'inc_rate_hb(%s, %s)' % (factor_id, stat_type)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def inc_rate_tb(self, field_name, stat_type=0, start=None, end=None, dates=None, ids=None):
        """同比增长率
        stat_type:
            0: 最新报表(累计)
            1：最新单季度
            2: ttm
        """
        factor_id = self.table_info.factor_id(field_name)
        func_name = 'inc_rate_tb(%s, %s)' % (factor_id, stat_type)
        return self._wrap_loaddata(field_name, func_name, start, end, dates, ids)

    def load_netprofit_ttm_incl_express(self, start=None, end=None, dates=None, ids=None):
        """包含业绩快报的TTM净利润"""
        func_name = 'load_netprofit_ttm_incl_express()'
        field_dict = {"'net_profit_excl_min_int_inc'": func_name}
        return PanelQuery(field_dict, start_date=start, end_date=end, dates=dates, stock_list=ids)

    def load_netasset_incl_express(self, start=None, end=None, dates=None, ids=None):
        """包含业绩快报的股东权益"""
        func_name = 'load_netprofit_ttm_incl_express()'
        field_dict = {"'tot_shrhldr_eqy_excl_min_int'": func_name}
        return PanelQuery(field_dict, start_date=start, end_date=end, dates=dates, stock_list=ids)

    @DateRange2Dates
    def get_index_members(self, idx, start_date=None, end_date=None, dates=None):
        r = []
        func = 'getBKMembers'
        idx = "'%s'" % idx
        for d in dates:
            dd = d.strftime("%Y%m%d")
            script_str = _gstr_from_func(func, [dd, idx])
            data = tsl.RemoteExecute(script_str, {})
            data = parse1DArray(data, "IDs", 1)
            data['date'] = d
            r.append(data)
        data = pd.concat(r)
        data['sign'] = 1
        data['IDs'] = data['IDs'].str[2:]
        return data.set_index(['date', 'IDs'])


if __name__ == '__main__':
    field = {"'list_days'": 'amount()'}
    # data = PanelQuery(field, start_date='20180101', end_date='20180110')
    a = TSLDBOnline()
    data = a.get_index_members('中证1000', dates=['20180116'])
    print(data)
