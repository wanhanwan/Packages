from missing_value_manage import missing_value_func_dict
from grouping_manage import grouping_funcs_dict
from utils.excel_io import *
from utils.datetime_func import Datetime2DateStr
import copy
import os
import shutil

def load_factor(self,factor, sector_name, dates):
    """从数据库中读取数据"""
    sector_name = self._config.universe
    start_date = self._config.start_date
    end_date = self._config.end_date
    dates = self._env._trade_calendar.get_trade_days(start_date, end_date, self._config.freq)
    if sector_name == '全A':
        stocks = self._env._sector.get_history_ashare(dates)
    else:
        stocks = self._env._sector.get_index_members(sector_name,dates=dates)
    ids = stocks.index.levels[1].unique().tolist()
    factor_data = self._env._data_source.load_factor(factor.axe,ids=ids,dates=dates)
    factor.data = factor_data.reindex(stocks.index)
    factor.update_info()

def drop_nan(self, factor, inplace=False):
    """处理缺失值"""
    try:
        nan_func = missing_value_func_dict[self._config.nan_method]
    except Exception:
        raise KeyError("method not supported")
    new_data = nan_func(factor)
    if inplace:
        factor.data = new_data
        factor.update_info()
        return
    else:
        new_factor = self.copy(factor,factor.name+'_nan_dropped')
        new_factor.data = new_data
        new_factor.update_info()
    return new_factor

def drop_untradable_stocks(self,factor, inplace=False):
    """去掉不可以交易的股票"""
    all_dates = factor.data.index.levels[0].tolist()
    stocks_trade_status = self._env._data_source.get_stock_trade_status(dates=all_dates)
    common = factor.data.merge(stocks_trade_status[['no_trading']],
                               left_index=True,right_index=True,how='left')
    new_data = common[common['no_trading']!=True][[factor.axe]]
    if inplace:
        factor.data = new_data
        factor.update_info()
        return
    else:
        new_factor = self.copy(factor,factor.name+'_nan_dropped')
        new_factor.data = new_data
        new_factor.update_info()
    return new_factor

def grouping(self,factor):
    methods = self._config.grouping_method
    for method in methods:
        grouping_funcs_dict[method](factor,self._env._data_source, self._config)

def gen_stock_list(self, factor):
    stock_list = {}
    methods = self._config.grouping_method
    for method in methods:
        group_info = factor.grouping_info[method]
        group_info = group_info[group_info['group_id']==1]
        group_info['Weight'] = 1 / group_info.groupby(level=0)['group_id'].count()
        group_info.reset_index(inplace=True)
        group_info['Dates'] = group_info['date'].apply(Datetime2DateStr)
        stock_list[method] = group_info[['Dates', 'IDs', 'Weight']]
    factor.stock_list = stock_list


        
        








