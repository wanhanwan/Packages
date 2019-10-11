# coding: utf-8
import numpy as np
import pandas as pd
from PkgConstVars import H5_PATH, FACTOR_PATH

from ..utils.datetime_func import DateStr2Datetime
from ..utils.tool_funcs import parse_industry, dummy2name
from .converter import IndustryConverter
from .csv_db import CsvDB
from .h5db import H5DB
from .trade_calendar import tc
from .tseries import resample_func, resample_returns

class Sector(object):
    def __init__(self, h5db):
        self.h5DB = h5db

    def get_st(self, dates=None, ids=None, idx=None):
        """某一个时间段内st的股票"""
        if not isinstance(dates, list):
            dates = [dates]
        st_list = self.h5DB.load_factor2('ashare_st', '/base/', dates=dates, ids=ids, idx=idx,
                                         stack=True).query('ashare_st==1.0')
        return st_list

    def get_history_ashare(self, dates, min_listdays=None,
                           drop_st=False):
        """获得某一天的所有上市A股"""
        if isinstance(dates, str):
            dates = [dates]
        stocks = self.h5DB.load_factor2('ashare', '/base/', dates=dates, stack=True)
        if min_listdays is not None:
            stocks = stocks.query('ashare>@min_listdays')
        if drop_st:
            st = self.get_st(dates=dates)
            stocks = stocks[~stocks.index.isin(st.index)]
        return stocks
    
    def get_industry_dummy(self, industry, ids=None, dates=None, idx=None, drop_first=True):
        """行业哑变量"""
        indu_id = parse_industry(industry)
        indu_info = self.h5DB.load_as_dummy2(indu_id, '/base/dummy/', dates=dates, ids=ids, idx=idx)
        if isinstance(drop_first, bool) and drop_first:
            indu_info = indu_info.iloc[:, 1:]
        elif isinstance(drop_first, str):
            indu_info = indu_info.drop(drop_first, axis=1)
        indu_info.rename_axis(['date', 'IDs'], inplace=True)
        return indu_info

    def get_industry_info(self, industry, ids=None, dates=None, idx=None):
        """行业信息
        返回Series, value是行业名称, name是行业分类
        """
        industry_dummy = self.get_industry_dummy(industry, ids, dates, idx, False)
        industry_info = dummy2name(industry_dummy)
        industry_info.name = industry
        return industry_info
    
    def get_index_weight(self, index_ticker, dates=None):
        """指数成分股权重"""
        file_name = f'consWeight{index_ticker}'
        try:
            data = self.h5DB.load_factor2(file_name, '/base/').sort_index()
        except OSError:
            file_name = f'cons{index_ticker}'
            data = self.h5DB.load_factor2(file_name, '/base/').sort_index()
        df = data.reindex(pd.DatetimeIndex(dates, name='date'), method='ffill').stack().to_frame('consWeight')
        return df

    def get_industry_members(self, industry_name, classification='中信一级', dates=None):
        """某个行业的股票列表
        返回Series
        """
        dummy = self.get_industry_dummy(classification, dates=dates, drop_first=False)
        df = dummy[dummy[industry_name]==1]
        return dummy2name(df)


h5 = H5DB(H5_PATH)
h5_2 = H5DB(FACTOR_PATH)
csv = CsvDB()
sec = Sector(h5_2)

