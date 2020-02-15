# coding: utf-8
import numpy as np
import pandas as pd
from PkgConstVars import H5_PATH, FACTOR_PATH
from functools import lru_cache
from ..utils.tool_funcs import parse_industry, dummy2name
from .csv_db import CsvDB
from .h5db import H5DB
from .trade_calendar import tc


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
        """行业哑变量
        :param industry: 中信一级、申万一级等
        """
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
        df.index.names = ['date', 'IDs']
        return df

    def get_industry_members(self, industry_name, classification='中信一级', dates=None):
        """某个行业的股票列表
        返回Series
        """
        dummy = self.get_industry_dummy(classification, dates=dates, drop_first=False)
        df = dummy[dummy[industry_name]==1]
        return dummy2name(df)

    def get_fund_info(self, fund_code=None, fund_type=None, start_date=None, fund_manager=None, benchmark=None):
        """
        获取公募基金的基本信息
        数据源来自Wind终端-基金-专题统计-报表收藏-股票+混合型基金基本资料。这个数据需要定时更新。

        :param fund_code str or list
            基金代码
        :param fund_type str
            可用','连接
            基金类型 偏股混合型基金  中长期纯债型基金 被动指数型基金 混合债券型一级基金
                     灵活配置型基金  增强指数型基金  普通股票型基金  偏债混合型基金
                     商品型基金      混合债券型二级基金 股票多空     平衡混合型基金
                     短期纯债型基金   国际(QDII)混合型基金        被动指数型债券基金
                     国际(QDII)股票型基金  REITs
        :param start_date str
            基金成立起始日期
        :param fund_manager str
            基金经理
        :param benchmark str
            基准
        :return DataFrame 代码 名称 现任基金 历任基金经理 基金公司 成立日期 到期日期 发行份额 最新规模 比较基准 投资类型
        """
        @lru_cache()
        def get_fund_file():
            return pd.read_excel(
                self.h5DB.abs_factor_path('/fund/', '股票+混合型基金基本资料').replace('.h5', '.xlsx'),
                header=0,
                parse_dates=['成立日期']
            )
        fund_info = get_fund_file()
        if fund_code:
            fund_info = fund_info[fund_info['代码'].isin(fund_code)]
        if fund_type:
            fund_info = fund_info[fund_info['投资类型'].isin(fund_type.split(','))]
        if start_date:
            fund_info = fund_info[fund_info['成立日期'] <= pd.to_datetime(start_date)]
        if fund_manager:
            fund_info = fund_info[fund_info['现任基金经理'].str.contains(fund_manager)]
        if benchmark:
            fund_info = fund_info[fund_info['比较基准'].str.contains(benchmark)]
        return fund_info


h5_2 = H5DB(FACTOR_PATH)
sec = Sector(h5_2)


class Fund(object):
    @staticmethod
    def get_fund_by_stock(ticker, period, min_ratio=0.05):
        """
        获取重仓某只股票的基金

        :param ticker str
            股票代码
        :param period str
            财报日期 YYYYMMDD
        :param min_ratio float(%)
            占基金比重的最小阈值
        """
        data = h5_2.read_h5file('fund_portfolio', '/fund/')
        period = pd.to_datetime(period)
        funds = data.query("symbol==@ticker & stk_mkv_ratio>=@min_ratio & date==@period")
        if not funds.empty:
            fund_info = sec.get_fund_info(funds['IDs'].tolist())
            funds = pd.merge(
                funds, fund_info, left_on='IDs', right_on='代码'
            ).sort_values('stk_mkv_ratio', ascending=False)
            funds = funds[['IDs', 'ann_date', 'date', 'symbol', 'stk_mkv_ratio',
                           '名称', '现任基金经理', '成立日期', '最新规模(亿元)']]
        return funds


h5 = H5DB(H5_PATH)
csv = CsvDB()
fund = Fund()
