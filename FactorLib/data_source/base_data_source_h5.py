# coding: utf-8
from functools import lru_cache
from datetime import timedelta, datetime
from PkgConstVars import H5_PATH, FACTOR_PATH

import os
import pendulum
import calendar
import numpy as np
import pandas as pd

from .h5db import H5DB
from .csv_db import CsvDB
from .bcolz_db import BcolzDB
from .trade_calendar import tc
from ..utils.tool_funcs import parse_industry, dummy2name, is_non_string_iterable


class Sector(object):
    def __init__(self, h5db):
        self.h5DB = h5db

    def get_st(self, dates=None, ids=None, idx=None):
        """某一个时间段内st的股票"""
        if not is_non_string_iterable(dates):
            dates = [dates]
        st_list = self.h5DB.load_factor2('ashare_st', '/base/', dates=dates, ids=ids, idx=idx,
                                         stack=True).query('ashare_st==1.0')
        return st_list

    def get_history_ashare(self, dates, min_listdays=None,
                           drop_st=False, history=False):
        """
        获得某一天的所有上市A股

        Parameters:
        -----------
        dates: str or list
            日期
        min_listdays: int
            上市最少天数(日历日)
        drop_st:
            是否去掉ST
        history: bool
            是否包含退市股
        """
        if not is_non_string_iterable(dates):
            dates = [dates]
        if history:
            stocks = self.h5DB.load_factor2('ashare', '/base/').loc[:pd.to_datetime(max(dates))].fillna(method='ffill')
            stocks = stocks.loc[pd.DatetimeIndex(dates)].stack().to_frame('ashare')
            min_listdays = 0
            drop_st = None
        else:
            stocks = self.h5DB.load_factor2('ashare', '/base/', dates=dates, stack=True)
        if min_listdays is not None:
            stocks = stocks.query('ashare>@min_listdays')
        if drop_st:
            st = self.get_st(dates=dates)
            stocks = stocks[~stocks.index.isin(st.index)]
        return stocks
    
    def get_industry_dummy(self, industry, ids=None, dates=None,
                           idx=None, drop_first=True, fill_method=None):
        """行业哑变量
        :param industry: 中信一级、申万一级等
        """
        indu_id = parse_industry(industry)
        indu_info = self.h5DB.load_as_dummy2(
            indu_id, '/base/dummy/', dates=dates, ids=ids, idx=idx, fill_method=fill_method
        )
        if isinstance(drop_first, bool) and drop_first:
            indu_info = indu_info.iloc[:, 1:]
        elif isinstance(drop_first, str):
            indu_info = indu_info.drop(drop_first, axis=1)
        indu_info.rename_axis(['date', 'IDs'], inplace=True)
        return indu_info

    def get_industry_info(self, industry, ids=None, dates=None, idx=None, fill_method=None):
        """行业信息
        返回Series, value是行业名称, name是行业分类
        """
        industry_dummy = self.get_industry_dummy(industry, ids, dates, idx, False, fill_method)
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
    
    @lru_cache()
    def get_fund_file(self):
        data =  pd.read_excel(
            self.h5DB.abs_factor_path('/fund/', '基金基本资料new').replace('.h5', '.xlsx'),
            header=0,
            parse_dates=['基金成立日', '基金到期日', '上市日期']
        )
        data.rename(
            columns={
                '证券代码': '代码',
                '证券简称': '名称',
                '基金管理人简称': '基金公司',
                '基金成立日':'成立日期',
                '基金到期日':'到期日期',
                '基金经理(现任)':'现任基金经理',
                '基金经理(历任)': '历任基金经理',
                '发行份额\r\n[单位] 百万份':'发行规模',
                '基金规模\r\n[单位] 百万元': '最新规模',
                '业绩比较基准': '比较基准',
                '投资类型(二级分类)': '投资类型'
                },
            inplace=True
            )
        data = data[data['代码'].apply(lambda x: len(x)==9)]
        data[['发行规模','最新规模']] /= 100.0
        return data

    def get_fund_info(self, fund_code=None, fund_type=None,
                      start_date=None, fund_manager=None,
                      invest_type_1=None, benchmark=None):
        """
        获取公募基金的基本信息
        数据源来自Wind终端-基金数据浏览器-我的模板-全部基金基本信息(包含未成立、已到期)。这个数据需要定时更新。

        :param fund_code str or list
            基金代码(带后缀)
        :param fund_type str
            可用','连接
            基金类型(二级分类) 偏股混合型基金  中长期纯债型基金 被动指数型基金 混合债券型一级基金
                     灵活配置型基金  增强指数型基金  普通股票型基金  偏债混合型基金
                     商品型基金      混合债券型二级基金 股票多空     平衡混合型基金
                     短期纯债型基金   国际(QDII)混合型基金        被动指数型债券基金
                     国际(QDII)股票型基金  REITs
        :param start_date str
            基金成立起始日期
        :param fund_manager str
            基金经理
        :param invest_type_1 str
            可用','连接
            一级投资类型：股票型基金、混合型基金、债券型基金、货币市场型基金、另类投资基金、QDII基金
        :param benchmark str
            基准
        :return DataFrame 代码 名称 现任基金 历任基金经理 基金公司 成立日期 到期日期 发行份额 最新规模 比较基准 投资类型
        """
        fund_info = self.get_fund_file()
        if fund_code:
            code = [x[:-3]+'.OF' for x in fund_code]
            fund_info = fund_info[fund_info['代码'].isin(code)]
        if fund_type:
            fund_info = fund_info[fund_info['投资类型'].isin(fund_type.split(','))]
        if start_date:
            fund_info = fund_info[fund_info['成立日期'] <= pd.to_datetime(start_date)]
        if fund_manager:
            fund_info = fund_info[fund_info['现任基金经理'].str.contains(fund_manager)]
        if invest_type_1:
            fund_info = fund_info[fund_info['投资类型(一级分类)'].isin(invest_type_1.split(','))]
        if benchmark:
            fund_info = fund_info[fund_info['比较基准'].str.contains(benchmark)]
        return fund_info
    
    @lru_cache()
    def _load_raw_data(self):
        contracts = h5_2.read_h5file('contracts', '/opt/')
        contracts_adj = h5_2.read_h5file('contracts_adjust', '/opt/').sort_values('adj_date')
        return contracts, contracts_adj
    
    @lru_cache()
    def get_history_option(self, date, exchange=None, underlying_ticker=None):
        """
        获取历史期权合约(上交所、深交所和中金所的股票期权)
        数据源：contracts.h5\contracts_adjust.h5

        Parameters:
        ----------
        date: str
            交易日期
        exchenge: str
            上交所XSHG、深交所XSHE、中金所CCFX
        underlying_ticker: str
            标的证券代码，如000300、510050、510300、159919
        
        Returns:
        --------
        DataFrame: date code trading_code symbol exchange_code list_date adj_date
        """
        c, cadj = self._load_raw_data()
        dt = pd.to_datetime(date)
        contracts_curr_dt = c[(c.list_date <= dt) & (c.last_trade_date >= dt)][
            ['code', 'trading_code', 'symbol', 'exchange_code', 'underlying_symbol', 'list_date']]
        contracts_curr_dt.set_index('code', inplace = True)
        contracts_curr_dt['adj_date'] = pd.to_datetime('1900-01-01')
        assert contracts_curr_dt.index.is_unique

        
        contracts_adj_curr_dt = contracts_curr_dt[contracts_curr_dt.index.isin(cadj.code.unique())]
        for i in contracts_adj_curr_dt.index:
            adj_record = cadj[(cadj.code == i) & (dt < cadj.adj_date)]
            if not adj_record.empty:
                contracts_curr_dt.loc[i, 'trading_code'] = adj_record['ex_trading_code'].iat[0]
                contracts_curr_dt.loc[i, 'symbol'] = adj_record['ex_name'].iat[0]
            adj_record2 = cadj[(cadj.code == i) & (dt >= cadj.adj_date)]
            if not adj_record2.empty:
                contracts_curr_dt.loc[i, 'trading_code'] = adj_record2['new_trading_code'].iat[-1]
                contracts_curr_dt.loc[i, 'symbol'] = adj_record2['new_name'].iat[-1]
                contracts_curr_dt.loc[i, 'adj_date'] = adj_record2['adj_date'].iat[-1]
        contracts_curr_dt['date'] = dt

        if exchange:
            contracts_curr_dt = contracts_curr_dt[contracts_curr_dt.exchange_code == exchange]
        if underlying_ticker:
            contracts_curr_dt = contracts_curr_dt[contracts_curr_dt.underlying_symbol == underlying_ticker]
        return contracts_curr_dt.reset_index()

    def get_main_dominant_future_contract(self, underlying, start_date=None, end_date=None):
        """
        根据昨日最大持仓获取期货主力合约。

        支持的期货品种为IF、IH、IC、T、TF

        Parameter:
        ---------
        underlying: str
            期货品种：IF、IH、IC、T、TF
        start_date: str
            开始日期
        end_date: str
            结束日期

        Return:
        -------
        Series(index=dates, values=ticker)
        """
        contracts = pd.read_json(
            os.path.join(FACTOR_PATH, 'futures', 'main_contracts', f'{underlying}.json'),
            encoding='GBK',
            compression='gzip',
            orient='index'
        )[0]
        if start_date:
            contracts = contracts.loc[start_date:]
        if end_date:
            contracts = contracts.loc[:end_date]
        return contracts

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
        funds = data.query("ticker==@ticker & ratio>=@min_ratio & date==@period")
        if not funds.empty:
            fund_info = sec.get_fund_info(funds['IDs'].tolist())
            funds = pd.merge(
                funds, fund_info, left_on='IDs', right_on='代码'
            ).sort_values('ratio', ascending=False)
            funds = funds[['IDs', 'report_date', 'ticker', 'ratio',
                           '名称', '现任基金经理', '成立日期', '最新规模(亿元)']]
        return funds
    
    @staticmethod
    def get_fund_stock_portfolio(ticker, period=None, scale_weight=True, datasource='jq'):
        """
        获取基金的个股配比

        Parameters:
        -----------
        ticker: str
            基金代码(无后缀)
        period: str
            财报日期 YYYYMMDD
        scale_weight: bool
            权重是否归一化
        datasource: str
            数据源，可选jq、None(tushare)
        
        Returns:
        --------
        DataFrame(index=[date, IDs],columns=ticker)
            持仓组合权重
        """
        file_name = ((datasource or '') + '_fund_portfolio').strip('_')
        weight_name = 'ratio' if datasource == 'tsl' else 'proportion'
        ticker += '.OF'

        data = h5_2.read_h5file(file_name, '/fund/')
        if period is None:
            funds = data.query("IDs==@ticker")
        else:
            period = pd.to_datetime(period)
            funds = data.query("IDs==@ticker & date==@period")
        if not funds.empty:
            funds = (
                funds
                .set_index(['date', 'symbol'])
                .sort_index()
                .rename_axis(['date', 'IDs'])
            )[weight_name]

            if scale_weight:
                funds = funds.groupby('date').transform(lambda x: x/x.sum())

            return funds.to_frame(ticker)
        else:
            return pd.DataFrame()

    @staticmethod
    def get_fund_industry_allocation(ticker, period, industry='申万一级', datasource=None):
        """
        获取基金行业配比

        Parameters:
        -----------
        ticker: str
            基金代码(无后缀)
        period: str
            财报日期 YYYYMMDD
        industry: str
            行业分类，如申万一级、中信一级等
        
        Returns:
        --------
        DataFrame(index=date, columns=industry_names)
        """
        portfolio = Fund.get_fund_stock_portfolio(
            ticker, period, scale_weight=True, datasource=datasource)
        if not portfolio.empty:
            indu = sec.get_industry_info(industry, idx=portfolio)
            if indu.empty:
                return pd.DataFrame()
            allocation = portfolio.groupby(indu)[ticker].sum()
            return pd.DataFrame(
                allocation.to_numpy()[None, :],
                index=pd.DatetimeIndex([period]),
                columns=allocation.index
            )
        else:
            return pd.DataFrame()

    @staticmethod
    def get_fund_nav(tickers, start_date=None, end_date=None, field='adj_nav'):
        """
        基金净值提取

        Parameters:
        -----------
        tickers: list of str
            基金代码(无后缀)
        """
        dates = tc.get_trade_days(start_date, end_date)
        nav = h5_2.load_factor2(field, '/fund/', ids=tickers, dates=dates)
        return nav

    @staticmethod
    def get_fund_maneger(ticker=None, person_id=None, person_name=None, date=None):
        """
        基金经理信息

        Parameters:
        ticker: str
            基金代码(无后缀)
        person_id: int
            基金经理代码(通联格式)
        person_name: str
            基金经理姓名
        date: datetime-like
            返回在改日期正在任职的基金经理
        
        Return:
        -------
        fund_info: DataFrame
            secID ticker secShortName personID name accessionDate dimissionDate
        """
        from .fund import load_manager_info
        from FactorLib.utils.tool_funcs import get_members_of_date
        fund_info = load_manager_info(
            personID=person_id, name=person_name, fund_ticker=ticker
            )
        if date:
            fund_info = get_members_of_date(
                date,
                'accessionDate',
                'dimissionDate',
                fund_info.columns,
                fund_info
            )
        return fund_info


class Option(object):

    @lru_cache()
    def _load_raw_data(self):
        contracts = h5_2.read_h5file('contracts', '/opt/')
        contracts_adj = h5_2.read_h5file('contracts_adjust', '/opt/').sort_values('adj_date')
        return contracts, contracts_adj

    @staticmethod
    def get_latest_trading_code_info(contract_id, date):
        """"
        获取最新期权合约的交易代码、调整日期和上市日期

        Parameters:
        -----------
        contract_id: str
            合约代码 如10000001 或 10000001.XSHG
        
        date: datetime or str
            当前日期

        Return:
        ----------
        Series
        """
        contracts = sec.get_history_option(date = date)
        contracts = contracts.query("code == '%s'" % contract_id.split('.')[0])
        assert len(contracts) == 1
        return contracts[['code', 'trading_code', 'adj_date', 'list_date']].iloc[0]
    
    @staticmethod
    def get_contract_daily_info(date, is_adjusted=None, contract_type=None, underlying_symbol=None,
                                level=None, level_non_adjusted=None, delivery_type=None, squeeze=True):
        """
        获取每日期权信息
        
        Return: DataFrame
        -----------------
        code trading_code rq_code contract_type exercise_price pre_close_underlying underlying_symbol
        position level level_non_adjusted delivery_type list_date adj_date
        """
        data = h5_2.read_h5file('contract_info', '/opt/')
        date = pd.to_datetime(date)
        
        query_str = "(date==@date)"
        if is_adjusted is not None:
            query_str += f" and (is_adjusted=={is_adjusted})"
        if contract_type:
            query_str += f" and (contract_type=='{contract_type}')"
        if underlying_symbol:
            query_str += f" and (underlying_symbol=='{underlying_symbol}')"
        if level:
            query_str += f" and (level=='{level}')"
        if level_non_adjusted:
            query_str += f" and (level_non_adjusted=='{level_non_adjusted}')"
        if delivery_type:
            query_str += f" and (delivery_type=='{delivery_type}')"
        rslt = data.query(query_str)
        if len(rslt)==1 and squeeze:
            return rslt.iloc[0, :]
        return rslt
    
    def get_delisted_dates(self, start, end, exchange='XSHG', retstr=None) -> list:
        """
        获取期权的到期交易日。

        上交所和深交所的到期月份是每月的第四个周三(法定节假日顺延)；  
        中金所的到期月份是每月第三个周五(法定节假日顺延).

        Parameter
        ---------
        start: str
            开始日期
        end: str
            结束日期
        exchange: str
            交易所代码 XSHG、XSHE、CFFX
        """
        c = calendar.Calendar(firstweekday=calendar.MONDAY)
        
        @lru_cache()
        def get_nth_week_day(year:int, month:int, n:int, weekday:int):
            monthcal = c.monthdatescalendar(year, month)
            day = [day for week in monthcal for day in week if day.weekday()==weekday and day.month==month][n-1]
            return day
        
        dates = pd.period_range(start, end, freq='1M')

        if exchange in ['XSHE', 'XSHG', 'SH', 'SZ']:
            days = [get_nth_week_day(year, month, 4, calendar.WEDNESDAY) for year, month in zip(dates.year, dates.month)]
        elif exchange in ['CFFX', 'CCFX']:
            days = [get_nth_week_day(year, month, 3, calendar.FRIDAY) for year, month in zip(dates.year, dates.month)]
        else:
            raise ValueError(f"not supported exchange code {exchange}")
        tdays = [tc.tradeDayOffset(x, 1, incl_on_offset_today=True, retstr=retstr) for x in days]
        if pd.Timestamp(start) > tdays[0]:
            tdays = tdays[1:]
        if pd.Timestamp(end) < tdays[-1]:
            tdays = tdays[:-1]
        return tdays


h5 = H5DB(H5_PATH)
csv = CsvDB()
fund = Fund()
option = Option()
bcolz_db = BcolzDB(FACTOR_PATH)
