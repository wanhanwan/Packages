# coding: utf-8
# author: wamhanwan
"""Tushare API"""
import tushare as ts
import pandas as pd
import numpy as np
from FactorLib.utils.tool_funcs import get_members_of_date


_token = '6135b90bf40bb5446ef2fe7aa20a9467ad10023eda97234739743f46'
SHEXG = 'SSE' # 上交所代码
SZEXG = 'SZSE' # 深交所代码


ts.set_token(_token)
pro_api = ts.pro_api()


class TushareDB(object):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            return cls._instance
        return cls._instance

    def __init__(self):
        self._api = ts
        self._pro_api = pro_api
        self._token = _token

    @classmethod
    def get_instance(cls):
        return TushareDB()

    def run_api(self, api_name, *args, **kwargs):
        return getattr(self._pro_api, api_name)(*args, **kwargs)

    def format_date(self, data, column=None):
        data[column] = pd.to_datetime(data[column], format='%Y%m%d')
        return data

    def stock_basic_get(self, list_status=None, exchange=None, is_hs=None,
                        fields=None):
        """基础数据-股票列表(只能取最新数据)

        Parameters:
        ----------
        list_status: str
            上市状态  L上市 D退市 P暂停上市
        exchange: str
            交易所 SHEXG上交所 SZEXG深交所
        is_hs: str
            是否深港通标的 N否 H沪股通 S深股通

        Fields:
        ------
        symbol 股票代码(ticker)
        name 股票简称
        industry 行业
        list_status 上市状态
        list_date 上市日期
        delist_date 退市日期
        is_hs 是否深港通标的
        """
        data1 = self.run_api('stock_basic', list_status='L', exchange=exchange, is_hs=is_hs,
                            fields=fields)
        data2 = self.run_api('stock_basic', list_status='P', exchange=exchange, is_hs=is_hs,
                            fields=fields)
        data3 = self.run_api('stock_basic', list_status='D', exchange=exchange, is_hs=is_hs,
                            fields=fields)
        l = [data1,data2,data3]
        if list_status:
            status_index =[['L', 'P', 'D'].index(x) for x in  ','.split(list_status)]
            return pd.concat([l[i] for i in status_index]).sort_values('symbol')
        return pd.concat(l).sort_values('symbol')

    def stock_st_get(self, date):
        """A股戴帽摘帽

        Paramters:
        ---------
        date: str
            日期 YYYYMMDD

        Returns:
        ------
        DataFrame: IDs name start_date end_date
        """
        data = self.run_api('namechange', end_date=date)
        data = data[data['name'].str.contains('ST')]
        data = data.fillna({'end_date': '21001231'})
        data = data[(data['start_date']<=date)&(data['end_date']>=date)]
        data = data[~data['ts_code'].duplicated(keep='last')]
        data['IDs'] = data['ts_code'].str[:6]
        return data

    def stock_onlist_get(self, date):
        """A股特定日期的股票列表
        Return
        ------
        DataFrame: symbol name list_date delist_date
        """
        all_stocks = self.stock_basic_get(
            fields='symbol,name,list_date,delist_date'
        ).fillna({'delist_date':'21001231'})
        indices = (all_stocks['list_date']<=date)&(all_stocks['delist_date']>date)
        return all_stocks[indices]

    def index_weight_get(self, index_code, date):
        """A股指数成分股权重

        Parameters:
        -----------
        index_code: str
            指数代码 399300.SZ沪深300 000905.SH中证500 000906.SH中证800
        date: str
            日期 YYYYMMDD

        Returns:
        --------
        DataFrame index_code con_code trade_date weight
        """
        start_date = (pd.to_datetime(date)-pd.Timedelta(days=30)).strftime('%Y%m%d')
        data = self.run_api('index_weight',index_code=index_code, start_date=start_date, end_date=date)
        data = data[data['trade_date']==data['trade_date'].max()]
        data['index_code'] = data['index_code'].str[:6]
        data['con_code'] = data['con_code'].str[:6]
        data['trade_date'] = date
        data['weight'] /= 100.0
        return data

    def quota_get(self, start_date, end_date, ticker=None, adj=None, freq='D', ma=None, asset='E'):
        """行情通用接口

        Paramters:
        ----------
        ticker: str
            证券代码, 必填且为单只证券代码
        start_date: str
            开始日期 YYYYMMDD
        end_date: str
            结束日期 YYYYMMDD
        adj: str
            复权类型 qfq前复权 hfq后复权
        freq：str
            数据频率 1\5\15\30\60min D
        ma: int
            均线数值
        asset: str
            资产类型 E股票 I指数 FT期货 FD基金 O期权 CB可转债

        Returns:
        --------
        DataFrame: ts_code, trade_date ...

        """
        if ticker and asset=='E':
            ts_code = ticker+'.SH' if ticker[0]=='6' else ticker+'.SZ'
        elif ticker:
            ts_code = ticker
        else:
            ts_code = None
        data = self._api.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                 freq=freq, adj=adj, ma=ma, asset=asset, adjfactor=True)
        if data is None:
            return pd.DataFrame()
        if not data.empty:
            data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        return data

    def stock_daily_basic_get(self, trade_date=None, ticker=None, start_date=None, end_date=None,
                              fields=None):
        """股票每日行情指标

        Parameters:
        ----------
        trade_date: str
            交易日期
        ticker: str
            股票代码 与交易日期必须至少一个参数非空。
        start_date: str
            开始日期
        end_date: str
            结束日期
        fields: str
            返回字段 ts_code股票代码 trade_date交易日期 close当日收盘价 turnover_rate换手率(%)
            turnover_rate_f换手率(自由流通股) volume_ratio量比 pe市盈率 pe_ttm市盈率(TTM)
            pb市净率 ps市销率 ps_ttm市销率(TTM) dv_ratio股息率 dv_ttm股息率(TTM) total_share总股本(万股)
            float_share流通股本(万) free_share自由流通股本(万) total_mv总市值(万) circ_mv流通市值(万)
        
        Returns:
        -------
        DataFrame: ts_code, trade_date ...
        """
        if ticker:
            ticker = ticker+'.SH' if ticker[0]=='6' else ticker+'.SZ'
        data = self.run_api('daily_basic', ts_code=ticker, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, fields=fields)
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        return data
    
    def stock_daily_price_get(self, ticker=None, trade_date=None, start_date=None, end_date=None,
                              fields=None):
        """
        A股日行情数据

        Parameters:
        -----------
        trade_date: str
            交易日期
        ticker: str
            股票代码 与交易日期必须至少一个参数非空。
        start_date: str
            开始日期
        end_date: str
            结束日期
        fields: str
            返回字段 ts_code股票代码 trade_date交易日期 close当日收盘价 open开盘价
                    high最高价 low最低价 pre_close前收盘价 change涨跌 pct_chg涨跌幅(未复权)
                    vol成交量(手) amount成交额(千元)

        Returns:
        -------
        DataFrame: ts_code, trade_date ...
        """
        if ticker:
            ticker = ticker+'.SH' if ticker[0]=='6' else ticker+'.SZ'
        data = self.run_api('daily', ts_code=ticker, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, fields=fields)
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        return data

    def stock_monthly_price_get(self, ticker='', trade_date='', start_date='', end_date='',
                                fields=None):
        """
        A股月行情数据

        Parameters:
        -----------
        trade_date: str
            交易日期
        ticker: str
            股票代码 与交易日期必须至少一个参数非空。
        start_date: str
            开始日期
        end_date: str
            结束日期
        fields: str
            返回字段 ts_code股票代码 trade_date交易日期 close当日收盘价 open开盘价
                    high最高价 low最低价 pre_close前收盘价 change涨跌 pct_chg涨跌幅(未复权)
                    vol成交量(手) amount成交额(千元)
        """
        if ticker:
            ticker = ticker+'.SH' if ticker[0]=='6' else ticker+'.SZ'
        data = self.run_api('monthly', ts_code=ticker, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, fields=fields)
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        return data

    def option_basic_get(self, exchange='SSE', fields=None):
        """
        期权合约信息
        :param exchange: str
            交易所 SSE上交所 SZSE深交所
        :param fields: str
            ts_codeTS代码 name合约名称 per_unit合约单位 opt_code标准合约代码 opt_type合约类型 call_put 期权类型
            exercise_price行权价 s_month结算价格 maturity_date到期日期 list_price挂牌基准价 list_date开始交易日期
            delist_date最后交割日期 quote_unit报价单位
        :return: DataFrame ts_code, name, opt_code, ...
        """
        data = self.run_api('opt_basic', exchange=exchange)
        if fields:
            data = data[fields.strip().split(',')]
        return data

    def option_daily_price_get(self, trade_date=None, ticker=None, start_date=None, end_date=None,
                               fields=None, exchange='SSE'):
        """
        期权日行情数据
        :param trade_date: str
            交易日期 YYYYMMDD
        :param ticker: str
            证券代码 深交所300ETF期权以9开头；上交所期权以1开头
        :param start_date: str
            起始日期
        :param end_date: str
            结束日期
        :param exchange: str
            交易所
        :param fields: str
            字段名称 ts_code合约代码 trade_date交易日期 close当日收盘价 open开盘价
                    high最高价 low最低价 pre_close前收盘价 pre_settle昨结算价
                    settle结算价 vol成交量(手) amount成交金额(万元) oi持仓量(手)
        :return: DataFrame ts_code, trade_date, ...
        """
        if ticker and ticker.find('.')<0:
            if ticker[0] == '1':
                ticker = ticker+'.SH'
            elif ticker[0] == '9':
                ticker = ticker+'.SZ'
            elif ticker[:2] == 'IO':
                ticker = ticker + '.CFX'
        data = self.run_api('opt_daily', exchange=exchange, ts_code=ticker,
                            trade_date=trade_date, start_date=start_date,
                            end_date=end_date)
        if fields:
            data = data[fields.strip().split(',')]
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        return data

    def fund_basic_get(self, market=None, fields=None):
        """
        公募基金数据基本信息
        :param market: str
            交易市场 E:场内 O场外
        :param fields: str
            返回字段  ts_code基金代码 name基金简称 management管理人 fund_type投资类型
            found_date成立日期 due_date到期日期 list_date上市日期 issue_date发行日期
            delist_date退市日期 issue_amount发行份额 invest_type投资风格  type基金类型
            benchmark基准
        """
        data = self.run_api('fund_basic', market=market)
        if fields:
            data = data[fields.strip().split(',')]
        return data
    
    def fund_portfolio_get(self, ts_code, start_period=None, end_period=None, fields=None):
        """
        公募基金持仓信息

        Parameters
        ----------
        ts_code : str
            基金代码(带后缀)
        start_period: str
            起始报告期
        end_period: str
            终止报告期
        fields: str
            返回字段  ts_code基金代码 ann_date公告日期  end_date截止日期
            symbol股票代码  mkv持有股票市值(元) amount持有股票数量(股)
            stk_mkt_ratio占股票市值比  stk_float_ratio占流通股本比

        Returns
        -------
        DataFrame ts_code end_date ...

        """
        data = self.run_api('fund_portfolio', ts_code=ts_code)
        if fields:
            data = data[fields.strip().split(',')]
        if start_period:
            data = data[data['end_date'] >= start_period]
        if end_period:
            data = data[data['end_date'] <= end_period]
        return data

    def fund_nav_get(self, ts_code=None, end_date=None, market=None, fields=None):
        """
        公募基金净值数据

        Parameters:
        -----------
        ts_code: str
            基金代码(带后缀)
        end_date: str
            净值日期YYYYMMDD
        market: str
            交易市场 E:场内 O场外
        fields: str
            ts_code基金代码 end_date截止日期 unit_nav单位净值 accum_div累计分红
            net_asset资产净值 total_netasset合计资产净值 adj_nav复权单位净值
        returns
        -------
        DataFrame: ts_code end_date ...

        """
        data = self.run_api('fund_nav', ts_code = ts_code, end_date = end_date,
                            market = market)
        if fields is not None:
            data = data[fields.strip().split(',')]
        data['end_date'] = pd.to_datetime(data['end_date'], format='%Y%m%d')
        return data
    
    def income_sheet(self, ticker=None, start_period=None, end_period=None, period=None,
                     report_type=None, fields=None):
        """
        A股利润表

        Parameters:
        -----------
        ticker: str
            股票代码
        start_period: str
            起始报告期
        end_period: str
            结束报告期
        period: str
            报告期
        report_type: str
            报告类型 
            1合并报表 2单季合并 3调整单季合并表 4调整合并报表 5调整前合并报表 
            6母公司报表 7母公司单季表 8母公司调整单季表 9母公司调整表 10母公司调整前报表 
            11调整前合并报表 12母公司调整前报表
        fields: str
            返回字段，字段太多，参照：https://tushare.pro/document/2?doc_id=33
        """
        if ticker:
            ticker = ticker + '.SH' if ticker[0] == '6' else ticker + '.SZ'
        if start_period is not None and end_period is not None:
            periods = pd.date_range(start_period, end_period, freq='1Q').strftime("%Y%m%d")
        else:
            periods = [period]
        
        df = []
        for p in periods:
            data = self.run_api('vip_income', ts_code=ticker, period=p, report_type=report_type,
                                fields=fields)
            df.append(df)
        data = pd.concat(df)
        return data

tushare_api = TushareDB()

