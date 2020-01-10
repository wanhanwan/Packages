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

    def __init__(self):
        self._api = ts
        self._pro_api = pro_api
        self._token = _token
        TushareDB._instance = self

    @classmethod
    def get_instance(cls):
        if TushareDB is not None:
            return TushareDB._instance
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
                                 freq=freq, adj=adj, ma=ma, asset=asset)
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
    
    def stock_daily_price_get(self, ticker=None, trade_dt=None, start_date=None, end_date=None,
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


tushare_api = TushareDB()
