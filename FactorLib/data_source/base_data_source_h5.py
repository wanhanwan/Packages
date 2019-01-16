# coding: utf-8
import numpy as np
import pandas as pd
from PkgConstVars import *

from ..const import MARKET_INDEX_DICT, SW_INDUSTRY_DICT, USER_INDEX_DICT
from ..utils.datetime_func import DateStr2Datetime
from ..utils.tool_funcs import parse_industry
from .converter import IndustryConverter
from .csv_db import CsvDB
from .h5db import H5DB
from .ncdb import NCDB
from .pkldb import PickleDB
from .trade_calendar import tc
from .tseries import resample_func, resample_returns


class base_data_source(object):
    def __init__(self, sector):
        self.h5DB = sector.h5DB
        self.trade_calendar = sector.trade_calendar
        self.sector = sector
        self.hdf5DB = sector.hdf5DB
        self.ncDB = sector.ncDB

    def load_factor(self, symbol, factor_path, ids=None, dates=None, start_date=None, end_date=None, idx=None):
        if idx is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date)
        return self.h5DB.load_factor(symbol, factor_path, ids=ids, dates=dates, idx=idx)

    def get_history_price(self, ids, dates=None, start_date=None, end_date=None,
                          freq='1d', type='stock', adjust=False):
        """ 提取股票收盘价
        adjust:是否复权
        """
        if type == 'stock':
            database = '/stocks/'
            symbol = 'adj_close' if adjust else 'close'
        else:
            database = '/indexprices/'
            symbol = 'close'
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date, freq, retstr=None)
        else:
            dates1 = self.trade_calendar.get_trade_days(start_date, end_date, freq, retstr=None)
            dates = pd.DatetimeIndex(dates).intersection(dates1)
        daily_price = self.h5DB.load_factor(symbol, database, dates=dates, ids=ids)
        return daily_price

    def get_period_return(self, ids, start_date, end_date, type='stock', incl_start=False):
        """
        计算证券的区间收益

        若incl_start=False
            区间收益 = 开始日收盘价 / 终止日收盘价 - 1, 即不包含起始日的收益
        反之，
            区间收益 = 开始日前收盘价 / 终止日收盘价 - 1, 即不包含起始日的收益

        返回: Series(index=IDs)
        """
        if incl_start:
            start_date = self.trade_calendar.tradeDayOffset(start_date, -1)
        prices = self.get_history_price(ids, dates=[start_date, end_date],
                                        type=type, adjust=True)
        prices = prices.swaplevel().sort_index().unstack()
        _cum_return_fun = lambda x: x.iloc[1] / x.iloc[0] - 1
        period_return = prices.apply(_cum_return_fun, axis=1)
        period_return.name = 'returns'
        return period_return.reindex(ids)

    def get_fix_period_return(self, ids, freq, start_date, end_date, type='stock'):
        """
        对证券的日收益率序列进行resample, 在开始日与结束日之间计算固定频率的收益率

        运算逻辑：
            提取[start_date, end_date]的所有收益率序列

            应用resample_return函数进行重采样

        """
        data_src = '/stocks/' if type == 'stock' else '/indexprices/'
        ret = self.load_factor('daily_returns_%', data_src, start_date=start_date, end_date=end_date, ids=ids) / 100
        return resample_returns(ret, convert_to=freq)

    def get_past_ndays_return(self, ids, window, start_date, end_date, type='stock'):
        """计算证券在过去N天的累计收益"""
        data_init_date = self.trade_calendar.tradeDayOffset(start_date, -window)
        price = self.get_history_price(ids, start_date=data_init_date, end_date=end_date,
                                       adjust=True, type=type).unstack().sort_index()
        cum_returns = (price / price.shift(window) - 1).stack()
        cum_returns.columns = ['return_%dd'%window]
        return cum_returns.loc[DateStr2Datetime(start_date):DateStr2Datetime(end_date)]

    def get_forward_ndays_return(self, ids, windows, freq='1d',
                                 dates=None, type='stock', idx=None):
        """计算证券未来N天的收益率"""
        from alphalens.utils import compute_forward_returns
        if idx is not None:
            ids = idx.index.get_level_values('IDs').unique().tolist()
            dates = idx.index.get_level_values('date').unique()
        max_date = self.trade_calendar.tradeDayOffset(max(dates), max(windows)+1, freq=freq, retstr=None)
        price = self.get_history_price(ids, start_date=min(dates), end_date=max_date,
                                       adjust=True, type=type).iloc[:, 0].unstack()
        ret = compute_forward_returns(price, periods=tuple(windows)).loc[dates]
        ret.index.names = ['date', 'IDs']
        if idx is not None:
            ret = ret.reindex(idx.index)
        return ret

    def get_periods_return(self, ids, dates, type='stock'):
        """
        计算dates序列中前后两个日期之间的收益率(start_date,end_date]

        :param ids: 股票ID序列

        :param dates: 日期序列

        :return:  stock_returns

        """
        def _cal_return(data):
            data_shift = data.shift(1)
            return data / data_shift - 1
        dates.sort()
        close_prices = self.get_history_price(
            ids, dates=dates, type=type, adjust=True).sort_index()
        # ret = close_prices.groupby('IDs', group_keys=False).apply(_cal_return
        close_flat = close_prices.iloc[:, 0].unstack()
        ret = close_flat.pct_change().stack().to_frame('returns')
        ret.index.names = ['date', 'IDs']
        # ret.columns = ['returns']
        return ret

    def get_history_bar(self, ids, start, end, adjust=False, type='stock', freq='1d'):
        """
         历史K线
        :param ids: stock ids

        :param start: start date

        :param end: end date

        :param adjust: 是否复权

        :param type: stock or index

        :param freq: frequency

        :return: high open low close avgprice volume amount turnover pctchange

        """
        from empyrical.stats import cum_returns_final
        agg_func_mapping = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vol': 'sum',
            'turn': 'sum',
            'amt': 'sum',
            'daily_returns_%': cum_returns_final
        }

        data_begin_date = self.trade_calendar.tradeDayOffset(start, -1, freq=freq)
        daily_dates = self.trade_calendar.get_trade_days(data_begin_date, end)
        if type == 'stock':
            data_dict = {'/stocks/': ['high', 'low', 'close', 'volume', 'daily_returns_%'],
                         '/stock_liquidity/': ['turn']}
        else:
            data_dict = {'/indexprices/': ['open', 'high', 'low', 'close', 'amt', 'vol', 'daily_returns_%']}
        data = self.h5DB.load_factors(data_dict, ids=ids, dates=daily_dates)
        data['daily_returns_%'] = data['daily_returns_%'] / 100
        bar = resample_func(data, convert_to=freq, func={x: agg_func_mapping[x] for x in data.columns})
        # bar.index.names = ['date', 'IDs']
        return bar

    def get_stock_trade_status(self, ids=None, dates=None, start_date=None, end_date=None, freq='1d'):
        """获得股票交易状态信息,包括停牌、ST、涨跌停"""
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date=start_date,end_date=end_date,freq=freq)
        elif not isinstance(dates, list):
            dates = [dates]
        if (start_date not in dates) and (start_date is not None):
            dates.append(start_date)
        if (end_date not in dates) and (end_date is not None):
            dates.append(end_date)
        dates.sort()

        trade_status = self.load_factor('no_trading', '/trade_status/', dates=dates)
        if not ids:
            return trade_status
        else:
            if not isinstance(ids, list):
                ids = [ids]
            idx = pd.MultiIndex.from_product([dates, ids])
            idx.names = ['date', 'IDs']
            return trade_status.reindex(idx, fill_value=0)

    def get_go_market_days(self, dates=None, start_date=None, end_date=None, ids=None, unit='d'):
        divider = {'d': 1, 'm': 30, 'y': 365}
        list_days = self.load_factor('list_days', '/stocks/', dates=dates,
                                     start_date=start_date, end_date=end_date, ids=ids)
        list_days /= divider[unit]
        dates = list_days.index.unique(level='date')
        all_stocks = self.sector.get_history_ashare(dates=dates, history=False)
        return list_days.reindex(all_stocks.index).dropna()


class sector(object):
    def __init__(self, h5, trade_calendar, hdf5=None, nc=None):
        self.h5DB = h5
        self.trade_calendar = trade_calendar
        self.hdf5DB = hdf5
        self.ncDB = nc

    def get_st(self, dates=None, start_date=None, end_date=None):
        """某一个时间段内st的股票"""
        dates = self.trade_calendar.get_trade_days(start_date,end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        st_list = self.h5DB.load_factor('is_st', '/stocks/', dates=dates)
        st_list = st_list.query('is_st == 1')
        return st_list['is_st']

    def get_st_details(self, stock_type, dates=None, start_date=None, end_date=None):
        """某个时间段内特别处理股票
        股票类型包括: 特别处理、暂停上市、退市、退市整理、特别转让服务、创业板暂停上市风险警示
        """
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date)
        from FactorLib.data_source.wind_financial_data_api import asharest
        stocks = asharest.get_type_stocks(dates, stock_type)
        return stocks

    def get_suspend(self, dates=None, start_date=None, end_date=None):
        """某一时间段停牌的股票"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        suspend = self.h5DB.load_factor('volume', '/stocks/', dates=dates)
        suspend = suspend.query('volume == 0')
        suspend_list = sec.get_st_details('暂停上市', dates=dates)
        new_idx = suspend.index.union(suspend_list.index)
        suspend = pd.DataFrame([0]*len(new_idx), index=new_idx, columns=['volume'])
        return suspend['volume']

    def get_uplimit(self, dates=None, start_date=None, end_date=None):
        """某一时间段内涨停的股票"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        factors = {'/stocks/': ['high', 'low', 'daily_returns_%']}
        uplimit = self.h5DB.load_factors(factors, dates=dates)
        uplimit = uplimit[(uplimit['high'] == uplimit['low']) & (uplimit['daily_returns_%'] > 9.5)]
        return uplimit['high']

    def get_downlimit(self, dates=None, start_date=None, end_date=None):
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        factors = {'/stocks/': ['daily_returns_%', 'high', 'low']}
        downlimit = self.h5DB.load_factors(factors, dates=dates)
        downlimit = downlimit[(downlimit['high']==downlimit['low'])&(downlimit['daily_returns_%']<-9.5)]
        return downlimit['high']

    def get_index_members(self, ids, dates=None, start_date=None, end_date=None):
        """某一个时间段内指数成分股,可以是市场指数也可以是行业指数
        目前,市场指数包括:
             万得全A(880011)、上证50(000016)、中证500(000905)、中证800(000906)、创业板综(399102)和
        沪深300(000300)。
        行业指数包括:中信一级行业和申万一级行业

        ids是指数成分代码而非指数名称
        """
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        all_stocks = self.get_history_ashare(dates)

        if isinstance(dates, str):
            dates = [dates]
        if ids == '全A':
            return all_stocks
        if ids in MARKET_INDEX_DICT:
            index_members = self.h5DB.load_factor('_%s' % ids, '/indexes/', dates=dates)
            index_members = index_members[index_members['_%s' % ids] == 1.0]
            return index_members[index_members.index.isin(all_stocks.index)]
        if ids in USER_INDEX_DICT:
            try:
                index_members = self.h5DB.load_factor('%s' % ids, '/indexes/', dates=dates)
            except FileNotFoundError:
                index_members = self.h5DB.load_factor('_%s' % ids, '/indexes/', dates=dates)
            index_members = index_members[index_members['%s' % ids] == 1.0]
            return index_members[index_members.index.isin(all_stocks.index)]
        for industry_name, rule in IndustryConverter._rules.items():
            if ids in IndustryConverter.all_ids(industry_name):
                temp = self.h5DB.load_factor(industry_name, '/indexes/', dates=dates)
                index_members = temp[temp[industry_name] == rule.name2id_func(ids)]
                return index_members[index_members.index.isin(all_stocks.index)]
            else:
                continue
        try:
            index_members = self.h5DB.load_factor('%s' % ids, '/indexes/', dates=dates)
            return index_members[index_members.index.isin(all_stocks.index)]
        except:
            raise KeyError("找不到指数ID对应成分股！")

    def get_stock_industry_info(self, ids, industry='中信一级', start_date=None, end_date=None, dates=None, idx=None):
        """股票行业信息"""
        try:
            symbol = parse_industry(industry)
        except KeyError:
            symbol = industry
        if idx is not None:
            industry_info = self.h5DB.load_factor(symbol, '/indexes/', idx=idx)
        else:
            dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
            if not isinstance(dates, list):
                dates = [dates]
            industry_info = self.h5DB.load_factor(symbol, '/indexes/', ids=ids, dates=dates)
        return IndustryConverter.convert(symbol, industry_info[symbol]).to_frame()

    def get_industry_dummy(self, ids=None, industry='中信一级', start_date=None, end_date=None, dates=None,
                           idx=None, drop_first=True):
        """股票行业哑变量
        Parameters
        ------------------
        ids : None or list of strings
            股票代码列表, 默认为None, 代表所有股票。只返回上述股票的行业哑变量
        industry : str
            中文行业名称， 详情查看const.py文件中的INDUSTRY_NAME_DICT
        
        """
        if idx is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        else:
            dates = idx.index.get_level_values("date").unique()

        try:
            industry_id = parse_industry(industry)
        except KeyError:
            industry_id = industry

        dummy = self.h5DB.load_as_dummy(industry_id, '/dummy/', ids=ids, dates=dates, idx=idx)
        if idx is not None:
            dummy = dummy.reindex(idx.index, fill_value=0)
        dummy = dummy.loc[:, (dummy != 0).any()]
        dummy = dummy[(dummy == 1).any(axis=1)]
        if drop_first:
            return dummy.iloc[:, 1:]
        return dummy

    def get_index_weight(self, ids, start_date=None, end_date=None, dates=None):
        """获取指数个股权重"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        symbol = '_{id}_weight'.format(id=ids)
        weight = self.h5DB.load_factor(symbol, '/indexes/', dates=dates)
        # weight = weight.unstack().reindex(pd.DatetimeIndex(dates), method='ffill').stack()
        weight.index.names = ['date', 'IDs']
        return weight

    def get_index_industry_weight(self, ids, industry_name='中信一级', start_date=None,
                                  end_date=None, dates=None):
        """获取指数的行业权重"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        index_weight = self.get_index_weight(ids, dates=dates)
        industry_data = self.get_stock_industry_info(None, industry=industry_name, idx=index_weight)
        industry_weight = index_weight.groupby(['date', industry_data.iloc[:, 0]]).sum()
        # index_industry_weight = common.reset_index().groupby(['date', symbol])[index_weight.columns[0]].sum()
        return industry_weight.iloc[:, 0]

    def get_history_ashare(self, dates, history=False):
        """获得某一天的所有上市A股"""
        if isinstance(dates, str):
            dates = [dates]
        stocks = self.h5DB.load_factor('ashare', '/indexes/', dates=dates)
        if history:
            stocks = stocks.unstack().expanding().max().stack()
        return stocks

    def get_ashare_onlist(self, dates, months_filter=24):
        """获得某一天已经上市的公司，并且上市日期不少于24个月"""
        ashare = self.h5DB.load_factor('list_days', '/stocks/', dates=dates)
        ashare = ashare[ashare.list_days > months_filter*30]
        return pd.DataFrame(np.ones(len(ashare)), index=ashare.index, columns=['ashare'])

    def get_stock_info(self, ids, date):
        """获得上市公司的信息。
        信息包括公司代码、公司简称、上市日期、所属行业(中信一级,wind一级)"""
        if not isinstance(date, list):
            date = [date]
        # if (not isinstance(ids, list)) and (ids is not None):
        #     ids = [ids]
        factors = {'/stocks/': ['name', 'list_date']}
        stock_name_listdate = self.h5DB.load_factors(factors, ids=ids)
        stock_name_listdate = stock_name_listdate.reset_index(level=0, drop=True)

        stock_members = self.get_stock_industry_info(ids, dates=date)
        stocks_members_wind = self.get_stock_industry_info(
            ids, dates=date, industry='中信二级')
        members = pd.concat([stock_members, stocks_members_wind], axis=1)
        members = members.reindex(pd.MultiIndex.from_product([pd.DatetimeIndex(date), ids],
                                                             names=['date', 'IDs']))

        stock_info = pd.merge(members,
                              stock_name_listdate,
                              left_index=True,
                              right_index=True)
        return stock_info

    def get_latest_unst(self, dates, months=6):
        """获得最近摘帽的公司"""
        idx = pd.DatetimeIndex(dates, name='date')
        unst = (self.h5DB.load_factor('unst', '/stocks/').reset_index().
                drop('unst', axis=1).assign(unst_date=lambda x: x.date).set_index(['date', 'IDs']).
                unstack().reindex(idx, method='ffill').stack().reset_index())
        latest_unst = unst[(unst['date']-unst['unst_date'])/pd.to_timedelta(1, unit='M') <= months]
        latest_unst['unst'] = 1
        return latest_unst.set_index(['date', 'IDs']).drop('unst_date', axis=1)


h5 = H5DB(H5_PATH)
ncdb = NCDB(NC_PATH)
riskDB = H5DB(RISKMODEL_PATH)
pkldb = PickleDB(PICKLE_PATH)
sec = sector(h5, tc, nc=ncdb)
data_source = base_data_source(sec)
csv = CsvDB()
