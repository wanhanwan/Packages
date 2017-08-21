import os
from datetime import timedelta, datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from const import SW_INDUSTRY_DICT, MARKET_INDEX_DICT
from data_source.h5db import H5DB
from data_source.trade_calendar import tc
from data_source.tseries import resample_func, resample_returns
from utils.datetime_func import DateStr2Datetime
from utils.tool_funcs import parse_industry, get_industry_names, financial_data_reindex, windcode_to_tradecode


class base_data_source(object):
    def __init__(self, sector):
        self.h5DB = sector.h5DB
        self.trade_calendar = sector.trade_calendar
        self.sector = sector
        self.dividend = None
        # self._load_dividends()
    
    def _load_dividends(self):
        def _save_convert(x):
            if isinstance(x, str):
                return datetime.strptime(x, "%Y-%m-%d")
            else:
                return x

        def _earliest_date(x):
            if 0 in x.values:
                try:
                    return _save_convert(x[x != 0].iloc[0])
                except IndexError:
                    return np.nan
            else:
                return x.apply(_save_convert).min()

        dividend_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dividend_csv_path = os.path.join(dividend_csv_path, 'resource/dividends.xlsx')
        xr = pd.ExcelFile(dividend_csv_path)
        l = []
        for sheet_name in xr.sheet_names:
            data = pd.read_excel(xr, sheetname=sheet_name, skiprows=3, header=1)[
                ['Wind代码', '分红预披露公告日', '预案公告日', '每股股利', '分红总额']]
            data['Wind代码'] = data['Wind代码'].apply(windcode_to_tradecode)
            data['period'] = sheet_name
            data['ann_dt'] = data[['分红预披露公告日', '预案公告日']].apply(_earliest_date, axis=1)
            data['ann_dt'] = data['ann_dt'].fillna(datetime.strptime(sheet_name+'0430', '%Y%m%d'))
            data = data.rename(columns={'Wind代码': 'IDs', '每股股利':'dividend_pershare', '分红总额':'total_dividend'})
            l.append(data)
        dividend_data = pd.concat(l)[['IDs','ann_dt','period','dividend_pershare','total_dividend']]
        self.dividend = dividend_data

    def load_factor(self, symbol, factor_path, ids=None, dates=None, start_date=None, end_date=None, idx=None):
        if idx is None:
            dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        return self.h5DB.load_factor(symbol, factor_path, ids=ids, dates=dates,idx=idx)

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
            dates1 = self.trade_calendar.get_latest_trade_days(start_date, end_date, freq, retstr=None)
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
        return resample_func(ret, convert_to=freq)

    def get_past_ndays_return(self, ids, window, start_date, end_date, type='stock'):
        """计算证券在过去N天的累计收益"""
        data_init_date = self.trade_calendar.tradeDayOffset(start_date, -window)
        price = self.get_history_price(ids, start_date=data_init_date, end_date=end_date,
                                       adjust=True, type=type).unstack().sort_index()
        cum_returns = (price / price.shift(window) - 1).stack()
        cum_returns.columns = ['return_%dd'%window]
        return cum_returns.loc[DateStr2Datetime(start_date):DateStr2Datetime(end_date)]

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
        close_prices = self.get_history_price(ids, dates=dates, type=type).sort_index()
        ret = close_prices.groupby('IDs', group_keys=False).apply(_cal_return)
        ret.index.names = ['date', 'IDs']
        ret.columns = ['returns']
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

    def get_latest_report(self, field, ids=None, dates=None, start_date=None,
                          end_date=None, freq='1d', report_type=None):
        """获得最近报告期的财务数据"""
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date=start_date,end_date=end_date,freq=freq)
        elif not isinstance(dates,list):
            dates = [dates]
        if (start_date not in dates) and (start_date is not None):
            dates.append(start_date)
        if (end_date not in dates) and (end_date is not None):
            dates.append(end_date)
        dates.sort()
        idx = self.get_latest_report_date(dates, report_type).loc[(slice(None), slice(ids)), :]
        raw_data = self.h5DB.load_factor(field, '/stock_financial_data/', ids=ids)
        return financial_data_reindex(raw_data, idx)
    
    def get_nlatest_reports(self, field, n, ids=None, dates=None, start_date=None,
                            end_date=None, freq='1d', report_type=None):
        """获得最近N期财务数据"""
        if dates is None:
            dates = self.trade_calendar.get_trade_days(start_date=start_date,end_date=end_date,freq=freq)
        elif not isinstance(dates,list):
            dates = [dates]
        if (start_date not in dates) and (start_date is not None):
            dates.append(start_date)
        if (end_date not in dates) and (end_date is not None):
            dates.append(end_date)
        dates.sort()
        idx = self.get_nlatest_report_dates(dates, n, report_type)
        raw_data = self.h5DB.load_factor(field, '/stock_financial_data/', ids=ids)
        return financial_data_reindex(raw_data, idx)
    
    def get_nlatest_report_dates(self, dates, n, report_type=None):
        """特定日期序列的前n个最大报告期"""
        ann_report_dates = self.get_report_ann_dt(report_type)
        _ = []
        if isinstance(dates, str):
            dates = [dates]
        for date in dates:
            temp = ann_report_dates[ann_report_dates['ann_dt']<=date]
            max_date = temp.groupby('IDs')['date'].nlargest(n).reset_index(level=1, drop=True)
            max_date.index = pd.MultiIndex.from_product([[date], max_date.index], names=['date', 'IDs'])
            _.append(max_date)
        return pd.concat(_).rename('max_report_date').to_frame()
    
    def get_latest_report_date(self, dates, report_type=None):
        """特定日期序列的最大报告期"""
        ann_report_dates = self.get_report_ann_dt(report_type)
        _ = []
        if isinstance(dates, str):
            dates = [dates]
        for date in dates:
            temp = ann_report_dates[ann_report_dates['ann_dt']<=date]
            max_date = temp.groupby('IDs')['date'].max()
            max_date.index = pd.MultiIndex.from_product([[date], max_date.index], names=['date', 'IDs'])
            _.append(max_date)
        return pd.concat(_).rename('max_report_date').to_frame()
    
    def get_report_ann_dt(self, report_type):
        ann_report_dates = self.h5DB.load_factor('ann_dt', '/stock_financial_data/').reset_index(level=0)
        if report_type is None:
            return ann_report_dates
        elif report_type == 'Q1':
            return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 3]
        elif report_type == 'Q2':
            return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 6]
        elif report_type == 'Q3':
            return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 9]
        else:
            return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 12]

    def to_dailyfinancial(self, factorname, type='realtime'):
        today = datetime.today().strftime("%Y%m%d")
        if type == 'realtime':
            data = self.get_latest_report(factorname, start_date='20070101', end_date=today)
            self.h5DB.save_factor(data[[factorname]], '/stock_financial_data/realtime_daily/')
        else:
            dates = pd.DatetimeIndex(self.trade_calendar.get_trade_days('20070101', today))

            @lru_cache()
            def _mapdate(date):
                if date.month == 3:
                    return pd.datetime(date.year, 4, 30)
                elif date.month == 6:
                    return pd.datetime(date.year, 8, 31)
                elif date.month == 9:
                    return pd.datetime(date.year, 10, 31)
                else:
                    return pd.datetime(date.year+1, 4, 30)
            data = self.h5DB.load_factor(factorname, '/stock_financial_data/').reset_index()
            data = data[pd.DatetimeIndex(data['date']).month != 12]
            data['realdate'] = data['date'].apply(_mapdate)
            data = data.set_index(['realdate', 'IDs'])[[factorname]].unstack().reindex(dates, method='ffill').stack()
            data.index.names = ['date', 'IDs']
            self.h5DB.save_factor(data, '/stock_financial_data/daily/')

    def get_dividends(self, ids, dates):
        """在指定截止日前最近年报的红利数据"""
        if self.dividend is None:
            self._load_dividends()
        dates = pd.DatetimeIndex(dates)
        l = []
        for date in dates:
            d = self.dividend[self.dividend['ann_dt'] <= date].sort_values('ann_dt').groupby('IDs').last()
            d['date'] = date
            d = d.set_index(['date'], append=True).swaplevel()
            l.append(d[['total_dividend']])
        if ids is None:
            return pd.concat(l)
        else:
            d = pd.concat(l)
        return d[d.index.get_level_values(1).isin(ids)]


class sector(object):
    def __init__(self, h5, trade_calendar):
        self.h5DB = h5
        self.trade_calendar = trade_calendar

    def get_st(self, dates=None, start_date=None, end_date=None):
        """某一个时间段内st的股票"""
        dates = self.trade_calendar.get_trade_days(start_date,end_date) if dates is None else dates
        if not isinstance(dates,list):
            dates = [dates]
        st_list = self.h5DB.load_factor('is_st', '/stocks/', dates=dates)
        st_list = st_list.query('is_st == 1')
        return st_list['is_st']

    def get_suspend(self, dates=None, start_date=None, end_date=None):
        """某一时间段停牌的股票"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        suspend = self.h5DB.load_factor('volume', '/stocks/', dates=dates)
        suspend = suspend.query('volume == 0')
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

    def get_downlimit(self,dates=None, start_date=None, end_date=None):
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates,list):
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
        """
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        all_stocks = self.get_history_ashare(dates)

        if isinstance(dates, str):
            dates = [dates]
        if ids == '全A':
            return all_stocks
        if ids in MARKET_INDEX_DICT:
            index_members = self.h5DB.load_factor('_%s' % ids, '/indexes/', dates=dates)
        elif ids in SW_INDUSTRY_DICT:
            temp = self.h5DB.load_factor('sw_level_1', '/indexes/', dates=dates)
            index_members = temp[temp['sw_level_1'] == int(ids)]
        else:
            temp = self.h5DB.load_factor('cs_level_1', '/indexes/', dates=dates)
            index_members = temp[temp['cs_level_1'] == int(ids[2:])]
        return  index_members[index_members.index.isin(all_stocks.index)]

    def get_stock_industry_info(self, ids, industry='中信一级', start_date=None, end_date=None, dates=None):
        """股票行业信息"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        if not isinstance(dates, list):
            dates = [dates]
        symbol = parse_industry(industry)
        industry_info = self.h5DB.load_factor(symbol, '/indexes/', ids=ids, dates=dates)
        return get_industry_names(symbol, industry_info)

    def get_index_weight(self, ids, start_date=None, end_date=None, dates=None):
        """获取指数个股权重"""
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        symbol = '_{id}_weight'.format(id=ids)

        weight = self.h5DB.load_factor(symbol, '/indexes/', dates=dates).sort_index()
        weight = weight.unstack().reindex(pd.DatetimeIndex(dates), method='ffill').stack()
        weight.index.names = ['date', 'IDs']
        return weight

    def get_index_industry_weight(self, ids, industry_name='中信一级', start_date=None,
                                  end_date=None, dates=None):
        """获取指数的行业权重"""
        symbol = parse_industry(industry_name)
        dates = self.trade_calendar.get_trade_days(start_date, end_date) if dates is None else dates
        index_weight = self.get_index_weight(ids, dates=dates)
        all_stocks = list(index_weight.index.levels[1].unique())
        industry_data = self.get_stock_industry_info(all_stocks, industry=industry_name, dates=dates)
        common = pd.concat([index_weight, industry_data], join='inner', axis=1)
        index_industry_weight = common.reset_index().groupby(['date', symbol])[index_weight.columns[0]].sum()
        return index_industry_weight

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
        ashare = self.get_history_ashare(dates).swaplevel()
        ashare_onlist_date = self.h5DB.load_factor(
            'list_date', '/stocks/').reset_index(level=0, drop=True)
        ashare_backdoordate = self.h5DB.load_factor(
            'backdoordate', '/stocks/').reset_index(level=0, drop=True)   # 借壳上市日期
        ashare_info = pd.merge(ashare, ashare_onlist_date, left_index=True, right_index=True, how='left')
        ashare_info = ashare_info.join(ashare_backdoordate).reset_index()
        ashare_info['backdoordate'] = ashare_info['backdoordate'].fillna('21000101')

        backdoordate = ashare_info['backdoordate'].apply(DateStr2Datetime)
        ashare_info['new_listdate'] = np.where(ashare_info['date']>=backdoordate, ashare_info['backdoordate'], ashare_info['list_date'])
        onlist_period = ashare_info['date'] - ashare_info['new_listdate'].apply(DateStr2Datetime)
        temp_ind = (onlist_period / timedelta(1)) > months_filter * 30
        return ashare_info.set_index(['date', 'IDs']).loc[temp_ind.values, ['list_date']].copy()

    def get_stock_info(self, ids, date):
        """获得上市公司的信息。
        信息包括公司代码、公司简称、上市日期、所属行业(中信一级)"""
        if not isinstance(date,list):
            date = [date]
        if (not isinstance(ids, list)) and (ids is not None):
            ids = [ids]
        factors = {'/stocks/': ['Name', 'list_date']}
        stock_name_listdate = self.h5DB.load_factors(factors, ids=ids)
        stock_name_listdate = stock_name_listdate.reset_index(level=0,drop=True)

        stock_members = self.get_stock_industry_info(ids, dates=date).swaplevel()
        stock_info = pd.merge(stock_members,stock_name_listdate,left_index=True, right_index=True,how='left')
        stock_info = stock_info.swaplevel()
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


h5 = H5DB("D:/data/h5")
sec = sector(h5, tc)
data_source = base_data_source(sec)

if __name__ == '__main__':
    data_source.get_history_bar(['000001','000002'], start ='20100101',end='20161231', freq='2w')



