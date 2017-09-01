import pandas as pd
import numpy as np
from functools import lru_cache
from WindPy import *

from FactorLib.utils.tool_funcs import get_industry_code, ReportDateAvailable, windcode_to_tradecode
from FactorLib.utils.datetime_func import DateStr2Datetime
from FactorLib.data_source.base_data_source_h5 import h5, sec
from FactorLib.data_source.wind_plugin import get_history_bar, get_wsd, _load_wsd_data
from FactorLib.data_source.data_api import get_trade_days, trade_day_offset
from FactorLib.data_source.update_data import index_members, sector_members, index_weights, industry_classes

w.start()

@lru_cache()
def get_ashare(date):
    d = w.wset("sectorconstituent","date={date};sectorid=a001010100000000".format(date=date))
    return d.Data[1]

@lru_cache()
def index_weight(index_id, date):
    d = w.wset("indexconstituent","date=%s;windcode=%s"%(date, index_id))
    ids = [x[:6] for x in d.Data[1]]
    weight = d.Data[3]
    return ids, weight

def updateSectorConstituent(dates, windcode):
    """更新某一个指数在时间序列上的成分股"""
    l = []
    for date in dates:
        d = w.wset("sectorconstituent","date={date};windcode={windcode}".format(
            date=date, windcode=windcode))
        d = d.Data[1]
        d = pd.DataFrame(d, columns=['IDs'])
        d['_%s'%windcode[:6]] = 1
        d['IDs'] = d['IDs'].str[:6]
        d['date'] = DateStr2Datetime(date)
        l.append(d)
    d = pd.concat(l, ignore_index=True)
    d = d.set_index(['date','IDs']).sort_index()
    return d

def updateSectorConstituent2(dates, sectorid, column_mark):
    """更新某一个板块的成分股"""
    l = []
    for date in dates:
        d = w.wset("sectorconstituent","date={date};sectorid={sectorid}".format(
            date=date, sectorid=sectorid))
        d = d.Data[1]
        d = pd.DataFrame(d, columns=['IDs'])
        d[column_mark] = 1
        d['IDs'] = d['IDs'].str[:6]
        d['date'] = DateStr2Datetime(date)
        l.append(d)
    d = pd.concat(l, ignore_index=True)
    d = d.set_index(['date','IDs']).sort_index()
    return d

def get_stock_industryname(stocks, date, industryid, industrytype):
    data = w.wsd(stocks, industryid, date, date, "industryType=%s"%industrytype)
    idx = pd.MultiIndex.from_product([[DateStr2Datetime(date)], [x[:6] for x in stocks]])
    d = pd.Series(data.Data[0], index=idx)
    return d

def index_weight_panel(dates, index_id):
    months = (trade_day_offset(x, 0, '1m') for x in dates)
    l = []
    for i, m in enumerate(months):
        ids, weight = index_weight(index_id, m)
        idx = pd.MultiIndex.from_product([[DateStr2Datetime(dates[i])], ids], names=['date','IDs'])
        l.append(pd.Series(weight, index=idx))
    d = pd.concat(l).to_frame().rename(columns={0:'_%s_weight'%index_id[:6]})
    return d


def onlist(start, end):
    """股票的上市日期"""
    d = get_ashare(end)
    idx = pd.MultiIndex.from_product([[DateStr2Datetime("19000101")],[x[:6] for x in d]],
                                        names=['date', 'IDs'])
    data = w.wsd(d, "ipo_date", end, end, "")
    list_date = [x.strftime("%Y%m%d") for x in data.Data[0]]
    list_date = pd.DataFrame(list_date, index=idx, columns=['list_date'])
    data = w.wsd(d, "backdoordate", end, end, "")
    backdoordate = [x.strftime("%Y%m%d") if x is not None else np.nan for x in data.Data[0]]
    backdoordate = pd.DataFrame(backdoordate, index=idx, columns=['backdoordate'])
    backdoordate.fillna('21000101', inplace=True)
    h5.save_factor(list_date, '/stocks/')
    h5.save_factor(backdoordate, '/stocks/')


def update_price(start, end):
    """更新价量行情数据"""
    # 股票价量数据
    field_names = "收盘价 涨跌幅 最高价 最低价 成交量"
    data = get_history_bar(field_names.split(),start,end,**{'复权方式':'不复权'})
    data.columns = ['close','daily_returns_%','high','low','volume']
    data['volume'] = data['volume'] / 100
    data['daily_returns'] = data['daily_returns_%'] / 100
    h5.save_factor(data,'/stocks/')

    field_names = "总市值 A股市值(不含限售股)"
    data = get_history_bar(field_names.split(),start,end)
    data.columns = ['total_mkt_value', 'float_mkt_value']
    data = data / 10000
    h5.save_factor(data,'/stocks/')

    # 股票后复权收盘价
    field_names = "收盘价"
    data = get_history_bar(field_names.split(),start,end,**{'复权方式':'后复权'})
    data.columns = ['adj_close']
    h5.save_factor(data,'/stocks/')

    field_names = "换手率 换手率(基准.自由流通股本)"
    data = get_history_bar(field_names.split(),start,end)
    data.columns = ['turn','freeturn']
    h5.save_factor(data,'/stock_liquidity/')

    # 指数价量数据
    field_names = "开盘价 最高价 最低价 收盘价 成交量 成交额 涨跌幅"
    data = get_history_bar(field_names.split(),start,end,id_type='index')
    data.columns = ['open','high','low','close','vol','amt', 'daily_returns_%']
    data['amt'] = data['amt'] / 10000
    data['vol'] = data['vol'] / 100
    h5.save_factor(data,'/indexprices/')


def update_stock_constest(start, end):
    field_names = "west_netprofit_FY1 west_netprofit_FY2 west_netprofit_FY3"
    data = get_wsd(field_names.split(), start, end) / 10000
    data.columns = ['netprofit_fy0', 'netprofit_fy1', 'netprofit_fy2']
    h5.save_factor(data, '/stock_est/')

    field_names = "west_eps_FY1 west_eps_FY2 west_eps_FY3"
    data = get_wsd(field_names.split(), start, end)
    data.columns = ['eps_fy0', 'eps_fy1', 'eps_fy2']
    h5.save_factor(data, '/stock_est/')


def update_sector(start, end):
    """更新成分股信息"""

    all_dates = get_trade_days(start, end)
    for index_id in index_members:
        d = updateSectorConstituent(all_dates, index_id)
        h5.save_factor(d, '/indexes/')

    for column_mark, sectorid in sector_members.items():
        d = updateSectorConstituent2(all_dates, sectorid, column_mark)
        if column_mark == 'ashare':
            h5.save_factor(d, '/indexes/')
        else:
            h5.save_factor(d, '/stocks/')


def update_idx_weight(start, end):
    """更新指数权重"""
    all_dates = get_trade_days(start, end)
    for index_id in index_weights:
        d = index_weight_panel(all_dates, index_id) / 100
        h5.save_factor(d, '/indexes/')


def update_industry_name(start, end):
    all_dates = get_trade_days(start, end)
    for column, indutryparams in industry_classes.items():
        l = []
        for idate in all_dates:
            ids = get_ashare(idate)
            l.append(get_stock_industryname(ids, idate, *indutryparams))
        industry = pd.concat(l).to_frame().rename(columns={0:column})
        industry = get_industry_code(column, industry)
        h5.save_factor(industry, '/indexes/')


def update_trade_status(start, end):
    dates = get_trade_days(start, end)

    st = sec.get_st(dates)
    suspend = sec.get_suspend(dates)
    uplimit = sec.get_uplimit(dates)
    downlimit = sec.get_downlimit(dates)

    trade_status = pd.concat([st,suspend,uplimit,downlimit], axis=1)
    trade_status = trade_status.where(pd.isnull(trade_status), other=1)
    trade_status.fillna(0, inplace=True)
    trade_status.columns = ['st', 'suspend', 'uplimit', 'downlimit']
    trade_status['no_trading'] = trade_status.any(axis=1).astype('int32')
    h5.save_factor(trade_status, '/trade_status/')


def update_industry_index_prices(start, end):
    from ...const import CS_INDUSTRY_CODES
    fields = ['open', 'high', 'low', 'close', 'pct_chg', 'volume']
    data = _load_wsd_data(CS_INDUSTRY_CODES, fields, start, end).astype('float32')
    data['volume'] /= 100
    h5.save_factor(data, '/indexprices/cs_level_1/')


if __name__ == '__main__':
    update_stock_constest('20170520', '20170827')