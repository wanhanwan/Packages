"""股票池分类"""
import pandas as pd
import numpy as np
from FactorLib.data_source.base_data_source_h5 import data_source, h5
from FactorLib.data_source.wind_financial_data_api.tool_funcs import get_go_market_days


def u_100001(start, end, **kwargs):
    """
    1. 连续两年扣非净利润不少于五千万
    2. 上市时间超过两年
    """
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end, '1d')
    list_days = get_go_market_days(dates)
    np_ly0 = data_source.load_factor('net_profit_deduct_nonprofit', '/stock_profit/', dates=dates)
    np_ly1 = data_source.load_factor('net_profit_deduct_nonprofit_ly1', '/stock_profit/', dates=dates)
    data = list_days.join([np_ly0, np_ly1])
    new = data.eval("list_days>0 & net_profit_deduct_nonprofit>50000000 & net_profit_deduct_nonprofit_ly1>50000000")
    datasource.h5DB.save_factor(new.to_frame('_100001').astype('int'), '/indexes/')


def u_100002(start, end, **kwargs):
    """
    最近一年出现资产重组的公司
    """
    from FactorLib.data_source.update_data.other_factors.merge_accquisition import merge_acc
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end)
    stocks = merge_acc(dates=dates)
    datasource.h5DB.save_factor(stocks, '/indexes/')


def u_100003(start, end, **kwargs):
    """3号股票池的筛选标准
        1. 上市满一年
        2. 最近一年之内没有发生资产重组
        3. 2014年之前市值小于30亿，2015年之后市值大于50亿
        """
    for date in data_source.trade_calendar.get_trade_days(start, end):
        date_dt = pd.to_datetime(date)
        # 上市年数
        list_days = get_go_market_days(date, unit='y') > 1.0
        # 市值
        if date_dt.year < 2014:
            mkt_value = h5.load_factor('float_mkt_value', '/stocks/', dates=[date]) > 3e5
        else:
            mkt_value = h5.load_factor('float_mkt_value', '/stocks/', dates=[date]) > 5e5
        # 资产重组
        merge_acc = h5.load_factor('merge_acc', '/indexes/', dates=[date]) == 1.0

        valid = list_days.iloc[:, 0] & mkt_value.iloc[:, 0]
        valid = valid[valid == 1]
        valid = valid[~valid.index.isin(merge_acc.index)].astype('int32')
        data_source.h5DB.save_factor(valid.to_frame('_100003'), '/indexes/')


def u_100004(start, end, **kwargs):
    """4号股票池筛选标准
    1. 2014年之前市值小于30亿,2014年之后市值小于50亿
    2. 处在风险预警池之中
    """
    dates = data_source.trade_calendar.get_trade_days(start, end)
    mkt_value = h5.load_factor('float_mkt_value', '/stocks/', dates=dates)
    stock_pool_1 = mkt_value[
        ((mkt_value.index.get_level_values('date').year<2014) & (mkt_value['float_mkt_value']<3e5)) |
    ((mkt_value.index.get_level_values('date').year>=2014) & (mkt_value['float_mkt_value']<5e5))]
    risky_stocks = data_source.sector.get_index_members('risky_stocks', dates=dates)
    new_idx = stock_pool_1.index.union(risky_stocks.index)
    new_pool = pd.DataFrame(np.ones(len(new_idx)), index=new_idx, columns=['_100004'])
    new_pool.index.names = ['date', 'IDs']
    data_source.h5DB.save_factor(new_pool, '/indexes/')


UniverseFuncListMonthly = []
UniverseFuncListDaily = [u_100002, u_100003]


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source

    # diversify_finance('20050104', '20171204', data_source=data_source)
    # excld_broker_banks('20050104', '20171204', data_source=data_source)
    # rescale_weight_afterdrop_brokers_and_banks('20050104', '20171204', data_source=data_source)
    u_100004('20180228', '20180528', data_source=data_source)
