"""股票池分类"""
import pandas as pd
from FactorLib.data_source.base_data_source_h5 import data_source


def u_100001(start, end, **kwargs):
    """
    1. 连续两年扣非净利润不少于一千万
    2. 上市时间超过两年
    """
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end, '1d')
    list_days = data_source.load_factor('list_days', '/stocks/', dates=dates)
    np_ly0 = data_source.load_factor('net_profit_deduct_nonprofit', '/stock_profit/', dates=dates)
    np_ly1 = data_source.load_factor('net_profit_deduct_nonprofit_ly1', '/stock_profit/', dates=dates)
    data = list_days.join([np_ly0, np_ly1])
    new = data.eval("list_days>0 & net_profit_deduct_nonprofit>50000000 & net_profit_deduct_nonprofit_ly1>50000000")
    datasource.h5DB.save_factor(new.to_frame('_100001').astype('int'), '/indexes/')


UniverseFuncListMonthly = []
UniverseFuncListDaily = [u_100001]


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source

    # diversify_finance('20050104', '20171204', data_source=data_source)
    # excld_broker_banks('20050104', '20171204', data_source=data_source)
    # rescale_weight_afterdrop_brokers_and_banks('20050104', '20171204', data_source=data_source)
    u_100001('20180101', '20180117', data_source=data_source)
