import pandas as pd
import numpy as np


def egrlf(start, end, **kwargs):
    """
    BARRA中EGIB因子
    未来3年企业一致预期净利润增长率
    """
    data_source = kwargs['data_source']
    net_profit = data_source.get_latest_report(
        'np_belongto_parcomsh', start_date=start, end_date=end, report_type='4Q')[['np_belongto_parcomsh']]
    net_profit_fy2 = data_source.load_factor('netprofit_fy2', '/stock_est/', start_date=start, end_date=end)
    merge_data = pd.merge(net_profit_fy2, net_profit, left_index=True, right_index=True, how='left')

    not_na_ind = pd.notnull(merge_data).all(axis=1)
    temp = merge_data[not_na_ind]
    res = 1+(temp['netprofit_fy2']*10000-temp['np_belongto_parcomsh']) / \
                          temp['np_belongto_parcomsh'].abs()
    new_factor = res.abs() ** (1/3) - 1
    new_factor = (new_factor * np.sign(res)).to_frame('EGRLF')
    data_source.h5DB.save_factor(new_factor.reindex(merge_data.index), '/barra/descriptors/')


def egrsf(start, end, **kwargs):
    """
    BARRA中EGIB_S因子
    未来1年企业一致预期净利润增长率

    """
    data_source = kwargs['data_source']
    net_profit = data_source.get_latest_report(
        'np_belongto_parcomsh', start_date=start, end_date=end, report_type='4Q')[['np_belongto_parcomsh']]
    net_profit_fy0 = data_source.load_factor('netprofit_fy0', '/stock_est/', start_date=start, end_date=end)
    merge_data = pd.merge(net_profit_fy0, net_profit, left_index=True, right_index=True, how='left')
    not_na_ind = pd.notnull(merge_data).all(axis=1)
    temp = merge_data[not_na_ind]
    new_factor = (temp['netprofit_fy0']*10000 - temp['np_belongto_parcomsh']) / temp['np_belongto_parcomsh'].abs()
    new_factor = new_factor.to_frame('EGRSF')
    data_source.h5DB.save_factor(new_factor.reindex(merge_data.index), '/barra/descriptors/')


GrowthFuncListDaily = []



if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    egrlf('20160101', '20170831', data_source=data_source)
    egrsf('20160101', '20170831', data_source=data_source)