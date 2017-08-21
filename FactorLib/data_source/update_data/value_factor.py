"""更新价值类因子"""
import pandas as pd
import numpy as np


# 市盈率ttm
def pe_ttm(start, end, **kwargs):
    mkt_value = kwargs['data_source'].load_factor('total_mkt_value', '/stocks/', start_date=start,
                                                  end_date=end)
    net_profit = kwargs['data_source'].get_latest_report(
        'netprofit_ttm2', start_date=start, end_date=end)
    pe_ttm = mkt_value['total_mkt_value'] * 10000 / net_profit['netprofit_ttm2']
    pe_ttm = pe_ttm.to_frame().rename(columns={0: 'pe_ttm'})
    kwargs['data_source'].h5DB.save_factor(pe_ttm, '/stock_value/')


# 市盈率
def pe(start, end, **kwargs):
    mkt_value = kwargs['data_source'].load_factor('total_mkt_value', '/stocks/', start_date=start,
                                                  end_date=end)
    net_profit = kwargs['data_source'].get_latest_report(
        'np_belongto_parcomsh', start_date=start, end_date=end, report_type='4Q')
    pe = mkt_value['total_mkt_value'] * 10000 / net_profit['np_belongto_parcomsh']
    pe = pe.to_frame().rename(columns={0: 'pe'})
    kwargs['data_source'].h5DB.save_factor(pe, '/stock_value/')


# 市净率
def pb(start, end, **kwargs):
    mkt_value = kwargs['data_source'].load_factor('total_mkt_value', '/stocks/', start_date=start,
                                                  end_date=end)
    book_value = kwargs['data_source'].get_latest_report(
        'eqy_belongto_parcomsh', start_date=start, end_date=end)
    pb = mkt_value['total_mkt_value'] * 10000 / book_value['eqy_belongto_parcomsh']
    pb = pb.to_frame().rename(columns={0: 'pb'})
    kwargs['data_source'].h5DB.save_factor(pb, '/stock_value/')


# BP(市净率的倒数)
def bp(start, end, **kwargs):
    data_source = kwargs['data_source']
    pb = data_source.load_factor('pb', '/stock_value/', start_date=start, end_date=end)
    bp = 1 / pb
    bp.columns = ['bp']
    data_source.h5DB.save_factor(bp, '/stock_value/')


# EP(市盈率的倒数)
def ep(start, end, **kwargs):
    data_source = kwargs['data_source']
    pe = data_source.load_factor('pe', '/stock_value/', start_date=start, end_date=end)
    ep = 1 / pe
    ep.columns = ['ep']
    data_source.h5DB.save_factor(ep, '/stock_value/')


def bp_divide_median(start, end, **kwargs):
    data_source = kwargs['data_source']
    dates = data_source.trade_calendar.get_trade_days(start, end)
    bp = data_source.load_factor('bp', '/stock_value/', dates=dates)
    ids = bp.index.get_level_values(1).unique().tolist()
    industry = data_source.sector.get_stock_industry_info(ids, dates=dates)

    bp = pd.concat([bp, industry], axis=1).reset_index(level=1).set_index('cs_level_1', append=True)
    bp = bp[pd.notnull(bp.index.get_level_values(1))]
    bp_median = bp.groupby(['date', 'cs_level_1'])['bp'].median()
    bp = pd.merge(bp,bp_median.to_frame(),left_index=True,right_index=True,how='left')
    bp['bp_divide_median'] = bp['bp_x'] / bp['bp_y']
    bp = bp.reset_index(level=1, drop=True).set_index('IDs', append=True)

    data_source.h5DB.save_factor(bp[['bp_divide_median']], '/stock_value/')


def ep_divide_median(start, end, **kwargs):
    data_source = kwargs['data_source']
    dates = data_source.trade_calendar.get_trade_days(start, end)
    ep = data_source.load_factor('ep', '/stock_value/', dates=dates)
    ids = ep.index.get_level_values(1).unique().tolist()
    industry = data_source.sector.get_stock_industry_info(ids, dates=dates)

    ep = pd.concat([ep, industry], axis=1).reset_index(level=1).set_index('cs_level_1', append=True)
    ep = ep[pd.notnull(ep.index.get_level_values(1))]
    ep_median = ep.groupby(['date', 'cs_level_1'])['ep'].median()
    ep = pd.merge(ep,ep_median.to_frame(),left_index=True,right_index=True,how='left')
    ep['ep_divide_median'] = ep['ep_x'] / ep['ep_y']
    ep = ep.reset_index(level=1, drop=True).set_index('IDs', append=True)

    data_source.h5DB.save_factor(ep[['ep_divide_median']], '/stock_value/')


ValueFuncListDaily = [pe_ttm, pe, pb, bp, ep, bp_divide_median, ep_divide_median]
ValueFuncListMonthly = []


if __name__ == '__main__':
    from data_source import data_source
    bp_divide_median('20170701', '20170731', data_source=data_source)
    ep_divide_median('20170701', '20170731', data_source=data_source)