"""其他因子"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from QuantLib.tools import df_rolling2
from FactorLib.const import INDUSTRY_NAME_DICT
from FactorLib.data_source.converter import IndustryConverter


# 特异度
def iffr(start, end, **kwargs):
    startdate = kwargs['data_source'].trade_calendar.tradeDayOffset(start, -30, incl_on_offset_today=True)
    all_dates = kwargs['data_source'].trade_calendar.get_trade_days(startdate, end)
    daily_ret = kwargs['data_source'].load_factor('daily_returns', '/stocks/', dates=all_dates)
    factors = {'/time_series_factors/': ['rf', 'mkt_rf', 'smb', 'hml']}
    three_factors = kwargs['data_source'].h5DB.load_factors(factors, dates=all_dates).reset_index(level=1, drop=True)
    
    daily_ret = daily_ret['daily_returns'].unstack()
    data = pd.concat([daily_ret, three_factors], axis=1)
    
    # 计算特质收益率因子

    r_square = df_rolling2(data, 20, _calRSquareApplyFunc)

    # 存储数据到数据库
    r_square = r_square.stack()
    r_square.index.names =['date','IDs']
    r_square = r_square.to_frame('iffr')
    kwargs['data_source'].h5DB.save_factor(r_square, '/stock_alternative/')


# %%构造一个线性回归的函数，计算拟合优度
def _calRSquare(y, x):
    '''线性回归计算拟合优度
    y: Series
    x: DataFrame,第一列为无风险收益率
    '''
    data_len = len(y)
    if pd.notnull(y).sum() / data_len < 0.7:
        return np.nan
    y = y - x[:, 0]
    x = x[:, 1:]
    ols = sm.OLS(y, x, missing='drop').fit()
    r_square = ols.rsquared
    return r_square


def _calRSquareApplyFunc(data_frame):
    r_square = np.apply_along_axis(_calRSquare, 0, data_frame[:, :-4], data_frame[:, -4:])
    return r_square


# 摘帽日期
def unst(start, end, **kwargs):
    start = kwargs['data_source'].trade_calendar.tradeDayOffset(start, -1, incl_on_offset_today=True)
    st = kwargs['data_source'].load_factor('is_st', '/stocks/', start_date=start, end_date=end)
    st_T = st.unstack()
    st_shift = st_T.shift(1)
    unst_stocks = ((st_T - st_shift) == -1).stack().astype('int').rename(columns={'is_st':'unst'})
    unst_stocks = unst_stocks[unst_stocks.unst==1]
    if not unst_stocks.empty:
        kwargs['data_source'].h5DB.save_factor(unst_stocks, '/stocks/')


# 把非银金融行业的股票细分成券商、保险、和其他
def diversify_finance(start, end, **kwargs):
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end)
    # 中信行业
    cs_level_1_id = datasource.h5DB.load_factor('cs_level_1', '/indexes/', dates=dates)
    cs_level_2_id = datasource.h5DB.load_factor('cs_level_2', '/indexes/', dates=dates)
    cs_level_2_id.columns = ['cs_level_1']
    cs_level_1_id.update(cs_level_2_id[cs_level_2_id['cs_level_1'].isin([5165, 5166, 5167])])
    cs_diversified_finance = cs_level_1_id.rename(columns={'cs_level_1': 'diversified_finance_cs'})
    datasource.h5DB.save_factor(cs_diversified_finance, '/indexes/')
    # 申万行业
    sw_level_1_id = datasource.h5DB.load_factor('sw_level_1', '/indexes/', dates=dates)
    sw_level_2_id = datasource.h5DB.load_factor('sw_level_2', '/indexes/', dates=dates)
    sw_level_2_id.columns = ['sw_level_1']
    sw_level_1_id.update(sw_level_2_id[sw_level_2_id['sw_level_1'].isin([801194.0, 801193.0, 801191.0, 801192.0])])
    sw_diversified_finance = sw_level_1_id.rename(columns={'sw_level_1': 'diversified_finance_sw'})
    datasource.h5DB.save_factor(sw_diversified_finance, '/indexes/')


# 中证500、沪深300、中证800指数剔除掉券商银行股之后权重归一化
def rescale_weight_afterdrop_brokers_and_banks(start, end, **kwargs):
    datasource = kwargs['data_source']
    industry_info = datasource.sector.get_stock_industry_info(ids=None, industry="申万细分非银", start_date=start, end_date=end)
    # 沪深300
    hs300_weight = datasource.sector.get_index_weight(ids='000300', start_date=start, end_date=end)
    hs300_weight = hs300_weight.join(industry_info).query('diversified_finance_sw not in ["银行", "券商"]')
    hs300_weight['_000300_weight'] = hs300_weight['_000300_weight'] / hs300_weight.groupby(level=0)['_000300_weight'].sum()
    new_index = hs300_weight[['_000300_weight']].rename(columns={'_000300_weight':'_000300_dropbrkbank_weight'})
    datasource.h5DB.save_factor(new_index, '/indexes/')

    # 中证500
    zz500_weight = datasource.sector.get_index_weight(ids='000905', start_date=start, end_date=end)
    zz500_weight = zz500_weight.join(industry_info).query('diversified_finance_sw not in ["银行", "券商"]')
    zz500_weight['_000905_weight'] = zz500_weight['_000905_weight'] / zz500_weight.groupby(level=0)[
        '_000905_weight'].sum()
    new_index = zz500_weight[['_000905_weight']].rename(columns={'_000905_weight': '_000905_dropbrkbank_weight'})
    datasource.h5DB.save_factor(new_index, '/indexes/')

    # 中证800
    zz800_weight = datasource.sector.get_index_weight(ids='000906', start_date=start, end_date=end)
    zz800_weight = zz800_weight.join(industry_info).query('diversified_finance_sw not in ["银行", "券商"]')
    zz800_weight['_000906_weight'] = zz800_weight['_000906_weight'] / zz800_weight.groupby(level=0)[
        '_000906_weight'].sum()
    new_index = zz800_weight[['_000906_weight']].rename(columns={'_000906_weight': '_000906_dropbrkbank_weight'})
    datasource.h5DB.save_factor(new_index, '/indexes/')


# 剔除银行和券商的全部A股(申万行业)
def excld_broker_banks(start, end, **kwargs):
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end)
    ashare = datasource.sector.get_history_ashare(dates)
    brokers = datasource.sector.get_index_members(ids='801193', dates=dates)
    banks = datasource.sector.get_index_members(ids='801192', dates=dates)
    new_list = ashare.drop(brokers.index).drop(banks.index).rename(columns={'ashare': '_101005'})
    datasource.h5DB.save_factor(new_list, '/indexes/')


# 行业哑变量
def update_indu_dummy(start, end, **kwargs):
    datasource = kwargs['data_source']
    for k, v in INDUSTRY_NAME_DICT.items():
        industry_info = datasource.sector.get_stock_industry_info(None, k, start, end)
        data_source.h5DB.save_as_dummy(industry_info[v], '/dummy/')

AlternativeFuncListMonthly = []
AlternativeFuncListDaily = [iffr, unst, diversify_finance, excld_broker_banks,
                            rescale_weight_afterdrop_brokers_and_banks, update_indu_dummy]

if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    # diversify_finance('20050104', '20171204', data_source=data_source)
    # excld_broker_banks('20050104', '20171204', data_source=data_source)
    # rescale_weight_afterdrop_brokers_and_banks('20050104', '20171204', data_source=data_source)
    update_indu_dummy('20050101', '20180502', data_source=data_source)
