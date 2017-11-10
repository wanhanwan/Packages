"""其他因子"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from QuantLib.tools import df_rolling2
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
    # 中信行业
    # cs_level_1 = datasource.sector.get_stock_industry_info(None, industry="中信一级", start_date=start, end_date=end)
    # cs_level_1_id = IndustryConverter.name2id("cs_level_1", cs_level_1['cs_level_1'])
    # cs_level_2 = datasource.sector.get_stock_industry_info(None, industry="中信二级", start_date=start, end_date=end)
    # cs_level_2_id = IndustryConverter.name2id("cs_level_2", cs_level_2['cs_level_2'])
    # cs_level_1_id.update(cs_level_2_id[cs_level_2_id.isin([5165, 5166, 5167])])
    # cs_diversified_finance = cs_level_1_id.to_frame("_cs_diversified_finance")
    # data_source.h5DB.save_factor(cs_diversified_finance, '/indexes/')
    # 申万行业
    sw_level_1 = datasource.sector.get_stock_industry_info(None, industry="申万一级", start_date=start, end_date=end)
    sw_level_1_id = IndustryConverter.name2id("sw_level_1", sw_level_1['sw_level_1'])
    sw_level_2 = datasource.sector.get_stock_industry_info(None, industry="申万二级", start_date=start, end_date=end)
    sw_level_2_id = IndustryConverter.name2id("sw_level_2", sw_level_2['sw_level_2'])
    sw_level_1_id.update(sw_level_2_id[sw_level_2_id.isin([801194, 801193, 801191])])
    sw_diversified_finance = sw_level_1_id.to_frame("_sw_diversified_finance")
    data_source.h5DB.save_factor(sw_diversified_finance, '/indexes/')



AlternativeFuncListMonthly = []
AlternativeFuncListDaily = [iffr, unst]

if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    diversify_finance('20170701', '20170816', data_source=data_source)