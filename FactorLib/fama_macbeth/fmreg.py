"""fama-macbeth回归"""
from data_source.base_data_source_h5 import data_source
from QuantLib.utils import StandardByQT
from utils.datetime_func import DateStr2Datetime
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp


def _default_formula(columns):
    independents = [x for x in columns if x != 'returns']
    return 'returns ~ ' + " +".join(independents)


def _ttest(data, popmean=0):
    t_value, pvalue = ttest_1samp(data, popmean)
    return pd.Series([t_value, pvalue], index=['t_value', 'p_value'])


def _cross_section_reg(data, formula):
    ols_result = smf.ols(formula, data=data, missing='drop').fit()
    params = ols_result.params.to_frame().T
    t_values = ols_result.tvalues.to_frame().T.rename(columns = lambda x:'t_value_' + x)
    # params['r_square'] = ols_result.rsquared
    return pd.concat([params, t_values], axis=1)

def _time_series_reg(params):
    return params.apply(_ttest)


def _get_dates(start, end, freq):
    return data_source.trade_calendar.get_trade_days(start, end, freq)


def _get_ids(universe, dates, freq):
    """指数的成分股，默认去掉了上市未满3个月的股票"""

    all_ids = data_source.sector.get_ashare_onlist(dates, months_filter=3)
    if universe != '全A':
        index_members = data_source.sector.get_index_members(universe, dates=dates)
        all_ids = all_ids[all_ids.index.isin(index_members.index)]
    return all_ids


def _load_data(factor_list, idx, freq):
    """提取因子数据
    数据包含了因子数据和股票的收益率
    """
    factors = []
    for factor in factor_list:
        factor_data = data_source.load_factor(factor[0], factor[1], idx=idx)
        factor_data_standard = StandardByQT(factor_data, factor[0])
        factors.append(factor_data_standard)
    factors = pd.concat(factors, axis=1)
    dates = np.unique(idx.get_level_values(0).to_pydatetime()).tolist()
    ids = idx.get_level_values(1).unique().tolist()
    ret_dates = dates + [DateStr2Datetime(data_source.trade_calendar.tradeDayOffset(dates[-1], 1, freq=freq, incl_on_offset_today=True))]
    ret = data_source.get_periods_return(ids, dates=ret_dates).groupby('IDs').shift(-1).dropna() * 100
    return pd.concat([ret, factors], axis=1, join='inner')


def fmreg_crosssection(data, formula=None):
    if formula is None:
        formula = _default_formula(data.columns)
    params = data.groupby(level=0).apply(_cross_section_reg, formula=formula).reset_index(level=1, drop=True)
    params_statistic = _time_series_reg(params[[x for x in data.columns if x != 'returns']])
    return params, params_statistic


def run_fama_macbeth(factors, config):
    dates = _get_dates(config.start_date, config.end_date, config.freq)
    ids = _get_ids(config.universe, dates, config.freq)
    factor_data = _load_data(factors, ids.index, config.freq)
    params, params_st = fmreg_crosssection(factor_data, config.formula)
    return params, params_st

def save_result(params, params_st, path):
    import os
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d_%H%M")
    excel_file = os.path.join(path, "fmbreg_%s.xlsx" % now)
    with pd.ExcelWriter(excel_file) as writer:
        params.to_excel(writer, sheet_name='截面回归系数')
        params_st.to_excel(writer, sheet_name='系数统计量')
