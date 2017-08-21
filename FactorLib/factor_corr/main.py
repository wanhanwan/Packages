from single_factor_test.config import parse_config
from utils import AttrDict
from factors.panel_factor import panelFactor
from data_source.base_data_source_h5 import h5, tc, sec, data_source
from datetime import datetime

import pandas as pd
import importlib.util as ilu
import os

def run(config_path):
    config = parse_config(config_path)
    config = AttrDict(config)
    
    # 加载因子
    print("加载因子...")
    factors = load_factors(config)
    factor_list = []
    for factor in factors:
        factor_list.append(panelFactor(*factor))
    load_factor_data(factor_list, config)    
    
    print("计算...")
    corr, corr_mean = cal_corr(factor_list)
    auto_corr, auto_corr_mean = cal_auto_corr(factor_list)
    
    print("存储...")
    write_to_excel(corr, corr_mean, auto_corr, auto_corr_mean, config)

def load_factors(config):
    spec = ilu.spec_from_file_location("factor_list", config.factor_list_file)
    factor_list = ilu.module_from_spec(spec)
    spec.loader.exec_module(factor_list)
    return factor_list.factor_list

def load_factor_data(factors, config):
    # 为因子加载数据
    dates = tc.get_trade_days(config.start_date, config.end_date, config.freq)
    sector_name = config.universe
    
    if sector_name == '全A':
        stocks = sec.get_history_ashare(dates)
    else:
        stocks = sec.get_index_members(sector_name, dates=dates)
    ids = stocks.index.get_level_values(1).unique().tolist()
    for factor in factors:
        factor_data = data_source.load_factor(factor.name, factor.axe, ids=ids, dates=dates)
        if not factor.data.empty:
            old = factor.data[~factor.data.index.isin(factor_data.index)]
        else:
            old = pd.DataFrame()
        factor.data = old.append(factor_data).reindex(stocks.index)

def cal_corr(factors):
    data = pd.concat([x.data for x in factors], axis=1, join='outer')
    corr = data.groupby(level=0).apply(lambda x: x.corr(method='spearman'))
    corr_mean = corr.unstack().mean().unstack()
    return corr, corr_mean

def cal_auto_corr(factors):
    """计算因子自相关性"""
    corr_func = lambda x: x.corr(method='spearman').iat[0, 1]
    l = []
    for factor in factors:
        data_shift = factor.data.sort_index().unstack().shift(1).stack()
        data = pd.concat([factor.data, data_shift], axis=1, ignore_index=True, join='inner')
        corr = data.groupby(level=0).apply(corr_func)
        corr.name = factor.name
        l.append(corr)
    auto_corr = pd.concat(l, axis=1)
    auto_corr_mean = auto_corr.mean()
    return auto_corr, auto_corr_mean

def write_to_excel(corr, corr_mean, auto_corr, auto_corr_mean, config):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    excel_path = config.output_dir
    excel_file = os.path.join(excel_path, "corr_%s.xlsx" % now)
    with pd.ExcelWriter(excel_file) as writer:
        corr.to_excel(writer, sheet_name='因子相关系数详情')
        corr_mean.to_excel(writer, sheet_name='因子相关系数均值')
        auto_corr.to_excel(writer, sheet_name='因子自相关系数详情')
        auto_corr_mean.to_excel(writer, sheet_name='因子自相关系数均值')

run("D:/FactorLib/factor_corr/config.yml")