"""生成股票列表"""
from data_source import data_source
from utils.tool_funcs import parse_industry
import pandas as pd


def typical(factor, name, direction=None, industry_neutral=True, benchmark=None, industry_name='中信一级', prc=0.05, **kwargs):
    factor_data = factor.reset_index()
    prc = 1 - prc if (direction == 1 or direction is None) else prc
    if industry_neutral:
        industry_str = parse_industry(industry_name)

        all_ids = factor_data['IDs'].unique().tolist()
        all_dates = factor_data['date'].unique().tolist()

        # 个股的行业信息与因子数据匹配
        industry_info = data_source.sector.get_stock_industry_info(
            all_ids, industry=industry_name, dates=all_dates).reset_index()
        factor_data = pd.merge(factor_data, industry_info, how='left')
        quantile_per_industry = factor_data.groupby(['date', industry_str])[name].quantile(prc)
        quantile_per_industry.name = 'quantile_value'
        factor_data = factor_data.join(quantile_per_industry, on=['date', industry_str], how='left')

        # 股票选择，stocks=DataFrame[日期 IDs 因子值 行业 行业分位数]
        if direction == 1:
            stocks = factor_data[factor_data[name] >= factor_data['quantile_value']]
        else:
            stocks = factor_data[factor_data[name] <= factor_data['quantile_value']]

        # 配置权重
        benchmark_weight = data_source.sector.get_index_industry_weight(
            benchmark, industry_name=industry_name, dates=all_dates)  # 基准指数的行业权重
        stock_counts_per_industry = stocks.groupby(['date', industry_str])['IDs'].count()
        weight = (benchmark_weight / stock_counts_per_industry).rename('Weight')
        stocks = stocks.join(weight, on=['date', industry_str], how='left').set_index(['date', 'IDs'])[['Weight']]
        stocks.dropna(inplace=True)
        sum_weight = stocks.groupby(level=0)['Weight'].sum()
        stocks['Weight'] = stocks['Weight'] / sum_weight
    else:
        quantile_per_date = factor_data.groupby('date')[name].quantile(prc)
        quantile_per_date.name = 'quantile_value'
        factor_data = factor_data.join(quantile_per_date, on='date', how='left')

        # 股票选择，stocks=DataFrame[日期 IDs 因子值 分位数]
        if direction == 1:
            stocks = factor_data[factor_data[name] >= factor_data['quantile_value']]
        else:
            stocks = factor_data[factor_data[name] <= factor_data['quantile_value']]
        stock_counts_per_date = stocks.groupby('date')['IDs'].count()
        stocks = stocks.join(1 / stock_counts_per_date.rename('Weight'), on='date').set_index(['date', 'IDs'])
    return stocks[['Weight']]


FuncList = {'typical': typical}