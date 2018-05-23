"""股票组合分析工具"""
from FactorLib.data_source.base_data_source_h5 import data_source
import pandas as pd
import numpy as np


def format_benchmark(func):
    def wrapper(portfolio, benchmark, *args, **kwargs):
        assert isinstance(portfolio, (pd.Series, pd.DataFrame))
        if isinstance(portfolio, pd.Series):
            portfolio = portfolio.to_frame('portfolio_weight')
        if isinstance(benchmark, str):
            dates = portfolio.index.get_level_values('date').unique()
            benchmark = data_source.sector.get_index_weight(ids=benchmark,
                                                            dates=dates)
        portfolio, benchmark = portfolio.align(benchmark, axis='index', fill_value=0.0)
        return func(portfolio, benchmark, *args, **kwargs)
    return wrapper


@format_benchmark
def cal_diversion(portfolio_weight, benchmark_weight):
    """计算持仓权重与基准的偏离度

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        持仓权重(index=[date, IDs], columns=weight)

    benchmark_weight: str or pd.DataFrame
        基准权重
    """
    diff = pd.DataFrame(data=np.abs(portfolio_weight.values-benchmark_weight.values),
                        index=portfolio_weight.index,
                        columns=['diversion'])
    return diff.groupby('date').sum()


@format_benchmark
def cal_diversion_of_large_weight(portfolio_weight, benchmark_weight, n=10,
                                  base='portfolio'):
    """计算前N大重仓股的权重偏离度

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        持仓权重(index=[date, IDs], columns=weight)

    benchmark_weight: str or pd.DataFrame
        基准权重

    n: int
        取前N大重仓

    base: str
        取值: portfolio or benchmark 以组合或者基准的重仓股为基准
    """
    if base == 'portfolio':
        base_weight = portfolio_weight.groupby('date', group_keys=False).apply(lambda x: x.nlargest(n, x.columns))
        other_weight = benchmark_weight.reindex(base_weight.index)
    else:
        base_weight = benchmark_weight.groupby('date', group_keys=False).apply(lambda x: x.nlargest(n, x.columns))
        other_weight = portfolio_weight.reindex(base_weight.index)
    diff = pd.DataFrame(data=np.abs(base_weight.values - other_weight.values),
                        index=base_weight.index,
                        columns=['diversion'])
    return diff.groupby('date').sum()


def cal_total_weight_in_benchmark(portfolio_weight, benchmark):
    """计算组合中属于基准成分股中的股票权重之和"""
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').sum()


def cal_total_num_in_benchmark(portfolio_weight, benchmark):
    """计算组合中属于基准成分股中的股票数量"""
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').size()


def cal_ratio_in_benchmark(portfolio_weight, benchmark):
    """计算组合中属于基准成分股中的股票数量比例"""
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').size() / portfolio_weight.groupby('date').size()


if __name__ == '__main__':
    p = data_source.sector.get_index_weight('000300', dates=['20180521'])
    p.columns = ['a']
    print(cal_ratio_in_benchmark(p, '000300'))