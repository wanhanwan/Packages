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
    """计算组合中属于基准成分股中的股票权重之和

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        持仓权重(index=[date, IDs], columns=weight)

    benchmark: str or pd.DataFrame
        基准名称
    """
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').sum()


def cal_total_num_in_benchmark(portfolio_weight, benchmark):
    """计算组合中属于基准成分股中的股票数量

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        持仓权重(index=[date, IDs], columns=weight)

    benchmark: str or pd.DataFrame
        基准名称
    """
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').size()


def cal_ratio_in_benchmark(portfolio_weight, benchmark):
    """计算组合中属于基准成分股中的股票数量比例

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        持仓权重(index=[date, IDs], columns=weight)

    benchmark: str or pd.DataFrame
        基准名称
    """
    if isinstance(benchmark, str):
        dates = portfolio_weight.index.get_level_values('date').unique()
        benchmark = data_source.sector.get_index_members(ids=benchmark, dates=dates)
    portfolio_in_benchmark = portfolio_weight[portfolio_weight.index.isin(benchmark.index)]
    return portfolio_in_benchmark.groupby('date').size() / portfolio_weight.groupby('date').size()


@format_benchmark
def compare_large_weight_with_benchmark(portfolio_weight, benchmark_weight, n=10, base='portfolio'):
    """与基准比较权重股的权重

    Parameters:
    ------------------------
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
        base_weight.columns = [base]
        other_weight = benchmark_weight.reindex(base_weight.index)
        other_weight.columns = ['benchmark']
    else:
        base_weight = benchmark_weight.groupby('date', group_keys=False).apply(lambda x: x.nlargest(n, x.columns))
        base_weight.columns = [base]
        other_weight = portfolio_weight.reindex(base_weight.index)
        other_weight.columns = ['portfolio']
    diff = base_weight.join(other_weight)[['portfolio', 'benchmark']]
    diff['Diff'] = diff.portfolio - diff.benchmark
    return diff


@format_benchmark
def compare_total_weight_with_benchmark(portfolio_weight, benchmark_weight, n=10, base='portfolio'):
    """与基准比较权重股总权重的差别"""
    if base == 'portfolio':
        base_weight = portfolio_weight.groupby('date', group_keys=False).apply(lambda x: x.nlargest(n, x.columns))
        base_weight.columns = [base]
        other_weight = benchmark_weight.reindex(base_weight.index)
        other_weight.columns = ['benchmark']
    else:
        base_weight = benchmark_weight.groupby('date', group_keys=False).apply(lambda x: x.nlargest(n, x.columns))
        base_weight.columns = [base]
        other_weight = portfolio_weight.reindex(base_weight.index)
        other_weight.columns = ['portfolio']
    diff = pd.concat([base_weight.groupby('date').sum(),
                      other_weight.groupby('date').sum()], axis=1)[['portfolio', 'benchmark']]
    diff['Diff'] = diff.portfolio - diff.benchmark
    return diff


def cal_future_return(portfolio_weight, group=None, window_len='20d'):
    """计算股票组合未来的收益率

    Parameters:
    -------------------------
    portfolio_weight: pd.DataFrame or Series
        组合的权重
    group: pd.Series
        可以用来给组合权重分组的分类器
    window_len: str
        未来的时间窗口
    """
    def cal_weighted_ret(x, y):
        return x.dot(y)

    future_ret = data_source.get_forward_ndays_return(None, idx=portfolio_weight, windows=[1], freq=window_len)[1]
    grouper = ['date']
    if group is not None:
        grouper.append(group)
    weight_name = portfolio_weight.columns[0]
    a = portfolio_weight.groupby(grouper, group_keys=False)[weight_name].agg(
        lambda x: cal_weighted_ret(x.values, future_ret.reindex(x.index).values))
    return a


if __name__ == '__main__':
    from alphalens.utils import quantize_factor
    p = data_source.sector.get_index_weight('000300', dates=['20170228'])
    p.columns = ['a']
    group = quantize_factor(p.rename(columns=lambda x: 'factor'), quantiles=10).rename(index={'asset': 'IDs'})
    print(cal_future_return(p, group=group))
