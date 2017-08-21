import pandas as pd
import numpy as np
from factor_performance.toolfuncs import *
from empyrical import stats
from scipy import stats as stats_scp
import pyfolio as pf
import pickle
'''给定收益率序列，计算相关指标'''


class ReturnSeries(object):

    """docstring for FundPerformance"""

    def __init__(self, benchmark_used='000905',
                 freq='daily', free_rate=0.):
        self.Ret = pd.Series()
        self.BenchmarkRet = pd.Series()
        self.activeRet = pd.Series()
        self.Freq = freq
        self.FreeRate = free_rate
        self.Benchmark = benchmark_used

    def get_state(self):
        _d={}
        _d['Ret'] = self.Ret
        _d['Freq'] = self.Freq
        _d['FreeRate'] = self.FreeRate
        _d['Benchmark'] = self.Benchmark
        _d['BenchmarkRet'] = self.BenchmarkRet
        _d['activeRet'] = self.activeRet
        return  _d

    def set_state(self, state):
        self.Ret = state['Ret']
        self.Freq = state['Freq']
        self.FreeRate = state['FreeRate']

        self.BenchmarkRet = state['BenchmarkRet']
        self.Benchmark = state['Benchmark']
        self.activeRet = state['activeRet']

    def update_stock_return(self, return_series):
        old = self.Ret[~self.Ret.index.isin(return_series.index)]
        self.Ret = old.append(return_series).sort_index()
        self.activeRet = stats._adjust_returns(
            self.Ret, self.BenchmarkRet).reindex(self.Ret.index)

    def update_benchmark_return(self, return_series):
        old = self.BenchmarkRet[~self.BenchmarkRet.index.isin(return_series.index)]
        self.BenchmarkRet = old.append(return_series)
        self.activeRet = stats._adjust_returns(
            self.Ret, self.BenchmarkRet).reindex(self.Ret.index)

    # 设置收益率序列
    def SetYieldSeries(self, yield_series, freq='daily'):
        self.Ret = yield_series
        self.activeRet = stats._adjust_returns(
            self.Ret, self.BenchmarkRet).reindex(self.Ret.index)
        self.Freq = freq
        return 1

    # 设置无风险收益率
    def SetFreeRate(self, free_rate):
        self.FreeRate = free_rate
        return 1

    # 计算年化收益
    def AnnualYield(self, hedge=True):
        ret = self.activeRet if hedge else self.Ret
        return stats.annual_return(ret, period=self.Freq)

    # 计算最大回撤
    def MaxDrawDown(self, hedge=True):
        ret = self.activeRet if hedge else self.Ret
        return stats.max_drawdown(ret)

    # 计算年化波动率
    def AnnualVol(self, hedge=True):
        ret = self.activeRet if hedge else self.Ret
        return stats.annual_volatility(ret,
                                       period=self.Freq)

    # 计算夏普比率
    def SharpRatio(self, simple_interest=True, hedge=True):
        ret = self.activeRet if hedge else self.Ret
        return stats.sharpe_ratio(ret,
                                  risk_free=self.FreeRate,
                                  period=self.Freq,
                                  simple_interest=simple_interest)

    # 计算月收益率
    def MonthlyRet(self, hedge=True):
        month_year = pd.Series(self.Ret.index).apply(
            lambda x: x.year * 100 + x.month)
        is_month_end = (~month_year.duplicated('last')).tolist()
        if hedge:
            monthRet = pd.Series(
                stats.aggregate_returns(self.activeRet, 'monthly'))
            monthRet.index = self.activeRet.index[is_month_end]
        else:
            monthRet = pd.Series(stats.aggregate_returns(self.Ret, 'monthly'))
            monthRet.index = self.Ret.index[is_month_end]
        return monthRet

    # 计算基准的月收益率
    def BenchmarkMonthlyRet(self):
        benchmarkRet = self.BenchmarkRet.reindex(self.Ret.index)
        month_year = pd.Series(self.Ret.index).apply(
            lambda x: x.year * 100 + x.month)
        is_month_end = (~month_year.duplicated('last')).tolist()
        monthRet = pd.Series(stats.aggregate_returns(benchmarkRet, 'monthly'))
        monthRet.index = self.Ret.index[is_month_end]
        return monthRet

    # 计算胜率
    def WinRate(self, period='monthly'):

        return stats.win_rate(self.Ret, self.BenchmarkRet, period=period)

    def YearlyPerformance(self):
        monthlyRet = self.MonthlyRet(False)                         # 单边做多模型月度收益率
        benchmarkMonthlyRet = self.BenchmarkMonthlyRet()
        hedgeMonthlyRet = self.MonthlyRet(True)                     # 对冲组合月收益率
        
        a = stats.aggregate_returns(monthlyRet, 'yearly')           # 多头组合分年收益
        b = stats.aggregate_returns(benchmarkMonthlyRet, 'yearly')  # 基准组合分年收益
        
        _l = []
        for i, year in enumerate(a.index):
            hedgeMonthlyRet_current_year = hedgeMonthlyRet.ix[str(year)]
            monthlyRet_current_year = monthlyRet.ix[str(year)]
            benchmarkMonthlyRet_current_year = benchmarkMonthlyRet.ix[str(year)]
            
            hdSharp_current_year = stats.sharpe_ratio(
                hedgeMonthlyRet_current_year, annualization=12)
            hdMaxDown_current_year = stats.max_drawdown(hedgeMonthlyRet_current_year)
            hdReturn_current_year = stats.annual_return(hedgeMonthlyRet_current_year, annualization=12)
            hdWinRate_current_year = stats.win_rate(monthlyRet_current_year, benchmarkMonthlyRet_current_year)
            _l.append([hdSharp_current_year, hdReturn_current_year, hdMaxDown_current_year, hdWinRate_current_year])
        # 计算全年收益表现
        hdSharp_all = stats.sharpe_ratio(hedgeMonthlyRet, annualization=12)
        hdMaxDown_all = stats.max_drawdown(hedgeMonthlyRet)
        hdReturn_all = stats.annual_return(hedgeMonthlyRet, annualization=12)
        hdWinRate_all = stats.win_rate(hedgeMonthlyRet, benchmarkMonthlyRet)
        _l.append([hdSharp_all, hdReturn_all, hdMaxDown_all, hdWinRate_all])
        result = pd.DataFrame(_l, columns=['夏普比率', '年化收益', '最大回撤', '胜率'], index=list(a.index) + ['All'])
        return result
    
    def get_summary(self, freq='m'):
        if freq == 'm':
            monthlyRet = self.MonthlyRet(False)
            monthlyBenchmarkRet = self.BenchmarkMonthlyRet()
            hedgeMonthlyRet = self.MonthlyRet()
            
            _dict = {
                'annual_return': stats.annual_return(hedgeMonthlyRet, annualization=12), 
                'annual_vol': stats.annual_volatility(hedgeMonthlyRet, annualization=12), 
                'IR': stats.information_ratio(monthlyRet, monthlyBenchmarkRet),
                't_stats': stats_scp.ttest_1samp(hedgeMonthlyRet, 0)[0],
                'win_rate': stats.win_rate(monthlyRet, monthlyBenchmarkRet),
                'max_drawdown': stats.max_drawdown(hedgeMonthlyRet),
                'drawn_down_start': pf.timeseries.get_top_drawdowns(hedgeMonthlyRet, 1)[0][0],
                'draw_down_end': pf.timeseries.get_top_drawdowns(hedgeMonthlyRet, 1)[0][1],
            }
        else:
            _dict = {
                'annual_return': stats.annual_return(self.activeRet, annualization=12), 
                'annual_vol': stats.annual_volatility(self.activeRet, annualization=12), 
                'IR': stats.information_ratio(self.Ret, self.BenchmarkRet),
                't_stats': stats_scp.ttest_1samp(self.activeRet, 0)[0],
                'win_rate': stats.win_rate(self.Ret, self.BenchmarkRet),
                'max_drawdown': stats.max_drawdown(self.activeRet),
                'drawn_down_start': pf.timeseries.get_top_drawdowns(self.activeRet, 1)[0],
                'draw_down_end': pf.timeseries.get_top_drawdowns(self.activeRet, 1)[1],
            }
        return pd.Series(_dict)

class GroupReturn(object):
    def __init__(self, n_groups):
        self.total_groups = n_groups
        self.group_returns = {x:ReturnSeries() for x in range(1, n_groups+1)}

    def to_frame(self):
        _l = [x.Ret for i, x in self.group_returns.items()]
        return_df = pd.concat(_l,axis=1,ignore_index=True)
        return_df.columns = range(1,self.total_groups+1)
        return return_df

    def get_state(self):
        _d = {}
        _d['total_groups'] = self.total_groups
        _d['group_returns'] = {i:x.get_state() for i,x in self.group_returns.items()}
        return _d

    def set_state(self, state):
        self.total_groups = state['total_groups']
        for i, group_return in self.group_returns.items():
            self.group_returns[i].set_state(state['group_returns'][i])

    def update_info(self, stock_returns, benchmark_returns):
        for i, x in self.group_returns.items():
            x.update_stock_return(stock_returns[i])
            x.update_benchmark_return(benchmark_returns)
    
    def get_benchmark_return(self):
        return self.group_returns[1].BenchmarkRet


class FactorGroupReturn(object):
    def __init__(self):
        self.group_methods = None
        self.group_returns = None
        self.n_groups = 0

    def initialize(self, methods, n_groups):
        self.group_methods = methods
        self.n_groups = n_groups
        self.group_returns = {x:GroupReturn(n_groups) for x in methods}

    def to_frame(self):
        _l = []
        for method in self.group_methods:
            group = self.group_returns[method].to_frame()
            columns = pd.MultiIndex.from_product([[method], group.columns])
            group.columns = columns
            _l.append(group)
        return pd.concat(_l, axis=1)

    def from_frame(self, stock_frame, benchmark_series):
        """从frame中更新数据
        frame格式: dataframe(index=date,columns=[method,group_id])
        """
        for method in self.group_methods:
            try:
                temp_stocks = stock_frame.ix[:, method]
            except Exception as e:
                continue
            self.group_returns[method].update_info(temp_stocks, benchmark_series)

    def get_state(self):
        _d = {}
        _d['group_methods'] = self.group_methods
        _d['group_returns'] = {i:x.get_state() for i,x in self.group_returns.items()}
        _d['n_groups'] = self.n_groups
        return _d

    def set_state(self, state):
        self.group_methods = state['group_methods']
        self.n_groups = state['n_groups']
        for i, x in self.group_returns.items():
            x.set_state(state['group_returns'][i])

    def get_group_return(self, group_id):
        frame = self.to_frame()
        return frame.xs(group_id, axis=1, level=1)
    
    def get_benchmark_return(self):
        _l = []
        for method in self.group_methods:
            _l.append(self.group_returns[method].get_benchmark_return())
        benchmark_return = pd.concat(_l, axis=1)
        benchmark_return.columns = self.group_methods
        return benchmark_return

class LongShortReturn(object):
    def __init__(self):
        self.methods = None
        self.long_short_returns = None

    def initialize(self, methods):
        self.methods = methods
        self.long_short_returns = {x:ReturnSeries() for x in methods}

    def get_state(self):
        _d = {}
        _d['methods'] = self.methods
        _d['long_short_returns'] = {i:x.get_state() for i,x in self.long_short_returns.items()}
        return _d

    def to_frame(self):
        """
        输出格式:
            dataframe(index:date,columns=methods,value=active_return)
        """
        _l = []
        for method in self.methods:
            _l.append(self.long_short_returns[method].activeRet.rename(method))
        return pd.concat(_l,axis=1)

    def set_state(self, state):
        self.methods = state['methods']
        for x in self.methods:
            self.long_short_returns[x].set_state(state['long_short_returns'][x])

    def update_info(self, stock_frame, benchmark_frame):
        for method in self.methods:
            try:
                stock_return = stock_frame.ix[:, method]
                benchmark_return = benchmark_frame.ix[:, method]
            except:
                continue
            self.long_short_returns[method].update_stock_return(stock_return)
            self.long_short_returns[method].update_benchmark_return(benchmark_return)
