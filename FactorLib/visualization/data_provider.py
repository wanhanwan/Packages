"""可视化模块中的数据源
可视化模块主要是对已有策略集和单因子测试的结果进行可视化分析。

建立DataProvider类，为可视化提供数据
"""

from FactorLib.factor_performance.analyzer import Analyzer
from FactorLib.data_source.base_data_source_h5 import data_source
import pandas as pd
import fastcache
import os
import re


@fastcache.clru_cache()
def _load_return_analysis(f):
    return pd.read_excel(f, sheetname='returns').iloc[:, 1:].to_dict(orient='list')


class StrategyPerformanceResultProvider(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.all_dates = []
        self.update_dates()
        self.max_date = max(self.all_dates)

    def update_dates(self):
        self.all_dates = []
        for f in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path, f)) and re.match('^[0-9]{8}$', f):
                self.all_dates.append(f)

    def load_data(self, date):
        p = os.path.join(self.root_path, date)
        return _load_return_analysis(os.path.join(p, "returns_analysis_%s.xlsx"%date))


class SingleStrategyResultProvider(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.all_strategies = []
        self.strategy_summary = None
        self.update_strategy()

    def update_strategy(self):
        summary_file = os.path.join(self.root_path, 'summary.csv')
        summary_df = pd.read_csv(summary_file, converters={'benchmark': lambda x: str(x).zfill(6)}, encoding='GBK')
        self.strategy_summary = summary_df
        self.all_strategies = summary_df['name'].tolist()

    @fastcache.clru_cache()
    def get_analyzer(self, strategy):
        pkl_path = os.path.join(self.root_path, "%s/backtest/BTresult.pkl"%strategy)
        benchmark_name = self.strategy_summary.loc[self.strategy_summary.name==strategy, 'benchmark'].iloc[0]
        return Analyzer(pkl_path, benchmark_name)

    @fastcache.clru_cache()
    def load_return(self, strategy):
        analyzer = self.get_analyzer(strategy)
        nav = pd.concat([analyzer.abs_nav, analyzer.benchmark_nav, analyzer.rel_nav], axis=1, join='inner')
        nav.columns = ['absolute', 'benchmark', 'relative']
        return nav

    @fastcache.clru_cache()
    def load_info(self, strategy):
        return str(self.strategy_summary.loc[self.strategy_summary.name==strategy, :].iloc[0])

    @fastcache.clru_cache()
    def load_strategy_performance(self, strategy):
        analyzer = self.get_analyzer(strategy)
        return str(analyzer.total_performance)

    @fastcache.clru_cache()
    def load_strategy_rel_yr_performance(self, strategy):
        analyzer = self.get_analyzer(strategy)
        return str(analyzer.rel_yearly_performance)

    @fastcache.clru_cache()
    def load_stock_info(self, date):
        return data_source.sector.get_stock_info(ids=None, date=date)

    @fastcache.clru_cache()
    def load_stock_return(self, date):
        return data_source.load_factor('daily_returns_%', '/stocks/', dates=[date]) / 100

    @fastcache.clru_cache()
    def load_positions(self, date, strategy):
        latest_trade_date = data_source.trade_calendar.tradeDayOffset(date, 0)
        analyzer = self.get_analyzer(strategy)
        stock_weight = analyzer.portfolio_weights(latest_trade_date).reset_index(level=0, drop=True)
        stock_info = self.load_stock_info(latest_trade_date).reset_index(level=0, drop=True)
        stock_return = self.load_stock_return(latest_trade_date).reset_index(level=0, drop=True).rename(
            columns={'daily_returns_%': 'daily_return'})
        return pd.concat([stock_weight, stock_info, stock_return], axis=1, join='inner')
