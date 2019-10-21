# coding: utf-8
# author: wamhanwan
from rqrisk import Risk, DAILY, MONTHLY, WEEKLY
from pandas.tseries.offsets import as_timestamp
from functools import lru_cache
from warnings import warn
import pandas as pd
import numpy as np

from QuantLib.tools import return2nav
from QuantLib.portfolio import plotting


def _add_date_index(x, period):
    if isinstance(x, np.ndarray):
        return pd.Series(
            x, index=pd.date_range('2000-01-01', periods=len(x), freq=period[0])
        )
    if isinstance(x, pd.Series):
        x = pd.Series(x.to_numpy(), index=pd.DatetimeIndex(x.index))
        return x


class Analyser(object):
    def __init__(self, returns, benchmark_returns=0., risk_free_rate=0., period=DAILY):
        self._returns = _add_date_index(returns, period)

        if isinstance(benchmark_returns, (int, float)):
            self._benchmark_returns = pd.Series(benchmark_returns, index=self._returns.index)
        else:
            self._benchmark_returns = _add_date_index(benchmark_returns, period)

        self._risk_free_rate = risk_free_rate
        self._period = period

        self._start = max(self._returns.index.min(),
                          self._benchmark_returns.index.min()
                          )
        self._end = min(self._returns.index.max(),
                        self._benchmark_returns.index.max()
                        )

        self.start_date = self._start
        self.end_date = self._end

    @classmethod
    def create_from_nav(cls, navs, benchmark_navs=1., risk_free_rate=0., period=DAILY):
        navs = _add_date_index(navs, period)
        returns = navs.pct_change(fill_method=None).iloc[1:]

        if isinstance(benchmark_navs, (int, float)):
            benchmark_returns = 0.
        else:
            benchmark_returns = _add_date_index(benchmark_navs, period).pct_change(fill_method=None).iloc[1:]
        return cls(returns, benchmark_returns, risk_free_rate, period)


    def set_date_range(self, start_date, end_date):
        if start_date is not None:
            s = as_timestamp(start_date)
            if s < self._start:
                warn("start_date out of bounds")
            self.start_date = s
        if end_date is not None:
            e = as_timestamp(end_date)
            if e > self._end:
                warn("end_date out of bounds")
            self.end_date = e

    def reset_date_range(self):
        self.start_date = self._start
        self.end_date = self._end

    @property
    def returns(self):
        return self._returns[self.start_date:self.end_date]

    @property
    def navs(self):
        return return2nav(self.returns)

    @property
    def benchmark_navs(self):
        return return2nav(self.benchmark_returns)

    @property
    def benchmark_returns(self):
        return self._benchmark_returns[self.start_date:self.end_date]

    @property
    def risk_free_rate(self):
        return self._risk_free_rate

    @property
    def period(self):
        return self._period

    @lru_cache()
    def get_risker(self, start_date, end_date):
        self.set_date_range(start_date, end_date)
        return Risk(self.returns.to_numpy(),
                    self.benchmark_returns.to_numpy(),
                    self.risk_free_rate,
                    self._period)

    def __getattr__(self, item):
        return getattr(self.get_risker(self.start_date, self.end_date), item)

    @property
    def yearly_report(self):
        years = range(self.start_date.year, self.end_date.year+1)
        yearly_data = pd.DataFrame(index=list(years),
                                   columns=['累计收益',
                                            '年化收益',
                                            '基准年化收益',
                                            '年化波动率',
                                            '最大回撤',
                                            '夏普比率',
                                            '年化跟踪误差',
                                            '信息比率'],
                                    dtype='float64')
        for y in years:
            risker = self.get_risker(as_timestamp(f'{y}-01-01'), as_timestamp(f'{y}-12-31'))
            data = [
                risker.return_rate,
                risker.annual_return,
                risker.benchmark_annual_return,
                risker.annual_volatility,
                risker.max_drawdown,
                risker.sharpe,
                risker.annual_tracking_error,
                risker.information_ratio
            ]
            yearly_data.loc[y, :] = np.asarray(data, dtype='float64')
        self.reset_date_range()
        return yearly_data

    @property
    def report(self):
        risker = self.get_risker(self.start_date, self.end_date)
        data = [[
            risker.return_rate,
            risker.annual_return,
            risker.benchmark_annual_return,
            risker.annual_volatility,
            risker.max_drawdown,
            risker.sharpe,
            risker.annual_tracking_error,
            risker.information_ratio
        ]]
        df = pd.DataFrame(
            data = np.asarray(data, dtype='float64'),
            index=['全样本'],
            columns=['累计收益',
                   '年化收益',
                   '基准年化收益',
                   '年化波动率',
                   '最大回撤',
                   '夏普比率',
                   '年化跟踪误差',
                   '信息比率'])
        return df
    
    def plot_nav_year_by_year(self):
        fig, axes = plotting.plot_nav_year_by_year(self.returns, self.benchmark_returns)
        return fig, axes


if __name__ == '__main__':
    import pandas as pd
    navs = pd.read_csv(r"G:\Research\我的坚果云\job\strategies\testing\safty_factor\strategy\strategy\portfolio.csv",
                       index_col=0, parse_dates=True, usecols=[0,4,6])
    benchmark_navs = pd.read_csv(r"G:\Research\我的坚果云\job\strategies\testing\safty_factor\strategy\strategy\benchmark_portfolio.csv",
                       index_col=0, parse_dates=True, usecols=[0,4,6])
    analyzer = Analyser(navs['unit_net_value']/navs['static_unit_net_value']-1.0,
                        benchmark_navs['unit_net_value']/benchmark_navs['static_unit_net_value']-1.0)
    print(analyzer.yearly_report)
    import matplotlib.pyplot as plt
    analyzer.plot_nav_year_by_year()
    plt.show()
