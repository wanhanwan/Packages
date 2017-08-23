import pandas as pd
import numpy as np
import os
from empyrical import stats
from ..data_source.base_data_source_h5 import data_source
from ..data_source.tseries import resample_returns, resample_func
from functools import partial
from datetime import datetime
from pandas.tseries.offsets import MonthBegin, QuarterBegin, YearBegin
from ..factors import load_factor


class Analyzer(object):
    def __init__(self, pkl_path, benchmark_name):
        self.rootdir = os.path.dirname(pkl_path)
        self.table = pd.read_pickle(pkl_path)
        self.portfolio_return = self._return_of_portfolio
        self.benchmark_return = self._return_of_benchmark(benchmark_name)
        self.active_return = stats._adjust_returns(self.portfolio_return, self.benchmark_return)

    @property
    def _return_of_portfolio(self):
        unit_net_value = self.table['portfolio']['unit_net_value']
        unit_net_value.index.name = 'date'
        return (unit_net_value / unit_net_value.shift(1).fillna(1) - 1).rename('portfolio_return')

    def _return_of_benchmark(self, name):
        try:
            ret = data_source.load_factor('daily_returns_%', '/indexprices/', ids=[name]) / 100
        except Exception as e:
            print(e)
            return pd.Series(np.zeros(len(self.portfolio_return)), index=self.portfolio_return.index, name=name)
        return ret.reset_index(level=1, drop=True)['daily_returns_%'].reindex(self.portfolio_return.index).rename(name)

    def resample_active_return_of(self, frequence):
        portfolio_return = self.resample_portfolio_return(frequence)
        benchmark_return = self.resample_benchmark_return(frequence)
        active_return = stats._adjust_returns(portfolio_return, benchmark_return)
        return active_return

    def resample_portfolio_return(self, frequence):
        return resample_returns(self.portfolio_return, convert_to=frequence)

    def resample_benchmark_return(self, frequence):
        return resample_returns(self.benchmark_return, convert_to=frequence)

    @property
    def abs_nav(self):
        return (1 + self.portfolio_return).cumprod()

    @property
    def rel_nav(self):
        return (1 + self.active_return).cumprod()

    @property
    def abs_annual_return(self):
        return stats.annual_return(self.portfolio_return)

    @property
    def rel_annual_return(self):
        return stats.annual_return(self.portfolio_return) - stats.annual_return(self.benchmark_return)

    @property
    def abs_total_return(self):
        return stats.cum_returns_final(self.portfolio_return)

    @property
    def rel_total_return(self):
        return stats.cum_returns_final(self.active_return) - stats.cum_returns_final(self.benchmark_return)

    @property
    def abs_annual_volatility(self):
        return stats.annual_volatility(self.portfolio_return)

    @property
    def rel_annual_volatility(self):
        return stats.annual_volatility(self.active_return)

    @property
    def abs_sharp_ratio(self):
        return stats.sharpe_ratio(self.portfolio_return, simple_interest=True)

    @property
    def rel_sharp_ratio(self):
        return stats.sharpe_ratio(self.active_return, simple_interest=True)

    @property
    def abs_maxdrawdown(self):
        return stats.max_drawdown(self.portfolio_return)

    @property
    def rel_maxdrawdown(self):
        return stats.max_drawdown(self.active_return)

    @property
    def abs_weekly_performance(self):
        df = self.resample_portfolio_return('1w').rename('weekly_return')
        return df

    @property
    def rel_weekly_performance(self):
        df = self.resample_benchmark_return('1w').rename('weekly_return')
        return self.abs_weekly_performance - df

    @property
    def abs_monthly_performance(self):
        func_dict = {
                'cum_return': (lambda x: stats.cum_returns_final(self.portfolio_return.reindex(x.index)) -
                                         stats.cum_returns_final(self.benchmark_return.reindex(x.index))),
                'volatility': lambda x: np.std(x, ddof=1) * 20 ** 0.5,
                'weekly_win_rate': lambda x: stats.win_rate(x, self.benchmark_return, 'weekly'),
                'daily_win_rate': lambda x: stats.win_rate(x, self.benchmark_return, 'daily')
            }
        df = resample_func(self.portfolio_return, convert_to='1m', func=func_dict)
        return df

    @property
    def rel_monthly_performance(self):
        func_dict = {
                'cum_return': (lambda x: stats.cum_returns_final(self.portfolio_return.reindex(x.index)) -
                                         stats.cum_returns_final(self.benchmark_return.reindex(x.index))),
                'volatility': lambda x: np.std(x, ddof=1) * 20 ** 0.5,
                'weekly_win_rate': lambda x: stats.win_rate(x, 0, 'weekly'),
                'daily_win_rate': lambda x: stats.win_rate(x, 0, 'daily')
            }
        df = resample_func(self.active_return, convert_to='1m', func=func_dict)
        return df

    @property
    def abs_yearly_performance(self):
        func_dict = {
                'cum_return': (lambda x: stats.cum_returns_final(self.portfolio_return.reindex(x.index)) -
                                         stats.cum_returns_final(self.benchmark_return.reindex(x.index))),
                'volatility': lambda x: np.std(x, ddof=1) * 250 ** 0.5,
                'sharp_ratio': lambda x: stats.sharpe_ratio(x, simple_interest=True),
                'maxdd': stats.max_drawdown,
                'IR': lambda x: stats.information_ratio(x, self.benchmark_return),
                'monthly_win_rate': lambda x: stats.win_rate(x, self.benchmark_return, 'monthly'),
                'weekly_win_rate': lambda x: stats.win_rate(x, self.benchmark_return, 'weekly'),
                'daily_win_rate': lambda x: stats.win_rate(x, self.benchmark_return, 'daily')
                }
        df = resample_func(self.portfolio_return, convert_to='1y', func=func_dict)
        return df

    @property
    def rel_yearly_performance(self):
        func_dict = {
                'cum_return': lambda x: stats.cum_returns_final(self.portfolio_return.reindex(x.index)),
                'benchmark_return': lambda x: stats.cum_returns_final(self.benchmark_return.reindex(x.index)),
                'volatility': lambda x: np.std(x, ddof=1) * 250 ** 0.5,
                'sharp_ratio': lambda x: stats.sharpe_ratio(x, simple_interest=True),
                'maxdd':stats.max_drawdown,
                'IR':partial(stats.information_ratio, factor_returns=0),
                'monthly_win_rate': partial(stats.win_rate, factor_returns=0, period='monthly'),
                'weekly_win_rate': partial(stats.win_rate, factor_returns=0, period='weekly'),
                'daily_win_rate': partial(stats.win_rate, factor_returns=0, period='daily')
            }
        df = resample_func(self.active_return, convert_to='1y', func=func_dict)
        df.insert(2, 'active_return', df['cum_return']-df['benchmark_return'])
        return df

    def range_pct(self, start, end, rel=True):
        try:
            if rel:
                return stats.cum_returns_final(self.active_return.loc[start:end])
            else:
                return stats.cum_returns_final(self.portfolio_return.loc[start:end])
        except:
            return np.nan

    def range_maxdrawdown(self, start, end, rel=True):
        try:
            if rel:
                return stats.max_drawdown(self.active_return.loc[start:end])
            else:
                return stats.max_drawdown(self.portfolio_return.loc[start:end])
        except:
            return np.nan

    def portfolio_weights(self, date):
        weight = (self.table['stock_positions'].loc[date, 'market_value'] /
                  self.table['stock_account'].loc[date, 'total_value']).to_frame('Weight')
        weight['IDs'] = self.table['stock_positions'].loc[date, 'order_book_id'].str[:6]
        return weight.set_index('IDs', append=True)

    def returns_sheet(self, cur_day=None):
        if cur_day is None:
            cur_day = pd.to_datetime(data_source.trade_calendar.get_latest_trade_days(
                datetime.today().strftime("%Y%m%d")))
        else:
            cur_day = pd.to_datetime(cur_day)
        dates = [
                    cur_day,
                    cur_day.to_period('W').start_time,
                    cur_day + MonthBegin(-1),
                    cur_day + QuarterBegin(-1),
                    cur_day + MonthBegin(-6),
                    cur_day + YearBegin(-1),
                    cur_day + YearBegin(-2)
                ]
        returns = list(map(lambda x: self.range_pct(x, cur_day), dates)) + \
                  [self.rel_annual_return, self.rel_total_return]
        return pd.DataFrame([returns], columns=['日回报', '本周以来', '本月以来', '本季以来', '近6个月', '今年以来',
                                                '近两年', '年化回报', '成立以来'])


class FactorAnalyzer(Analyzer):
    def __init__(self, pkl_path):
        factor = load_factor(pkl_path)
        self.portfolio_return = (factor.group_return.get_group_return(1)['typical'].
                                 rename('portfolio_returns'))
        self.portfolio_return.index.name = 'date'
        self.benchmark_return = (factor.group_return.get_benchmark_return()['typical']
                                 .rename('benchmark_return')
                                 .reindex(self.portfolio_return.index))
        self.benchmark_return.index.name = 'date'
        self.active_return = stats._adjust_returns(self.portfolio_return, self.benchmark_return).rename('active_return')
        self.long_short_return = factor.long_short_return.to_frame()['typical'].rename('long_short_returns').\
            reindex(self.portfolio_return.index)
        self.long_short_return.index.name = 'date'

    def ls_range_pct(self, start, end):
        try:
            return stats.cum_returns_final(self.long_short_return.loc[start:end])
        except:
            return np.nan

    @property
    def ls_annual_return(self):
        return stats.annual_return(self.long_short_return)

    @property
    def ls_total_return(self):
        return stats.cum_returns_final(self.long_short_return)

    def ls_returns_sheet(self, cur_day=None):
        if cur_day is None:
            cur_day = pd.to_datetime(data_source.trade_calendar.get_latest_trade_days(
                datetime.today().strftime("%Y%m%d")))
        else:
            cur_day = pd.to_datetime(cur_day)
        dates = [
            cur_day,
            cur_day.to_period('W').start_time,
            cur_day + MonthBegin(-1),
            cur_day + QuarterBegin(-1),
            cur_day + MonthBegin(-6),
            cur_day + YearBegin(-1),
            cur_day + YearBegin(-2)
        ]
        returns = list(map(lambda x: self.ls_range_pct(x, cur_day), dates)) + \
                  [self.ls_annual_return, self.ls_total_return]
        return pd.DataFrame([returns], columns=['日回报', '本周以来', '本月以来', '本季以来', '近6个月', '今年以来',
                                                '近两年', '年化回报', '成立以来'])

    def resample_ls_return_of(self, frequence):
        return resample_returns(self.long_short_return, convert_to=frequence)


if __name__ == '__main__':
    # analyzer = Analyzer(r"D:\data\factor_investment_strategies\兴业风格_价值\backtest\BTresult.pkl",
    #                     benchmark_name='000905')
    ff = FactorAnalyzer(r"D:\factors\全市场_过去12个月\bp_divide_median\bp_divide_median.pkl")
    a = ff.resample_ls_return_of('1m')
    b = ff.abs_yearly_performance
