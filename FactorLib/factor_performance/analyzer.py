import pandas as pd
import numpy as np
import os
from empyrical import stats
from ..data_source.base_data_source_h5 import data_source, tc
from ..data_source.tseries import resample_returns, resample_func
from functools import partial
from datetime import datetime
from pandas.tseries.offsets import MonthBegin, QuarterBegin, YearBegin
from ..factors import load_factor
from ..riskmodel.attribution import RiskExposureAnalyzer, RiskModelAttribution
from ..utils.tool_funcs import uqercode_to_windcode


class Analyzer(object):
    def __init__(self, pkl_path, benchmark_name):
        self.rootdir = os.path.dirname(pkl_path)
        self.table = pd.read_pickle(pkl_path)
        self.portfolio_return = self._return_of_portfolio
        self.benchmark_return = self._return_of_benchmark(benchmark_name)
        self.active_return = stats._adjust_returns(self.portfolio_return, self.benchmark_return)
        self.benchmark_name = benchmark_name

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
    def benchmark_nav(self):
        return (1 + self.benchmark_return).cumprod()

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
        return self.abs_total_return - self.total_benchmark_return

    @property
    def total_benchmark_return(self):
        return stats.cum_returns_final(self.benchmark_return)

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

    @property
    def total_performance(self):
        """策略全局表现"""
        win_rate_func = lambda x, y: stats.win_rate(x, factor_returns=0, period=y)
        df = pd.DataFrame([[self.abs_total_return, self.total_benchmark_return, self.rel_total_return],
                           [self.abs_maxdrawdown, stats.max_drawdown(self.benchmark_return), self.rel_maxdrawdown],
                           [self.abs_annual_volatility, stats.annual_volatility(self.benchmark_return), self.rel_annual_volatility],
                           [self.abs_annual_return, stats.annual_return(self.benchmark_return), self.rel_annual_return],
                           [self.abs_sharp_ratio, stats.sharpe_ratio(self.benchmark_return, simple_interest=True), self.rel_sharp_ratio],
                           [win_rate_func(self.portfolio_return, 'monthly'), win_rate_func(self.benchmark_return, 'monthly'),
                            win_rate_func(self.active_return, 'monthly')]],
                          columns=['portfolio', 'benchmark', 'hedge'],
                          index=['total_return', 'maxdd', 'volatility', 'annual_return', 'sharp', 'win_rate'])
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

    def portfolio_weights(self, dates, scale=False):
        """组合成分股权重
        Paramters:
        --------------
        scale: bool
            股票权重是否归一化处理，默认False
        """
        if not isinstance(dates, list):
            dates = [dates]
        dates = pd.DatetimeIndex(dates)
        weight = (self.table['stock_positions'].loc[dates, 'market_value'] /
                  self.table['stock_account'].loc[dates, 'total_value']).to_frame('Weight')
        weight['IDs'] = self.table['stock_positions'].loc[dates, 'order_book_id'].str[:6]
        weight.index.name = 'date'
        weight = weight.set_index('IDs', append=True)
        if scale:
            sum_of_weight = weight.sum(level=0)
            weight = weight / sum_of_weight
        return weight

    def portfolio_risk_expo(self, data_src_name, dates):
        """组合风险因子暴露"""
        dates = pd.DatetimeIndex(dates)
        positions = self.portfolio_weights(dates)
        data_src = RiskExposureAnalyzer.from_df(positions, barra_datasource=data_src_name,
                                                benchmark=self.benchmark_name)
        barra_expo, indus_expo, risk_expo = data_src.cal_multidates_expo(dates)
        return barra_expo, indus_expo, risk_expo

    def range_attribute(self, start_date, end_date, data_src_name='xy'):
        """在一个时间区间进行归因分析"""
        dates = tc.get_trade_days(start_date, end_date)
        ret_ptf = self.portfolio_return.loc[dates]
        barra_expo, indus_expo, risk_expo = self.portfolio_risk_expo(data_src_name, dates=dates)
        risk_model = RiskModelAttribution(ret_ptf, barra_expo, indus_expo, self.benchmark_name, data_src_name)
        return risk_model.range_attribute(start_date, end_date)

    def trade_records(self, start_date, end_date):
        """
        导出交易记录
        交易记录的格式为：
            DataFrame(columns=[证券代码,操作类型(买入，卖出),交易价格,交易日期])
        """
        trades = self.table['trades'].copy()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        trades.index = pd.DatetimeIndex(trades.index)
        trades = trades.loc[start_date:end_date, ['order_book_id', 'side', 'last_quantity', 'last_price']]
        trades['order_book_id'] = trades['order_book_id'].apply(uqercode_to_windcode)
        trades.loc[trades['side']=='SELL', 'last_quantity'] = trades.loc[trades['side']=='SELL', 'last_quantity'] * -1
        trades['side'] = trades['side'].map({'SELL': '卖出', 'BUY': '买入'})
        trades = trades.reset_index().loc[:, ['order_book_id', 'side', 'last_quantity', 'last_price', 'datetime']]
        trades.columns = ['证券代码', '操作类型', '交易数量', '交易价格', '交易日期']
        return trades

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
                    cur_day + QuarterBegin(n=-1, startingMonth=1),
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
            cur_day + QuarterBegin(n=-1, startingMonth=1),
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
    analyzer = Analyzer(r"D:\data\factor_investment_strategies\兴业风格_价值\backtest\BTresult.pkl",
                        benchmark_name='000905')
    # ff = FactorAnalyzer(r"D:\factors\全市场_过去12个月\bp_divide_median\bp_divide_median.pkl")
    a = analyzer.portfolio_weights('20170830')