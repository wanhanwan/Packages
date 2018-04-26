# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from empyrical import stats
from functools import wraps
from ..data_source.base_data_source_h5 import data_source, tc, H5DB
from ..data_source.tseries import resample_returns, resample_func
from functools import partial
from datetime import datetime
from collections import Iterable
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
        self.pkl_path = pkl_path

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
                'cum_return': lambda x: stats.cum_returns_final(self.portfolio_return.reindex(x.index)),
                'volatility': lambda x: np.std(x, ddof=1) * 250 ** 0.5,
                'sharp_ratio': lambda x: stats.sharpe_ratio(x, simple_interest=True),
                'maxdd': stats.max_drawdown,
                'IR': lambda x: stats.information_ratio(x, self.benchmark_return),
                'monthly_win_rate': lambda x: stats.win_rate(x, 0.0, 'monthly'),
                'weekly_win_rate': lambda x: stats.win_rate(x, 0.0, 'weekly'),
                'daily_win_rate': lambda x: stats.win_rate(x, 0.0, 'daily')
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

    def range_performance(self, start, end):
        portfolio_return = self.portfolio_return.loc[start:end]
        benchmark_return = self.benchmark_return.loc[start:end]
        active_return = self.active_return.loc[start:end]
        a = self.__class__(self.pkl_path, self.benchmark_name)
        a.portfolio_return = portfolio_return
        a.benchmark_return = benchmark_return
        a.active_return = active_return
        return a.total_performance

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
        if not isinstance(dates, Iterable) or isinstance(dates, str):
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

    def portfolio_risk_expo(self, data_src_name, dates, bchmrk_name=None):
        """组合风险因子暴露"""
        bchmrk_name = self.benchmark_name if bchmrk_name is None else bchmrk_name
        dates = pd.DatetimeIndex(dates)
        positions = self.portfolio_weights(dates, scale=True)
        data_src = RiskExposureAnalyzer.from_df(positions, barra_datasource=data_src_name,
                                                benchmark=bchmrk_name)
        barra_expo, indus_expo, risk_expo = data_src.cal_multidates_expo(dates)
        return barra_expo, indus_expo, risk_expo

    def range_attribute(self, start_date, end_date, data_src_name='xy', bchmrk_name=None):
        """在一个时间区间进行归因分析"""
        bchmrk_name = self.benchmark_name if bchmrk_name is None else bchmrk_name
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        ret_ptf = self.portfolio_return.loc[dates]
        barra_expo, indus_expo, risk_expo = self.portfolio_risk_expo(data_src_name, dates=dates)
        risk_model = RiskModelAttribution(ret_ptf, barra_expo, indus_expo, bchmrk_name, data_src_name)
        return risk_model.range_attribute(start_date, end_date)

    def range_attribute_from_strategy(self, sm, strategy_name, start_date, end_date, data_src_name='xy', bchmrk_name=None):
        bchmrk_name = self.benchmark_name if bchmrk_name is None else bchmrk_name
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        ret_ptf = self.portfolio_return.loc[dates]
        barra_expo, indus_expo = sm.import_risk_expo(start_date=start_date, end_date=end_date, strategy_name=strategy_name,
                                                     data_source=data_src_name, bchmrk_name=bchmrk_name)
        risk_model = RiskModelAttribution(ret_ptf, barra_expo, indus_expo, bchmrk_name, data_src_name)
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

    def get_dividends(self):
        """导出分红记录"""
        if os.path.isfile(os.path.join(self.rootdir, 'dividends.pkl')):
            d = pd.read_pickle(os.path.join(self.rootdir, 'dividends.pkl'))
            if not d.empty:
                d['order_book_id'] = d['order_book_id'].apply(uqercode_to_windcode)
                d['trading_date'] = pd.DatetimeIndex(d['trading_date'])
            return d
        else:
            return pd.DataFrame(columns=['trading_date', 'order_book_id', 'dividends'])

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


def add_bckmrk_ret(func):
    from alphalens.utils import get_forward_returns_columns
    @wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]
        if obj._bchmrk_return is None:
            dates = obj._cf_and_fr.index.get_level_values('date').unique().tolist()
            windows = get_forward_returns_columns(obj._cf_and_fr.columns)
            bchmrk_return = data_source.get_forward_ndays_return(
                ids=[obj._benchmark], windows=windows, freq=obj._freq, dates=dates, type='index')
            obj._bchmrk_return = bchmrk_return.reset_index('IDs', drop=True)
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


class AlphalensAnalyzer(object):
    """基于alphalens工具包进行业绩分析
    初始化时的输入变量是alphalens.utils.get_clean_factor_and_forward_return函数的输出
    """
    def __init__(self, clean_factor_and_forward_return, freq='1d', benchmark=None):
        self._cf_and_fr = clean_factor_and_forward_return
        self._freq = freq
        self._benchmark = benchmark
        self._bchmrk_return = None

    def get_quantile_return(self, by_group=False, demeaned=True, by_date=False,
                            group_adjust=False):
        """计算分组的平均收益率

        Parameters:
        ----------------------
        by_group: bool
            是否分组计算收益
        demeaned: bool
            把收益率转换成超额收益，其中基准收益是每组股票的算数平均收益率
        by_date: bool
            是否按照每个日期分别计算收益
        group_adjust: bool
            是否按组转换成超额收益
        """
        from alphalens.performance import mean_return_by_quantile
        return mean_return_by_quantile(self._cf_and_fr, by_date, by_group, demeaned,
                                       group_adjust)[0]
    
    @add_bckmrk_ret
    def get_excess_return(self, by_group=False, by_date=False):
        """计算分组下的超额收益率

        Parameters:
        -----------------------
        by_group: bool
            是否分组计算收益
        by_date: bool
            是否分时间计算收益
        """
        from alphalens.utils import get_forward_returns_columns
        from alphalens.performance import mean_return_by_quantile
        cf_and_fr = self._cf_and_fr.copy()
        return_columns = get_forward_returns_columns(cf_and_fr.columns)
        cf_and_fr[return_columns] = cf_and_fr[return_columns].sub(self._bchmrk_return, level='date',
                                                                  axis='index')
        return mean_return_by_quantile(cf_and_fr, by_group=by_group, by_date=by_date,
                                       demeaned=False)[0]

    def get_long_short_return(self, by_group=False, by_date=False, factor_direction=1):
        """计算第一组最后一组的多空收益

        Parameters:
        -----------------------------
        by_group: bool
            是否分组计算收益
        by_date: bool
            是否按照每个日期分别计算收益
        factor_direction: int
            因子的方向, 1代表正向，-1代表负向。
        """
        mean_return = self.get_quantile_return(by_group, by_date=by_date)
        if factor_direction == 1:
            long_quantile = mean_return.index.get_level_values('factor_quantile').max()
            short_quantile = 1
        else:
            long_quantile = 1
            short_quantile = mean_return.index.get_level_values('factor_quantile').max()
        ls = mean_return.xs(long_quantile, level='factor_quantile') - \
             mean_return.xs(short_quantile, level='factor_quantile')
        return ls

    def get_icir(self, by_group=False, by_date=True):
        """计算因子的ICIR

        Parameters:
        -----------------------------
        by_group: bool
            是否分组计算ICIR
        by_date: bool
            是否分日期计算
        """
        from alphalens.performance import factor_information_coefficient
        ic_series = factor_information_coefficient(self._cf_and_fr, by_group=by_group)
        if not by_date:
            return ic_series.groupby('group').mean()
        return ic_series

    def plot_ic_series(self, by_group=False, by_date=True, ax=None):
        """绘制IC时间序列图
        """
        from alphalens.plotting import plot_ic_ts, plot_ic_by_group
        ic_ts = self.get_icir(by_group, by_date)
        if by_group:
            return plot_ic_by_group(ic_ts, ax=ax)
        else:
            return plot_ic_ts(ic_ts, ax)
        


if __name__ == '__main__':
    analyzer = Analyzer(r"D:\data\factor_investment_strategies\兴业风格_价值\backtest\BTresult.pkl",
                        benchmark_name='000905')
    # ff = FactorAnalyzer(r"D:\factors\全市场_过去12个月\bp_divide_median\bp_divide_median.pkl")
    a = analyzer.portfolio_weights('20170830')