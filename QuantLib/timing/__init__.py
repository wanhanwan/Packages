#!python
# -*- coding: utf-8 -*-
#
# __init__.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/5/7 下午3:24:19
from QuantLib.timing.factor_quantile_test import ThreeQuantilesTest
from QuantLib.timing.utils import is_discrete_signal

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)


class TimingSignalTester(object):
    """
    择时信号的统一检验框架 
    """

    def __init__(self, df_signal, benchmark, real_signal=None):
        if isinstance(df_signal, pd.Series):
            self._df_signal = df_signal.to_frame()
        else:
            self._df_signal = df_signal
        self._discret_signal = np.sign(self._df_signal)
        
        self._benchmark = benchmark
        if real_signal is None:
            self._real_signal = np.sign(self._benchmark)
        else:
            self._real_signal = real_signal
        
        self._scored_signal = None
    
    @property
    def scored_signal(self):
        if self._scored_signal is None:
            discrete = is_discrete_signal(self._df_signal)
            self._scored_signal = self._df_signal.loc[:, ~discrete]
        return self._scored_signal
    
    def win_rate_and_pnl(self):
        up_times = lambda x: np.sum(x.to_numpy() == 1.0)
        down_times = lambda x: np.sum(x.to_numpy() == -1.0)
        flat_times = lambda x: np.sum(x.to_numpy() == 0.0)
        
        def _win_rate_up(x):
            correct = (x == 1.0) & (x == self._real_signal.reindex(x.index))
            if np.sum(x.to_numpy() == 1.0):
                return np.sum(correct.to_numpy()) / np.sum(x.to_numpy() == 1.0)
            else:
                return 0.0
        
        def _win_rate_down(x):
            correct = (x == -1.0) & (x == self._real_signal.reindex(x.index))
            if np.sum(x.to_numpy() == -1.0):
                return np.sum(correct.to_numpy()) / np.sum(x.to_numpy() == -1.0)
            return 0.0
        
        def _win_rate(x):
            correct = x == self._real_signal.reindex(x.index)
            return np.sum(correct.to_numpy()) / len(x)
        
        def _pnl(x):
            r = x * self._benchmark.reindex(x.index)
            return -np.mean(r[r > 0].to_numpy()) / np.mean(r[r < 0].to_numpy())
        
        functions = [
            up_times, down_times, flat_times,
            _win_rate_up, _win_rate_down, _win_rate,
            _pnl
        ]

        res = self._discret_signal.groupby(
            lambda x: x.year
        ).agg(functions)
        res.columns = res.columns.set_levels(['Up', 'Down', 'Flat', 'WRU', 'WRD', 'WR', 'PNL'],
                                             level=1)
        res_all_sample = self._discret_signal.agg(functions)
        res_all_sample.index = res.columns.levels[1]
        res_all_sample = res_all_sample.unstack().to_frame('全样本').T
        res = res.append(res_all_sample).sort_index(axis=1, level=1)

        return res

    def long_short_performance(self, commission=0.0):
        """
        多空组合的表现

        Return:
        -------
            ls_ret: DataFrame
                多空收益时间序列
            ir_df: DataFrame
                多空组合IR
        """
        from QuantLib.portfolio import NavAnalyser
        cost = self._discret_signal.diff().fillna(0.0) * commission * 2
        ls_ret = self._discret_signal.mul(self._benchmark, axis='index') - cost

        ir_df = []
        for name, series in ls_ret.iteritems():
            a = NavAnalyser(series, benchmark, risk_free_rate=0.02)
            ir = a.yearly_report['信息比率'].append(a.report['信息比率'])
            ir_df.append(ir)
        ir_df = pd.concat(ir_df, axis=1, keys=ls_ret.columns)
        return ls_ret, ir_df

    def average_return_with_different_scores(self):
        """计算在全样本范围内，每个信号的不同分值下的基准平均收益"""
        score = self.scored_signal
        res = {}
        b = self._benchmark.reindex(score.index)
        for name, score_i in score.iteritems():
            res[name] = b.groupby(score_i).mean()
        return res

    def write_report(self, report_save_path='SignalCompare.xlsx'):
        # 胜率和盈亏比
        wr_pnl = self.win_rate_and_pnl()

        # 多空组合回报
        ls_ret, ir = self.long_short_performance(0.001)

        # 平均回报
        average_return = self.average_return_with_different_scores()

        from StyleFrame import StyleFrame, Styler, utils
        with pd.ExcelWriter(report_save_path, engine='xlsxwriter',
                            datetime_format='yyyy/mm/dd') as writer:
            wb = writer.book
            

signal = pd.DataFrame(np.random.randint(-1, 2, (30, 2)),
                      index=pd.date_range('2010-01-31', periods=30, freq='1M'),
                      columns=['A', 'B'])
benchmark = pd.Series((np.random.rand(30) - 0.5) / 10, index=signal.index)
tester = TimingSignalTester(signal, benchmark)
print(tester.average_return_with_different_scores())

