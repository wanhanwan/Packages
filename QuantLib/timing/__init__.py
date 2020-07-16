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

    def __init__(self, df_signal, benchmark,
                 real_signal=None):
        if isinstance(df_signal, pd.Series):
            self._df_signal = df_signal.to_frame()
        else:
            self._df_signal = df_signal
        self._discret_signal = None
        
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

    @property
    def discret_signal(self):
        if self._discret_signal is None:
            self._discret_signal = np.sign(self._df_signal)
        return self._discret_signal

    @property
    def real_signal(self):
        return self._real_signal

    @property
    def benchmark_return(self):
        return self._benchmark

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
            correct = (x != 0.0) & (x == self._real_signal.reindex(x.index))
            if np.sum(x.to_numpy() != 0.0):
                return np.sum(correct.to_numpy()) / np.sum(x.to_numpy() != 0.0)
            return 0.0
        
        def _pnl(x):
            r = x * self._benchmark.reindex(x.index)
            return -np.mean(r[r > 0].to_numpy()) / np.mean(r[r < 0].to_numpy())
        
        functions = [
            up_times, down_times, flat_times,
            _win_rate_up, _win_rate_down, _win_rate,
            _pnl
        ]

        res = self.discret_signal.groupby(
            lambda x: x.year
        ).agg(functions)
        res.columns = res.columns.set_levels(['Up', 'Down', 'Flat', 'WRU', 'WRD', 'WR', 'PNL'],
                                             level=1)
        res_all_sample = self.discret_signal.agg(functions)
        res_all_sample.index = res.columns.levels[1]
        res_all_sample = res_all_sample.unstack().to_frame('全样本').T
        res = res.append(res_all_sample)
        res.columns = res.columns.swaplevel()
        res = res[['Up','Down','Flat','WR','PNL','WRU','WRD']]

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
        cost = self.discret_signal.diff().fillna(0.0).abs() * commission
        ls_ret = self.discret_signal.mul(self._benchmark, axis='index') - cost

        ir_df = []
        for name, series in ls_ret.iteritems():
            a = NavAnalyser(series, self._benchmark, risk_free_rate=0.02)
            ir = a.yearly_report['信息比率'].append(a.report['信息比率'])
            ir_df.append(ir)
        ir_df = pd.concat(ir_df, axis=1, keys=ls_ret.columns)
        return ls_ret, ir_df

    def long_only_performance(self, commission=0.0):
        """
        多头组合的表现

        Return:
        -------
            ls_ret: DataFrame
                多头收益时间序列
            ir_df: DataFrame
                多头组合IR
        """
        from QuantLib.portfolio import NavAnalyser
        signals = self.discret_signal.replace(-1.0, 0)
        cost = signals.diff().fillna(0.0).abs() * commission
        ls_ret = signals.mul(self._benchmark, axis='index') - cost

        ir_df = []
        for name, series in ls_ret.iteritems():
            a = NavAnalyser(series, self._benchmark, risk_free_rate=0.02)
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
            res[name] = b.groupby(score_i).agg(['mean', 'std'])
        return res

    def write_report(self, report_save_path='SignalCompare.xlsx'):
        from QuantLib.tools import return2nav
        # 胜率和盈亏比
        wr_pnl = self.win_rate_and_pnl()
        # 多空组合回报
        ls_ret, ir = self.long_short_performance(0.001)
        ls_nav = return2nav(ls_ret)
        ls_nav = ls_nav.join(
            return2nav(
                self._benchmark.reindex(
                    ls_ret.index, fill_value=0.0).to_frame('Benchmark')
                       ),
            how='left'
        )
        # 多头组合回报
        long_only_ret, long_only_ir = self.long_only_performance(0.001)
        long_only_nav = return2nav(long_only_ret)
        long_only_nav = long_only_nav.join(
            return2nav(
                self._benchmark.reindex(
                    long_only_ret.index, fill_value=0.0).to_frame('Benchmark')
            ),
            how='left'
        )
        # 平均回报
        average_return = self.average_return_with_different_scores()

        with pd.ExcelWriter(report_save_path, engine='xlsxwriter',
                            datetime_format='yyyy/mm/dd') as writer:
            wr_pnl.style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='胜率与盈亏比', float_format='%.3f')

            ls_nav.style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='多空组合', float_format='%.3f')
            ir.style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='多空组合', float_format='%.3f',
                         startcol=ls_nav.shape[1]+3)
            writer.sheets['多空组合'].write(0, ls_nav.shape[1]+3, 'IR')
            ls_ret.style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='多空组合', float_format='%.3f',
                         startcol=ls_nav.shape[1]*2+4)
            writer.sheets['多空组合'].write(0, ls_nav.shape[1]*2+4, '多空回报')

            long_only_nav.style.set_properties(**{'text-align': 'center'}). \
                to_excel(writer, sheet_name='多头组合', float_format='%.3f')
            long_only_ir.style.set_properties(**{'text-align': 'center'}). \
                to_excel(writer, sheet_name='多头组合', float_format='%.3f',
                         startcol=long_only_nav.shape[1] + 3)
            writer.sheets['多头组合'].write(0, long_only_nav.shape[1] + 3, 'IR')
            long_only_ret.style.set_properties(**{'text-align': 'center'}). \
                to_excel(writer, sheet_name='多头组合', float_format='%.3f',
                         startcol=long_only_nav.shape[1] * 2 + 4)
            writer.sheets['多头组合'].write(0, long_only_nav.shape[1] * 2 + 4, '多头回报')

            for i, (name, avg_ret) in enumerate(average_return.items()):
                avg_ret.to_excel(writer, sheet_name='分数-平均回报',
                                 float_format='%.3f', startcol=i*4)
                writer.sheets['分数-平均回报'].write(0, i*4, name)

            self.discret_signal.join(
                self.real_signal.rename('真实信号'),
                how='left'
            ).style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='信号详情', float_format='%.0f')
            self.benchmark_return.reindex(self.discret_signal.index).\
                to_frame('基准指数回报').\
                style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='信号详情',
                         startcol=self.discret_signal.shape[1]+2,
                         float_format='%.3f',
                         index=False)
            self._df_signal.style.set_properties(**{'text-align': 'center'}).\
                to_excel(writer, sheet_name='信号详情', float_format='%.3f',
                         startcol=self.discret_signal.shape[1]+4)
            writer.sheets['信号详情'].write(0, self.discret_signal.shape[1]+4, '原始信号')
