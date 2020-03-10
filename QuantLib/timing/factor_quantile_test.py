# -*- coding: utf-8 -*-
#
# factor_quantile_test.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2020/3/1 下午2:23:15
from empyrical.stats import cum_returns_final
from statsmodels.tsa import stattools as tsa

import pandas as pd
import numpy as np


class ThreeQuantilesTest(object):
    """择时因子检验框架

    该框架测试的一般步骤如下:
    1. 因子的平稳性检验
    2. 样本内预测能力。通过观察全样本，看当期指标对下一期市场收益是否具有显著
       影响能力。考虑到经济数据的非正态性，我们使用三分位点法和领先-滞后分析法。
    3. 样本外预测能力。我们对下一期进行预测时只依据预测之前的数据，因而规避了前视
       数据的问题。我们仍然使用三分位点法进行分析。
    """
    def __init__(self, factor, benchmark_return):
        self.factor = factor
        self.benchmark_return = benchmark_return
    
    def adf_test(self, series):
        """对序列进行平稳性检验，返回adf检验的P值"""
        rslt = tsa.adfuller(series, regression='ct')
        return rslt[1]
    
    def test_insample(self, forward_days=20, shift=1, print_result=False):
        """样本内检验
        通过观察全样本数据进行三分位点法分析.

        Parameters:
        -----------
        forward_days: int
            向前预测区间长度
        shift: int
            预测变量滞后长度
        
        Returns: Dict
        --------
        {'t':    T值, 负值代表自变量与因变量负相关；反之亦然。
         'nobs': 有效样本数量
         'F1':   自变量低分位组对应因变量的均值
         'F3':   自变量高分位组对应因变量的均值
         'adf_pvalue'： adf检验P值；小于0.05时，拒绝原假设，认为序列平稳。
         'F1_Series': 自变量低分位组对应的因变量序列。
         'F3_Series': 自变量高分位组对应的因变量序列。}
        """
        q = 3
        if shift > 0:
            factor = self.factor.shift(shift).dropna()
        else:
            factor = self.factor.dropna()
        
        if forward_days > 1:
            returns = self.benchmark_return.rolling(
                forward_days).agg(cum_returns_final).shift(-forward_days).dropna()
        else:
            returns = self.benchmark_return.shift(-forward_days).dropna()
        
        factor, returns = factor.align(returns, axis='index', join='inner')

        adf = self.adf_test(factor)

        labels = pd.qcut(factor, q=q, labels=list(range(1, q+1)))
        nobs = labels.value_counts()
        stats = returns.groupby(labels).agg(
            ['mean', 'std']
        )
        n1, n3 = nobs[1], nobs[3]
        s1, s3 = stats.at[1., 'std'], stats.at[3., 'std']
        f1, f3 = stats.at[1., 'mean'], stats.at[3., 'mean']

        t = (f1 - f3) / \
            (
                ((n1-1)*s1**2+(n3-1)*s3**2)/\
                (n1+n3-2) * (1/n1 + 1/n3)
            ) ** 0.5
        
        rslt = {'t': t, 'nobs': nobs.sum(), 'F1': f1, 'F3': f3, 'adf_pvalue': adf,
                'F1_Series': returns[labels==1],
                'F3_Series': returns[labels==3]}
        if print_result:
            print(
                """样本内检验结果\n
                adf_p: %.2f\n
                T值：%.2f, 样本量: %d \n
                F1 : %.2f, F3: %.2f
                """%(adf, t, rslt['nobs'], f1, f3)
            )
        return rslt

    def test_outsample(self, forward_days=20, shift=1, rolling_periods=None):
        """
        样本外检验(滚动样本计算分位数)

        Parameters:
        -----------
        forward_days: int
            向前预测区间长度
        shift: int
            预测变量滞后长度
        rolling_periods: int
            滚动计算自变量分位数的样本量
        
        Returns: Dict
        --------
        {'t':    T值, 负值代表自变量与因变量负相关；反之亦然。
        'nobs': 有效样本数量
        'F1':   自变量低分位组对应因变量的均值
        'F3':   自变量高分位组对应因变量的均值
        'adf_pvalue'： adf检验P值；小于0.05时，拒绝原假设，认为序列平稳。
        'F1_Series': 自变量低分位组对应的因变量序列。
        'F3_Series': 自变量高分位组对应的因变量序列。}
        """
        q = 3
        if shift > 0:
            factor = self.factor.shift(shift).dropna()
        else:
            factor = self.factor.dropna()
        
        if forward_days > 1:
            returns = self.benchmark_return.rolling(
                forward_days).agg(cum_returns_final).shift(-forward_days).dropna()
        else:
            returns = self.benchmark_return.shift(-forward_days).dropna()
        
        adf = self.adf_test(factor)
        
        if rolling_periods is not None:
            labels = factor.rolling(rolling_periods, min_periods=q).apply(
                lambda x: pd.qcut(x, q, labels=list(range(1, q+1)))[-1], raw=True
            )
        else:
            labels = factor.expanding(min_periods=q).apply(
                lambda x: pd.qcut(x, q, labels=list(range(1, q+1)))[-1], raw=True
            )
        labels.dropna(inplace=True)
        labels, returns = labels.align(returns, axis='index', join='inner')

        nobs = labels.value_counts()
        stats = returns.groupby(labels).agg(
            ['mean', 'std']
        )
        n1, n3 = nobs[1], nobs[3]
        s1, s3 = stats.at[1., 'std'], stats.at[3., 'std']
        f1, f3 = stats.at[1., 'mean'], stats.at[3., 'mean']

        t = (f1 - f3) / \
            (
                ((n1-1)*s1**2+(n3-1)*s3**2)/\
                (n1+n3-2) * (1/n1 + 1/n3)
            ) ** 0.5
        
        rslt = {'t': t, 'nobs': nobs.sum(), 'F1': f1, 'F3': f3, 'adf_pvalue': adf,
                'F1_Series': returns[labels==1],
                'F3_Series': returns[labels==3]}
        print(
            """样本外检验结果
            adf_p: %.2f\n
            T值：%.2f, 样本量: %d \n
            F1 : %.2f, F3: %.2f
            """%(adf, t, rslt['nobs'], f1, f3)
        )
        return rslt
        