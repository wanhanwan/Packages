# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:11:06 2017
构造一个函数，输入单因子数据，输出合并因子。
合并方法参照海通单因子IC加权
@author: wanshuai
"""

import pandas as pd
import numpy as np
import QuantLib as qlib
from QuantLib.factor_validation import cal_ic

def combine_factors_by_ic(factor_list):
    '''通过IC最优加权，将单因子合并成复合因子

    输入
    ---------------------------
    factor_list:list of DataFrames

    '''
    factor_frame = pd.concat(factor_list, axis=1, join='inner')
    all_dates_of_factor = factor_frame.index.levels[0]    

    # 标准化数据
    factor_standard_frame = factor_frame.apply(
        lambda x: qlib.Standard(x.reset_index(), x.name)[x.name+'_after_standard'])

    # 计算ic值
    ic_frame = factor_standard_frame.apply(
        lambda x: cal_ic(x.to_frame(), x.name, 18).iloc[:, 0])

    # 计算因子相关性矩阵
    factor_cov = factor_standard_frame.groupby(level=0).cov()

    # 计算因子协方差矩阵、ic序列的n月移动平均
    month_ma_window = 24 * 1
    factor_cov_ma = factor_cov.unstack().rolling(month_ma_window,
                                                 min_periods=1).mean()
    factor_cov_ma = factor_cov_ma.stack()

    ic_ma_frame = ic_frame.rolling(month_ma_window,
                                   min_periods=1).mean().reindex(all_dates_of_factor)
    ic_ma_frame = ic_ma_frame.shift(1)

    # 计算负荷因子
    compound_factors = []
    factor_weight = []
    dates = ic_ma_frame.index.tolist()

    for iDate in dates[1:]:
        w = np.dot(np.linalg.inv(factor_cov_ma.ix[iDate]),
                   ic_ma_frame.ix[iDate])
        cf = np.dot(factor_standard_frame.ix[iDate], w.T)

        i_index = pd.MultiIndex.from_product([[iDate],
                                              factor_standard_frame.ix[iDate].index],
                                             names=['Date', 'IDs'])
        cf = pd.Series(cf, index=i_index)

        i_index2 = pd.MultiIndex.from_product([[iDate],
                                               factor_frame.columns])
        w = pd.Series(w, index=i_index2)

        compound_factors.append(cf)
        factor_weight.append(w)

    compound_factors = pd.concat(compound_factors)
    factor_weight = pd.concat(factor_weight)

    factor_weight = factor_weight.unstack()
    compound_factors = compound_factors.reset_index()
    compound_factors.rename(columns={0: 'compound_factor'}, inplace=True)

    return factor_weight, compound_factors
