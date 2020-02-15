#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nav_calculator.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2019/8/20 下午3:53:36
import numpy as np
import pandas as pd
from QuantLib.tools import return2nav


def _fast_forward(weights, ret):
    """
    weights: array
    ret: array
    """
    total_ret = np.sum(weights * ret)
    divider = np.sum(np.abs(weights)*(1.0+ret))
    new_weights = (np.abs(weights) * (1.0+ret) / divider) * np.sign(weights)
    return total_ret, new_weights


def get_nav_and_weights(rtn_df, rebalance_weights, init_weights):
    """计算净值

    Parameters:
    ==========
    ret_df: DataFrame
        收益率序列
    rebalance_weights: DataFrame
        权重序列
    init_weights: dict or Series
        初始权重
    """
    rtn_df = rtn_df.copy()
    rtn_df['rebalance'] = 0.0
    asset_names = rtn_df.columns
    if isinstance(init_weights, dict):
        weights = pd.Series(init_weights)
    else:
        weights = init_weights.copy()
    rebalance_weights = rebalance_weights.copy()
    rebalance_weights = rebalance_weights.reindex(columns=asset_names)
    # 加入平衡项，平衡项的收益为0，不影响结果。
    rebalance_weights['rebalance'] = 1.0 - rebalance_weights.sum(axis=1)
    weights = weights.reindex(index=asset_names)
    weights.at['rebalance'] = 1.0 - weights.sum()
#    weights = weights.to_numpy()
    
    rebalance_dates = rebalance_weights.index
    rtns_series = pd.Series(index=rtn_df.index)
    weights_df = pd.DataFrame(columns=asset_names, index=rtn_df.index)
    for dt, rtn in rtn_df.iterrows():
        if dt in rebalance_dates:
            weights = rebalance_weights.loc[dt, :].to_numpy()
        weights_df.loc[dt,:] = weights
        r, weights = _fast_forward(weights, rtn.to_numpy())
        rtns_series.at[dt] = r
    nav = return2nav(rtns_series)
    return nav, weights_df


def get_nav_and_weights2(rtn_df,
                         capital_weights,
                         init_capital_weights,
                         leverages,
                         init_leverages):
    """杠杆型资产(期货)的回测框架"""
    def _fast_forward2(ret, cw_t, l_t):
        total_ret = np.sum(ret * l_t)
        new_l_t = (cw_t*np.abs(l_t)*(1.0+ret))/(1.0+np.sum(cw_t*l_t*ret))*np.sign(l_t)
        new_cw_t = cw_t*(1.0+l_t*ret)/np.sum(cw_t*(1.0+l_t*ret))
        return total_ret, new_cw_t, new_l_t
        
    asset_names = rtn_df.columns
    if isinstance(init_capital_weights, dict):
        cw = pd.Series(init_capital_weights).reindex(asset_names).to_numpy()
    assert isinstance(capital_weights, pd.DataFrame)
    capital_weights = capital_weights.reindex(columns=asset_names, fill_value=0.0)
    if isinstance(leverages, (int,float)):
        leverages = pd.DataFrame(
                data=np.ones(capital_weights.shape)*leverages,
                index=capital_weights.index,
                columns=capital_weights.columns)
    else:
        assert isinstance(leverages, pd.DataFrame)
        leverages = leverages.reindex(columns=asset_names)
    if isinstance(init_leverages, dict):
        l = pd.Series(init_leverages).reindex(asset_names).to_numpy()
    elif isinstance(init_leverages, (int, float)):
        l = pd.Series(init_leverages, index=asset_names).to_numpy()
    else:
        assert isinstance(init_leverages, pd.Series)
        l = init_leverages.reindex(asset_names).to_numpy()
    
    rebalance_dates = capital_weights.index
    rtns_series = pd.Series(index=rtn_df.index)
    leverage_df = pd.DataFrame(columns=asset_names, index=rtn_df.index)
    capital_df = pd.DataFrame(columns=asset_names, index=rtn_df.index)
    for dt, rtn in rtn_df.iterrows():
        if dt in rebalance_dates:
            cw = capital_weights.loc[dt, :].to_numpy()
            l = leverages.loc[dt, :].to_numpy()
        leverage_df.loc[dt,:] = l
        capital_df.loc[dt, :] = cw
        r, cw, l = _fast_forward2(rtn.to_numpy(), cw, l)
        rtns_series.at[dt] = r
    nav = return2nav(rtns_series)
    return nav, capital_df, leverage_df


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    ret_df = pd.DataFrame((np.random.randn(1000, 2))/100.0,
                          index=pd.date_range('2010-01-01', periods=1000),
                          columns=list('ab'))
    weights_df = pd.DataFrame([[0.5,0.5]], index=[pd.to_datetime('2010-2-28')], columns=['a', 'b'])
    nav, weights = get_nav_and_weights(ret_df, weights_df, {'a': 0.5, 'b': 0.5})
    nav_test = return2nav(ret_df.mean(axis=1))
    pd.concat([nav, nav_test], axis=1).plot()
    plt.show()

