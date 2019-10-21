#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nav_calculator.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2019/8/20 下午3:53:36
import numpy as np
import pandas as pd
from QuantLib.tools import return2nav


def _get_rebalancing_asset(i, rebalancing_periods):
    return rebalancing_periods[
        i%rebalancing_periods==0].index.tolist()


def _get_init_weight(weight, dt, asset):
    if isinstance(weight, pd.Series):
        return weight[asset]
    else:
        loc = weight.columns.get_loc(asset)
        return weight.loc[:dt].iat[-1, loc]


def _buy(asset_name, wt, weights, cash_name, leverage_name,
         interest_name, credit_name):
    if wt > 0:
        cash_available = max(0, weights[cash_name] - 0.05)
        if leverage_name is not None:
            leverage_available = max(
                (weights[interest_name] + weights[credit_name]*0.7)-abs(weights[leverage_name]),
                0.0
                )
        else:
            leverage_available = 0.0
        wt = min(wt, cash_available+leverage_available)
        leverage_need = max(0, wt-cash_available)
        if leverage_need > 0.0:
            weights[leverage_name] -= leverage_need
            weights[cash_name] -= cash_available
        else:
            weights[cash_name] -= wt
        weights[asset_name] += wt
    else:
        if leverage_name is not None:
            weights[leverage_name] -= wt
            if weights[leverage_name] > 0.0:
                weights[cash_name] += weights[leverage_name]
                weights[leverage_name] = 0.0
        weights[asset_name] += wt


def _fast_forward(weights, ret):
    total_ret = sum((weights[x]*ret[x] for x in weights))
    divider = sum((weights[x]*(1+ret[x]) for x in weights))
    new_weights = {x: weights[x]*(1+ret[x])/divider for x in weights}
    return total_ret, new_weights


def _calc_order(curr_w, w, order, assets=None):
    if assets is None:
        assets = list(w)
    for k in assets:
        if k in assets:
            order[k] = w[k] - curr_w[k]


def get_nav_and_weights(ret_df, asset_weight, rebalance_periods,
            cash_name, leverage_name=None, interest_name=None,
            credit_name=None):
    """计算净值

    Parameters:
    ==========
    ret_df: DataFrame
        收益率序列
    weight: Series or DataFrame
        权重序列，Series会广播到整个时间段.
    rebalance_periods: int or dict
        再平衡时间段
    """
    ret_p = pd.Series(np.zeros(ret_df.shape[0]), index=ret_df.index)
    weights = {}
    all_assets = ret_df.columns.tolist()

    if isinstance(rebalance_periods, int):
        rebalance_periods = pd.Series(
            np.ones(ret_df.shape[1]) * rebalance_periods,
            index=ret_df.colunms
        )
    else:
        rebalance_periods = pd.Series(rebalance_periods)

    if isinstance(asset_weight, pd.Series):
        curr_weights = asset_weight.to_dict()
    else:
        curr_weights = asset_weight.iloc[0, :].to_dict()

    for i, (dt, ret) in enumerate(ret_df.iterrows()):
        i_ret, curr_weights = _fast_forward(curr_weights, ret)
        ret_p.iat[i] = i_ret
        weights[dt] = curr_weights.copy()

        # 主动调仓和再平衡同时进行，若两个存在冲突，主动调仓优先级更高
        init_weight = _get_init_weight(asset_weight, dt, all_assets)
        order = {}
        rebalancing_assets = _get_rebalancing_asset(i+1, rebalance_periods)
        if leverage_name is not None:
            init_weight.drop(leverage_name, inplace=True)
            rebalancing_assets = [x for x in rebalancing_assets if x!=leverage_name]
        _calc_order(curr_weights, init_weight, order, rebalancing_assets)
        if isinstance(asset_weight, pd.DataFrame) and dt in asset_weight.index:
            _calc_order(curr_weights, init_weight, order)
        
        sell_assets = [k for k in order if order[k]<0]
        for asset in sell_assets:
            _buy(asset, order[asset], curr_weights, cash_name, leverage_name,
                 interest_name, credit_name)

        buy_assets = [k for k in order if order[k]>0]
        for asset in buy_assets:
            _buy(asset, order[asset], curr_weights, cash_name, leverage_name,
                 interest_name, credit_name)

    nav = return2nav(ret_p)
    wt = pd.DataFrame(weights).T
    return nav, wt
