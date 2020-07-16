#!python
# -*- coding: utf-8 -*-
#
# calculate_return.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/5/22 上午10:15:14
import numpy as np
import pandas as pd
from typing import Tuple
from numpy import ndarray
from FactorLib.data_source.tseries import move_dtindex
from FactorLib.data_source.base_data_source_h5 import h5_2
from FactorLib.data_source.trade_calendar import traderule_alias_mapping

FQCLOSE = 'tushare_close_hfq'
FQOPEN = 'tushare_open_hfq'
BDAY = traderule_alias_mapping['d'] * 1


def adjust_trade(
    w0: ndarray,
    trade: ndarray,
    unable_buy: ndarray,
    unable_sell: ndarray
):
    """
    根据无法买入和卖出的股票列表调整
    原始交易清单。
    """
    trade_direction = trade - w0
    idx = np.logical_or((unable_buy == 1) & (trade_direction > 0.0),
                        (unable_sell == 1) & (trade_direction < 0.0)
    )
    new_trade = np.where(
        idx,
        w0,
        trade
    )
    new_trade[~idx] /= np.extract(~idx,new_trade).sum() / (1.0 - np.extract(idx,w0).sum())
    return new_trade


def calculate_portfolio_return_and_weights(
    trade_table: ndarray,
    return_chunks: Tuple[ndarray],
    unable_buy: ndarray,
    unable_sell: ndarray,
    commission: float = 0.0
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    根据调仓记录计算组合回报和每日资产权重

    计算资产权重时考虑到了被动仓位的影响。

    Parameters:
    -----------
    trade_table: 2D-Array
        每一行是资产截面权重，代表一次交易。
    return_chunks: Tuple
        每个元素是一个2D-Array, 代表两个交易日之间资产的收益率序列。
    unable_buy: 2D-Array
        行数与trade_table相同，数值非0即1，代表交易当日不能买入的股票。
    unable_sell: 2D-Array
        行数与trade_table相同，数值非0即1，代表交易当日不能卖出的股票。
    commission: float
        双边交易费率。

    Returns：
    --------
    weights: 2D-Array
        每日资产的期初权重。注意，这里是每日的期初权重，不包含当日的涨跌。
    returns: 1D-Array
        组合每日收益率，returns = weights * asset_daily_returns.
    turnovers: 1D-Array
        每个交易日的换手率(双边)
    """
    if not (
        trade_table.shape[0]
        == len(return_chunks)
        == unable_buy.shape[0]
        == unable_sell.shape[0]
    ):
        raise RuntimeError("时间维度不一致!")

    if not (trade_table.shape[1] == return_chunks[0].shape[1] ==
        unable_buy.shape[1] == unable_sell.shape[1]):

        raise RuntimeError("资产数量维度不一致!")
    
    T = sum([x.shape[0] for x in return_chunks])
    weights = np.zeros((T, trade_table.shape[1]), dtype='float64')
    returns = np.zeros(T, dtype='float64')
    turnovers = np.zeros(trade_table.shape[0], 'float64')
    w0 = np.zeros(trade_table.shape[1], 'float64')

    n = 0 # for counting
    for i in range(trade_table.shape[0]):
        t_i = return_chunks[i].shape[0]

        w1 = adjust_trade(w0, trade_table[i,:], unable_buy[i,:], unable_sell[i,:])
        weights_i = w1 * np.add(return_chunks[i], 1.0).cumprod(axis=0)
        weights_i /= np.sum(weights_i, axis=1)[:, None]

        weights[n, :] = w1
        weights[n+1:n+t_i, :] = weights_i[:-1]
        
        returns[n:n + t_i] = np.sum(weights[n:n+t_i, :] * return_chunks[i], axis=1)
        turnovers[i] = np.sum(np.abs(w1 - w0))
        returns[n] -= turnovers[i] * commission

        w0 = weights_i[-1, :]
        n += t_i
    
    return weights, returns, turnovers


def backtest_portfolio(portfolio: pd.Series,
                       portfolio_shift_days: int,
                       prices: pd.DataFrame,
                       prices_shift_days: int,
                       unable_buy: pd.DataFrame=None,
                       unable_sell: pd.DataFrame=None,
                       commission=0.0):
    """
    一个投资组合的回测

    Parameters:
    ----------
    portfolio: Series
        组合截面权重, Series(index=[date, IDs], values=weights)
    portfolio_shift_days: int
        出于实操的考虑，组合往往在持仓生成后的下一个交易日下单，并且下单当日的回报是根据
        调整前的权重还是调整后的权重计算，也跟使用什么类型的价格有关。如果用收盘价或者均
        价，那么下单当日回报应该用调整前权重，此时portfolio_shift_days=2;如果用开盘价
        ，那么下单当日回报应该用调整后权重，此时portfolio_shift_days=1.
    prices_shift_days: int
        如果价格是开盘价，当日回报=下一日开盘价/当日开盘价-1，因此price_shift_days=-1.
        其他情况下, price_shift_days=0.
    unable_buy: DataFrame
        交易当日不可买入的股票, DataFrame(index=date, columns=IDs, value= 0 or 1)
    unable_sell: DataFrame
        交易当日不可卖出的股票, DataFrame(index=date, columns=IDs, value= 0 or 1)
    commission: float
        双边交易费率
    """
    portfolio = portfolio.unstack().fillna(0.0)

    if unable_buy is None:
        unable_buy = pd.DataFrame(0, index=portfolio.index+BDAY, columns=portfolio.columns)
    else:
        unable_buy = unable_buy.reindex(index=portfolio.index + BDAY, columns=portfolio.columns, fill_value=1.0)

    if unable_sell is None:
        unable_sell = pd.DataFrame(0, index=unable_buy.index, columns=portfolio.columns)
    else:
        unable_sell = unable_sell.reindex(index=unable_buy.index, columns=portfolio.columns, fill_value=1.0)

    if portfolio_shift_days != 0:
        portfolio = move_dtindex(portfolio, portfolio_shift_days, '1d')

    return_table = (
        prices
        .pct_change(fill_method=None)
        .fillna(0.0)
    )
    if prices_shift_days != 0:
        return_table = move_dtindex(return_table, prices_shift_days, '1d')

    idx = np.where(portfolio.index>=return_table.index.min())[0][0]
    start_date = portfolio.index[idx]
    unable_buy = unable_buy.iloc[idx:, :]
    unable_sell = unable_sell.iloc[idx:, :]

    end_date = return_table.index.max()
    portfolio = portfolio.loc[start_date:end_date]
    unable_buy = unable_buy.iloc[:len(portfolio), :]
    unable_sell = unable_sell.iloc[:len(portfolio), :]

    return_table = return_table.loc[start_date:end_date, portfolio.columns]

    split_idx = np.searchsorted(return_table.index, portfolio.index, side='left')
    return_chunks = np.split(return_table.to_numpy(), split_idx, axis=0)[1:]

    weights, returns, turnovers = calculate_portfolio_return_and_weights(
        portfolio.to_numpy(),
        return_chunks,
        unable_buy.to_numpy(),
        unable_sell.to_numpy(),
        commission
    )
    weights = pd.DataFrame(weights, index=return_table.index, columns=return_table.columns)
    returns = pd.Series(returns, index=return_table.index, name='returns')
    turnovers = pd.Series(turnovers, index=portfolio.index, name='turnovers')

    return weights, returns, turnovers


def backtest_stock_portfolio(portfolio: pd.Series, price_type='open', commission=0.0):
    """
    股票组合的回测

    Parameters:

    -----------
    portfolio: Series
        股票组合, Series(index=[date,IDs], values=weight)
    
    price_type: str
        open\close\vwap
    
    commission: float
        交易费用

    Returns:
    --------

    weights: DataFrame
        日频持仓权重：DataFrame(index=date, columns=IDs)

    returns: Series
        组合收益率
    
    turnovers: Series
        组合换手率

    """
    all_stocks = portfolio.index.get_level_values(level=1).unique().tolist()
    if price_type == 'close':
        prices = h5_2.load_factor2(FQCLOSE, '/base/stock_prices/', ids=all_stocks)
        portfolio_shift_periods = 2
        returns_shift_periods = 0
    elif price_type == 'open':
        prices = h5_2.load_factor2(FQOPEN, '/base/stock_prices/', ids=all_stocks)
        portfolio_shift_periods = 1
        returns_shift_periods = -1
    else:
        raise ValueError(f"Not supported price type: {price_type}")

    trade_status = h5_2.load_factors3(
        {'/base/stock_prices/':['go_up_limit', 'go_down_limit', 'suspend']},
        dates = prices.index.tolist(),
        ids = prices.columns.tolist()
    )
    unable_buy = trade_status[['suspend', 'go_up_limit']].max(axis=1).unstack(fill_value=1)
    unable_sell = trade_status[['suspend', 'go_down_limit']].max(axis=1).unstack(fill_value=1)

    return backtest_portfolio(
        portfolio,
        portfolio_shift_periods,
        prices,
        returns_shift_periods,
        unable_buy,
        unable_sell,
        commission
    )
