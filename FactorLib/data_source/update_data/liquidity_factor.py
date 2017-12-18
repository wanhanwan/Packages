"""流动性因子"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import QuantLib as qlib
import numba
from numba import float64


# 市值调整换手率
def turnover_adjust_by_float_mv(start, end, **kwargs):
    all_days = kwargs['data_source'].trade_calendar.get_trade_days(start, end)
    float_mkt_value = kwargs['data_source'].load_factor("float_mkt_value", '/stocks/', dates=all_days)
    float_mkt_value['float_mkt_value'] = np.log(float_mkt_value['float_mkt_value'])
    turnover = kwargs['data_source'].load_factor("turn", '/stock_liquidity/', dates=all_days)
    
    # %% 两个因子合并在一个数据框里面
    data_frame = pd.merge(float_mkt_value, turnover, left_index=True,
                          right_index=True, how='outer').reset_index()
    
    # %%数据标准化
    zsz_drop_outlier = qlib.DropOutlier(data_frame, 'float_mkt_value', method='BoxPlot')
    zsz_drop_outlier = zsz_drop_outlier.reset_index()
    zsz_standard = qlib.Standard(zsz_drop_outlier, 'float_mkt_value_after_drop_outlier')
    turnover_drop_outlier = qlib.DropOutlier(data_frame, 'turn', method='BoxPlot')
    turnover_drop_outlier = turnover_drop_outlier.reset_index()
    turnover_standard = qlib.Standard(turnover_drop_outlier, 'turn_after_drop_outlier')
    
    # %%时间截面上计算市值调整后的换手率
    data_frame_processed = pd.concat([turnover_standard, zsz_standard],
                                     join='inner', axis=1)
    turnover_adjustby_zsz = data_frame_processed.groupby(level=0)[
        ['float_mkt_value_after_drop_outlier_after_standard',
             'turn_after_drop_outlier_after_standard']].apply(_resid_ols)
    
    # %%存储数据
    turnover_adjustby_zsz.columns = ['turnover_adjust_total_mkt_value']
    kwargs['data_source'].h5DB.save_factor(turnover_adjustby_zsz, '/stock_liquidity/')    
    
def _resid_ols(data):
    notNanInd = pd.notnull(data).all(axis=1)
    y = data.iloc[:, 1].values
    x = data.iloc[:, 0].values
    x = sm.add_constant(x)
    ols = sm.OLS(y, x, missing='drop').fit()
    data['resid'] = np.nan
    data.loc[notNanInd, 'resid'] = ols.resid
    return data[['resid']]


@numba.jit(float64(float64[:], float64), nogil=True)
def _avg_turn2(datanp, valid_prc=0.8):
    new_arr = np.zeros(len(datanp), dtype=np.float64)
    for x in np.arange(60, len(datanp)):
        if np.isnan(datanp[x-60:x]).sum() / len(datanp[x-60:x]) > 1 - valid_prc:
            new_arr[x-1] = np.nan
        else:
            new_arr[x-1] = np.nanmean(datanp[x-60:x])
    return new_arr


def _avg_turn3(df):
    new = _avg_turn2(df.values, 0.8)
    return new

def avg_turnover(start, end, **kwargs):
    """60日日均换手率"""
    datasource = kwargs['data_source']
    start = datasource.trade_calendar.tradeDayOffset(start, -60)
    trade_days = datasource.trade_calendar.get_trade_days(start, end)
    turn = datasource.h5DB.load_factor('turn', '/stock_liquidity/', dates=trade_days)
    avg_turn = turn.groupby(level=1)['turn'].transform(_avg_turn3)
    datasource.h5DB.save_factor(avg_turn[avg_turn!=0].to_frame('turn_60d'), '/stock_liquidity/')

LiquidityFuncListDaily = [turnover_adjust_by_float_mv, avg_turnover]
LiquidityFuncListMonthly = []


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    avg_turnover('20070101', '20171215', data_source=data_source)