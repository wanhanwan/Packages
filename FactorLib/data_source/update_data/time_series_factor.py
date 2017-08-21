"""计算时间序列因子"""
import pandas as pd
from itertools import product
from WindPy import *


def _get_rebalance_dates(dates):
    '''
        calculate rebalance dates for the given dates
        rebalance date is the end of June for each year
    '''
    rebalabce_dates = []
    year_start_date = dates[0][:4]
    if dates[0] <= (year_start_date + '0630'):
        year_start = int(year_start_date) - 1
    else:
        year_start = int(year_start_date)
    year_end_date = dates[-1][:4]
    if dates[-1] <= (year_end_date + '0630'):
        year_end = int(year_end_date) - 1
    else:
        year_end = int(year_end_date)
    rebalabce_dates = [str(x) + '0630' for x in range(year_start, year_end + 1)]
    return rebalabce_dates


def _load_stock_returns(start_date, end_date, **kwargs):
    '''
        load daily returns of all stocks within time range between start_date
    and end_date

    '''
    all_dates = kwargs['data_source'].trade_calendar.get_trade_days(start_date, end_date)
    daily_returns = kwargs['data_source'].load_factor('daily_returns', '/stocks/', dates = all_dates)
    return daily_returns.reset_index()


def _load_pb(start_date, end_date, **kwargs):
    '''
        load pb of all stocks within time range between start_date and end_date
    '''
    all_dates = kwargs['data_source'].trade_calendar.get_trade_days(start_date,end_date)
    daily_returns = kwargs['data_source'].load_factor('pb', '/stock_value/', dates=all_dates)
    return daily_returns.reset_index()


def _load_float_mv(start_date, end_date, **kwargs):
    '''
        load float mv of all stocks within time range between start_date and
    end_date

    '''
    all_dates = kwargs['data_source'].trade_calendar.get_trade_days(start_date,end_date)
    daily_float_mv = kwargs['data_source'].load_factor('float_mkt_value', '/stocks/', dates=all_dates)
    return daily_float_mv.reset_index()


def _get_pb_portfolio(rebalance_dates, **kwargs):
    '''
        construct high and low B/P portfolio on rebalance dates
    '''
    data_start_date = rebalance_dates[0][:6] + '01'
    pb = _load_pb(data_start_date, rebalance_dates[-1], **kwargs)
    f = lambda x:str(x.year * 100 + x.month)
    pb['year_month'] = pb.date.apply(f)
    pb = pb[pb['pb'] > 0]

    high_bp = []
    low_bp = []
    for iDate in rebalance_dates:
        all_stocks = kwargs['data_source'].sector.get_ashare_onlist(iDate, 24).index.get_level_values(1).unique()
        temp_data = pb[(pb['year_month'] == iDate[:6]) &
                       (pb['IDs'].isin(all_stocks))]
        iPB = temp_data.groupby('IDs')['pb'].mean()
        iPB = iPB.dropna()
        quantiles = iPB.quantile([0.33, 0.66])

        i_high_bp = iPB[iPB < quantiles.ix[0.33]].index
        i_low_bp = iPB[iPB > quantiles.ix[0.66]].index

        high_bp += list(product([iDate], i_high_bp))
        low_bp += list(product([iDate], i_low_bp))

    high_bp = pd.DataFrame(high_bp, columns=['date', 'IDs'])
    low_bp = pd.DataFrame(low_bp, columns=['date', 'IDs'])

    return high_bp, low_bp


def _get_mv_portfolio(rebalance_dates, **kwargs):
    '''
        construct high and low cap portfolio on rebalance dates
    '''
    data_start_date = rebalance_dates[0][:6] + '01'
    mv = _load_float_mv(data_start_date, rebalance_dates[-1], **kwargs)
    f = lambda x:str(x.year * 100 + x.month)
    mv['year_month'] = mv.date.apply(f)

    high_mv = []
    low_mv = []

    for iDate in rebalance_dates:
        all_stocks = kwargs['data_source'].sector.get_ashare_onlist(iDate, 24).index.get_level_values(1).unique()
        temp_data = mv[(mv['year_month'] == iDate[:6]) &
                       (mv['IDs'].isin(all_stocks))]
        iMV = temp_data.groupby('IDs')['float_mkt_value'].mean()
        iMV = iMV.dropna()
        median = iMV.median()

        i_high = iMV[iMV > median].index
        i_low = iMV[iMV < median].index

        high_mv += list(product([iDate], i_high))
        low_mv += list(product([iDate], i_low))

    high_mv = pd.DataFrame(high_mv, columns=['date', 'IDs'])
    low_mv = pd.DataFrame(low_mv, columns=['date', 'IDs'])

    return high_mv, low_mv


def _get_mkt_factor(all_days, stock_returns, **kwargs):
    '''
        calculate mkt factor on given days
    '''
    mv = _load_float_mv(all_days[0], all_days[-1], **kwargs)

    temp_data = pd.merge(stock_returns, mv)

    def weighted_average(data):
        return data.prod(axis=1).sum() / data.iloc[:, 1].sum()

    mkt = temp_data.groupby('date')[['daily_returns', 'float_mkt_value']].apply(
        weighted_average)

    return mkt


def _get_hml_factor(pb_portfolio, all_days, stock_returns):
    '''
        calculate smb factor on given days
    '''
    high_bp = pb_portfolio[0]
    low_bp = pb_portfolio[1]

    high_bp['1'] = 1
    low_bp['1'] = 1
    high_bp = (high_bp.set_index(['date', 'IDs']).unstack().
        reindex(all_days, method='pad')).stack().reset_index()
    high_bp['date'] = pd.DatetimeIndex(high_bp['date'])

    low_bp = (low_bp.set_index(['date', 'IDs']).unstack().
        reindex(all_days, method='pad')).stack().reset_index()
    low_bp['date'] = pd.DatetimeIndex(low_bp['date'])

    high_bp_returns = pd.merge(high_bp, stock_returns, how='left')
    low_bp_returns = pd.merge(low_bp, stock_returns, how='left')

    high_bp_returns_perday = high_bp_returns.groupby('date')['daily_returns'].mean()
    low_bp_returns_perday = low_bp_returns.groupby('date')['daily_returns'].mean()

    return high_bp_returns_perday - low_bp_returns_perday


def _get_smb_factor(mv_portfolio, all_days, stock_returns):
    '''
        calculate smb factor on given days
    '''
    high_mv = mv_portfolio[0]
    low_mv = mv_portfolio[1]
    high_mv['1'] = 1
    low_mv['1'] = 1
    high_mv = (high_mv.set_index(['date', 'IDs']).unstack().
        reindex(all_days, method='pad')).stack().reset_index()
    high_mv['date'] = pd.DatetimeIndex(high_mv['date'])

    low_mv = (low_mv.set_index(['date', 'IDs']).unstack().
        reindex(all_days, method='pad')).stack().reset_index()
    low_mv['date'] = pd.DatetimeIndex(low_mv['date'])

    high_mv_returns = pd.merge(high_mv, stock_returns, how='left')
    low_mv_returns = pd.merge(low_mv, stock_returns, how='left')

    high_mv_returns_perday = high_mv_returns.groupby('date')['daily_returns'].mean()
    low_mv_returns_perday = low_mv_returns.groupby('date')['daily_returns'].mean()

    return low_mv_returns_perday - high_mv_returns_perday


def _get_rf(all_days, **kwargs):
    w.start()
    start_date = kwargs['data_source'].trade_calendar.tradeDayOffset(all_days[0], -2, freq='1m')
    data = w.edb("M0043802", start_date, all_days[-1])
    date_range = [x.strftime("%Y%m%d") for x in data.Times]
    rf_data = pd.Series(data.Data[0], index=date_range, name='rf')
    rf_data = rf_data.reindex(all_days, method='pad')
    rf_data.index = pd.DatetimeIndex(rf_data.index)
    rf_data = rf_data.apply(lambda x: (1 + x / 100) ** (1 / 365) - 1)
    w.stop()
    return rf_data


def fama_french(start_date, end_date, **kwargs):
    all_days = kwargs['data_source'].trade_calendar.get_trade_days(start_date, end_date)
    rebalace_dates = _get_rebalance_dates(all_days)

    stock_returns = _load_stock_returns(start_date, end_date, **kwargs)
    pb_portfolio = _get_pb_portfolio(rebalace_dates, **kwargs)
    mv_portfolio = _get_mv_portfolio(rebalace_dates, **kwargs)

    rf = _get_rf(all_days, **kwargs)
    mkt = _get_mkt_factor(all_days, stock_returns, **kwargs)
    smb = _get_smb_factor(mv_portfolio, all_days, stock_returns)
    hml = _get_hml_factor(pb_portfolio, all_days, stock_returns)

    factors = pd.concat([rf, mkt, smb, hml], axis=1, join='inner')
    factors.columns = ['rf', 'mkt_rf', 'smb', 'hml']
    kwargs['data_source'].h5DB.save_factor(factors, '/time_series_factors/')

TimeSeriesFuncListDaily = [fama_french]
TimeSeriesFuncListMonthly = []