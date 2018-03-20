import pandas as pd
import numpy as np
from . import incomesheet, profitexpress, balancesheet, windissuingdate, winddescription
from ..base_data_source_h5 import tc
from .database import _reconstruct
from ..helpers import handle_ids
from pathlib import Path


def _read_mapping():
    file_path = Path(__file__).parent.parent.parent / 'resource' / 'financesheet_map.xlsx'
    mapping = pd.read_excel(file_path, header=0, index_col=0)
    return mapping


field_map = _read_mapping()


def _get_mapping(filed_name):
    map_series = field_map[filed_name]
    formal_tb = map_series.index[~np.isnan(map_series.values)][0]
    return formal_tb, map_series.to_dict()


@handle_ids
def get_latest_report(start_date=None, end_date=None, dates=None, ids=None, quarter=None):
    """获取指定日期的最新报告期"""
    data = windissuingdate.all_data

    if ids is not None:
        if isinstance(ids, list):
            ids_int = np.asarray(ids).astype('int32')
            data = data.query("IDs in @ids_int")
        if isinstance(ids, (pd.Series, pd.DataFrame)):
            dates = ids.index.get_level_values('date').strftime('%Y%m%d').unique().tolist()
            ids_int = ids.index.get_level_values('IDs').unique().astype('int32').tolist()
            data = data.query("IDs in @ids_int")

    if quarter is not None:
        data = data[data['date'] % 10000 // 100 == quarter*3]

    if isinstance(dates, list):
        dates = np.asarray(dates).astype('int32')
    elif isinstance(dates, pd.DatetimeIndex):
        dates = np.asarray(dates.strftime("%Y%m%d")).astype('int32')
    elif start_date is not None and end_date is not None:
        dates = np.asarray(tc.get_trade_days(start_date, end_date)).astype('int32')
    else:
        raise ValueError
    r = []
    for date in dates:
        tmp = data[(data['ann_dt'] <= date) & (data['ann_dt'] != 0)].groupby('IDs', as_index=False)['date'].max()
        tmp.rename(columns={'date': 'report_period'}, inplace=True)
        tmp['date'] = date
        r.append(tmp)
    rslt = _reconstruct(pd.concat(r).set_index(['date', 'IDs']))
    if isinstance(ids, (pd.DataFrame, pd.Series)):
        return rslt.reindex(ids.index)
    return rslt


def get_go_market_days(date, ids=None, uint='d'):
    """已经上市天数"""
    divider = {'d': 1, 'm': 30, 'y': 365}
    date_int = int(date)
    date_dt = pd.to_datetime(date)
    data = winddescription.all_data.query('delistdate > @date_int')
    go_mkt_days = (date_dt - pd.to_datetime(data['listdate'].astype('str').values)) / pd.to_timedelta(divider[uint], 'd')
    data['go_market_days'] = go_mkt_days
    data['date'] = date_int
    if ids is not None:
        ids = np.asarray(ids).astype('int32')
        data = data[data['IDs'].isin(ids)]
    return _reconstruct(data[['IDs', 'date', 'go_market_days']].set_index(['date', 'IDs'])).sort_index()


def load_newest(field_name, year, quarter, dates, ids=None):
    """包含业绩快报和正式财报的数据"""
    table_name, field_dict = _get_mapping(field_name)
    table_name, table_id = table_name.split('_')
