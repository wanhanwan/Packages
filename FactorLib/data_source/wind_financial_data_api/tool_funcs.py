import pandas as pd
import numpy as np
from . import incomesheet, profitexpress, balancesheet, windissuingdate
from ..base_data_source_h5 import tc
from pathlib import Path


def _read_mapping():
    file_path = Path(__file__).parent / 'resource' / 'financesheet_map.xlsx'
    mapping = pd.read_csv(file_path, header=0, index_col=0)
    return mapping


field_map = _read_mapping()


def _get_mapping(filed_name):
    map_series = field_map[filed_name]
    formal_tb = map_series.index[~np.isnan(map_series.values)][0]
    return formal_tb, map_series.to_dict()


def get_report_ann_dt(start_date=None, end_date=None, dates=None, ids=None, quarter=None):
    """指定日期的最新报告期"""
    data = windissuingdate.all_data

    if ids is not None:
        if isinstance(ids, list):
            ids_int = np.asarray(ids).astype('int32')
            data = data.query("IDs in @ids_int")
        if isinstance(ids, (pd.Series, pd.DataFrame)):
            dates = ids.index.get_level_values('date').strftime('%Y%m%d').unique().tolist()
            ids_int = ids.index.get_level_values('IDs').unique().astype('int32').tolist()
            data = data.query("IDs in @ids_int")

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
        tmp = data[(data['ann_dt'] <= date) & (data['ann_dt'] != 0)].groupby('IDs')['date'].max()






def load_newest(field_name, year, quarter, dates, ids=None):
    """包含业绩快报和正式财报的数据"""
    table_name, field_dict = _get_mapping(field_name)
    table_name, table_id = table_name.split('_')
    globals()[table_id].l
