import numpy as np
from ..base_data_source_h5 import tc
from .params import LOCAL_FINDB_PATH
from .database import _reconstruct
from ..h5db import H5DB


finance_db = H5DB(LOCAL_FINDB_PATH)


def load_dividends(ids=None, dates=None, start_date=None, end_date=None, idx=None, report_year=None):
    """加载每股股利"""
    def search_data(data, dates):
        new_data = data.reindex(dates, method='ffill')
        return new_data

    if idx is not None:
        dates = idx.index.unique(level='date').strftime('%Y%m%d').astype('int')
        ids = idx.index.unique(level='IDs').astype('int')
    else:
        if dates is not None:
            dates = np.asarray(dates, dtype='int')
        else:
            dates = np.asarray(tc.get_trade_days(start_date, end_date)).astype('int')
        if ids is not None:
            ids = np.asarray(ids, dtype='int')
    dividends = finance_db.read_h5file('cash_div', '/dividends/').sort_values(['IDs', 'date', 'ann_dt'])
    if ids is not None:
        dividends = dividends[dividends['IDs'].isin(ids)]
    dividends = dividends[(dividends['IDs']<900000) & (dividends['date']%10000//100==12)]
    if report_year is not None:
        dividends = dividends[dividends['date']//10000==report_year]
    dividends = dividends.groupby(['IDs', 'date', 'ann_dt']).sum()
    dividends_indexed = dividends.reset_index(['date', 'IDs'])
    rslt = dividends_indexed.groupby('IDs')['dividend'].apply(search_data, dates=dates)
    rslt.index.names = ['IDs', 'date']
    return _reconstruct(rslt.reset_index())
