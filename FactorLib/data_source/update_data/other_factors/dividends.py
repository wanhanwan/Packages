import pandas as pd
from pandas.tseries.offsets import YearBegin, YearEnd
from FactorLib.data_source.uqer_db import UqerDB
from FactorLib.data_source.base_data_source_h5 import H5DB
from FactorLib.data_source.wind_financial_data_api.params import LOCAL_FINDB_PATH
import TSLDataAPI as data_api

uqer_db = UqerDB.get_instance()
h5db = H5DB(LOCAL_FINDB_PATH)

def get_dividends_by_uqer(start, end, **kwargs):
    start = (pd.to_datetime(start) - YearBegin(1)).strftime("%Y%m%d")
    end = (pd.to_datetime(end) + YearEnd(1)).strftime("%Y%m%d")
    raw_dividends = uqer_db.run_api("EquDivGet", beginDate=start,
        endDate=end, field=["endDate", "ticker", "publishDate", "perCashDiv"])
    raw_dividends.dropna(inplace=True)
    raw_dividends['endDate'] = (raw_dividends['endDate'].str.replace('-', '')).astype('int')
    raw_dividends['publishDate'] = (raw_dividends['publishDate'].str.replace('-', '')).astype('int')
    raw_dividends['ticker'] = raw_dividends['ticker'].astype('int')
    raw_dividends.sort_values(['ticker', 'endDate', 'publishDate'], inplace=True)
    raw_dividends.rename(columns={'ticker': 'IDs', 'endDate': 'date', 'publishDate': 'ann_dt',
        'perCashDiv': 'dividend'}, inplace=True)
    h5db.save_h5file(raw_dividends, 'cash_div', '/dividends/')


def get_dividends(start, end, **kwargs):
    start = (pd.to_datetime(start) - YearBegin(2)).strftime("%Y%m%d")
    end = (pd.to_datetime(end) - YearEnd(1)).strftime("%Y%m%d")
    dvd = data_api.EqyDivGet(beginReportDate=start, endReportDate=end, field=[
        '红利比', '预案公布日','截止日']).reset_index().drop('date', axis=1)
    dvd['IDs'] = dvd['IDs'].astype('int')
    dvd.rename(columns={'s_div_prelanndate': 'ann_dt', 'report_date': 'date',
        'cash_dvd_per_sh_pre_tax': 'dividend'}, inplace=True)
    dvd = dvd[dvd['ann_dt'] != 0]
    h5db.save_h5file(dvd, 'cash_div', '/dividends/')



if __name__ == '__main__':
    get_dividends('20180621', '20180621')
