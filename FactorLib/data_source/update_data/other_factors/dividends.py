import pandas as pd
from pandas.tseries.offsets import YearBegin, YearEnd
from FactorLib.data_source.uqer_db import UqerDB
from FactorLib.data_source.base_data_source_h5 import H5DB
from FactorLib.data_source.wind_financial_data_api.params import LOCAL_FINDB_PATH

uqer_db = UqerDB.get_instance()
h5db = H5DB(LOCAL_FINDB_PATH)

def get_dividends(start, end, **kwargs):
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


if __name__ == '__main__':
    get_dividends('20100101', '20111231')
