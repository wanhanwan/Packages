from ..data_source.base_data_source_h5 import data_source
from QuantLib import stockFilter


def get_stocklist(dates, name, qualify_method):
    stocks = data_source.sector.get_index_members(ids=name, dates=dates)
    stocks = stocks[stocks.iloc[:, 0] == 1]
    return _qualify_stocks(stocks, qualify_method)


def _qualify_stocks(stocklist, method, **kwargs):
    if not hasattr(stockFilter, method):
        raise KeyError("%s doesn't exist in file stockFilter.py")
    else:
        return getattr(stockFilter, method)(stocklist, **kwargs)
