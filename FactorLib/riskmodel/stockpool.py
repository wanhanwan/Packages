from data_source import riskModelDB
from QuantLib import stockFilter


def get_stocklist(dates, name, qualify_method):
    stocks = riskModelDB.load_factor(name, "/stockpool/", dates=dates)
    stocks = stocks[stocks.iloc[:, 0] == 1]
    stocks = stocks.unstack().reindex(dates, method='ffill').stack()
    stocks.columns = ['date', 'IDs']
    return _qualify_stocks(stocks, qualify_method)


def _qualify_stocks(stocklist, method):
    if not hasattr(stockFilter, method):
        raise KeyError("%s doesn't exist in file stockFilter.py")
    else:
        return getattr(stockFilter, method)(stocklist)
