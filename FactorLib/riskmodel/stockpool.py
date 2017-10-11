from ..data_source.base_data_source_h5 import data_source
from QuantLib import stockFilter


def get_stocklist(dates, name, qualify_method, **kwargs):
    stocks = data_source.sector.get_index_members(ids=name, dates=dates)
    stocks = stocks[stocks.iloc[:, 0] == 1]
    return _qualify_stocks(stocks, qualify_method, **kwargs)


def _qualify_stocks(stocklist, method, **kwargs):
    if not hasattr(stockFilter, method):
        raise KeyError("%s doesn't exist in file stockFilter.py")
    else:
        return getattr(stockFilter, method)(stocklist, **kwargs)


def _parse_estuconfig(config_file):
    estu_config = dict()
    estu_config['ESTU'] = config_file.ESTU
    estu_config['CU'] = config_file.CU
    return estu_config


def get_estu(dates, estu_config):
    stocklist = data_source.h5DB.load_factor(estu_config['ESTU'], '/indexes/', dates=dates)
    return stocklist[stocklist.iloc[:, 0] == 1]

