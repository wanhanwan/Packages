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
    estu_config['func'] = config_file.ESTU['func']
    estu_config['func_args'] = config_file.ESTU['func_args']
    estu_config['CU'] = config_file.CU
    return estu_config


def get_estu(dates, estu_config):
    if estu_config is None:
        kwargs = {}
    else:
        kwargs = estu_config['func_args']
    return get_stocklist(dates, estu_config['CU'], qualify_method=estu_config['func'], **kwargs)
