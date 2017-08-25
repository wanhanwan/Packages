from FactorLib.data_source.base_data_source_h5 import data_source, riskDB


def _load_style(stocklist, factor_dict):
    data = data_source.h5DB.load_factor()


def load_factors(stocklist, factor_dict):
    return