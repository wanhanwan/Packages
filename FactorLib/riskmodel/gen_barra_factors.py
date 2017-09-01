from FactorLib.single_factor_test.config import parse_config
from FactorLib.single_factor_test import factor_list
from FactorLib.utils import AttrDict
from FactorLib.riskmodel.stockpool import get_stocklist
from FactorLib.riskmodel import riskmodel_data_source


def get_estu(dates, config):
    if config.ESTU.func_args is None:
        args = {}
    else:
        args = config.ESTU.func_args.__dict__
    return get_stocklist(dates, config.CU, qualify_method=config.ESTU.func, **args)


def prepare_factors(config):
    descriptors = getattr(factor_list, config.risk_descriptors)
    if not isinstance(descriptors, list):
        descriptors = [descriptors]
    for other_factor in config.others:
        descriptors.append(getattr(factor_list, other_factor))
    return descriptors


def prepare_csoperate_funcs(config):
    func_modules = []
    for func_name, args in config.cross_section_operate:
        func_description = {'func_name':func_name, 'args':args.__dict__}
        func_modules.append(func_description)
    return func_modules


def prepare_save_info(config):
