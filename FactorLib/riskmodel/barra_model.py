import os.path as path
import pandas as pd
import numpy as np
from ..utils.tool_funcs import import_module, searchNameInStrList
from .riskmodel_data_source import RiskModelDataSourceOnH5
from ..data_source.base_data_source_h5 import h5
from ..utils.datetime_func import DateRange2Dates


class BarraModel(object):
    """Barra风险模型"""
    def __init__(self, name, model_path):
        self.name = name
        if not path.isfile(model_path+'/config.py'):
            raise FileNotFoundError
        if not path.isfile(model_path+'/DS-Barra.pkl'):
            raise FileNotFoundError
        data_source = RiskModelDataSourceOnH5(h5_db=h5)
        data_source.load_info('barra', 'model_path'+'/DS-Barra.pkl')
        self.data_source = data_source
        config = import_module('config', model_path+'/config.py')
        self.getFactorReturnArgs()

    def getFactorReturnArgs(self):
        """初始化参数"""
        args = dict()
        args['行业因子'] = searchNameInStrList(self.data_source.factor_names, ['cs_level','wind_level','sw_level'])
        args['回归权重因子'] = searchNameInStrList(self.data_source.factor_names, ['float_mkt_value'])
        args['风格因子'] = [x for x in self.data_source.factor_names if x not in [args['行业因子', args['回归权重因子']]]]
        args['忽略行业'] = []
        if not hasattr(self, 'all_industries'):
            self.all_industries = self.data_source.get_factor_unique_data(args['行业因子'])
        if np.nan in self.all_industries:
            self.all_industries.remove(np.nan)
        self.genFactorReturnArgs = args
        return

    def setGenFactorReturnArgs(self, attr, value):
        """设置因子收益率生成参数"""
        if attr in self.genFactorReturnArgs:
            self.genFactorReturnArgs[attr] = value

    @DateRange2Dates
    def set_dimension(self, start=None, end=None, dates=None):
        self.data_source.update_estu(dates)