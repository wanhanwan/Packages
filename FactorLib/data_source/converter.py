# coding: utf-8

import pandas as pd
import numpy as np
from os.path import join, abspath, dirname
from collections import namedtuple
from ..const import (
                        SW_INDUSTRY_DICT,
                        CS_INDUSTRY_DICT,
                        WIND_INDUSTRY_DICT,
                        SW_INDUSTRY_DICT_REVERSE,
                        CS_INDUSTRY_DICT_REVERSE,
                        WIND_INDUSTRY_DICT_REVERSE
)

Rule = namedtuple('Rule', ['mapping', 'convertfunc', 'name2id_func'])


def read_industry_mapping():
    p = abspath(dirname(__file__)+'/..') + '/resource/level_2_industry_dict.xlsx'
    with pd.ExcelFile(p) as file:
        if pd.__version__ >= '0.21.0':
            sw_level_2 = dict(file.parse(sheet_name='sw_level_2', header=0, converters={'Code': str}).values)
            cs_level_2 = dict(file.parse(sheet_name='cs_level_2', header=0, converters={'Code': str}).values)
            divrsfd_finan_sw = dict(file.parse(sheet_name='divrsfd_finan_sw', header=0,
                                                   converters={'Code': str}).values)    # 把申万非银金融改成其二级行业
            divrsfd_finan_cs = dict(file.parse(sheet_name='divrsfd_finan_cs', header=0,
                                               converters={'Code': str}).values)        # 中信同理
        else:
            sw_level_2 = dict(file.parse(sheetname='sw_level_2', header=0, converters={'Code': str}).values)
            cs_level_2 = dict(file.parse(sheetname='cs_level_2', header=0, converters={'Code': str}).values)
            divrsfd_finan_sw = dict(file.parse(sheet_name='divrsfd_finan_sw', header=0,
                                               converters={'Code': str}).values)
            divrsfd_finan_cs = dict(file.parse(sheet_name='divrsfd_finan_cs', header=0,
                                               converters={'Code': str}).values)
    # file.close()
    return sw_level_2, cs_level_2, divrsfd_finan_sw, divrsfd_finan_cs
SW_LEVEL_2_DICT, CS_LEVEL_2_DICT, divrsfd_finan_sw, divrsfd_finan_cs = read_industry_mapping()


class Converter(object):
    def __init__(self, rules):
        self._rules = rules
        self.unmapping = {}
        for name, rule in self._rules.items():
            self.unmapping[name] = {rule.mapping[x]: x for x in rule.mapping}

    def convert(self, name, data):
        """行业代码转行业名称"""
        try:
            r = self._rules[name]
        except KeyError:
            return data

        if isinstance(data, list):
            keys = [r.convertfunc(x) for x in data]
            return [r.mapping[x] for x in keys]
        elif isinstance(data, pd.Series):
            return data.dropna().apply(r.convertfunc).map(r.mapping).reindex(data.index)
        else:
            return data

    def name2id(self, name, data):
        """行业名称转行业代码"""
        try:
            r = self._rules[name]
            unmapping = self.unmapping[name]
        except KeyError:
            return data

        if isinstance(data, list):
            names = [unmapping[x] for x in data]
            codes = [r.name2id_func(x) for x in names]
            return codes
        elif isinstance(data, pd.Series):
            return data.dropna().map(unmapping).apply(r.name2id_func).reindex(data.index)
        else:
            return data

    def all_values(self, name):
        """返回所有行业名称"""
        try:
            r = self._rules[name]
        except KeyError as e:
            raise e

        return [v for k, v in r.mapping.items()]

    def all_ids(self, name):
        """返回所有行业代码"""
        try:
            r = self._rules[name]
        except KeyError as e:
            raise e
        return [x for x in r.mapping]


IndustryConverter = Converter({
    'cs_level_1': Rule(CS_INDUSTRY_DICT, lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_1': Rule(SW_INDUSTRY_DICT, lambda x: str(int(x)), int),
    'diversified_finance_sw': Rule(divrsfd_finan_sw, lambda x: str(int(x)), int),
    'diversified_finance_cs': Rule(divrsfd_finan_cs, lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'wind_level_1': Rule(WIND_INDUSTRY_DICT, lambda x: str(int(x)), int),
    'cs_level_2': Rule(CS_LEVEL_2_DICT, lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_2': Rule(SW_LEVEL_2_DICT, lambda x: str(int(x)), int)
})


# ncdb中数据类型decode/encode
# 日期类型
DATE_ECODING = {'dtype': 'uint16', 'scale_factor': 1, '_FillValue': 0, 'units': 'days since 1970-01-01', 'zlib': True,
'complevel': 9}
PRICE_ENCODING = {'dtype': 'uint16', 'scale_factor': 10e-3, '_FillValue': 0, 'zlib': True,
'complevel': 9}
BIGNUM_ENCODING = {'dtype': 'int64', 'scale_factor': 10e-5, '_FillValue': -9999,'zlib': True,
'complevel': 9}
INTEGER_ENCODING = {'dtype': 'int32', 'scale_factor': 1, '_FillValue': -9999, 'zlib': True,
'complevel': 9}
BOOL_ENCODING = {'dtype': 'uint8', 'scale_factor': 1, '_FillValue': 2, 'zlib': True,
'complevel': 9}


def parse_nc_encoding(dtype):
    if dtype in [np.float32, np.float64]:
        return BIGNUM_ENCODING
    elif dtype in [np.int32, np.int16, np.int64]:
        return INTEGER_ENCODING
    elif dtype in [np.bool]:
        return BOOL_ENCODING
    elif dtype in [np.datetime64]:
        return DATE_ECODING
    else:
        return {}