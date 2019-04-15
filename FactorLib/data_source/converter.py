# coding: utf-8

import pandas as pd
import numpy as np
from os.path import abspath, dirname
from collections import namedtuple


# mapping: 字典，{String类型的Wind行业代码：行业中文名称}
# convertfunc: Int类型行业代码转成String类型的Wind行业代码；
# name2id_func: String类型的Wind行业代码转成Int类型行业代码
Rule = namedtuple('Rule', ['mapping', 'convertfunc', 'name2id_func'])


class Converter(object):
    def __init__(self, rules):
        """初始化
        每条规则下mapping代表代码(带有前缀或后缀)-行业中文名称，
        unmapping代表行业中文名称-代码(带有前缀或后缀)
        """
        self._rules = rules
        self.unmapping = {}
        for name, rule in self._rules.items():
            self.unmapping[name] = {rule.mapping[x]: x for x in rule.mapping}

    def convert(self, name, data):
        """Int类型的行业代码转行业中文名称
        对应rule.convertfunc

        如果data的类型是Series, data.values应为Int类型行业代码。
        """
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
        """行业中文名称转Int类型行业代码

        行业代码是int类型，对应rule.name2id_func

        如果data的类型是Series, data.values应为String类型的Wind行业代码。
        """
        try:
            r = self._rules[name]
            unmapping = self.unmapping[name]
        except KeyError:
            return data

        if isinstance(data, list):
            try:
                names = [unmapping[x] for x in data]
            except KeyError:
                unmapping = {x.replace('Ⅱ',''):y for x,y in unmapping.items()}
                names = [unmapping[x] for x in data]
            codes = [r.name2id_func(x) for x in names]
            return codes
        elif isinstance(data, pd.Series):
            try:
                return data.dropna().map(unmapping).apply(r.name2id_func).reindex(data.index)
            except ValueError:
                unmapping = {x.replace('Ⅱ', ''): y for x, y in unmapping.items()}
                return data.dropna().map(unmapping).apply(r.name2id_func).reindex(data.index)
        else:
            return data

    def all_values(self, name):
        """返回所有行业中文名称"""
        try:
            r = self._rules[name]
        except KeyError as e:
            raise e

        return [v for k, v in r.mapping.items()]

    def all_ids(self, name):
        """返回所有行业Wind代码"""
        try:
            r = self._rules[name]
        except KeyError as e:
            raise e
        return [x for x in r.mapping]


def read_industry_mapping():
    p = abspath(dirname(__file__)+'/..') + '/resource/level_2_industry_dict.xlsx'
    data = {}
    with pd.ExcelFile(p) as file:
        for sheet_name in file.sheet_names:
            data[sheet_name] = dict(file.parse(sheet_name=sheet_name, header=0, converters={'Code': str}).values)
    return data
industry_dict = read_industry_mapping()


IndustryConverter = Converter({
    'cs_level_1': Rule(industry_dict['cs_level_1'], lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_1': Rule(industry_dict['sw_level_1'], lambda x: str(int(x)), int),
    'diversified_finance_sw': Rule(industry_dict['divrsfd_finan_sw'], lambda x: str(int(x)), int),
    'diversified_finance_cs': Rule(industry_dict['divrsfd_finan_cs'], lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'wind_level_1': Rule(industry_dict['wind_level_1'], lambda x: str(int(x)), int),
    'wind_level_2': Rule(industry_dict['wind_level_2'], lambda x: str(int(x)), int),
    'cs_level_2': Rule(industry_dict['cs_level_2'], lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_2': Rule(industry_dict['sw_level_2'], lambda x: str(int(x)), int)
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


if __name__ == '__main__':
    int_code = IndustryConverter.name2id('cs_level_1', ['石油石化'])
    print(int_code)
