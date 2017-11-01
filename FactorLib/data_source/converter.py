import pandas as pd
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
    file = pd.ExcelFile(p)
    if pd.__version__ >= '0.21.0':
        sw_level_2 = dict(file.parse(sheet_name='sw_level_2', header=0).values)
        cs_level_2 = dict(file.parse(sheet_name='cs_level_2', header=0).values)
    else:
        sw_level_2 = dict(file.parse(sheetname='sw_level_2', header=0).values)
        cs_level_2 = dict(file.parse(sheetname='cs_level_2', header=0).values)
    file.close()
    return sw_level_2, cs_level_2

SW_LEVEL_2_DICT, CS_LEVEL_2_DICT = read_industry_mapping()


class Converter(object):
    def __init__(self, rules):
        self._rules = rules
        self.unmapping = {}
        for name, rule in self._rules.items():
            self.unmapping[name] = {rule.mapping[x]: x for x in rule.mapping}

    def convert(self, name, data):
        try:
            r = self._rules[name]
        except KeyError:
            return data

        if isinstance(data, list):
            keys = [r.convertfunc(x) for x in data]
            return [r.mapping[x] for x in keys]
        elif isinstance(data, pd.Series):
            return data.apply(r.convertfunc).map(r.mapping)
        else:
            return data

    def name2id(self, name, data):
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
            return data.map(unmapping).apply(r.name2id_func)
        else:
            return data

    def all_values(self, name):
        try:
            r = self._rules[name]
        except KeyError as e:
            raise e

        return [v for k, v in r.mapping.items()]


IndustryConverter = Converter({
    'cs_level_1': Rule(CS_INDUSTRY_DICT, lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_1': Rule(SW_INDUSTRY_DICT, lambda x: str(int(x)), int),
    'wind_level_1': Rule(WIND_INDUSTRY_DICT, lambda x: str(int(x)), int),
    'cs_level_2': Rule(CS_LEVEL_2_DICT, lambda x: 'CI'+str(int(x)).zfill(6), lambda x: int(x[2:])),
    'sw_level_2': Rule(SW_LEVEL_2_DICT, lambda x: str(int(x)), int)
})