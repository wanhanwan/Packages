import pandas as pd
from os.path import join, abspath, dirname
from collections import namedtuple
from ..const import SW_INDUSTRY_DICT, CS_INDUSTRY_DICT, WIND_INDUSTRY_DICT

Rule = namedtuple('Rule', ['mapping', 'convertfunc'])


def read_industry_mapping():
    p = abspath(dirname(__file__)+'/..') + '/resource/level_2_industry_dict.xlsx'
    file = pd.ExcelFile(p)
    sw_level_2 = dict(file.parse(sheetname='sw_level_2', header=0).values)
    cs_level_2 = dict(file.parse(sheetname='cs_level_2', header=0).values)
    file.close()
    return sw_level_2, cs_level_2

SW_LEVEL_2_DICT, CS_LEVEL_2_DICT = read_industry_mapping()


class Converter(object):
    def __init__(self, rules):
        self._rules = rules

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

IndustryConverter = Converter({
    'cs_level_1': Rule(CS_INDUSTRY_DICT, lambda x: 'CI'+str(int(x)).zfill(6)),
    'sw_level_1': Rule(SW_INDUSTRY_DICT, lambda x: str(int(x))),
    'wind_level_1': Rule(WIND_INDUSTRY_DICT, lambda x: str(int(x))),
    'cs_level_2': Rule(CS_LEVEL_2_DICT, lambda x: 'CI'+str(int(x)).zfill(6)),
    'sw_level_2': Rule(SW_LEVEL_2_DICT, lambda x: str(int(x)))
})