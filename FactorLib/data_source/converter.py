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
        
        某些二级行业会带有罗马数字Ⅱ, 函数输入时可以忽略。

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
    
    def name2windid(self, name, data):
        """行业中文名称转Wind行业代码(无后缀的字符串)
        
        某些二级行业会带有罗马数字Ⅱ, 函数输入时可以忽略。
        """
        try:
            unmapping = self.unmapping[name]
        except KeyError:
            return data
        if isinstance(data, list):
            try:
                names = [unmapping[x] for x in data]
            except KeyError:
                unmapping = {x.replace('Ⅱ',''):y for x,y in unmapping.items()}
                names = [unmapping[x] for x in data]
        elif isinstance(data, pd.Series):
            try:
                names = data.dropna().map(unmapping).reindex(data.index)
            except ValueError:
                unmapping = {x.replace('Ⅱ', ''): y for x, y in unmapping.items()}
                names = data.dropna().map(unmapping).reindex(data.index)
        else:
            raise RuntimeError("不支持的数据类型")
        return names
    
    def windid2name(self, name, windid):
        """"Wind行业代码(无后缀)转行业中文名称(省略罗马数字)"""
        try:
            r = self._rules[name]
            mapping = {k:v.replace('Ⅱ', '') for k,v in r.mapping.items()}
        except KeyError:
            return windid
        if isinstance(windid, list):
            names = [mapping[x] for x in windid]
        elif isinstance(windid, pd.Series):
            names = windid.dropna().map(mapping).reindex(windid.index)
        else:
            raise RuntimeError("不支持的数据类型")
        return names
        
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



if __name__ == '__main__':
    int_code = IndustryConverter.name2id('cs_level_1', ['石油石化'])
    print(int_code)
