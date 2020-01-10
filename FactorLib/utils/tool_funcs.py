# coding: utf-8
"""一些工具函数"""
import pandas as pd
import numpy as np
import re
from ..const import INDUSTRY_NAME_DICT


try:
    from fastcache import clru_cache
except:
    from functools import lru_cache
    clru_cache = lru_cache


def dict_reverse(_dict):
    return {_dict[x]:x for x in _dict}


def parse_industry(industry):
    return INDUSTRY_NAME_DICT[industry]


def anti_parse_industry(industry):
    return dict_reverse(INDUSTRY_NAME_DICT)[industry]


def tradecode_to_windcode(tradecode):
    return tradecode + '.SH' if tradecode[0] == '6' else tradecode + '.SZ'


def windcode_to_tradecode(windcode):
    return windcode[:6]

def windcode_to_intcode(windcode):
    return int(windcode[:6])

@clru_cache()
def intcode_to_tradecode(intcode):
    return str(intcode).zfill(6)

def tradecode_to_intcode(tradecode):
    return int(tradecode)

@clru_cache()
def uqercode_to_windcode(uqercode):
    return uqercode[:6]+'.SH' if uqercode[-4:]=='XSHG' else uqercode[:6]+'.SZ'

@clru_cache()
def tradecode_to_uqercode(tradecode):
    return tradecode+'.XSHG' if tradecode[0] == '6' else tradecode+'.XSHE'

@clru_cache()
def tradecode_to_tslcode(code):
    if code[:2].isalpha():
        return code
    return 'SH'+code if code[0] == '6' else 'SZ'+code

@clru_cache()
def windcode_to_tslcode(windcode):
    dot_indice = windcode.find(".")
    if dot_indice>0:
        return windcode[dot_indice+1:]+windcode[:dot_indice]
    return windcode

@clru_cache()
def tslcode_to_tradecode(code):
    match = re.match("[a-zA-Z]*([0-9]*)", code)
    if match:
        return match.group(1)
    return code

def drop_patch(code):
    return code.split(".")[0]


def import_module_from_file(module_name, module_path):
    """
    从源文件引入一个模块

    Parameters:
    ===========
    module_name: str
        模块的文件名
    module_path: str
        模块的文件路径(包含文件名)
    """
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise FileNotFoundError("模块不存在: %s" % module_path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ensure_dir_exists(dir_path):
    import os
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


# 在一个日期区间中可能发布的财务报告的报告期
def ReportDateAvailable(start_date, end_date):
    def _(date):
        if '0101' <= date[4:] <= '0430':
            return str(int(date[:4]) - 1)+'1231'
        elif '0701' <= date[4:] <= '0830':
            return date[:4] + '0630'
        elif '1001' <= date[4:] <= '1030':
            return date[:4] + '0930'
        else:
            return date
    if start_date !=  end_date:
        report_dates = pd.date_range(_(start_date), _(end_date), freq='Q')
        return report_dates.strftime("%Y%m%d")
    else:
        return _(start_date)


# 某个时间区间内的所有报告期(季度)
def get_all_report_periods(start, end):
    periods = pd.date_range(start, end, freq='Q', name='date')
    return periods


def distribute_equal(n, m):
    """把整数n平均分成m份"""
    Quotient = n // m
    Remainder = n % m
    Res = [Quotient] * m
    for i in range(Remainder):
        Res[i] += 1
    return Res


# 返回一个可用名称
def get_available_names(name, used_names):
    i = 2
    while (name + str(i)) in used_names:
        i += 1
    return name + str(i)


# 返回recource文件夹的绝对路径
def get_resource_abs_path():
    from pathlib import Path
    curr_path = Path(__file__)
    tar_path = curr_path.parents[1] / 'resource'
    return tar_path


def get_members_of_date(date, entry_dt_field, remove_dt_field, return_field, data):
    """获取某一天的成分股
    函数需要输入一张包含纳入纳出日期的表，在纳入纳出日期之间的数据会被作为结果输出.

    Parameters:
    ===========
    date : str
        需要哪天的成分股,"YYYYMMDD"
    entry_dt_field : str
        data中纳入日期对应的列名
    remove_dt_field : str
        data中纳出日期对应的列名
    return_field : str
        data中需要返回的列名
    data : DataFrame
        一张包含纳入纳出日期的表, index为股票6位数字代码(String)。
    """
    date = int(date)
    # data = data.sort_index()
    data.index.name = 'IDs'
    rslt = data.loc[(data[entry_dt_field]<=date)&
                    ((data[remove_dt_field].isnull())|(data[remove_dt_field]>=date)),
                    return_field]
    return rslt


def get_members_of_dates(dates, entry_dt_field, remove_dt_field, return_field, data):
    """获取某个时间序列的指数成分股"""
    all_ids = data.index.unique()
    rslt = pd.DataFrame(index=dates, columns=all_ids, dtype='O')
    for dt in dates:
        idata = get_members_of_date(dt, entry_dt_field, remove_dt_field, return_field, data)
        rslt.loc[dt, :] = idata
    return rslt


def dummy2name(dummy):
    """Dummy变量转换成对应的哑变量名称
    Paramters:
    ==========
    dummy: DataFrame
        哑变量，索引是二维索引。
    """
    columns = dummy.columns
    names = dummy.apply(lambda x: columns[np.where(x)[0][0]], axis=1, raw=True)
    return names
