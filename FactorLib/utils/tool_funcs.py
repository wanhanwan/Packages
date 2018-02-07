# coding: utf-8
"""一些工具函数"""
from FactorLib.const import (INDUSTRY_NAME_DICT,
                   SW_INDUSTRY_DICT,
                   CS_INDUSTRY_DICT,
                   SW_INDUSTRY_DICT_REVERSE,
                   CS_INDUSTRY_DICT_REVERSE,
                   WIND_INDUSTRY_DICT_REVERSE,
                   WIND_INDUSTRY_DICT
                   )
import pandas as pd
import os
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


def write_df_to_excel(sheet, start_point, df, index=True, columns=True):
    if index:
        df = df.reset_index()
    df_shape = df.shape
    if columns:
        for i, x in enumerate(df.columns):
            _ = sheet.cell(column=start_point[1]+i, row=start_point[0],
                           value=x)
            if isinstance(x, pd.Timestamp):
                _.number_format = 'yyyy/mm/dd'
        start_point = (start_point[0]+1,start_point[1])
    for r in range(df_shape[0]):
        for c in range(df_shape[1]):
            col = start_point[1] + c
            row = start_point[0] + r
            _ = sheet.cell(column=col, row=row, value=df.iloc[r, c])
            if isinstance(df.iloc[r, c], (int, float)):
                _.number_format = '0.000'
            elif isinstance(df.iloc[r, c], pd.Timestamp):
                _.number_format = 'yyyy/mm/dd'
    end_point = (start_point[0] + df_shape[0]-1, start_point[1] + df_shape[1]-1)
    return end_point


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
def tradecode_to_tslcode(code):
    return 'SH'+code if code[0] == '6' else 'SZ'+code

@clru_cache()
def windcode_to_tslcode(windcode):
    return windcode[-2:] + windcode[:6]

@clru_cache()
def tslcode_to_tradecode(code):
    return code[2:]

def drop_patch(code):
    return code.split(".")[0]


def import_mod(mod_name):
    try:
        from importlib import import_module
        return import_module(mod_name)
    except Exception as e:
        return None


def import_module(module_name, module_path):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(module_name, module_path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ensure_dir_exists(dir_path):
    import os
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_industry_names(industry_symbol, industry_info):
    if industry_symbol == 'sw_level_1':
        series = pd.Series(SW_INDUSTRY_DICT).to_frame().rename(columns={0:industry_symbol})
        series.index = series.index.astype("int32")
    elif industry_symbol == 'cs_level_1':
        series = pd.Series(CS_INDUSTRY_DICT).to_frame().rename(columns={0: industry_symbol})
        series.index = [int(x[2:]) for x in series.index]
    elif industry_symbol == 'wind_level_1':
        series = pd.Series(WIND_INDUSTRY_DICT).to_frame().rename(columns={0: industry_symbol})
        series.index = [float(x) for x in series.index]
    elif industry_symbol == 'cs_level_2':
        level_2_excel = os.path.abspath(os.path.dirname(__file__) +'/..') + os.sep + "resource" + os.sep + "level_2_industry_dict.xlsx"
        level_2_dict = pd.read_excel(level_2_excel, sheetname=industry_symbol, header=0)
        level_2_dict['Code'] = level_2_dict['Code'].apply(lambda x: int(x[2:]))
        series = level_2_dict.set_index('Code').rename(columns={'Name': industry_symbol})
    elif industry_symbol == 'sw_level_2':
        level_2_excel = os.path.abspath(os.path.dirname(__file__) +'/..') + os.sep + "resource" + os.sep + "level_2_industry_dict.xlsx"
        level_2_dict = pd.read_excel(level_2_excel, sheetname=industry_symbol, header=0)
        level_2_dict['Code'] = level_2_dict['Code'].apply(int)
        series = level_2_dict.set_index('Code').rename(columns={'Name': industry_symbol})
    industry_info.columns = ['industry_code']
    return industry_info.join(series, on='industry_code', how='left')[[industry_symbol]]


def get_industry_code(industry_symbol, industry_info):
    if industry_symbol in ['sw_level_2', 'cs_level_2']:
        level_2_excel = "D:/Packages/FactorLib" + os.sep + "resource" + os.sep + "level_2_industry_dict.xlsx"
        level_2_dict = pd.read_excel(level_2_excel, sheetname=industry_symbol, header=0)
    industry_info.columns = ['industry_code']
    if industry_symbol == 'cs_level_2':
        level_2_dict['Code'] = level_2_dict['Code'].apply(lambda x: int(x[2:]))
        series = level_2_dict.set_index('Name').rename(columns={'Code': industry_symbol})
        return industry_info.join(series, on='industry_code', how='left')[[industry_symbol]]
    elif industry_symbol == 'sw_level_2':
        level_2_dict['Code'] = level_2_dict['Code'].apply(int)
        series = level_2_dict.set_index('Name').rename(columns={'Code': industry_symbol})
        temp = industry_info.join(series, on='industry_code', how='left')[[industry_symbol]]
        temp = temp.unstack().fillna(method='backfill').stack().astype('int32')
        return temp
    elif industry_symbol == 'sw_level_1':
        industry_info[industry_symbol] = industry_info['industry_code'].map(SW_INDUSTRY_DICT_REVERSE)
        industry_info.dropna(inplace=True)
        industry_info[industry_symbol] = industry_info[industry_symbol].str[:6].astype('int32')
        return industry_info[[industry_symbol]]
    elif industry_symbol == 'cs_level_1':
        industry_info[industry_symbol] = industry_info['industry_code'].map(CS_INDUSTRY_DICT_REVERSE)
        industry_info.dropna(inplace=True)
        industry_info[industry_symbol] = industry_info[industry_symbol].str[2:].astype('int32')
        return industry_info[[industry_symbol]]
    elif industry_symbol == 'wind_level_1':
        industry_info[industry_symbol] = industry_info['industry_code'].map(WIND_INDUSTRY_DICT_REVERSE)
        industry_info.dropna(inplace=True)
        industry_info[industry_symbol] = industry_info[industry_symbol].str[:6].astype('int32')
        return industry_info[[industry_symbol]]


# 将某报告期回溯N期
def RollBackNPeriod(report_date, n_period):
    Date = report_date
    for i in range(1,n_period+1):
        if Date[-4:]=='1231':
            Date = Date[0:4]+'0930'
        elif Date[-4:]=='0930':
            Date = Date[0:4]+'0630'
        elif Date[-4:]=='0630':
            Date = Date[0:4]+'0331'
        elif Date[-4:]=='0331':
            Date = str(int(Date[0:4])-1)+'1231'
    return Date


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
    report_dates = pd.date_range(_(start_date), _(end_date), freq='Q')
    return report_dates.strftime("%Y%m%d")


# 对财务数据进行重新索引
def financial_data_reindex(data, idx):
    idx2 = idx.reset_index(level=1)
    idx2.index = pd.DatetimeIndex(idx2.index)
    new_data = idx2.join(data, on=['max_report_date', 'IDs'])
    return new_data.set_index('IDs', append=True)


# 某个时间区间内的所有报告期(季度)
def get_all_report_periods(start, end):
    periods = pd.date_range(start, end, freq='Q', name='date')
    return periods


# 更新dict
def deep_update_dict(from_dict, to_dict):
    import collections
    for (key, value) in from_dict.items():
        if (key in to_dict.keys() and
                isinstance(to_dict[key], collections.Mapping) and
                isinstance(value, collections.Mapping)):
            deep_update_dict(value, to_dict[key])
        else:
            to_dict[key] = value


def distribute_equal(n, m):
    """把整数n平均分成m份"""
    Quotient = n // m
    Remainder = n % m
    Res = [Quotient] * m
    for i in range(Remainder):
        Res[i] += 1
    return Res


# 在给定的字符串列表str_list中寻找第一个含有name_list中给定字符串的字符串名字,如果没有找到，返回str_list的第一个元素
def searchNameInStrList(str_list, name_list):
    Rslt = None
    for iStr in str_list:
        for iNameStr in name_list:
            if iStr.find(iNameStr) != -1:
                Rslt = iStr
                break
        if Rslt is not None:
            break
    if Rslt is None:
        Rslt = str_list[0]
    return Rslt


# 生成多维索引
def generate_mulindex(method='product', **kwargs):
    if method == 'product':
        return pd.MultiIndex.from_product(kwargs.values(), names=list(kwargs))
    elif method == 'array':
        return pd.MultiIndex.from_arrays(kwargs.values(), names=list(kwargs))
    raise KeyError


# 返回一个可用名称
def get_available_names(name, used_names):
    i = 2
    while (name + str(i)) in used_names:
        i += 1
    return name + str(i)