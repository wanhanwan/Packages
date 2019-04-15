# coding: utf-8
"""一些工具函数"""
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
def tradecode_to_uqercode(tradecode):
    return tradecode+'.XSHG' if tradecode[0] == '6' else tradecode+'.XSHE'

@clru_cache()
def tradecode_to_tslcode(code):
    if code[:2].isalpha():
        return code
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
