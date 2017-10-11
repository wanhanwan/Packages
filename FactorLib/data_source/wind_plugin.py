# wind插件API
from WindPy import *
from . import data_api
from .base_data_source_h5 import h5
from ..utils.tool_funcs import tradecode_to_windcode, drop_patch
from .trade_calendar import as_timestamp
from ..const import MARKET_INDEX_WINDCODE
import pandas as pd
import xlrd
import os

current_path = os.path.abspath(os.path.dirname(__file__)+'/../resource')
workbook_path = os.path.join(current_path, 'wind_addin.xlsx')

argInfoWB = xlrd.open_workbook(workbook_path) 
argInfo = pd.read_excel(argInfoWB,sheetname='ArgInfo',engine='xlrd')

w.start()
# 行情数据接口
def get_history_bar(field_names, start_date, end_date, id_type='stock', **kwargs):
    if not isinstance(field_names,list):
        field_names = [field_names]

    if id_type == 'stock':
        field_info = pd.read_excel(argInfoWB,sheetname='收盘行情',engine='xlrd')
         # 按照字段循环取数据
        _l = []
        for fieldName in field_names:
            field_name = field_info[field_info['FactorName']==fieldName]['FieldName'].iat[0]
            args = field_info[field_info['FactorName']==fieldName]['Args'].iat[0]
          
            params = _parse_args(eval(args),**kwargs)
            all_days = data_api.tc.get_trade_days(start_date, end_date)
            all_ids = data_api.get_history_ashare(all_days).index.levels[1].unique()
  
            data = w.wsd(
                list(map(tradecode_to_windcode, all_ids)), field_name, start_date, end_date, params)
            _l.append(_bar_to_dataframe(data))
        data = pd.concat(_l,axis=1)
    elif id_type == 'index':
        field_info = pd.read_excel(argInfoWB,sheetname='指数收盘行情',engine='xlrd')
        all_ids = [MARKET_INDEX_WINDCODE[x] for x in MARKET_INDEX_WINDCODE]
        _l = []
        for fieldName in field_names:
            field_name = field_info[field_info['FactorName']==fieldName]['FieldName'].iat[0]
            args = field_info[field_info['FactorName']==fieldName]['Args'].iat[0]
          
            params = _parse_args(eval(args),**kwargs)
            data = w.wsd(all_ids, field_name, start_date, end_date, params)
            _l.append(_bar_to_dataframe(data))
        data = pd.concat(_l,axis=1)
    return data


def get_wsd(field_names, start_date, end_date, **kwargs):
    all_days = data_api.tc.get_trade_days(start_date, end_date)
    all_ids = data_api.get_history_ashare(all_days).index.levels[1].unique()
    all_ids = list(map(tradecode_to_windcode, all_ids))
    data = _load_wsd_data(all_ids, field_names, start_date, end_date, **kwargs)
    return data


# WSQ实时行情接口
def realtime_quote(fieldnames, ids=None):
    if not isinstance(fieldnames, list):
        fieldnames = [fieldnames]
    if ids is None:
        ids = data_api.get_history_ashare(datetime.today().strftime("%Y%m%d")).index.get_level_values(1)
    _l = []
    for field in fieldnames:
        data = w.wsq(list(map(tradecode_to_windcode, ids)), field)
        _l.append(_bar_to_dataframe(data))
    data = pd.concat(_l, axis=1)
    return data


def _parse_args(args,**kwargs):
    """解析参数信息，返回WindAPI字符串"""
    arg_str = []
    for arg in args:
        arg_name_str = argInfo[argInfo['ArgName']==arg]['ArgNameStr'].iat[0]
        if arg not in kwargs:
            arg_value_str = argInfo[argInfo['ArgName']==arg]['DefaultValue'].iat[0]
        else:
            arg_value_str = argInfo[
            (argInfo['ArgName']==arg) & (argInfo['ArgValue']==kwargs[arg])]['ArgValueStr'].iat[0]
            if (arg_value_str == 'NaN') or pd.isnull(arg_value_str):
                continue
        arg_str.append(
            "{arg_name}={arg_value}".format(
                arg_name=arg_name_str, arg_value=arg_value_str)
            )
    return ";".join(arg_str)


def _bar_to_dataframe(data):
    """把windAPI数据转换成dataframe"""
    if data.ErrorCode != 0:
        raise ValueError(str(data.ErrorCode))
    ids = list(map(drop_patch, data.Codes))
    dates = [as_timestamp(x) for x in data.Times]
    col = pd.Index(ids, name='IDs')
    if len(dates) == 1:
        df = pd.DataFrame(data.Data)
    else:
        df = pd.DataFrame(data.Data).T
    df.index = pd.DatetimeIndex(dates, name='date')
    df.columns = col
    df = df.stack().to_frame().sort_index().rename(columns={0: data.Fields[0].lower()})
    return df


def _params2dict(params):
    _d = {}
    if params:
        for key_value in params.split(";"):
            key, value = key_value.split("=")
            _d[key] = value
    return _d

def _param2str(param_dict):
    _s = []
    for k, v in param_dict.items():
        _s.append("%s=%s"%(k, v))
    return ";".join(_s)

def _adjust_params(params, kwargs):
    p_dict = _params2dict(params)
    p_dict.update(kwargs)
    return _param2str(p_dict)

def _load_wsd_data(ids, fields, start, end, **kwargs):
    if isinstance(fields, str):
        fields = [fields]
    params = _adjust_params("", kwargs)
    ids = ",".join(ids)
    _l = []
    for field in fields:
        d = w.wsd(ids, field, start, end, params)
        _l.append(_bar_to_dataframe(d))
    data = pd.concat(_l, axis=1)
    return data

if __name__ == '__main__':
    from const import CS_INDUSTRY_DICT
    codes = [x+'.WI' for x in CS_INDUSTRY_DICT]
    pct_change = _load_wsd_data(codes, 'pct_chg', '20110101', '20170709')
    pct_change.columns=['changeper']
    h5.save_factor(pct_change, '/indexprices/cs_level_1/')

