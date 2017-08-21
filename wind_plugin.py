# wind插件API
from WindPy import *
from data_source import data_api
from utils.tool_funcs import tradecode_to_windcode, windcode_to_tradecode
import pandas as pd
import xlrd

argInfoWB = xlrd.open_workbook("D:/MultiFactor/Resource/WindAddin.xlsx") 
argInfo = pd.read_excel(argInfoWB,sheetname='ArgInfo',engine='xlrd')

# 行情数据接口
def get_history_bar(field_names, start_date, end_date, **kwargs):
    field_info = pd.read_excel(argInfoWB,sheetname='收盘行情',engine='xlrd')
    if not isinstance(field_names,list):
        field_names = [field_names]
    # 按照字段循环取数据
    _l = []
    w.start()
    for fieldName in field_names:
        field_name = field_info[field_info['FactorName']==field_name]['FieldName'].iat[0]
        args = field_info[field_info['FactorName']==field_name]['Args'].iat[0]
        
        params = _parse_args(args,**kwargs)
        all_days = data_api.tc.get_trade_days(start_date, end_date)
        all_ids = data_api.get_history_ashare(all_days).index.levels[1].unique()

        data = w.wsd(
            list(map(tradecode_to_windcode, all_ids)), field_name, start_date, end_date, params)
        _l.append(_bar_to_dataframe(data))
    data = pd.concat(_l,axis=1)
    w.close()
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
            (ArgInfo['ArgName']==arg) & (ArgInfo['ArgValue']==kwargs[arg])]['ArgValueStr'].iat[0]
        arg_str.append(
            "{arg_name}={arg_value}".format(
                arg_name=arg_name_str, arg_value=arg_value_str)
            )
    return ",".join(arg_str)

def _bar_to_dataframe(data):
    """把windAPI数据转换成dataframe"""
    ids = list(map(windcode_to_tradecode, data.Codes))
    dates = pd.DatetimeIndex(data.Times, name='date')
    col = pd.Index(ids, name='IDs')
    df = pd.DataFrame(data.Data).T
    df.index = dates
    df.columns = col
    df = df.stack().to_frame().sort_index().rename(columns={0:data.Fields[0]})
    return df


