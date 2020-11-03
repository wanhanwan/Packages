import pandas as pd
import numpy as np
import warnings
import re

from datetime import datetime
from pandas._libs import OutOfBoundsDatetime


def suffix_ids(ids):
    if re.match(r'^[0,3,6][0-9]{5}$', ids[0]):
        return [x+'.SH' if x[0]=='6' else x+'.SZ' for x in ids]
    return ids


def _argdict_to_arglist(arg_dict):
    return [x+'='+str(y) for x, y in arg_dict.items()]


def _safe_format_date(date):
    if isinstance(date, str):
        date = date.replace('-', '')
        return "%s-%s-%s" % (date[:4], date[4:6], date[6:8])
    elif isinstance(date, (pd.Timestamp, datetime)):
        return date.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("非法的日期格式!")


def _as_timestamp(obj):
    if isinstance(obj, pd.Timestamp):
        return obj
    try:
        return pd.Timestamp(obj)
    except OutOfBoundsDatetime:
        pass
    return obj


class WindAddIn(object):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if WindAddIn._instance is None:
            WindAddIn._instance = object.__new__(cls)
        return WindAddIn._instance
    
    def __init__(self):
        self.w = None

    def connect(self):
        if self.w is None:
            from WindPy import w
            self.w = w
            self.w.start()

    def wss_convert_to_df(self, data, dt):
        ids = data.Codes
        dates = [_as_timestamp(dt)] * len(ids)
        res = list(zip(*([ids, dates] + data.Data)))
        columns = ['IDs', 'date'] + [x.lower() for x in data.Fields]
        df = pd.DataFrame(res, columns=columns)
        return df
    
    def wsd_convert_to_df(self, data):
        dates = [_as_timestamp(x) for x in data.Times]
        ids = data.Codes * len(dates)
        res = list(zip(*([ids, dates] + data.Data)))
        columns = ['IDs', 'date'] + [x.lower() for x in data.Fields]
        df = pd.DataFrame(res, columns=columns)
        return df
    
    def edb_convert_to_df(self, data):
        dates = [_as_timestamp(x) for x in data.Times]
        if (len(dates) > 1 and len(data.Codes)>1) or len(data.Codes)==1:
            df = pd.DataFrame(np.asarray(data.Data, dtype='float').T,
                              columns=data.Codes,
                              index=dates)
        else:
            df = pd.DataFrame(data.Data, columns=data.Codes, index=dates)
        return df
    
    def wss(self, filed_names, ids, dates, date_field='tradeDate', arg_dict=None):
        """封装WindAPI的wss函数
        
        Return:
        =======
        DataFrame:(index=[date, IDs], columns=fileds)
        """
        self.connect()
        ids = suffix_ids(ids)
        arg_dict = {} if arg_dict is None else arg_dict

        res = [None] * len(dates)
        for i, d in enumerate(dates):
            date_arg = "%s=%s"%(date_field, d)
            arg_list = [date_arg] + _argdict_to_arglist(arg_dict)
            data = self.w.wss(ids, filed_names, *arg_list)
            if data.ErrorCode != 0:
                warnings.warn(("wss数据提取失败!"
                               "错误代码:{error_code};"
                               "数据截面日期:{dt}").format(
                                    error_code=str(data.ErrorCode),
                                    dt=d))
            res[i] = self.wss_convert_to_df(data, d)
        df = pd.concat(res).set_index(['date', 'IDs'])
        return df
    
    def wsd(self, field_names, ids, start_dt, end_dt, arg_dict=None):
        self.connect()
        ids = suffix_ids(ids)
        arg_dict = {} if arg_dict is None else arg_dict

        res = [None] * len(ids)
        for i, stock in enumerate(ids):
            arg_list = _argdict_to_arglist(arg_dict)
            data = self.w.wsd(stock, field_names, start_dt, end_dt, *arg_list)
            if data.ErrorCode != 0:
                warnings.warn(("wsd数据提取失败!"
                               "错误代码:{error_code};"
                               ).format(error_code=str(data.ErrorCode)))
            res[i] = self.wsd_convert_to_df(data)
        df = pd.concat(res).set_index(['date', 'IDs'])
        return df
    
    def edb(self, field_code, start_date, end_date,
            fill_method=None, frequency=None,
            usenames=None):
        """从Wind终端下载宏观数据.

        Parameters: 
        ===========
        field_code: str
            宏观数据指标的Wind ID. 例如，"工业企业:利润总额:累计同比"所对应的Wind ID是"M0000557".
        
            更多指标可在Wind客户端输入"CG"查看.
        start_date: str or datetime-like
            数据的开始日期, YYYYMMDD
        end_date: str or datetime-like
            数据的结束日期, YYYYMMDD.
        fill_method: str, None by default
            无交易数据处理.默认是沿用上期数值，如果是None，则用NaN替代.
        frequency: str, None by default
            数据采样频率
        
        Return:
        ===========
        pandas.DataFrame
        """
        self.connect()
        start_date = _safe_format_date(start_date)
        end_date = _safe_format_date(end_date)
        
        params_str = ''
        if fill_method == 'Previous':
            params_str += 'Fill=Previous'

        data = self.w.edb(field_code, start_date, end_date, params_str)
        final = self.edb_convert_to_df(data)
        if usenames is not None:
            if isinstance(final, pd.Series):
                final.name = usenames
            else:
                final.columns = usenames
        return final


if __name__ == '__main__':
    wind_api = WindAddIn()
    d = wind_api.wsd('close, high', ['000001', '600000'], start_dt='20190410', end_dt='20190410')
    print(d)
