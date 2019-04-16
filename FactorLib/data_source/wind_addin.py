import pandas as pd
import warnings
import re

from pandas.tseries.offsets import as_timestamp


def suffix_ids(ids):
    if re.match(r'^[0,3,6][0-9]{5}$', ids[0]):
        return [x+'.SH' if x[0]=='6' else x+'.SZ' for x in ids]
    return ids


def _argdict_to_arglist(arg_dict):
    return [x+'='+str(y) for x, y in arg_dict.items()]


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
        dates = [as_timestamp(dt)] * len(ids)
        res = list(zip(*([ids, dates] + data.Data)))
        columns = ['IDs', 'date'] + [x.lower() for x in data.Fields]
        df = pd.DataFrame(res, columns=columns)
        return df
    
    def wsd_convert_to_df(self, data):
        dates = [as_timestamp(x) for x in data.Times]
        ids = data.Codes * len(dates)
        res = list(zip(*([ids, dates] + data.Data)))
        columns = ['IDs', 'date'] + [x.lower() for x in data.Fields]
        df = pd.DataFrame(res, columns=columns)
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


if __name__ == '__main__':
    wind_api = WindAddIn()
    d = wind_api.wsd('close, high', ['000001', '600000'], start_dt='20190410', end_dt='20190410')
    print(d)
