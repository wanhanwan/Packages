import pandas as pd
from iFinDPy import *
from const import THS_USERID, THS_PASSWORD, THS_BAR_DEFAULT_PARAMS, CS_INDUSTRY_CODES
from utils.tool_funcs import tradecode_to_windcode, drop_patch
from data_source.base_data_source_h5 import sec, tc, h5

THS_iFinDLogin(THS_USERID, THS_PASSWORD)


def _params2dict(params):
    _d = {}
    if not params:
        return _d
    for key_value in params.split(","):
        key, value = key_value.split(":")
        _d[key] = value
    return _d


def _param2str(param_dict):
    _s = []
    for k, v in param_dict.items():
        _s.append("%s:%s"%(k, v))
    return ",".join(_s)


def _adjust_params(params, pricetype, period, kwargs):
    p_dict = _params2dict(params)
    if pricetype is not None:
        p_dict['pricetype'] = pricetype
    if period is not None:
        p_dict['period'] = period
    p_dict.update(kwargs)
    return _param2str(p_dict)


def _updateHistoryBar(ids, start, end, fields, pricetype, period='D', **kwargs):
    if isinstance(fields, str):
        fields = [fields]
    params = _adjust_params(THS_BAR_DEFAULT_PARAMS, pricetype, period, kwargs)
    all_dates = tc.get_trade_days(start, end)
    fields = ",".join(fields)

    _l = []
    for d in all_dates:
        # print(d)
        date_param = '-'.join([d[:4], d[4:6], d[-2:]])
        codes_params = ",".join(ids)
        data = THS_HistoryQuotes(codes_params, fields, params, date_param, date_param)
        data_frame = THS_Trans2DataFrame(data)
        _l.append(data_frame)
    data = pd.concat(_l)
    data['IDs'] = data['thscode'].apply(drop_patch)
    data.rename(columns={'time': 'date'}, inplace=True)
    data['date'] = pd.DatetimeIndex(data['date'])
    data.set_index(['date', 'IDs'], inplace=True)
    data = data[pd.notnull(data.index.get_level_values(0))]
    return data


def _updateBasic(stocklist, field, params, dtype='float32'):
    all_dates = stocklist.index.get_level_values(0).unique()

    _l = []
    for d in all_dates:
        date_param = d.strftime("%Y-%m-%d")
        iparams = params.replace('date', date_param)
        codes_param = [tradecode_to_windcode(x) for x in stocklist.loc[d].index]
        codes_param = ",".join(codes_param)
        data = THS_BasicData(codes_param, field, iparams)
        data_frame = THS_Trans2DataFrame(data)
        _l.append(data_frame)
    data = pd.concat(_l)
    data['IDs'] = data['thscode'].apply(drop_patch)
    data.rename(columns={'time': 'date'}, inplace=True)
    data['date'] = pd.DatetimeIndex(data['date'])
    data.set_index(['date', 'IDs'], inplace=True)
    data = data[pd.notnull(data.index.get_level_values(0))].drop('thscode', axis=1).astype(dtype)
    return data


def financial_data(start, end, fields, param_dict, dtypes_dict={}):
    from utils.tool_funcs import ReportDateAvailable

    ids = sec.get_history_ashare(tc.get_trade_days("20070101", end))
    reportdts = pd.DatetimeIndex(ReportDateAvailable(start, end),name='date')
    ids = ids.unstack().reindex(reportdts, method='ffill').stack()
    for field in fields:
        if field in dtypes_dict:
            data = _updateBasic(ids, field, param_dict[field], dtypes_dict[field])
        else:
            data = _updateBasic(ids, field, param_dict[field])
        h5.save_factor(data, '/stock_financial_data/ths/')


if __name__ == '__main__':
    # 中信一级行业的历史行情数据
    fields = ['open', 'high', 'low', 'close', 'changeper', 'volume']
    data = _updateHistoryBar(CS_INDUSTRY_CODES, '20170708', '20170813', fields, 1)
    h5.save_factor(data, '/indexprices/cs_level_1/')

