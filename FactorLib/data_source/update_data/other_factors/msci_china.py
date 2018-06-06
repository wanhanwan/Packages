# 更新MSCI中国A股成分股
from FactorLib.utils.tool_funcs import get_resource_abs_path, windcode_to_tradecode
from FactorLib.utils.datetime_func import DateRange2Dates
import pandas as pd


@DateRange2Dates
def get_msci_members(start_date=None, end_date=None, dates=None):
    msci_list = pd.read_csv(get_resource_abs_path()/'msci_index_list.csv', header=0, converters={'IDs': windcode_to_tradecode},
                       parse_dates=['into_date', 'out_date']).set_index('IDs')
    dates = pd.to_datetime(dates)
    rslt = []
    for d in dates:
        members = msci_list[(msci_list['into_date']<=d)&((msci_list['out_date'].isnull()) | (msci_list['out_date']>d))]
        rslt.append(pd.Series([1]*len(members), index=pd.MultiIndex.from_product([[d], members.index])))
    rslt = pd.concat(rslt).to_frame('msci_china').sort_index()
    rslt.index.names = ['date', 'IDs']
    return rslt


def save_msci_members(start_date, end_date, **kwargs):
    msci_members = get_msci_members(start_date=start_date, end_date=end_date)
    kwargs['data_source'].h5DB.save_factor(msci_members, '/indexes/')


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    save_msci_members(start_date='20170701', end_date='20180605', data_source=data_source)

