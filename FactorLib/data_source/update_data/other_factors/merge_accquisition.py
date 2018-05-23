from FactorLib.data_source.uqer_db import UqerDB
import pandas as pd
import numpy as np


def merge_in_one_year(data4, year, max_date=None):
    if max_date is None:
        return data4[data4['year']==year]['ticker'].unique()
    else:
        return data4[(data4['year']==year) & (data4['publishDate']<=max_date)]['ticker'].unique()


def merge_acc(dates):
    """返回指定日期最近一年发生并购事件的股票
    遍历每一天，回顾当年和上一年份上市公司是否发生并购行为
    """
    raw_data = pd.read_hdf(r"D:\data\h5\temp\ma_all.h5", "data")
    dates = pd.DatetimeIndex(dates)
    rslt = [np.nan] * len(dates)
    for i, d in enumerate(dates):
        year = d.year
        all_ids = np.union1d(merge_in_one_year(raw_data, year, d), merge_in_one_year(raw_data, year-1))
        rslt[i] = pd.DataFrame(np.ones(len(all_ids), dtype='int'),
                               index=pd.MultiIndex.from_product([[d], all_ids], names=['date', 'IDs']),
                               columns=['merge_acc'])
    return pd.concat(rslt)


def update_raw_from_uqer(start, end, **kwargs):
    u = UqerDB.get_instance()
    u.connect()
    s = kwargs['data_source'].trade_calendar.get_trade_days(start, end)[0][:4]+'0101'
    s = kwargs['data_source'].trade_calendar.tradeDayOffset(s, 1, incl_on_offset_today=True)
    e = s[:4]+'1231'
    all_ids = kwargs['data_source'].sector.get_history_ashare(dates=[s], history=False)
    data = u.run_api('EquRestructuringGet', ticker=list(all_ids[all_ids['ashare']==1.0].index.get_level_values('IDs')),
                     field=['ticker', 'program', 'restructuringType',
                            'iniPublishDate', 'isSucceed', 'institNameSub', 'publishDate'],
                     beginDate=s, endDate=e)
    data2 = data[
        (data['program'].isin(['3', '17', '8', '12'])) & (data['isSucceed'] == 1) & data['restructuringType'].isin(
            ['1', '4', '7'])]
    data2 = data2.dropna()
    if not data2.empty:
        data2[['program', 'restructuringType']] = data2[['program', 'restructuringType']].astype('int')
        data2['iniPublishDate'] = pd.DatetimeIndex(data2['iniPublishDate'])
        data2['publishDate'] = pd.DatetimeIndex(data2['publishDate'])
        data2['year'] = data2['publishDate'].dt.year
        kwargs['data_source'].h5DB.save_h5file(data2, 'ma_all', '/temp/')


if __name__ == '__main__':
    import sys
    sys.path.append("D:/scripts")
    from FactorLib.data_source.base_data_source_h5 import data_source
    from save_factor import save
    save('20180101', '20180305', path='/indexes/', factor_name='merge_acc', func=merge_acc)
    # update_raw_from_uqer('20140101', '20141231', data_source=data_source)
    # data = pd.read_hdf('D:/data/merge_2018.h5', 'data')
    # data_source.h5DB.save_h5file(data, 'ma_all', '/temp/')