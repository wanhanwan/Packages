# coding: utf-8
import pandas as pd
import numpy as np
import calendar


def period_backward(dates, quarter=None):
    """计算上年报告期, 默认返回上年同期"""
    year = dates // 10000
    month = dates % 10000 // 100
    day = dates % 10000 % 100
    if quarter is None:
        return (year - 1) * 10000 + month * 100 + day
    else:
        c = calendar
        day = np.asarray([c.monthrange(y, m) for y, m in zip(year, month)])
        return (year - 1) * 10000 + month * 100 + day


class DataLoader(object):
    """财务数据库的数据加载器"""

    def ttm(self, raw_data, field_name, dates, ids=None):
        """加载ttm数据

        Parameters
        ==============
        raw_data: pd.DataFrame
            原始的财务数据, 数据框中必须包含一下字段:
                IDs: 股票代码, 格式必须与参数ids一致
                date：财报会计报告期, yyyymmdd的int格式
                ann_dt: 财报的公告期, yyyymmdd的int格式
                quarter: 会计报告期所在季度, int
                year: 会计报告期所在年份, int
        field_name: str
            待计算的财务数据字段名称
        dates: list/array
            日期序列, yyyymmdd的int格式
        ids: same type as IDs in raw_data
        """
        r = []
        for date in dates:
            if ids is not None:
                data = raw_data.query("ann_dt <= @date & IDs in @ids")
            else:
                data = raw_data.query("ann_dt <= @date")
            latest = data.groupby('IDs')[[field_name, 'date']].last()
            latest_year = data.query("quarter == 4").groupby('IDs')[field_name].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            last_year = self.last_year(tmp, latest['date'])
            ittm = latest[field_name] + latest_year - last_year
            ittm.index = pd.MultiIndex.from_product([[date], ittm.index], names=['date', 'IDs'])
            r.append(ittm)
        return pd.concat(r)

    @staticmethod
    def last_year(raw_data, curr_period, quarter=None):
        """最新报告期的上年财报数据

        Parameters:
        ===============
        raw_data: pd.Series
            索引：
            IDs: 股票代码, 格式必须与参数ids一致
            date：财报会计报告期, yyyymmdd的int格式
            值：财务数据
        curr_period: pd.Serise
            基准报告期, 以股票代码为索引, 会计报告期为数值
        quarter: int
            选择上年的季度, 默认(None)为上年同期
        """
        new_period = period_backward(curr_period.values, quarter)
        new_idx = pd.MultiIndex.from_arrays([curr_period.index, new_period], names=['IDs', 'date'])
        return raw_data.reindex(new_idx).reset_index(level='date', drop=True)


if __name__ == '__main__':
    loader = DataLoader()
    np = pd.read_hdf(r"D:\data\finance\income\net_profit_excl_min_int_inc.h5", "data")
    ttm = loader.ttm(np, 'net_profit_excl_min_int_inc', [20180110], ids=[1, 2])
    print(ttm)