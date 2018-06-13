# coding: utf-8
import pandas as pd
import numpy as np
import calendar
from pandas.tseries.offsets import QuarterEnd


def period_backward(dates, quarter=None, back_nyear=1, back_nquarter=None):
    """计算上年报告期, 默认返回上年同期"""
    if back_nquarter is not None:
        dates = pd.to_datetime(dates.astype('str')) - QuarterEnd(back_nquarter)
        return np.asarray(dates.strftime("%Y%m%d")).astype('int')
    year = dates // 10000
    month = dates % 10000 // 100
    c = calendar
    if quarter is not None:
        if isinstance(quarter, int):
            month = np.ones(len(dates)).astype('int') * quarter * 3
        else:
            month = quarter * 3
    day = np.asarray([c.monthrange(y-back_nyear, m)[1] for y, m in zip(year, month)])
    return (year - back_nyear) * 10000 + month * 100 + day


def quarter2intdate(year, quarter):
    if quarter in [1, 4]:
        day = 31
    else:
        day = 30
    return year * 10000 + quarter * 3 * 100 + day


def min_report_date(date):
    """返回当下日期理论上最晚的报告期
    计算原则：
        11月-次年4月份对应上年三季报
        5-8月份对应上年年报
        9-10月对应当年半年报
    """
    year = date // 10000
    month = date % 10000 // 100
    if 1 <= month <= 4:
        return (year - 1)*10000 + 930
    elif 5<= month <= 8:
        return (year - 1)*10000 + 1231
    elif 9 <= month <=10:
        return year*10000 + 630
    else:
        return year * 10000 + 930


def incr_rate(old, new):
    """计算增长率
    遵循如下计算规则:
    1. 分子分母其中一项是NaN，增长率也是NaN
    2. 分母为0时， 增长率是NaN
    3. 分母取绝对值，修正其为负数的情况

    Parameters:
    ============
    old: pd.Series
    new: pd.Series
    """
    old, new = old.align(new, axis=0, join='outer')
    old.replace(0.0, np.nan, inplace=True)
    return (new - old) / old.abs()


def avg(old, new):
    """计算期初期末平均值
    """
    old, new = old.align(new, axis=0, join='outer')
    old.fillna(new, inplace=True)
    return (old + new) / 2.0


def div(x, y):
    """return x/y

    parameters:
    ================
    x : pd.DataFrame

    y : pd.DataFrame
    """
    x, y = x.align(y, axis=0, join='inner')
    return x.iloc[:, 0] / y.iloc[:, 0]


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

            filter_date = min_report_date(date)
            data = data.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            latest = data.groupby('IDs')[[field_name, 'date']].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            last_year = self._last_nyear(tmp, latest['date'], quarter=4)
            last_quarter = self._last_nyear(tmp, latest['date'])
            ittm = latest[field_name] + last_year - last_quarter
            ittm.index = pd.MultiIndex.from_product([[date], ittm.index], names=['date', 'IDs'])
            r.append(ittm)
        return pd.concat(r)

    def last_nyear_ttm(self, raw_data, field_name, dates, n=1, ids=None):
        """回溯N年之前的TTM数值
        例如 2017-03-31报告期在1年之前的TTM值是2015-03-31至2016-03-31的数据
        """
        if ids is not None:
            raw_data = raw_data.query("IDs in @ids")
        r = []
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            filter_date = min_report_date(date)
            data = data.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            latest_periods = data.groupby('IDs')['date', 'quarter'].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            # 过去n年同期数据
            last_period = self._last_nyear(tmp, curr_period=latest_periods['date'],
                                           quarter=latest_periods['quarter'].values,
                                           back_nyear=n)
            # 过去n年年报
            last_year = self._last_nyear(tmp, curr_period=latest_periods['date'],
                                         quarter=4, back_nyear=n)

            # 过去n+1年同期数据
            last_period_2 = self._last_nyear(tmp, curr_period=latest_periods['date'],
                                             quarter=latest_periods['quarter'].values,
                                             back_nyear=n+1)
            ittm = last_period + last_year - last_period_2
            ittm.index = pd.MultiIndex.from_product([[date], ittm.index], names=['date', 'IDs'])
            r.append(ittm)
        return pd.concat(r)

    def last_nyear(self, raw_data, field_name, dates, n=1, ids=None, quarter=None):
        """回溯N年之前的财务数据"""
        if ids is not None:
            raw_data = raw_data.query("IDs in @ids")
        if quarter is not None:
            raw_data = raw_data.query('quarter==@quarter')
        r = []
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            filter_date = min_report_date(date)
            data = data.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            latest_periods = data.groupby('IDs')['date'].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            idata = self._last_nyear(tmp, latest_periods, quarter=quarter, back_nyear=n)
            idata.index = pd.MultiIndex.from_product([[date], idata.index], names=['date', 'IDs'])
            r.append(idata)
        return pd.concat(r)

    def last_nperiod(self, raw_data, field_name, dates, n=1, ids=None):
        """回溯N个报告期之前的财务数据"""
        if ids is not None:
            raw_data = raw_data.query("IDs in @ids")
        r = []
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            filter_date = min_report_date(date)
            data = data.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            latest_periods = data.groupby('IDs')['date'].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            idata = self._last_nperiod(tmp, latest_periods, back_quarter=n)
            idata.index = pd.MultiIndex.from_product([[date], idata.index], names=['date', 'IDs'])
            r.append(idata)
        return pd.concat(r)

    @staticmethod
    def _last_nyear(raw_data, curr_period, quarter=None, back_nyear=1):
        """以给定的报告期为基准， 回溯n年之前某个季度的财报数据
        Parameters:
        ===============
        raw_data: pd.Series
            索引：
            IDs: 股票代码, 格式必须与参数ids一致
            date：财报会计报告期, yyyymmdd的int格式
            值：财务数据
        curr_period: pd.Serise
            基准报告期, 以股票代码为索引, 会计报告期为数值
        quarter: int or list/array
            选择特定的季度。如果是标量，默认广播到所有股票；
            如果是向量，向量长度保持与股票数量相同。默认(None)为上年同期
        """
        new_period = period_backward(curr_period.values, quarter, back_nyear)
        new_idx = pd.MultiIndex.from_arrays([curr_period.index, new_period], names=['IDs', 'date'])
        return raw_data.reindex(new_idx).reset_index(level='date', drop=True)

    @staticmethod
    def _last_nperiod(raw_data, curr_period, back_quarter=1):
        """以给定的报告期为基准， 回溯n个报告期之前某个季度的财报数据
        Parameters:
        ===============
        raw_data: pd.Series
            索引：
            IDs: 股票代码, 格式必须与参数ids一致
            date：财报会计报告期, yyyymmdd的int格式
            值：财务数据
        curr_period: pd.Serise
            基准报告期, 以股票代码为索引, 会计报告期为数值
        back_quarter: int
            回溯n个季度
        """
        new_period = period_backward(curr_period.values, back_nquarter=back_quarter)
        new_idx = pd.MultiIndex.from_arrays([curr_period.index, new_period], names=['IDs', 'date'])
        return raw_data.reindex(new_idx).reset_index(level='date', drop=True)

    @staticmethod
    def latest_period(raw_data, field_name, dates, ids=None, quarter=None):
        """最近报告期财务数据
        Parameters:
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
        if quarter is not None:
            raw_data = raw_data.query("quarter==@quarter")
        if ids is not None:
            raw_data = raw_data.query("IDs in @ids")
        r = []
        for date in dates:
            tmp = raw_data.query("ann_dt <= @date")

            filter_date = min_report_date(date)
            tmp = tmp.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            latest_period = tmp.groupby('IDs')[field_name].last()
            latest_period.index = pd.MultiIndex.from_product([[date], latest_period.index], names=['date', 'IDs'])
            r.append(latest_period)
        return pd.concat(r)

    def inc_rate_tb(self, raw_data, field_name, dates, n=1, ids=None, quarter=None):
        """同比增长率"""
        new = self.latest_period(raw_data, field_name, dates, ids, quarter=quarter)
        old = self.last_nyear(raw_data, field_name, dates, n, ids, quarter=quarter)
        inc_r = incr_rate(old, new)
        return inc_r.to_frame("inc_rate")

    def inc_rate_hb(self, raw_data, field_name, dates, ids=None):
        """环比增长率"""
        if ids is not None:
            raw_data = raw_data.query("IDs in @ids")
        r = []
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            filter_date = min_report_date(date)
            data = data.groupby('IDs').filter(lambda x: x['date'].max() >= filter_date)

            new = raw_data.groupby('IDs')['date', field_name].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            old = self._last_nperiod(tmp, new['date'], back_quarter=1)
            inc_r = incr_rate(old, new[field_name])
            inc_r.index = pd.MultiIndex.from_product([[date], inc_r.index], names=['date', 'IDs'])
            r.append(inc_r)
        return pd.concat(r)

    def ttm_avg(self, raw_data, field_name, dates, ids=None):
        """最近12个月的平均值(期初+期末)/2， 一般用于资产负债表项目
        """
        new = self.latest_period(raw_data, field_name, dates=dates, ids=ids)
        old = self.last_nyear(raw_data, field_name, dates, 1, ids)
        return avg(old, new)

    def sq_avg(self, raw_data, field_name, dates, ids=None):
        """单季度平均值(期初+期末)/2, 一般用于资产负债表
        """
        new = self.latest_period(raw_data, field_name, dates=dates, ids=ids)
        old = self.last_nperiod(raw_data, field_name, dates=dates, ids=ids)
        return avg(old, new)


if __name__ == '__main__':
    loader = DataLoader()
    nprof = pd.read_hdf(r"D:\data\finance\sq_income\net_profit_excl_min_int_inc.h5", "data")
    ttm = loader.sq_avg(nprof, 'net_profit_excl_min_int_inc', [20180110], ids=[1, 2])
    print(ttm)