# coding: utf-8
import pandas as pd
import numpy as np
import calendar
from pandas.tseries.offsets import QuarterEnd


def period_backward(dates, quarter=None, back_nyear=1, back_nquarter=None):
    """返回N年之前的同期日期

    Parameters:
    -----------
    dates: list of int date
        原始日期序列。
    quarter: int or list of int, Default None
        季度参数(1，2，3，4)，结果会返回N年之前该季度的最后一天。
    back_nyear: int
        回溯N年之前的同期日期。
    back_nquarter: int, default None
        回溯N个季度之前的日期, 如果该参数不是None, quarter
        和back_nyear两个参数无效。

    Examples:
    ----------
    >>> period_backward([20101231], back_nyear=2)
    [20081231]
    >>> period_backward([20101231], back_nquarter=2)
    [20090630]
    >>> period_backward([20101231], quarter=1)
    [20090331]
    """
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
            month = quarter.astype('int') * 3
    day = np.asarray([c.monthrange(y-back_nyear, m)[1] for y, m in zip(year, month)])
    return (year - back_nyear) * 10000 + month * 100 + day


def quarter2intdate(year, quarter):
    if quarter in [1, 4]:
        day = 31
    else:
        day = 30
    return year * 10000 + quarter * 3 * 100 + day


def safe_convert_int_to_date(df, columns):
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            df[c] = pd.to_datetime(df[c].apply(lambda x: str(int(x))), format='%Y%md')
    return df


def safe_convert_date_to_int(df, columns):
    for c in columns:
        if pd.api.types.is_datetime64_dtype(df[c].dtype):
            df[c] = df[c].dt.strftime("%Y%m%d").astype('int64')
    return df


def safe_extract_year_from_date(ser):
    if pd.api.types.is_numeric_dtype(ser):
        year = ser // 10000
    else:
        year = ser.dt.year
    return year.astype('int64')


def safe_extract_quarter_from_date(ser):
    if pd.api.types.is_numeric_dtype(ser):
        quarter = ser % 10000 // 100 / 4
    else:
        quarter = ser.dt.quarter
    return quarter.astype('int64')


def min_report_date(date, quarter=None):
    """返回当下日期理论上最晚的报告期
    计算原则：
        11月-次年4月份对应上年三季报
        5-8月份对应上年年报
        9-10月对应当年半年报
    """
    year = date // 10000
    month = date % 10000 // 100
    if 1 <= month <= 4:
        new_d = (year - 1)*10000 + 930
    elif 5<= month <= 8:
        new_d = (year - 1)*10000 + 1231
    elif 9 <= month <=10:
        new_d = year*10000 + 630
    else:
        new_d = year * 10000 + 930
    if quarter:
        cur_q = new_d % 10000 // 100 / 3
        if quarter == cur_q:
            return new_d
        elif quarter > cur_q:
            return period_backward(np.array([new_d]), quarter=quarter)[0]
        else:
            return period_backward(np.array([new_d]),
                                   back_nquarter=cur_q-quarter)[0]
    else:
        return new_d


def incr_rate(old, new):
    """计算增长率
    遵循如下计算规则:
    1. 分子分母其中一项是NaN，增长率也是NaN
    2. 分母为0时， 增长率是NaN
    3. 分母取绝对值，修正其为负数的情况

    Parameters:
    ============
    old: pd.DataFrame
    new: pd.DataFrame
    """
    old, new = old.align(new, join='outer')
    old.replace(0.0, np.nan, inplace=True)
    return (new - old) / old.abs()


def avg(old, new):
    """计算期初期末平均值
    """
    old, new = old.align(new, join='outer')
    old.fillna(new, inplace=True)
    return (old + new) / 2.0


def div(x, y):
    """return x/y

    parameters:
    ================
    x : pd.DataFrame

    y : pd.DataFrame
    """
    x, y = x.align(y, join='inner')
    y.replace(0.0, np.nan, inplace=True)
    return x / y


class DataLoader(object):
    """财务数据库的数据加载器"""

    def ttm(self, raw_data, field_name, dates, ids, dtype='float64'):
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
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")
            latest = data.groupby('IDs')[[field_name, 'date']].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            last_year = self._last_nyear(tmp, latest['date'], quarter=4)
            last_quarter = self._last_nyear(tmp, latest['date'])
            ittm = latest[field_name] + last_year - last_quarter
            r.loc[date, :] = ittm
        return r

    def last_nyear_ttm(self, raw_data, field_name, dates, ids, dtype='float64', n=1):
        """回溯N年之前的TTM数值
        例如 2017-03-31报告期在1年之前的TTM值是2015-03-31至2016-03-31的数据
        """
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            latest_periods = data.groupby('IDs')['date', 'quarter'].last()
            pre_periods = pd.Series(period_backward(latest_periods['date'].values, back_nyear=n),
                                    index=latest_periods.index)
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            # 过去n年同期数据
            last_period = self._last_nyear(tmp, curr_period=latest_periods['date'],
                                           quarter=latest_periods['quarter'].values,
                                           back_nyear=n)
            # 过去n年年报
            last_year = self._last_nyear(tmp, curr_period=pre_periods,
                                         quarter=4, back_nyear=1)

            # 过去n+1年同期数据
            last_period_2 = self._last_nyear(tmp, curr_period=latest_periods['date'],
                                             quarter=latest_periods['quarter'].values,
                                             back_nyear=n+1)
            ittm = last_period + last_year - last_period_2
            r.loc[date, :] = ittm
        return r

    def last_nperiod_ttm(self, raw_data, field_name, dates, ids, dtype='float64', n=1):
        """回溯N年之前的TTM数值
        例如 2017-03-31报告期在1年之前的TTM值是2015-03-31至2016-03-31的数据
        """
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            latest_periods = data.groupby('IDs')['date', 'quarter'].last()
            pre_periods = pd.Series(period_backward(latest_periods['date'].values, back_nquarter=n),
                                    index=latest_periods.index)
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            # 过去n期数据
            last_period = self._last_nperiod(tmp, curr_period=latest_periods['date'],
                                             back_quarter=n)
            # 过去n年年报
            last_year = self._last_nyear(tmp, curr_period=pre_periods,
                                         quarter=4, back_nyear=1)

            # 过去n+1年同期数据
            last_period_2 = self._last_nperiod(tmp, curr_period=latest_periods['date'],
                                               back_quarter=n+4)
            ittm = last_period + last_year - last_period_2
            r.loc[date, :] = ittm
        return r

    def last_nyear(self, raw_data, field_name, dates, ids, dtype='float64', n=1, quarter=None):
        """回溯N年之前同期的财务数据

        Parameters:
        -----------
        raw_data: DataFrame
            原始数据，必须含有一下字段：
            1. IDs: 股票代码, 格式必须与参数ids一致
            2. date：财报会计报告期, yyyymmdd的int格式
            3. ann_dt: 财报的公告期, yyyymmdd的int格式
            4. quarter: 会计报告期所在季度, int
            5. year: 会计报告期所在年份, int
        field_name: str
            财务数据的字段名称
        dates: list of int
            日期序列
        ids: list
            股票序列
        dtype: str
            财务数据的数据类型
        n: int
            回溯N年之前
        quarter: int
            只使用某个季度(报告期)的财务数据
        """
        if quarter is not None:
            raw_data = raw_data.query('quarter==@quarter')
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            latest_periods = data.groupby('IDs')['date'].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            idata = self._last_nyear(tmp, latest_periods, quarter=quarter, back_nyear=n)
            r.loc[date, :] = idata
        return r

    def last_nyears(self, raw_data, field_name, dates, ids=None, n=1, quarter=None, dtype='float64'):
        """回溯0~N年之前同期的财务数据"""
        if quarter is not None:
            raw_data = raw_data.query('quarter==@quarter')
        if ids is not None:
            raw_data = raw_data.query('IDs in @ids')
        raw_data = raw_data.sort_values(['IDs', 'ann_dt', 'date'])
        r = [None] * len(dates)
        for i, date in enumerate(dates):
            data = raw_data.query("ann_dt <= @date")
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            idata = tmp.groupby('IDs').tail(n)
            r[i] = idata
        df = pd.concat(r, keys=pd.to_datetime(np.asarray(dates, dtype='int').astype('str'), format='%Y%m%d'))
        df = df.unstack()
        df.index.names = ['date', 'IDs']
        return df

    def last_nperiod(self, raw_data, field_name, dates, ids, dtype='float64', n=1):
        """回溯N个报告期之前的财务数据"""
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            latest_periods = data.groupby('IDs')['date'].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()

            idata = self._last_nperiod(tmp, latest_periods, back_quarter=n)
            r.loc[date, :] = idata
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
    def latest_period(raw_data, field_name, dates, ids, dtype='float64', quarter=None):
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
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            tmp = raw_data.query("ann_dt <= @date")

            latest_period = tmp.groupby('IDs')[field_name].last()
            r.loc[date, :] = latest_period
        return r

    def inc_rate_tb(self, raw_data, field_name, dates, n=1, ids=None, quarter=None):
        """同比增长率"""
        new = self.latest_period(raw_data, field_name, dates, ids, quarter=quarter)
        old = self.last_nyear(raw_data, field_name, dates, n, ids, quarter=quarter)
        inc_r = incr_rate(old, new)
        return inc_r

    def inc_rate_hb(self, raw_data, field_name, dates, ids, dtype='float64'):
        """环比增长率"""
        def _incr_rate(old, new):
            old, new = old.align(new, axis=0, join='outer')
            old.replace(0.0, np.nan, inplace=True)
            return (new - old) / old.abs()

        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")

            new = raw_data.groupby('IDs')['date', field_name].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            old = self._last_nperiod(tmp, new['date'], back_quarter=1)
            inc_r = _incr_rate(old, new[field_name])
            r.loc[date, :] = inc_r
        return r

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


data_loader = DataLoader()
def run_data_loader(func_name, data, field_name, idx=None, dates=None, ids=None, **kwargs):
    """
    函数式运行DataLoader实例

    Parameters:
    -----------
    func_name: str
        DataLoader成员函数名称, 如ttm
    data: DataFrame
        必须有的列：IDs date ann_dt，日期可以是datetime格式，也可以是int格式。
        可选的列：year、quarter
    
    Returns:
        DataFrame
    """
    data = data.copy()
    assert {field_name, 'IDs', 'date'} < set(data.columns)
    safe_convert_date_to_int(data, ['date', 'ann_dt'])
    if 'year' not in data.columns:
        data['year'] = safe_extract_year_from_date(data['date'])
    if 'quarter' not in data.columns:
        data['quarter'] = safe_extract_quarter_from_date(data['date'])
    
    if idx is not None:
        dates = [int(x.strftime('%Y%m%d')) for x in idx.index.get_level_values('date').unique()]
        ids = list(idx.index.get_level_values('IDs').unique())
    df = getattr(data_loader, func_name)(data, field_name, dates = dates, ids = ids, **kwargs)
    if df.index.nlevels == 1 and pd.api.types.is_numeric_dtype(df.index.dtype):
        df.index = pd.DatetimeIndex([str(x) for x in df.index])
    return df

class ConsistentPeriodDataLoader(object):

    def _load_period_data(self, data, period, filed_name):
        return data[data['date']==period][filed_name]


    def ttm(self, raw_data, field_name, dates, ids, dtype):
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
        r = pd.DataFrame(index=dates, columns=ids, dtype=dtype)
        for date in dates:
            data = raw_data.query("ann_dt <= @date")
            latest = data.groupby('IDs')[[field_name, 'date']].last()
            tmp = data.groupby(['IDs', 'date'])[field_name].last()
            last_year = self._last_nyear(tmp, latest['date'], quarter=4)
            last_quarter = self._last_nyear(tmp, latest['date'])
            ittm = latest[field_name] + last_year - last_quarter
            r.loc[date, :] = ittm
        return r
