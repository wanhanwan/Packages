# encoding: utf-8
# author: wanhanwan
"""金融数据加载器

金融数据的特点，首先是有报告期、公告期；其次财务数据会区分调整前和调整后。
为了避免用到未来数据，我们需要提供一套标准化的数据处理接口。
"""
import pandas as pd
import numpy as np
from FactorLib.utils.tool_funcs import ReportDateAvailable


class FinancialDataLoader(object):
    """数据加载函数的集合

    每一个成员函数都接受一个raw_df(dataframe)，作为参数输入。
    raw_df(dataframe)必须存在如下字段：
    IDs-股票代码
    date-报告期
    ann_dt-公告日期
    quarter-财报季度
    """
    def lastest_period(self, raw_df, field_name, dates, ids=None, quarter=None):
        """最近财报期的数据"""
        if quarter is not None:
            if not isinstance(quarter, list):
                quarter = [quarter]
            raw_df = raw_df.query("quarter in @quarter")
        if ids is not None:
            raw_df = raw_df.query("IDs in @ids")
        dates = pd.to_datetime(dates)
        raw_df.sort_values(['IDs', 'date', 'ann_dt'], inplace=True)
        r = [None] * len(dates)
        for i, dt in enumerate(dates):
            tmp = raw_df.query("ann_dt <= @dt")
            data = tmp.groupby('IDs')[field_name].last()
            r[i] = data
        df = pd.concat(r, keys=dates).rename_axis(['date', 'IDs']).to_frame()
        return df

    def latest_period_with_consistent_report_dt(self, raw_df, field_name,
                                                dates, ids=None, quarter=None):
        """最近财报期的数据，每一个横截面统一财报日期"""
        dates = pd.to_datetime(dates)
        report_dates = pd.to_datetime([ReportDateAvailable(x, x) for x in dates.strftime("%Y%m%d")])
        if quarter is not None:
            if not isinstance(quarter, list):
                quarter = [quarter]
            raw_df = raw_df.query("quarter in @quarter")
        if ids is not None:
            raw_df = raw_df.query("IDs in @ids")
        raw_df.sort_values(['IDs', 'date', 'ann_dt'], inplace=True)
        r = [None] * len(dates)
        for i, dt in enumerate(dates):
            report_dt = report_dates[i]
            tmp = raw_df.query("(ann_dt<=@dt)&(date==@report_dt)")
            data = tmp.groupby('IDs')[field_name].last()
            r[i] = data
        df = pd.concat(r, keys=dates).rename_axis(['date', 'IDs']).to_frame()
        return df

    def last_n_periods(self, raw_df, field_name, dates, n, ids=None, quarter=None):
        """最近N个财报日期的数据
        每只股票在每个横截面上返回N个数据, 如果数据不足，返回已存在的全部数据。
        """
        if quarter is not None:
            if not isinstance(quarter, list):
                quarter = [quarter]
            raw_df = raw_df.query("quarter in @quarter")
        if ids is not None:
            raw_df = raw_df.query("IDs in @ids")
        raw_df.sort_values(['IDs', 'date', 'ann_dt'], inplace=True)
        raw_df.set_index(['IDs', 'date'], inplace=True)
        dates = pd.to_datetime(dates)
        r = [None] * len(dates)
        for i, dt in enumerate(dates):
            tmp = raw_df.query("ann_dt <= @dt")
            tmp = tmp[~tmp.index.duplicated(keep='last')]
            data = tmp.groupby('IDs')[field_name].tail(n)
            r[i] = data
        df = pd.concat(r, keys=dates).rename_axis(['date', 'IDs', 'report_dt']).to_frame()
        return df

    def last_n_periods_with_consistend_report_dt(self, raw_df, field_name, dates,
                                                 n, ids=None, quarter=None):
        """最近N个财报日期的数据，每个横截面同一财报期
        每只股票在每个横截面上返回N个数据, 如果数据不足，返回已存在的全部数据。
        """
        dates = pd.to_datetime(dates)
        report_dates = pd.to_datetime([ReportDateAvailable(x, x) for x in dates.strftime("%Y%m%d")])
        if quarter is not None:
            if not isinstance(quarter, list):
                quarter = [quarter]
            raw_df = raw_df.query("quarter in @quarter")
        if ids is not None:
            raw_df = raw_df.query("IDs in @ids")
        raw_df.sort_values(['IDs', 'date', 'ann_dt'], inplace=True)
        raw_df.set_index(['IDs', 'date'], inplace=True)
        r = [None] * len(dates)
        for i, dt in enumerate(dates):
            report_dt = report_dates[i]
            tmp = raw_df.query("(ann_dt<=@dt)")
            tmp = tmp[(~tmp.index.duplicated(keep='last'))&(tmp.index.get_level_values('date')<=report_dt)]
            data = tmp.groupby('IDs')[field_name].tail(n)
            r[i] = data
        df = pd.concat(r, keys=dates).rename_axis(['date', 'IDs', 'report_dt']).to_frame()
        return df
