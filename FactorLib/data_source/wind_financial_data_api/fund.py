# coding : utf-8
"""公募基金数据库""" 
from .database import MutualFundDesc, MutualFundNav, MutualFundSector
from ...utils.datetime_func import DateRange2Dates
import pandas as pd
import numpy as np


fund_desc = MutualFundDesc()
fund_nav = MutualFundNav()
fund_sector = MutualFundSector()
_all_fundsectors = {'普通股票型基金': 101, '被动指数型基金': 102,
                    '增强指数型基金': 103, '偏股混合型基金': 201,
                    '平衡混合型基金': 202, '偏债混合型基金': 203,
                    '灵活配置型基金': 204}
_buildin_fund_sectors = {'股票基金': ['普通股票型基金', '偏股混合型基金'],
                         '混合基金': ['平衡混合型基金', '灵活配置型基金'],
                         '指数增强基金': ['被动指数型基金'],
                         '全部基金': ['普通股票型基金', '被动指数型基金',
                                      '增强指数型基金', '偏股混合型基金',
                                      '平衡混合型基金', '灵活配置型基金'],
                         '主动管理基金': ['普通股票型基金','增强指数型基金',
                                          '偏股混合型基金','平衡混合型基金',
                                          '灵活配置型基金']}


# 按照风格分类
def get_fund_by_sector(sector, date, field=None):
    """按照基金风格返回基金列表

    Parameters:
    -----------
    sector : str or list of str
        Wind基金风格分类。主要包括：
            101: 普通股票型基金
            102: 被动指数型基金
            103: 增强指数型基金
            201: 偏股混合型基金
            202: 平衡混合型基金
            203: 偏债混合型基金
            204: 灵活配置型基金
            股票基金、指数增强基金、混合基金
            全部基金、主动管理基金
    date : str or datetime
    field : str or list of str
        返回字段
        可选参数：IDs，in_date，out_date，cur_sign
                 sector
    """
    if isinstance(date, str):
        date = int(date)
    else:
        date = int(date.strftime("%Y%m%d"))
    if sector in _buildin_fund_sectors:
        sector = _buildin_fund_sectors[sector]
    if isinstance(sector, str):
        sector = [sector]
    sector_codes = [_all_fundsectors[x] for x in sector]
    sector_name_code_map = dict(zip(sector_codes, sector))
    data = fund_sector.all_data.query(
        "sector in @sector_codes & in_date<=@date & out_date>@date")
    data['sector'] = data['sector'].map(sector_name_code_map)
    data.drop_duplicates(subset='IDs', inplace=True)
    if field:
        return data[field]
    return data


def get_fund_description(sec_ids):
    """基金基本信息
    基本信息包括：
    基金简称、基金代码、成立日期、是否指数基金、前端代码、后端代码
    """
    if len(sec_ids) > 0:
        data = fund_desc.all_data.query("IDs in @sec_ids")
    else:
        data = fund_desc.all_data
    # data['issue_dt'] = pd.to_datetime(data['issue_dt'].astype('str'))
    data['setup_dt'] = pd.to_datetime(data['setup_dt'].astype('str'))
    return data.set_index('IDs')


@DateRange2Dates
def get_fund_nav(sec_ids=None, start_date=None, end_date=None,
                 dates=None, sector=None, field=None):
    """获取基金净值

    Parameters:
    -----------
    sec_ids : list of str
        基金代码，例如000001.OF
    start_date : str or datetime
        start trade date
    end_date : str or datetime
        end trade date
    dates : list of date
    sector : str
        fund sector
        see _all_fund_sectors and _buildin_fund_sectors
        for more details
    field : list of str
        columns to return. Options include 单位净值、复权单位净值和累计净值。
    """
    if field is None:
        field = ['单位净值', '复权单位净值', '累计净值']
    field_ids = fund_nav.data_dict.wind_factor_ids(fund_nav.table_name, field)
    max_date = dates.max().strftime("%Y%m%d")
    dates_int = dates.strftime("%Y%m%d").astype('int32')
    if sector:
        sec_ids = get_fund_by_sector(sector, date=max_date, field='IDs').values
    idx = pd.MultiIndex.from_product([dates_int, sec_ids], names=['date', 'IDs'])
    idx2 = pd.MultiIndex.from_product([dates, sec_ids], names=['date', 'IDs'])
    rslt = pd.DataFrame(columns=field_ids, index=idx2)
    for i_field_id, i_field in zip(field_ids, field):
        data = fund_nav.load_h5(i_field).set_index(['date', 'IDs'])
        rslt[i_field_id] = data.reindex(idx)[i_field_id].values
    return rslt










