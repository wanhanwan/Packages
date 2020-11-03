#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tushare_fund_managers.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 9/28/2020, 3:13:07 PM
"""
所有公募基金经理的任职数据。

更新方式：全量更新
基金类型：所有
更新方式：优矿网站下载csv文件，放到/fund/
"""
import os
import pandas as pd
from PkgConstVars import FACTOR_PATH

def cache_data(func):

    def parse_code(x: str):
        if x == 'OFCN':
            return x[:6] + '.OF'
        elif x == 'XSHG':
            return x[:6] + '.SH'
        else:
            return x[:6] + '.SZ'

    raw_data = pd.read_csv(
        os.path.join(FACTOR_PATH,'fund','基金经理信息.csv'),
        header=0,
        parse_dates=['accessionDate', 'dimissionDate'],
        encoding='GBK',
        converters={'secID': parse_code},
        dtype={'personID': int}
        )
    raw_data['ticker'] = raw_data['secID'].str[:6]
    
    def wrapper(**kwargs):
        return func(raw_data, **kwargs)

    return wrapper

@cache_data
def load_manager_info(raw_data=None,
                      personID=None,
                      name=None,
                      fund_code=None,
                      fund_ticker=None
                    ):
    """根据基金经理ID或者基金经理姓名检索其任职信息

    基金经理任职表是一张全量的csv文件，路径：‘/fund/'。需要在优矿
    平台进行更新。

    Parameters:
    -----------
    personID: int
        基金经理ID
    name: str
        基金经理姓名
    """
    if personID:
        raw_data = raw_data.query("personID==@personID")
    if name:
        raw_data = raw_data.query("name==@name")
    if fund_code:
        raw_data = raw_data.query("secID==@fund_code")
    if fund_ticker:
        raw_data = raw_data.query("ticker==@fund_ticker")
    return raw_data
