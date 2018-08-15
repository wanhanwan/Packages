# -*- coding: utf-8 -*-
#
# datayes.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2018-8-14 14:38:23
"""通过OpenDataTools中的datayes模块封装通联数据"""
from opendatatools import datayes
from FactorLib.utils.tool_funcs import get_resource_abs_path
import pandas as pd


def _read_table():
    pth = get_resource_abs_path() / 'datayes_tableinfo.xlsx'
    table = pd.read_excel(pth, sheet_name='FieldID', header=0,
                          index_col=0)
    return table


class DataYesDB(object):
    _instance = None

    def __new__(cls, user_name='wanhanwan', password='1991822929'):
        if cls._instance is None:
            cls._instance = super(DataYesDB, cls).__new__(cls)
            return cls._instance
        else:
            return cls._instance
    
    def __init__(self, user_name='wanhanwan', password='1991822929'):
        self.user_name = user_name
        self.password = password
        self.table_info = _read_table()
        self.is_login = 0
    
    def login(self):
        if not self.is_login:
            datayes.login(self.user_name, self.password)
            self.is_login = 1
    
    def get_top_items(self):
        return datayes.get_top_items()
    
    def get_sub_items(self, item_id):
        df, msg = datayes.get_sub_items(item_id)
        return df
    
    def get_series(self, item_name):
        item_id = self.table_info.at[item_name, 'FieldID']
        df, info, msg = datayes.get_series(str(item_id))
        return df
    
