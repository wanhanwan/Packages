# coding: utf-8
from sqlalchemy import create_engine
from functools import lru_cache
from .params import *
import pandas as pd
import numpy as np
import sqlalchemy as sa
import os


class BaseDB(object):
    def __init__(self, user_name, pass_word, db_name, ip_address, db_type='oracle',
                 port=1521):
        self.db_type = db_type
        self.user_name = user_name
        self.db_name = db_name
        self.ip_address = ip_address
        self.db_engine = create_engine(self.create_conn_string(user_name, pass_word, db_name, ip_address, port, db_type))
        self.db_inspector = sa.inspect(self.db_engine)

    @staticmethod
    def create_conn_string(user_name, pass_word, db_name, ip_address, port, db_type):
        return "{db_type}://{user_name}:{pass_word}@{ip_address}:{port}/{db_name}".format(
            db_type=db_type, user_name=user_name, pass_word=pass_word, ip_address=ip_address, port=port, db_name=db_name)

    def exec_query(self, query_str, **kwargs):
        with self.db_engine.connect() as conn, conn.begin():
            data = pd.read_sql(query_str, conn, **kwargs)
        return data

    @lru_cache()
    def list_columns_of_table(self, table_name):
        columns = self.db_inspector.get_columns(table_name)
        return pd.DataFrame(columns)[['comment', 'name', 'type']]

    def get_column_type(self, table_name, column):
        data_type = self.list_columns_of_table(table_name)
        return data_type[data_type['name'] == column]['type'].iloc[0]

    def load_panel_data(self, table_name, columns, _between=None, _in=None, _equal=None):
        """
        从底层数据库中加载某张表的字段

        Paramters:
        =============================
        _between: Dict, SQL中的BETWEEN关键字
            >> load_panel_data(table_name, columns, _between={'Filed_1':(start, end)})
        _in: Dict, SQL中的IN关键字
            >> load_panel_data(table_name, columns, _in={'Filed_1':[Value1, Value2,...]})
        _equal: Dicr, SQL中的=关键字
            >> load_panel_data(table_name, columns, _equal={'Filed_1':Value1})
        """
        select, and_, text = sa.select, sa.and_, sa.text
        literal_column, table = sa.sql.literal_column, sa.sql.table
        s = select([literal_column("%s.%s" % (table_name, x)) for x in columns])
        text_list = []
        if self.db_type == 'oracle':
            from sqlalchemy.dialects import oracle
            VARCHAR, NUMBER, DATE = oracle.VARCHAR, oracle.NUMBER, oracle.DATE
        else:
            from sqlalchemy.dialects import mysql
            VARCHAR, NUMBER, DATE = mysql.VARCHAR, mysql.NUMBER, mysql.DATE

        if _between is not None:
            for field, value in _between.items():
                if isinstance(self.get_column_type(table_name, field), VARCHAR):
                    text_list.append(text("{table}.{field} BETWEEN '{start}' AND '{end}'".format(
                        table=table_name,
                        field=field,
                        start=value[0],
                        end=value[1])))
                elif isinstance(self.get_column_type(table_name, field), NUMBER):
                    text_list.append(text("{table}.{field} BETWEEN {start} AND {end}".format(
                        table=table_name,
                        field=field,
                        start=value[0],
                        end=value[1])))
                elif isinstance(self.get_column_type(table_name, field), DATE):
                    start = "to_date('%s','yyyy/mm/dd')" % value[0].strftime("%Y/%m/%d")
                    end = "to_date('%s','yyyy/mm/dd')" % value[1].strftime("%Y/%m/%d")
                    text_list.append(text("{table}.{field} BETWEEN {start} AND {end}".format(
                        table=table_name,
                        field=field,
                        start=start,
                        end=end)))

        if _in is not None:
            for field, value in _in.items():
                if isinstance(self.get_column_type(table_name, field), VARCHAR):
                    value_str = "('" + "',".join(value) + "')"
                elif isinstance(self.get_column_type(table_name, field), NUMBER):
                    value_str = "(" + ",".join(value) + ")"
                else:
                    value_str = ""
                text_list.append(text("{table}.{field} IN {value_str}".format(
                    table=table_name,
                    field=field,
                    value_str=value_str
                )))
        if _equal is not None:
            for field, value in _equal.items():
                if isinstance(self.get_column_type(table_name, field), VARCHAR):
                    value = "'%s'" % value
                else:
                    value = str(value)
                text_list.append(text("{table}.{field} = {value}".format(
                    table=table_name,
                    field=field,
                    value=value
                )))
        if text_list:
            s = s.where(and_(*text_list))
        s = s.select_from(table(table_name))

        data = self.exec_query(s)
        return data

class WindDB(BaseDB):
    """Wind数据库"""
    data_dict = WindTableInfo()

    def __init__(self, user_name=WIND_USER, pass_word=WIND_PASSWORD, db_name=WIND_DBNAME,
        ip_address=WIND_IP, db_type=WIND_DBTYPE, port=WIND_PORT):
        super(BaseDB, self).__init__(user_name, pass_word, db_name, ip_address, db_type, port)

    def load_factors(self, factors, table, _in=None, _between=None, _equal=None):
        


class WindEstDB(WindDB):
    """Wind一致预期数据库Wrapper"""

    def __init__(self):
        super(WindDB, self).__init__()



class WindTableInfo(object):
    """docstring for WindTableInfo"""
    table_info, factor_info = _read_est_dict()

    def wind_table_name(self, table_name):
        """中文表明对应的Wind数据库表名"""
        return WindTableInfo.table_info[WindTableInfo.table_info['TableName']==table_name]['WindTableName'].iloc[0]

    def list_factors(self, table_name):
        """列出某张表下所有字段对照"""
        return WindTableInfo.factor_info[WindTableInfo.factor_info['TableName']==table_name][['Name', 'WindID']].set_index('Name')

    def wind_factor_ids(self, table_name, factor_names):
        """某张表中文字段对应的Wind数据库字段"""
        all_factors = self.list_factors(table_name)
        return all_factors.loc[factor_names, 'WindID'].tolist()

    
def _read_est_dict():
    file_pth = os.path.abspath(__file__+"/../../")
    file_pth = os.path.join(file_pth, 'resource', 'wind_tableinfo.xlsx')
    wind_table_info = pd.read_excel(file_pth, sheet_name='TableInfo', header=0, encoding='GBK')
    wind_factor_info = pd.read_excel(file_pth, sheet_name='FactorInfo', header=0, encoding='GBK')
    return wind_table_info, wind_factor_info
        

if __name__ == '__main__':
    from datetime import datetime
    wind = BaseDB('Filedb', 'Filedb', 'cibfund', '172.20.65.27')
    data =  wind.load_panel_data('ashareincome', ['s_info_windcode', 'ann_dt', 'actual_ann_dt',
                                                  'report_period', 'TOT_OPER_REV', 'NET_PROFIT_INCL_MIN_INT_INC', 'statement_type'],
                                 _between={'opdate': (datetime(2017, 5, 11), datetime(2017,5,12))})

