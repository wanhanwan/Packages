# coding: utf-8
from sqlalchemy import create_engine
from fastcache import clru_cache
from params import *
from FactorLib.data_source.base_data_source_h5 import ncdb, tc
from FactorLib.utils.tool_funcs import ensure_dir_exists
from collections import Iterator
from data_loader import DataLoader
import pandas as pd
import numpy as np
import sqlalchemy as sa
import os
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings(action='ignore', category=PerformanceWarning)


def _read_est_dict():
    file_pth = os.path.abspath(__file__+"/../../../")
    file_pth = os.path.join(file_pth, 'resource', 'wind_tableinfo.xlsx')
    wind_table_info = pd.read_excel(file_pth, sheet_name='TableInfo', header=0, encoding='GBK')
    wind_factor_info = pd.read_excel(file_pth, sheet_name='FactorInfo', header=0, encoding='GBK')
    return wind_table_info, wind_factor_info


def _drop_invalid_stocks(data):
    """去掉非法股票"""
    invalid_ids = [x for x in data['IDs'].unique() if x[0] not in ['6', '0', '3']]
    return data.query("IDs not in @invalid_ids")


def _reconstruct(data):
    """重新调整数据结构，把整数股票ID转成字符串，日期转成datetime
    """
    data = data.reset_index()
    data['IDs'] = data['IDs'].astype('str').str.zfill(6)
    data['date'] = pd.to_datetime(data['date'].astype('str'))
    return data.set_index(['date', 'IDs'])

class WindTableInfo(object):
    """docstring for WindTableInfo"""
    table_info, factor_info = _read_est_dict()

    @staticmethod
    def wind_table_name(table_name):
        """中文表明对应的Wind数据库表名"""
        return WindTableInfo.table_info[WindTableInfo.table_info['TableName']==table_name]['WindTableName'].iloc[0]

    @staticmethod
    def list_factors(table_name):
        """列出某张表下所有字段对照"""
        return WindTableInfo.factor_info[WindTableInfo.factor_info['TableName']==table_name][['Name', 'WindID']].set_index('Name')

    def wind_factor_ids(self, table_name, factor_names):
        """某张表中文字段对应的Wind数据库字段"""
        all_factors = self.list_factors(table_name)
        if isinstance(factor_names, str):
            return all_factors.at[factor_names, 'WindID']
        return all_factors.loc[factor_names, 'WindID'].tolist()

    @clru_cache()
    def list_default_factors(self, table_name):
        """某张表的缺省因子，不论从这张表取什么字段，缺省字段都会被提取出来。

        Return
        =========
        DataFrame(index:[字段中文名], columns:[WindID])
        """
        tmp = WindTableInfo.factor_info.query("TableName==@table_name & DefaultColumns==1")[['Name', 'WindID']].set_index('Name')
        return tmp

    @clru_cache()
    def get_table_index(self, table_name):
        tmp = WindTableInfo.factor_info.query("TableName==@table_name & DFIndex!='None'")[['Name', 'WindID', 'DFIndex']].set_index(
            'Name')
        return tmp


class BaseDB(object):
    def __init__(self, user_name, pass_word, db_name, ip_address, db_type='oracle',
                 port=1521):
        self.db_type = db_type
        self.user_name = user_name
        self.db_name = db_name
        self.ip_address = ip_address
        self.port = port
        self.password = pass_word
        self.db_engine = None
        self.db_inspector = None

    def connectdb(self):
        self.db_engine = create_engine(self.create_conn_string(
            self.user_name, self.password, self.db_name, self.ip_address, self.port, self.db_type))
        self.db_inspector = sa.inspect(self.db_engine)

    @staticmethod
    def create_conn_string(user_name, pass_word, db_name, ip_address, port, db_type):
        return "{db_type}://{user_name}:{pass_word}@{ip_address}:{port}/{db_name}".format(
            db_type=db_type, user_name=user_name, pass_word=pass_word, ip_address=ip_address, port=port, db_name=db_name)

    def exec_query(self, query_str, **kwargs):
        with self.db_engine.connect() as conn, conn.begin():
            data = pd.read_sql(query_str, conn, **kwargs)
        return data

    @clru_cache()
    def list_columns_of_table(self, table_name):
        columns = self.db_inspector.get_columns(table_name.lower())
        return pd.DataFrame(columns)[[u'name', u'type']]

    def get_column_type(self, table_name, column):
        data_type = self.list_columns_of_table(table_name)
        return data_type[data_type[u'name'] == column][u'type'].iloc[0]

    def load_panel_data(self, table_name, columns, _between=None, _in=None, _equal=None, **kwargs):
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
        s = select([literal_column("%s.%s" % (table_name, x)).label(x) for x in columns])
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
                    value_str = "('" + "','".join(value) + "')"
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

        data = self.exec_query(s, **kwargs)
        return data
        

class WindDB(BaseDB):
    """Wind数据库"""
    data_dict = WindTableInfo()

    def __init__(self, user_name=WIND_USER, pass_word=WIND_PASSWORD, db_name=WIND_DBNAME,
        ip_address=WIND_IP, db_type=WIND_DBTYPE, port=WIND_PORT):
        super(WindDB, self).__init__(user_name, pass_word, db_name, ip_address, db_type, port)

    def load_factors(self, factors, table, _in=None, _between=None, _equal=None, **kwargs):
        """提取某张表的数据"""

        def _query_iterator(data):
            for idata in data:
                return _wrap_data(idata)

        def _wrap_data(idata):
            if idata.empty:
                return idata
            idata = idata.rename(columns=table_index)
            idata['IDs'] = idata['IDs'].str[:6]
            return _drop_invalid_stocks(idata)

        wind_table_name = WindDB.data_dict.wind_table_name(table)
        wind_factor_ids = list(set(WindDB.data_dict.wind_factor_ids(table, factors) +
                          WindDB.data_dict.list_default_factors(table)['WindID'].tolist()))
        if _in is not None:
            _in = {WindDB.data_dict.wind_factor_ids(table, [x])[0]: y for x, y in _in.items()}
        if _between is not None:
            _between = {WindDB.data_dict.wind_factor_ids(table, [x])[0]: y for x, y in _between.items()}
        if _equal is not None:
            _equal = {WindDB.data_dict.wind_factor_ids(table, [x])[0]: y for x, y in _equal.items()}
        table_index = WindDB.data_dict.get_table_index(table).set_index('WindID')['DFIndex'].to_dict()
        data = self.load_panel_data(wind_table_name, wind_factor_ids, _between, _in, _equal, **kwargs)
        if kwargs.get('chuncksize', None):
            return _query_iterator(data)
        else:
            return _wrap_data(data)

    def save_factor(self, data, path, name, if_exists='append'):
        """存储数据"""
        if isinstance(data, Iterator):
            for idata in data:
                if if_exists == 'replace':
                    self.save_factor(idata, path, name, if_exists=if_exists)
                    if_exists = 'append'
                else:
                    self.save_factor(idata, path, name, if_exists=if_exists)
        else:
            ncdb.save_factor(data, name, path, if_exists, append_type='concat')


class WindEstDB(WindDB):

    """Wind一致预期数据库Wrapper"""
    def __init__(self):
        super(WindEstDB, self).__init__()


class WindFinanceDB(WindDB):
    """Wind财务数据库Wrapper"""
    table_name = ""
    data_loader = DataLoader()

    def __init__(self):
        super(WindFinanceDB, self).__init__()

    def gen_dataframe(self, data):
        """按字段逐一生成DataFrame"""
        default_columns = self.data_dict.get_table_index(self.table_name)['DFIndex'].tolist() + ['quarter', 'year']
        if isinstance(data, Iterator):
            for idata in data:
                if idata.empty:
                    yield idata
                columns = [x for x in idata.columns if x not in default_columns]
                for c in columns:
                    yield c, idata[default_columns+[c]]
        else:
            if data.empty:
                yield None, data    
            columns = [x for x in data.columns if x not in default_columns]
            for c in columns:
                yield c, data[default_columns + [c]]

    def save_data(self, data, table_id, if_exists='append'):
        tar_pth = os.path.join(LOCAL_FINDB_PATH, table_id)
        ensure_dir_exists(tar_pth)
        for c, d in self.gen_dataframe(data):
            if d.empty:
                return
            if if_exists == 'replace':
                self.save_factor(d, tar_pth, c, if_exists='replace')
                if_exists = 'append'
            else:
                self.save_factor(d, tar_pth, c, if_exists='append')

    @staticmethod
    def add_quarter_year(idata):
        if idata.empty:
            return idata
        idata.dropna(subset=['ann_dt'], inplace=True)
        idata['quarter'] = pd.to_datetime(idata['date']).dt.quarter
        idata['year'] = pd.to_datetime(idata['date']).dt.year
        idata['date'] = idata['date'].astype('int')
        idata['ann_dt'] = idata['ann_dt'].astype('int')
        idata['stat_type'] = idata['stat_type'].map(WindIncomeSheet.statement_type_map)
        idata['IDs'] = idata['IDs'].astype('int')
        idata = idata.sort_values(['IDs', 'date', 'ann_dt', 'stat_type']).reset_index(drop=True)
        return idata

    def save_factor(self, data, path, factor_name, if_exists='append'):
        tar_file = os.path.join(path, factor_name+'.h5')
        if (if_exists == 'replace') or (not os.path.isfile(tar_file)):
            data.to_hdf(tar_file, "data", mode='w', complib='blosc', complevel=9)
        else:
            table_index = self.data_dict.get_table_index(self.table_name)['DFIndex'].tolist()
            raw_data = pd.read_hdf(tar_file, "data")
            new_data = raw_data.append(data).drop_duplicates(subset=table_index, keep='last')
            new_data = new_data.sort_values(['IDs', 'date', 'ann_dt', 'stat_type']).reset_index(drop=True)
            new_data.to_hdf(tar_file, "data", mode='w', complib='blosc', complevel=9)

    @clru_cache()
    def load_h5(self, file_name):
        wind_id = self.data_dict.wind_factor_ids(self.table_name, file_name)
        data_pth = os.path.join(LOCAL_FINDB_PATH, self.table_id, wind_id+'.h5')
        data = pd.read_hdf(data_pth, "data")
        return data


class WindIncomeSheet(WindFinanceDB):
    """Wind利润表"""
    table_name = u'中国A股利润表'
    table_id = 'income'
    statement_type_map = {'408004000': 4, '408050000': 3, '408001000': 2, '408005000': 1}

    def __init__(self):
        super(WindIncomeSheet, self).__init__()

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据
        报表类型采用合并报表、合并报表调整、合并报表更正前、合并调整更正前
        """
        statement_type = {u'报表类型': ['408001000', '408004000', '408005000', '408050000']}
        if _in is None:
            _in = statement_type
        else:
            _in.update(statement_type)
        data = self.load_factors(factors, WindIncomeSheet.table_name, _in, _between, _equal, **kwargs)
        return self.add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindIncomeSheet, self).save_data(data, WindIncomeSheet.table_id, if_exists)

    def load_ttm(self, factor_name, start=None, end=None, dates=None, ids=None):
        """ 加载TTM数据
        """
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.ttm(data, wind_id, dates, ids)
        return _reconstruct(new)


if __name__ == '__main__':
    from datetime import datetime
    wind = WindIncomeSheet()
    # wind.connectdb()
    # data = wind.download_data([u'净利润(不含少数股东损益)'], _between={u'报告期': ('20070101', '20171231')})
    # wind.save_data(data)
    ttm = wind.load_ttm('净利润(不含少数股东损益)', ids=['000001'], start='20170101', end='20171231')
    print(ttm)
    # data =  wind.load_panel_data('ashareincome', ['s_info_windcode', 'ann_dt', 'actual_ann_dt',
    #                                               'report_period', 'TOT_OPER_REV', 'NET_PROFIT_INCL_MIN_INT_INC', 'statement_type'],
    #                              _between={'opdate': (datetime(2017, 5, 11), datetime(2017,5,12))})

