# coding: utf-8
from sqlalchemy import create_engine
from fastcache import clru_cache
from FactorLib.data_source.wind_financial_data_api.params import *
from FactorLib.data_source.wind_financial_data_api.data_loader import DataLoader, quarter2intdate, period_backward, avg
from FactorLib.data_source.base_data_source_h5 import ncdb, tc
from FactorLib.utils.tool_funcs import ensure_dir_exists
from FactorLib.data_source.helpers import handle_ids
from FactorLib.utils.datetime_func import DateRange2Dates
from pathlib import Path
from collections import Iterator
import pandas as pd
import numpy as np
import sqlalchemy as sa
import os, sys
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings(action='ignore', category=PerformanceWarning)


def _read_est_dict():
    file_pth = os.path.abspath(__file__+"/../../../")
    file_pth = os.path.join(file_pth, 'resource', 'wind_tableinfo.xlsx')
    wind_table_info = pd.read_excel(file_pth, sheet_name='TableInfo', header=0, encoding='GBK')
    wind_factor_info = pd.read_excel(file_pth, sheet_name='FactorInfo', header=0, encoding='GBK')
    return wind_table_info, wind_factor_info


def _drop_invalid_stocks(data, field_name='IDs'):
    """去掉非法股票"""
    invalid_ids = [x for x in data[field_name].unique() if x[0] not in ['6', '0', '3'] or not x.isdigit()]
    return data.query("%s not in @invalid_ids" % field_name)


def _reconstruct(data):
    """重新调整数据结构，把整数股票ID转成字符串，日期转成datetime
    """
    data = data.reset_index()
    data['IDs'] = data['IDs'].astype('str').str.zfill(6)
    data['date'] = pd.to_datetime(data['date'].astype('str'))
    return data.set_index(['date', 'IDs'])


def _search_columns(data_c, column_list):
    r = []
    for c in column_list:
        if c in data_c:
            r.append(c)
    return r


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
        if not factor_names:
            return []
        all_factors = self.list_factors(table_name)
        if sys.version_info.major == 3:
            if isinstance(factor_names, str):
                return all_factors.at[factor_names, 'WindID']
        else:
            if isinstance(factor_names, unicode):
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
            'Name').astype({'WindID': 'str'})
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
        if db_type == 'mysql':
            return "{db_type}+pymysql://{user_name}:{pass_word}@{ip_address}:{port}/{db_name}".format(
                db_type=db_type, user_name=user_name, pass_word=pass_word, ip_address=ip_address,
                port=port, db_name=db_name)
        return "{db_type}://{user_name}:{pass_word}@{ip_address}:{port}/{db_name}".format(
                db_type=db_type, user_name=user_name, pass_word=pass_word, ip_address=ip_address,
                port=port, db_name=db_name)

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
        s = select([literal_column("%s" % x) for x in columns])
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
                    text_list.append(text("{field} BETWEEN '{start}' AND '{end}'".format(
                        # table=table_name,
                        field=field,
                        start=value[0],
                        end=value[1])))
                elif isinstance(self.get_column_type(table_name, field), NUMBER):
                    text_list.append(text("{field} BETWEEN {start} AND {end}".format(
                        # table=table_name,
                        field=field,
                        start=value[0],
                        end=value[1])))
                elif isinstance(self.get_column_type(table_name, field), DATE):
                    start = "to_date('%s','yyyy/mm/dd hh24:mi:ss')" % value[0].strftime("%Y/%m/%d %H:%M:%S")
                    end = "to_date('%s','yyyy/mm/dd hh24:mi:ss')" % value[1].strftime("%Y/%m/%d %H:%M:%S")
                    text_list.append(text("{field} BETWEEN {start} AND {end}".format(
                        # table=table_name,
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
                yield _wrap_data(idata)

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
        if kwargs.get('chunksize', None):
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


class WindFinanceDB(WindDB):
    """Wind财务数据库Wrapper"""
    table_name = ""
    table_id = ""
    data_loader = DataLoader()
    statement_type_map = {}

    def __init__(self):
        super(WindFinanceDB, self).__init__()

    def gen_dataframe(self, data):
        """按字段逐一生成DataFrame"""
        default_columns = self.data_dict.get_table_index(self.table_name)['DFIndex'].tolist() + ['quarter', 'year']
        if isinstance(data, Iterator):
            for idata in data:
                if idata.empty:
                    yield None, idata
                columns = [x for x in idata.columns if x not in default_columns]
                columns2 = list(set(idata.columns).intersection(set(default_columns)))
                if not columns:
                    yield self.table_id, idata[columns2]
                for c in columns:
                    yield c, idata[columns2+[c]]
        else:
            if data.empty:
                yield None, data    
            columns = [x for x in data.columns if x not in default_columns]
            columns2 = list(set(data.columns).intersection(set(default_columns)))
            if not columns:
                yield self.table_id, data[columns2]
            for c in columns:
                yield c, data[columns2 + [c]]

    def save_data(self, data, table_id, if_exists='append'):
        tar_pth = os.path.join(LOCAL_FINDB_PATH, table_id)
        ensure_dir_exists(tar_pth)
        for c, d in self.gen_dataframe(data):
            if d.empty:
                continue
            if if_exists == 'replace':
                self.save_factor(d, tar_pth, c, if_exists='replace')
                if_exists = 'append'
            else:
                self.save_factor(d, tar_pth, c, if_exists='append')

    def add_quarter_year(self, idata):
        if idata.empty:
            return idata
        idata.dropna(subset=['ann_dt'], inplace=True)
        idata['quarter'] = pd.to_datetime(idata['date']).dt.quarter
        idata['year'] = pd.to_datetime(idata['date']).dt.year
        idata['date'] = idata['date'].astype('int')
        idata['ann_dt'] = idata['ann_dt'].astype('int')
        idata['stat_type'] = idata['stat_type'].map(self.statement_type_map)
        idata['IDs'] = idata['IDs'].astype('int')
        idata = idata.sort_values(['IDs', 'date', 'ann_dt', 'stat_type']).reset_index(drop=True)
        return idata

    def _wrap_add_quarter_year(self, data):
        for idata in data:
            yield self.add_quarter_year(idata)

    def save_factor(self, data, path, factor_name, if_exists='append'):
        tar_file = os.path.join(path, factor_name+'.h5')
        if (if_exists == 'replace') or (not os.path.isfile(tar_file)):
            data.to_hdf(tar_file, "data", mode='w', complib='blosc', complevel=9)
        else:
            table_index = self.data_dict.get_table_index(self.table_name)['DFIndex'].tolist()
            raw_data = pd.read_hdf(tar_file, "data")
            new_data = raw_data.append(data).drop_duplicates(subset=table_index, keep='last')
            c = _search_columns(new_data.columns, ['IDs', 'date', 'ann_dt', 'stat_type'])
            new_data = new_data.sort_values(c)
            new_data.reset_index(drop=True, inplace=True)
            new_data.to_hdf(tar_file, "data", mode='w', complib='blosc', complevel=9)

    @clru_cache()
    def load_h5(self, file_name):
        try:
            wind_id = self.data_dict.wind_factor_ids(self.table_name, file_name)
        except KeyError:
            wind_id = self.table_id
        data_pth = os.path.join(LOCAL_FINDB_PATH, self.table_id, wind_id+'.h5')
        data = pd.read_hdf(data_pth, "data")
        return data

    @handle_ids
    def load_latest_period(self, factor_name, start=None, end=None, dates=None, ids=None,
                           quarter=None):
        """最新报告期数据"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.latest_period(data, wind_id, dates, ids, quarter)
        return _reconstruct(new)

    @handle_ids
    def load_spec_period(self, factor_name, year, quarter, start=None, end=None, dates=None, ids=None):
        """指定报告期"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name).query("year==@year & quarter==@quarter")
        new = self.data_loader.latest_period(data, wind_id, dates, ids)
        return _reconstruct(new)

    @handle_ids
    def load_last_nyear(self, factor_name, n, start=None, end=None, dates=None, ids=None,
                        quarter=None):
        """回溯N年之前的财务数据"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.last_nyear(data, wind_id, dates, n, ids, quarter)
        return _reconstruct(new)

    @handle_ids
    def load_incr_tb(self, factor_name, n, start=None, end=None, dates=None, ids=None, quarter=None):
        """同比序列"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.inc_rate_tb(data, wind_id, dates, n, ids, quarter=quarter)
        return _reconstruct(new)

    @handle_ids
    def load_incr_hb(self, factor_name, start=None, end=None, dates=None, ids=None):
        """环比序列"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.inc_rate_hb(data, wind_id, dates, ids)
        return _reconstruct(new)

    @handle_ids
    def load_spec_avg(self, factor_name, year, quarter, back_nquarter=1, start=None, end=None,
                      dates=None, ids=None):
        """两个报告期的平均值，常用在资产负债表上"""
        new = self.load_spec_period(factor_name, year, quarter, start, end, dates, ids)
        old_period = period_backward(np.asarray([quarter2intdate(year, quarter)]), back_nquarter=back_nquarter)[0]
        old = self.load_spec_period(factor_name, old_period//10000, int(old_period % 10000//100/3), start, end, dates,
                                    ids)
        return avg(old, new)

    @handle_ids
    def load_last_nyear_avg(self, factor_name, quarter, back_nyear=1, start=None, end=None,
                       dates=None, ids=None):
        """n年之前报告期期初与期末的平均值"""
        new = self.load_latest_period(factor_name, start, end, dates, ids, quarter)
        old = self.load_last_nyear(factor_name, back_nyear, start, end, dates, ids, quarter)
        return avg(old, new)


class WindConsensusDB(WindFinanceDB):
    """Wind中国A股一致预期汇总数据库

    statement_type 是Wind底层表中的综合值周期类型字段, 明细:
        263001000 : 30天
        263002000 : 90天
        263003000 : 180天
        263004000 : 大事后180天

    """
    table_name = u"中国A股盈利预测汇总"
    table_id = "ashareconsensusdata"
    statement_type_map = {"263001000": 30, "263002000": 90, "263003000": 180, "263004000": 2180}
    year_type = {'FY1': 1, 'FY2': 2, 'FY3': 3}

    def __init__(self):
        super(WindConsensusDB, self).__init__()

    def add_quarter_year(self, idata):
        if idata.empty:
            return idata
        idata.dropna(subset=['ann_dt'], inplace=True)
        idata['quarter'] = pd.to_datetime(idata['date']).dt.quarter
        idata['year'] = pd.to_datetime(idata['date']).dt.year
        idata['date'] = idata['date'].astype('int')
        idata['ann_dt'] = idata['ann_dt'].astype('int')
        idata['stat_type'] = idata['stat_type'].map(self.statement_type_map)
        idata['IDs'] = idata['IDs'].astype('int')
        idata['year_type'] = idata['year_type'].map(self.year_type).fillna(-1)
        idata = idata.sort_values(['IDs', 'date', 'ann_dt', 'stat_type', 'year_type']).reset_index(drop=True)
        return idata

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """下载数据"""

        def _wrap_add_quarter_year(data):
            for idata in data:
                yield self.add_quarter_year(idata)

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        else:
            return _wrap_add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindConsensusDB, self).save_data(data, self.table_id, if_exists)

    @handle_ids
    def load_fy1(self, factor_name, start=None, end=None, dates=None, ids=None, stat_type=90):
        from .est_data_loader import load_fy1
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = load_fy1(data, wind_id, dates, ids, stat_type)
        return _reconstruct(new)

    @handle_ids
    def load_fy2(self, factor_name, start=None, end=None, dates=None, ids=None, stat_type=90):
        from .est_data_loader import load_fy2
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = load_fy2(data, wind_id, dates, ids, stat_type)
        return _reconstruct(new)

    @handle_ids
    def load_fy3(self, factor_name, start=None, end=None, dates=None, ids=None, stat_type=90):
        from .est_data_loader import load_fy3
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = load_fy3(data, wind_id, dates, ids, stat_type)
        return _reconstruct(new)

    @handle_ids
    def load_spec_year(self, factor_name, spec_year, start=None, end=None, dates=None, ids=None, stat_type=90):
        from .est_data_loader import load_spec_year
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = load_spec_year(data, wind_id, spec_year, dates, ids, stat_type)
        return _reconstruct(new)


def _read_inst_id():
    p = Path(__file__)
    file_path = p.parent.parent.parent / 'resource' / 'est_inst_name_id.csv'
    return file_path


class WindEarningEst(WindFinanceDB):
    """Wind盈利预测明细"""
    table_name = u'中国A股盈利预测明细'
    table_id = 'earningest'
    statement_type_map = {'0001': 0}
    inst_id = pd.read_csv(_read_inst_id(), index_col=0, header=0,
                          encoding='gbk')

    def __init__(self):
        super(WindEarningEst, self).__init__()

    def add_quarter_year(self, idata):
        idata.dropna(subset=['IDs', 'ann_dt', 'inst_name'], inplace=True)
        if idata.empty:
            return idata
        idata.fillna(np.nan, inplace=True)
        idata['stat_type'] = '0001'
        idata['inst_name'] = idata['inst_name'].str.decode('GBK')
        idata['inst_name'] = idata['inst_name'].map(self.inst_id['id'])
        idata['quarter'] = pd.to_datetime(idata['date']).dt.quarter
        idata['year'] = pd.to_datetime(idata['date']).dt.year
        idata['date'] = idata['date'].astype('int')
        idata['ann_dt'] = idata['ann_dt'].astype('int')
        idata['stat_type'] = idata['stat_type'].map(self.statement_type_map)
        idata['IDs'] = idata['IDs'].astype('int')
        idata = idata.sort_values(['IDs', 'date', 'ann_dt', 'stat_type']).reset_index(drop=True)
        return idata

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据
        """
        def _wrap_add_quarter_year(data):
            for idata in data:
                yield self.add_quarter_year(idata)

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        else:
            return _wrap_add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindEarningEst, self).save_data(data, self.table_id, if_exists)


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
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        return self._wrap_add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindIncomeSheet, self).save_data(data, self.table_id, if_exists)

    @handle_ids
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

    @handle_ids
    def load_last_nyear_ttm(self, factor_name, n, start=None, end=None, dates=None, ids=None):
        """加载N年之前的TTM数据
        """
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.last_nyear_ttm(data, wind_id, dates, n, ids)
        return _reconstruct(new)


class WindCashFlow(WindIncomeSheet):
    """Wind现金流量表"""
    table_name = u'中国A股现金流量表'
    table_id = 'cashflow'


class WindSQIncomeSheet(WindIncomeSheet):
    table_id = 'sq_income'
    statement_type_map = {'408002000': 1, '408003000': 2}

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据
        报表类型采用合并报表(单季度)、合并报表(单季度调整)
        """
        statement_type = {u'报表类型': ['408002000', '408003000']}
        if _in is None:
            _in = statement_type
        else:
            _in.update(statement_type)
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        return self._wrap_add_quarter_year(data)


class WindBalanceSheet(WindFinanceDB):
    """Wind资产负债表"""
    table_name = u'中国A股资产负债表'
    table_id = 'balance'
    statement_type_map = {'408004000': 4, '408050000': 3, '408001000': 2, '408005000': 1}

    def __init__(self):
        super(WindBalanceSheet, self).__init__()

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据
        报表类型采用合并报表、合并报表调整、合并报表更正前、合并调整更正前
        """
        statement_type = {u'报表类型': ['408001000', '408004000', '408005000', '408050000']}
        if _in is None:
            _in = statement_type
        else:
            _in.update(statement_type)
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        return self._wrap_add_quarter_year(data)

    @handle_ids
    def load_ttm_avg(self, factor_name, start=None, end=None, dates=None, ids=None):
        """最近12个月的平均值(期初+期末)/2， 一般用于资产负债表项目
        """
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.ttm_avg(data, wind_id, dates, ids)
        return _reconstruct(new)

    @handle_ids
    def load_sq_avg(self, factor_name, start=None, end=None, dates=None, ids=None):
        """单季度平均值(期初+期末)/2, 一般用于资产负债表
        """
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.sq_avg(data, wind_id, dates, ids)
        return _reconstruct(new)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindBalanceSheet, self).save_data(data, self.table_id, if_exists)


class WindProfitExpress(WindFinanceDB):
    table_name = u'中国A股业绩快报'
    table_id = 'profit_express'
    statement_type_map = {'0001': 0}

    def add_quarter_year(self, data):
        if data.empty:
            return data
        else:
            data['stat_type'] = '0001'
            return data

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据
        """

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        return self._wrap_add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindProfitExpress, self).save_data(data, self.table_id, if_exists)


class WindProfitNotice(WindFinanceDB):
    """中国A股业绩预告

    业绩预告类型明细:
        不确定 454001000
        略减 454002000
        略增 454003000
        扭亏 454004000
        其他 454005000
        首亏 454006000
        续亏 454007000
        续盈 454008000
        预减 454009000
        预增 454010000
    """
    table_id = 'profit_notice'
    table_name = u'中国A股业绩预告'
    statement_type_map = {454001000: 0, 454002000: 1, 454003000: 2,
                          454004000: 3, 454005000: 4, 454006000: 5,
                          454007000: 6, 454008000: 7, 454009000: 8,
                          454010000: 9}

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """
        取数据
        """
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if not isinstance(data, Iterator):
            return self.add_quarter_year(data)
        return self._wrap_add_quarter_year(data)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindProfitNotice, self).save_data(data, self.table_id, if_exists)


class WindAshareCapitalization(WindFinanceDB):
    """
    中国A股股本
    """
    table_name = u'中国A股股本'
    table_id = 'ashare_capitalization'

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据"""
        def _wrapper(idata):
            for i in idata:
                i['ann_dt'] = i['ann_dt'].astype('int32')
                i['IDs'] = i['IDs'].astype('int32')
                yield i
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if isinstance(data, Iterator):
            return _wrapper(data)
        if data.empty:
            return data
        data['ann_dt'] = data['ann_dt'].astype('int32')
        data['IDs'] = data['IDs'].astype('int32')
        return data

    @handle_ids
    def load_latest(self, factor_name, start=None, end=None, dates=None, ids=None):
        """某日最新数据"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        if start is not None and end is not None:
            dates = np.asarray(tc.get_trade_days(start, end, retstr='%Y%m%d')).astype('int')
        else:
            dates = np.asarray(dates).astype('int')
        if ids is not None:
            ids = np.asarray(ids).astype('int')
        data = self.load_h5(factor_name)
        new = self.data_loader.latest_period(data, wind_id, dates, ids, quarter=None)
        return _reconstruct(new)

    def load_avg(self, factor_name, start=None, end=None, dates=None, ids=None):
        """两个日期之间的平均值"""
        wind_id = self.data_dict.wind_factor_ids(self.table_name, factor_name)
        data = self.load_latest(factor_name, dates=[start, end], ids=ids).reset_index()
        data = pd.pivot_table(data, values=wind_id, index='IDs', columns='date', fill_value=np.nan)
        data.replace(0.0, np.nan, inplace=True)
        r = data.mean(axis=1)
        return r.to_frame(wind_id)

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindAshareCapitalization, self).save_data(data, self.table_id, if_exists)


class WindAindexMembers(WindFinanceDB):
    """Wind指数成分"""
    table_name = u'中国A股指数成分股'
    table_id = 'aindexmembers'

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据"""
        def _wrapper(idata):
            for i in idata:
                i.dropna(subset=['in_date'], inplace=True)
                i['IDs'] = i['IDs'].astype('int32')
                i['out_date'] = i['out_date'].fillna('22000000').astype('int32')
                i['in_date'] = i['in_date'].astype('int32')
                yield i
        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if isinstance(data, Iterator):
            return _wrapper(data)
        if data.empty:
            return data
        data.dropna(subset=['in_date'], inplace=True)
        data['IDs'] = data['IDs'].astype('int32')
        data['out_date'] = data['out_date'].fillna('22000000').astype('int32')
        data['in_date'] = data['in_date'].astype('int32')
        return data

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindAindexMembers, self).save_data(data, self.table_id, if_exists)

    @DateRange2Dates
    def get_members(self, idx, start_date=None, end_date=None, dates=None):
        l = []
        raw_data = self.load_h5(self.table_id)
        for d in {int(x.strftime('%Y%m%d')) for x in dates}:
            m = raw_data.query("idx_id=='%s' & in_date<=@d & out_date>=@d" % idx).copy()
            m['date'] = d
            m['sign'] = 1
            l.append(m[['IDs', 'date', 'sign']])
        r = pd.concat(l).set_index(['date', 'IDs'])
        return _reconstruct(r)


class WindAindexMembersWind(WindAindexMembers):
    """Wind指数成分"""
    table_name = u'中国A股万得指数成分股'
    table_id = 'aindexmemberswind'


class WindChangeWindcode(WindFinanceDB):
    """Wind代码变更表"""
    table_name = u'中国A股Wind代码变更表'
    table_id = 'asharechangewindcode'

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据"""
        def _reconstruct(raw):
            raw.dropna(subset=['change_dt'], inplace=True)
            raw = _drop_invalid_stocks(raw, 'new_id')
            raw['IDs'] = raw['IDs'].astype('int32')
            raw['change_dt'] = raw['change_dt'].astype('int32')
            raw['new_id'] = raw['new_id'].str[:6].astype('int32')
            return raw

        def _wrapper(idata):
            for i in idata:
                i = _reconstruct(i)
                yield i

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if isinstance(data, Iterator):
            return _wrapper(data)
        if data.empty:
            return data
        data = _reconstruct(data)
        return data

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindChangeWindcode, self).save_data(data, self.table_id, if_exists)

    @property
    def all_data(self):
        data = self.load_h5(self.table_id)
        return data


class WindIssuingDate(WindFinanceDB):
    """中国A股定期报告披露日期"""
    table_name = u'中国A股定期报告披露日期'
    table_id = 'ashareissuingdatepredict'
    
    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据"""
        def _reconstruct(raw):
            raw.fillna(0, inplace=True)
            raw['IDs'] = raw['IDs'].astype('int32')
            raw['date'] = raw['date'].astype('int32')
            raw['pre_ann_dt'] = raw['pre_ann_dt'].astype('int32')
            raw['ann_dt'] = raw['ann_dt'].astype('int32')
            return raw

        def _wrapper(idata):
            for i in idata:
                i = _reconstruct(i)
                yield i

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if isinstance(data, Iterator):
            return _wrapper(data)
        if data.empty:
            return data
        data = _reconstruct(data)
        return data

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindIssuingDate, self).save_data(data, self.table_id, if_exists)

    @property
    def all_data(self):
        data = self.load_h5(self.table_id)
        return data


class WindAshareDesc(WindFinanceDB):
    """A股基本资料"""
    table_name = u'中国A股基本资料'
    table_id = 'asharedescription'

    def download_data(self, factors, _in=None, _between=None, _equal=None, **kwargs):
        """取数据"""
        exec_market = {'434004000': 0, '434003000': 1, '434001000': 2}
        def _reconstruct(raw):
            raw.dropna(subset=['listdate'], inplace=True)
            raw.fillna({'delistdate': 21000000}, inplace=True)
            raw['IDs'] = raw['IDs'].astype('int32')
            raw[['listdate', 'delistdate']] = raw[['listdate', 'delistdate']].astype('int32')
            raw['shsc'] = raw['shsc'].astype('int32')
            raw['listboard'] = raw['listboard'].map(exec_market)
            return raw

        def _wrapper(idata):
            for i in idata:
                i = _reconstruct(i)
                yield i

        data = self.load_factors(factors, self.table_name, _in, _between, _equal, **kwargs)
        if isinstance(data, Iterator):
            return _wrapper(data)
        if data.empty:
            return data
        data = _reconstruct(data)
        return data

    def save_data(self, data, table_id=None, if_exists='append'):
        super(WindAshareDesc, self).save_data(data, self.table_id, if_exists)

    @property
    def all_data(self):
        data = self.load_h5(self.table_id)
        return data


if __name__ == '__main__':
    # from FactorLib.data_source.stock_universe import StockUniverse
    from datetime import datetime
    wind = WindConsensusDB()
    wind.connectdb()
    data = wind.download_data([u'预告净利润变动幅度下限(%)', u'预告净利润变动幅度上限(%)', u'预告净利润下限(万元)',
                               u'预告净利润上限(万元)'],
                              _between={u'报告期': ('20061231', '20180331')}, chunksize=10000)
    wind.save_data(data)
    # u = StockUniverse('000905')
    # ttm = wind.load_latest_period('净利润(不含少数股东损益)', ids=u, start='20170101', end='20171231')
    # print(ttm)


