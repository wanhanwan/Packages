# encoding: utf-8
# author: wanhanwan

import bcolz
import datetime
import pandas as pd
import numpy as np

from os import path
from copy import copy
from pathlib import Path
from itertools import chain
from fastcache import clru_cache
from more_itertools import windowed
from pandas.api.types import(is_numeric_dtype,
                             is_datetime64_dtype,
                             is_string_dtype)
from FactorLib.utils.tool_funcs import ensure_dir_exists
from FactorLib.utils.disk_persist_provider import DiskPersistProvider


def _convert_datetime_to_int(df: pd.DataFrame, format='%Y%m%d'):
    """把时期转整数"""
    def _cvt(ser):
        return (
        ser.year * 1e10 + ser.month * 1e8 + ser.day * 1e6
        + ser.hour * 1e4 + ser.minute * 1e2 + ser.second
                )
    result = df.apply(_cvt)
    if format == "%Y%m%d":
        result = result // 1e6
    return result


@clru_cache()
def date_to_string(v, f):
    return v.strftime(f)


def check_dataframe_dtypes(df: pd.DataFrame):
    dtypes = {}
    for col_name, series in df.iteritems():
        if is_string_dtype(series):
            dtypes[col_name] = 'string'
        elif is_datetime64_dtype(series):
            dtypes[col_name] = 'datetime'
        elif is_numeric_dtype(series):
            dtypes[col_name] = 'numeric'
        else:
            raise TypeError("unrecongnized data type!")
    return dtypes


_default_data_dype = {
    'numeric': np.dtype('float64'),
    'datetime': np.dtype('datetime64[ns]'),
    'string': np.dtype('U32')
}


def _create_na_mapper(dtype, float_na):
    """
    string的空值是''
    datetime的空值是1900-1-1
    float的空值是nan
    """
    mapper = {}
    for key, tye in dtype.items():
        if tye == 'datetime':
            mapper[key] = datetime.datetime(1900, 1, 1)
        elif tye == 'numeric':
            mapper[key] = float_na[key]
        elif tye == 'string':
            mapper[key] = ''
    return mapper


def convert_data(arr, dtype, round, na):
    if dtype == 'string':
        return arr
    if dtype == 'datetime':
        if np.issubdtype(arr.dtype, np.integer):
            if arr[0] < 1e9:
                arr *= int(1e6)
            return pd.to_datetime(arr.astype(str), format='%Y%m%d%H%M%S').to_numpy()
        else:
            return arr
    if dtype == 'numeric':
        arr = arr / 10 ** round
        arr = np.where(arr==na, np.nan, arr)
        return arr


def convert_to_int(df, dtypes, rounds, datetime_format='%Y%m%d', convert_datetime=True):
    for name, tpe in dtypes.items():
        if tpe == 'numeric':
            df[name] = (df[name] * 10 ** rounds[name]).astype('int64')
        elif tpe == 'datetime' and convert_datetime:
            df[name] = _convert_datetime_to_int(df[name], datetime_format).astype('int64')
        elif tpe == 'string':
            pass
    return df


def id_range(id_series, datetime_format):
    assert id_series.is_monotonic
    id_series = id_series.to_numpy()
    last_indices = np.where(id_series[1:]!=id_series[:-1])[0]
    if last_indices.size == 0:
        return {id_series[0]: (0, len(id_series))}
    range_ = {id_series[x]: (int(x), int(y)) for x, y in
                windowed(chain([0], last_indices + 1, [len(id_series)]), 2)}
    return range_


def append_along_index(df1, df2):
    df1, df2 = df1.align(df2, axis='columns')
    new = pd.DataFrame(np.vstack((df1.values, df2.values)),
                       columns=df1.columns,
                       index=df1.index.append(df2.index))
    new.sort_index(inplace=True)
    return new


class BcolzFile(object):
    def __init__(self, pth):
        self._pth = pth
        self._table = bcolz.open(pth, mode='r')

    @property
    def line_map(self):
        return self._table.attrs['line_map']


class BcolzDB(object):
    """
    BcolzDB

    BloscDB基于bcolz库，是一个列式数据库解决方案。支持
    浮点、日期和字符串三种数据格式的读写，兼容Pandas和Numpy
    数据格式。
    """
    def __init__(self, root_dir):
        self.root_dir = str(root_dir)
        self.cached_table_handlers = {}
        self.cached_table_silces = {}

    def _p(self, *args):
        return path.join(
            self.root_dir,
            *(x.strip('/') for x in args)
        ) + '.bcolz'

    def check_table_existence(self, table_path):
        return path.isdir(table_path)

    def _load_table_handler(self, pth):
        """pth绝对路径"""
        if pth in self.cached_table_handlers:
            return self.cached_table_handlers[pth]
        self.cached_table_handlers[pth] = BcolzFile(pth)
        return self.cached_table_handlers[pth]

    def load_factor_attr(self, factor_path, name):
        pth = path.join(self.root_dir, factor_path.strip('/'), name, 'attrs.pkl')
        return pd.read_pickle(pth)

    def _load_table_slice(self, pth: str, field: str):
        """pth: bcolz文件绝对路径"""
        handler = self._load_table_handler(pth)
        assert field in handler._table.names
        return handler._table.cols[field]
    
    def _load_table_column(self, pth: str, f: str, s: int, e: int, attr:dict):
        return convert_data(
            self._load_table_slice(pth, f)[s:e],
            attr['dtype_mapper'][f],
            attr['round_mapper'][f],
            attr['na_mapper'][f]
        )

    def load_factor(self,
                    name,
                    factor_path,
                    idx_name,
                    idx_start=None,
                    idx_end=None,
                    idx=None,
                    set_idx_name=None,
                    reindex_idx=None,
                    fields=None,
                    **kwargs
                    ):
        """
        加载一个因子
        """
        if idx is not None:
            idx_start, idx_end = min(idx), max(idx)
        pth = self._p(factor_path, name, idx_name)
        attr = pd.read_pickle(Path(pth).parent / 'attrs.pkl')
        handler = self._load_table_handler(pth)

        if idx_start is None and idx_end is None:
            s, e  = 0, attr['max_rows']
        elif idx_start is None:
            s = 0
            e = handler.line_map[idx_end][1]
        elif idx_end is None:
            s = handler.line_map[idx_start][0]
            e = attr['max_rows']
        else:
            s = handler.line_map[idx_start][0]
            e = handler.line_map[idx_end][1]

        fields = fields or handler._table.names
        dtypes = [
            (
                f,
                _default_data_dype[attr['dtype_mapper'][f]]
            )
        for f in fields
        ]
        arr = np.empty(shape=e-s, dtype=dtypes)
        for f in fields:
            arr[f] = self._load_table_column(pth, f, s, e, attr)
        df = pd.DataFrame(arr, columns=fields)
        if set_idx_name is not None:
            df.set_index(set_idx_name, inplace=True)
        if reindex_idx is not None:
            df = df.reindex(reindex_idx.index)
        return df

    def save_factor(self,
                    df: pd.DataFrame,
                    save_path: str,
                    name: str,
                    nan: int = -1,
                    na_mapper: dict = None,
                    round: int = 4,
                    round_mapper: dict = None,
                    datetime_format: str = '%Y%m%d',
                    save_datetime_as_int: bool = True,
                    sorter: list = None,
                    if_exists = 'update'
                    ):
        """
            保存一个DataFrame. 如果DataFrame有MultiIndex, 则为每一个level
        创建一个索引，这通常用于为资产代码和日期分别建立横截面模型和时间
        序列模型。
            注：索引的数据类型必须是整数或字符串(日期类型会转成%Y%m%d的字符串)。
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df.index, pd.MultiIndex):
            index_names = df.index.names
        else:
            index_names = [df.index.name]
        df.reset_index(inplace=True)

        for idx in index_names:
            pth = self._p(save_path, name, idx)
            if self.check_table_existence(pth):
                factor_attrs = self.load_factor_attr(save_path, name)
                if if_exists == 'update':
                    old = self.load_factor(
                        name,
                        save_path,
                        idx,
                        attrs = factor_attrs
                    )
                    d = old.append(df).drop_duplicates(subset=index_names, keep='last')
                else:
                    d = df.copy()
            else:
                ensure_dir_exists(path.dirname(pth))
                d = df.copy()

            dtype_mapper = check_dataframe_dtypes(d)

            na_mapper = {} if na_mapper is None else na_mapper
            for n in dtype_mapper:
                if n not in na_mapper:
                    na_mapper[n] = nan
            df_na_value = _create_na_mapper(dtype_mapper, na_mapper)
            for n, value in df_na_value.items():
                d[n].fillna(value, inplace=True)

            round_mapper = {} if round_mapper is None else round_mapper
            for n in dtype_mapper:
                if n not in round_mapper:
                    round_mapper[n] = round

            d = convert_to_int(d, dtype_mapper, round_mapper, datetime_format, save_datetime_as_int)

            sort_columns = copy(sorter or [idx])
            if idx not in sort_columns:
                sort_columns = [idx] + sort_columns
            else:
                sort_columns.remove(idx)
                sort_columns = [idx] + sort_columns
            d = d.sort_values(sort_columns).reset_index(drop=True)
            line_map = id_range(d[idx], datetime_format)
            if is_datetime64_dtype(is_datetime64_dtype(line_map.keys())):
                line_map = {x.strftime(datetime_format):y for x, y in line_map.items()}
            else:
                line_map = {str(x):y for x, y in line_map.items()}

            ct = bcolz.ctable.fromdataframe(d, rootdir=pth, mode='w')
            ct.attrs['line_map'] = line_map

        attrs = {
            'index_names' : index_names,
            'dtype_mapper': dtype_mapper,
            'na_mapper': na_mapper,
            'round_mapper': round_mapper,
            'max_rows': d.shape[0],
            'datetime_format': datetime_format
        }
        dumper = DiskPersistProvider(path.join(self.root_dir, save_path.strip('/'), name))
        dumper.dump(attrs, 'attrs')
