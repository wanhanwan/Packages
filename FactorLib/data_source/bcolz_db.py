# encoding: utf-8
# author: wanhanwan
import os
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


def _create_na_mapper(dtype, user_mapper, float_na):
    """
    string的空值是''
    datetime的空值是1900-1-1
    float的空值是nan
    """
    dft_na = {
        'datetime': pd.Timestamp('1900-01-01'),
        'numeric': float_na,
        'string': ''
    }
    mapper = {
        key: user_mapper.get(key, dft_na[value]) for
            key, value in dtype.items()
    }
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


def id_range(id_series):
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


def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b


def remove_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to remove.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return a[keep_names]



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
    
    def _idx_of(self, line_map, idx_start=None, idx_end=None):
        keys = np.asarray(list(line_map.keys()), dtype=str)
        s = line_map[min(keys[keys >= idx_start])][0] if idx_start else None
        e = line_map[max(keys[keys<=idx_end])][1] if idx_end else None
        return s, e

    def load_factor(self,
                    name,
                    factor_path,
                    idx_name,
                    idx_start=None,
                    idx_end=None,
                    idx: pd.Index = None,
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
            idx_name = idx.name
        pth = self._p(factor_path, name, idx_name)
        attr = pd.read_pickle(Path(pth).parent / 'attrs.pkl')
        handler = self._load_table_handler(pth)

        if idx_start is None and idx_end is None:
            s, e  = 0, attr['max_rows']
        elif idx_start is None:
            s = 0
            e = self._idx_of(handler.line_map, idx_end=idx_end)[1]
        elif idx_end is None:
            s = self._idx_of(handler.line_map, idx_start=idx_start)[0]
            e = attr['max_rows']
        else:
            s, e = self._idx_of(handler.line_map, idx_start, idx_end)

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
                    if_exists: str = 'update',
                    save_index: bool = True
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

        dtype_mapper = check_dataframe_dtypes(df)
        na_mapper = _create_na_mapper(dtype_mapper, na_mapper or {}, nan)
        df.fillna(na_mapper, inplace=True)

        dft_round_mapper = {key: round for key in dtype_mapper}
        if round_mapper:
            round_mapper.update(dft_round_mapper)
        else:
            round_mapper = dft_round_mapper
        df = convert_to_int(df, dtype_mapper, round_mapper, datetime_format, save_datetime_as_int)

        def update_matrix(old, new, unique_names):
            mask = np.ones(old.shape[0], dtype='bool')
            for n in unique_names:
                np.logical_and(mask, np.in1d(old[n], new[n]), out=mask)
            old = np.delete(old, np.where(mask), axis=0)
            new = np.hstack((old, new)) # todo  invalid type promotion
            return new

        for idx in index_names:
            pth = self._p(save_path, name, idx)
            if self.check_table_existence(pth):
                factor_attrs = self.load_factor_attr(save_path, name)
                if if_exists == 'update':
                    old = self._load_table_handler(pth)._table[:]
                    new = df.to_records(
                        index=False,
                        column_dtypes={x:old.dtype[x] for x in df.columns}
                    )
                    new = update_matrix(old, new, factor_attrs['index_names'])
                else:
                    new = df.to_records(index=False)
            else:
                ensure_dir_exists(path.dirname(pth))
                new = df.to_records(index=False)

            sort_columns = copy(sorter or [idx])
            if idx not in sort_columns:
                sort_columns = [idx] + sort_columns
            else:
                sort_columns.remove(idx)
                sort_columns = [idx] + sort_columns
            new.sort(order=sort_columns)
            line_map = id_range(new[idx])
            if np.issubdtype(new[idx].dtype, np.datetime64):
                line_map = {pd.Timestamp(x).strftime(datetime_format):y for x, y in line_map.items()}
            else:
                line_map = {str(x):y for x, y in line_map.items()}
            if not save_index:
                ct = bcolz.ctable(remove_fields(new, [idx]), rootdir=pth, mode='w')
                ct.attrs['line_map'] = list(new[idx])
            else:
                ct = bcolz.ctable(new, rootdir=pth, mode='w')
                ct.attrs['line_map'] = line_map

        attrs = {
            'index_names' : index_names,
            'dtype_mapper': dtype_mapper,
            'na_mapper': na_mapper,
            'round_mapper': round_mapper,
            'max_rows': new.shape[0],
            'datetime_format': datetime_format
        }
        dumper = DiskPersistProvider(path.join(self.root_dir, save_path.strip('/'), name))
        dumper.dump(attrs, 'attrs')

    def save_factor_by_date(self,
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
                            if_exists='update'
                            ):
        for dt, data in df.groupby('date', group_keys=False, as_index=False):
            try:
                data = data.reset_index('date', drop=True)
            except:
                pass
            self.save_factor(data,
                             f'{save_path}{name}/',
                             dt.strftime(datetime_format),
                             nan,
                             na_mapper,
                             round,
                             round_mapper,
                             datetime_format,
                             save_datetime_as_int,
                             sorter,
                             'replace',
                             False
                             )

    @clru_cache()
    def _load_factor_by_date(self, name, factor_path, date):
        h = self._load_table_handler(self._p(factor_path, name, date, 'IDs'))
        data = self.load_factor(date, factor_path+'/'+name+'/', 'IDs')
        data.index = pd.Index(h.line_map, name='IDs')
        return data

    def load_factor_by_date(self,
                            name: str,
                            factor_path: str,
                            start_date: str,
                            end_date: str,
                            ids: list=None,
                            fields: list=None
                            ):
        df = []
        dates = []
        for dt in (x for x in os.listdir(self.root_dir+factor_path+'/'+name)
                   if start_date<=x<=end_date):
            _d = self._load_factor_by_date(name, factor_path, dt)
            df.append(_d)
            dates.append(dt)
        df = pd.concat(df, keys=pd.DatetimeIndex(dates, name='date'))
        if ids:
            df = df.loc[pd.IndexSlice[:, ids], :]
        if fields:
            df = df[fields]
        return df
