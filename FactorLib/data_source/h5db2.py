# coding: utf-8

"""h5db性能加强版
保留h5db的框架，使用3D-Array保存原始数据提升读写性能
"""

import pandas as pd
import numpy as np
import os
import h5py
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime, IntDate2Datetime, Datetime2IntDate,\
    MatlabDatetime2Datetime, Datetime2MatlabDatetime
from ..utils.tool_funcs import tradecode_to_intcode, windcode_to_intcode, intcode_to_tradecode
from collections import namedtuple

_DataTypeDict = {0: 'WFloat', 1: 'BFloat', 2: 'Date'}
_Rule = namedtuple('Rule', ['data_type', 'multiplier'])
# 数值转换
_Converters = {
    'WFloat': _Rule(0, 10000),
    'BFloat': _Rule(1, 100),
    'Date': _Rule(2, 1)
}


class H5DB2(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path + '/../feather')
        self.snapshots_path = os.path.abspath(data_path + '/../snapshots')
        self.data_dict = None
        self._update_info()

    def _update_info(self):
        pass

    def set_data_path(self, path):
        self.data_path = path
        self._update_info()

    # ---------------------------因子管理---------------------------------------
    # 查看因子是否存在
    def check_factor_exists(self, factor_name, factor_dir='/'):
        return factor_name in self.data_dict[self.data_dict['path'] == factor_dir]['name'].values

    # 查看文件是否存在
    def check_file_exists(self, file_name, file_dir='/'):
        return os.path.isfile(self.abs_factor_path(file_dir, file_name))

    # 删除文件
    def delete_factor(self, factor_name, factor_dir='/'):
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        try:
            os.remove(factor_path)
        except Exception as e:
            print(e)
            pass
        self._update_info()

    # 列出单个文件的因子名称
    def list_file_factors(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with h5py.File(file_path, "r") as file:
            return (file['data'].attrs['factor_names']).split(",")

    # 列出文件的shape
    def list_file_shape(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with h5py.File(file_path, "r") as file:
            return tuple(file['data'].attrs['shape'])

    # 列出文件的日期
    def list_file_dates(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with h5py.File(file_path, "r") as file:
            return file['date'][...]

    # 列出文件的IDs
    def list_file_ids(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with h5py.File(file_path, "r") as file:
            return file['IDs'][...]

    # 列出因子数据类型
    def list_factor_types(self, file_name, file_dir, factors=None):
        file_path = self.abs_factor_path(file_dir, file_name)
        with h5py.File(file_path, "r") as file:
            if factors is None:
                return [_DataTypeDict[x] for x in file['dtypes'][...]]
            all_factors = ",".split((file['data'].attrs['factor_names']).decode('utf8'))
            return [_DataTypeDict[file['dtypes'][all_factors.index(x)]] for x in factors]

    # 重命名文件
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        os.rename(factor_path, temp_factor_path)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        if not os.path.isdir(self.data_path + factor_dir):
            os.makedirs(self.data_path + factor_dir)

    # 因子的时间区间
    def get_date_range(self, file_name, factor_path):
        with h5py.File(self.abs_factor_path(factor_path, file_name), "r") as file:
            max_date = MatlabDatetime2Datetime(file['date'][0])
            min_date = MatlabDatetime2Datetime(file['date'][-1])
        return Datetime2DateStr(min_date), Datetime2DateStr(max_date)

    # --------------------------数据管理-------------------------------------------

    def _read_raw(self, file_path, dates_idx, ids_idx, factor_idx):
        """读取原始数据，numpy作为容器"""
        s_date, e_date = dates_idx
        s_id, e_id = ids_idx
        s_factor, e_factor = factor_idx

        with h5py.File(file_path, "r") as file:
            factor_data = file['data'][s_date:e_date, s_id:e_id, s_factor:e_factor].astype('float64')
            factor_data[factor_data == -10000*10000.0] = np.nan
        return factor_data / 10000

    def load_factor(self, file_name, file_dir=None, factor_names=None, dates=None, ids=None,
                    idx=None, df=True, h5_style=True):
        """" 读取单因子数据

        paramters:
        ==========
        idx: DataFrame
            返回data.align(idx, how='right')[0]
        dates: list or array-like
            dates终将把每个元素转换成int格式
        ids: list or array-like
            ids终将把每个元素转换成string格式
        df: bool
            是否已DataFrame格式返回. True by default.
        """

        file_shape = self.list_file_shape(file_name, file_dir)
        all_factors = self.list_file_factors(file_name, file_dir)
        all_ids = self.list_file_ids(file_name, file_dir)
        all_dates = self.list_file_dates(file_name, file_dir)
        if factor_names is None:
            factor_idx = (0, file_shape[2])
            factor_names = all_factors
        else:
            temp_factors = np.sort(factor_names)
            factor_idx = (all_factors.index(temp_factors[0]), all_factors.index(temp_factors[-1]))
        if dates is None:
            date_idx = (0, file_shape[0])
            dates = all_dates
        else:
            dates = np.sort(Datetime2MatlabDatetime(pd.DatetimeIndex(dates).values))
            date_idx = (np.searchsorted(all_dates, dates[0]), np.searchsorted(all_dates, dates[-1], side='right'))
        if ids is None:
            ids_idx = (0, file_shape[1])
            ids = all_ids
        else:
            ids = np.sort(ids).astype('int')
            ids_idx = (np.searchsorted(all_ids, ids[0]), np.searchsorted(all_ids, ids[-1], side='right'))
        factor_data = self._read_raw(self.abs_factor_path(file_dir, file_name), date_idx, ids_idx, factor_idx)
        date_idx2 = np.in1d(all_dates[date_idx[0]:date_idx[-1]], dates)
        ids_idx2 = np.in1d(all_ids[ids_idx[0]:ids_idx[-1]], ids)
        factor_idx2 = np.in1d(all_factors[factor_idx[0]:factor_idx[-1]], factor_names)
        factor_data = factor_data[np.ix_(date_idx2, ids_idx2, factor_idx2)]
        if df:
            if h5_style:
                ids_str_func = np.frompyfunc(intcode_to_tradecode, 1, 1)
                ids_str = ids_str_func(ids)
                datetimes = MatlabDatetime2Datetime(dates)
                df = self.arr3d2df(factor_data, datetimes, ids_str, factor_names)
            else:
                df = self.arr3d2df(factor_data, dates, ids, factor_names)
            if idx is not None:
                df = df.reindex(idx.index)
            return df
        return factor_data


    def save_factor(self, factor_data, file_name, file_dir, if_exists='append'):
        """往数据库中写数据
        数据格式：DataFrame(index=[date,IDs],columns=data)
        """
        if factor_data.index.nlevels == 1:
            if isinstance(factor_data.index, pd.DatetimeIndex):
                factor_data['IDs'] = '111111'
                factor_data.set_index('IDs', append=True, inplace=True)
            else:
                factor_data['date'] = DateStr2Datetime('19700101')
                factor_data.set_index('date', append=True, inplace=True)
        factor_data.sort_index(inplace=True)
        factor_data.reset_index(inplace=True)
        factor_data.fillna(-10000, inplace=True)
        datetime_data = factor_data.select_dtypes(include='datetime64').apply(Datetime2MatlabDatetime)
        factor_data[datetime_data.columns] = datetime_data
        str_data = factor_data.select_dtypes('object').apply(lambda x: x.astype('int64'))
        factor_data[str_data.columns] = str_data
        factor_data.set_index(['date', 'IDs'], inplace=True)
        othertype_data = (factor_data.select_dtypes(include=['float64', 'float32', 'int32', 'int64',
                                                             'uint8']) * 10000).astype('int64')
        factor_data[othertype_data.columns] = othertype_data
        factor_data = factor_data.astype('int64')

        self.create_factor_dir(file_dir)
        all_ids = factor_data.index.get_level_values(1).unique()
        all_dates = factor_data.index.get_level_values(0).unique()
        factor_names = ",".join(list(factor_data.columns))
        if not self.check_file_exists(file_name, file_dir):
            arr3d = self.df2arr3d(factor_data)
            shape = arr3d.shape
            self.save_arr3d(self.abs_factor_path(file_dir, file_name), arr3d, all_dates, all_ids, factor_names, shape)
        elif if_exists == 'append':
            old_frame = (self.load_factor(file_name, file_dir, h5_style=False).fillna(-10000) * 10000).astype('int64')
            new_frame = old_frame.append(factor_data)
            new_frame = new_frame[~new_frame.index.duplicated(keep='last')].sort_index()
            new_arr = self.df2arr3d(new_frame)
            new_ids = new_frame.index.get_level_values(1).unique()
            new_dates = new_frame.index.get_level_values(0).unique()
            new_shape = new_arr.shape
            new_factors = ",".join(list(new_frame.columns))
            available_name = self.get_available_factor_name(file_name, file_dir)
            self.save_arr3d(self.abs_factor_path(file_dir, available_name), new_arr, new_dates, new_ids, new_factors, new_shape)
            self.delete_factor(file_name, file_dir)
            self.rename_factor(available_name, file_name, file_dir)
        elif if_exists == 'replace':
            self.delete_factor(file_name, file_dir)
            arr3d = self.df2arr3d(factor_data)
            shape = arr3d.shape
            self.save_arr3d(self.abs_factor_path(file_dir, file_name), arr3d, all_dates, all_ids, factor_names, shape)
        else:
            self._update_info()
            raise KeyError("please make sure if_exists is valide")
        self._update_info()

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + os.path.join(factor_path, factor_name + '.hdf5')

    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while os.path.isfile(self.abs_factor_path(factor_path, factor_name + str(i))):
            i += 1
        return factor_name + str(i)

    def arr3d2df(self, arr3d, dim1, dim2, col_names):
        idx = pd.MultiIndex.from_product([dim1, dim2], names=['date', 'IDs'])
        new_arr = arr3d.reshape((-1, arr3d.shape[-1]))
        return pd.DataFrame(new_arr, index=idx, columns=col_names)

    def df2arr3d(self, df):
        n_factors = df.shape[1]
        values = df.unstack().fillna(-10000 * 10000).values
        arr3d = values.reshape((values.shape[0], -1, n_factors), order='F')
        return arr3d

    def save_arr3d(self, path, arr3d, date, IDs, factor_names, shape):
        with h5py.File(path, "w") as file:
            dset = file.create_dataset("data", dtype=np.int64, data=arr3d, compression='lzf')
            dset.attrs['factor_names'] = str(factor_names)
            dset.attrs['shape'] = np.array(shape)
            file.create_dataset("date", dtype=np.int64, data=date, compression='lzf')
            file.create_dataset("IDs", dtype=np.int64, data=IDs, compression='lzf')