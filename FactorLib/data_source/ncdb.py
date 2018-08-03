# coding: utf-8

"""基于hdf5的netcdf数据库
保留h5db的框架，使用xarry保存原始数据提升读写性能
支持单文件多因子存储
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
from os import path
from multiprocess import Lock
from .converter import parse_nc_encoding
from .helpers import handle_ids

lock = Lock()


class NCDB(object):
    def __init__(self, data_path):
        self.data_path = data_path
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
        pass

    # 查看文件是否存在
    def check_file_exists(self, file_name, file_dir='/'):
        return path.isfile(self.abs_factor_path(file_dir, file_name))

    # 删除文件
    def delete_factor(self, factor_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        try:
            os.remove(factor_path)
        except Exception as e:
            self._update_info()
            raise e

    # 列出单个文件的因子名称
    def list_file_factors(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            t = list(file.data_vars)
            t.sort()
        return t

    # 列出文件的日期
    def list_file_dates(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine='netcdf4') as file:
            return file.indexes['date']

    # 列出文件的IDs
    def list_file_ids(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            return file.indexes['IDs']

    # 重命名文件
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        os.rename(factor_path, temp_factor_path)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        factor_dir = factor_dir.strip('/\\')
        tar_dir = path.join(self.data_path, factor_dir)
        if not path.isdir(tar_dir):
            os.makedirs(tar_dir)

    # 因子的时间区间
    def get_date_range(self, file_name, file_path):
        dates = self.list_file_dates(file_name, file_path)
        return min(dates), max(dates)

    # --------------------------数据管理-------------------------------------------
    @ handle_ids
    def load_factor(self, file_name, file_dir=None, factor_names=None, dates=None, ids=None,
                    idx=None, ret='df', reset_index=False):
        """ 读取单因子数据
        paramters:
        ==========
        idx: DataFrame
            返回data.align(idx, how='right')[0]
        dates: list or array-like
            dates终将把每个元素转换成int格式
        ids: list or array-like
            ids终将把每个元素转换成string格式
        ret: str
            数据返回的格式，选项包括df,xarray,panel
        """

        if factor_names is not None:
            all_names = self.list_file_factors(file_name, file_dir)
            factor_names_drop = [x for x in all_names if x not in factor_names]
        else:
            factor_names_drop = None
        if idx is not None:
            dates = idx.index.unique(level='date')
            ids = idx.index.unique(level='IDs')
        else:
            if dates is None:
                dates = self.list_file_dates(file_name, file_dir)
            else:
                dates = pd.DatetimeIndex(dates)
            if ids is None:
                ids = self.list_file_ids(file_name, file_dir)
        file_path = self.abs_factor_path(file_dir, file_name)

        try:
            lock.acquire()
            with xr.open_dataset(file_path, engine="netcdf4",
                                 drop_variables=factor_names_drop,
                                 autoclose=True) as file:
                factor_data = file.sel(date=dates, IDs=ids)
            lock.release()
        except Exception as e:
            lock.release()
            raise e

        if ret == 'xarray':
            return factor_data
        elif ret == 'panel':
            return factor_data.to_dataframe().to_panel()
        else:
            factor_data = factor_data.to_dataframe()
        if reset_index:
            return factor_data.dropna(how='all').reset_index()
        else:
            df = factor_data.dropna(how='all')
            if df.index.names[0] == 'IDs':
                df.index = df.index.swaplevel('IDs', 'date')
            if idx is None:
                df.sort_index(inplace=True)
            else:
                df = df.reindex(idx.index)
            return df

    def save_factor(self, factor_data, file_name, file_dir,
                    if_exists='append'):
        """往数据库中写数据
        数据格式：DataFrame(index=[date,IDs],columns=data)
        """
        if factor_data.index.nlevels == 1:
            if isinstance(factor_data.index, pd.DatetimeIndex):
                factor_data['IDs'] = '111111'
                factor_data.set_index('IDs', append=True, inplace=True)
            else:
                factor_data['date'] = np.datetime64('1970-01-01')
                factor_data.set_index('date', append=True, inplace=True)
        if isinstance(factor_data.columns, pd.MultiIndex):
            factor_data = factor_data.stack()

        self.create_factor_dir(file_dir)
        new_dset = factor_data.to_xarray()
        file_path = self.abs_factor_path(file_dir, file_name)

        lock.acquire()
        try:
            if not self.check_file_exists(file_name, file_dir):
                new_dset.to_netcdf(file_path, "w", engine="netcdf4")
            elif if_exists == 'append':
                with xr.open_dataset(file_path, engine="netcdf4") as file:
                    new_data = new_dset.combine_first(file)
                new_data.to_netcdf(file_path, "w", engine="netcdf4")
            elif if_exists == 'replace':
                new_dset.to_netcdf(file_path, "w", engine="netcdf4")
            else:
                raise KeyError("please make sure if_exists is valide")
        except Exception as e:
            self._update_info()
            lock.release()
            raise e
        lock.release()
        self._update_info()

    def save_as_dummy(self, factor_data, file_name, file_dir, if_exists='append'):
        """把数据当成哑变量保存，节省空间
        factor_data： DataFrame(index:[date, IDs], columns:[industry_1, industry_2, ...])
        """
        from .converter import INTEGER_ENCODING
        factor_data = factor_data.drop('T00018', axis=0, level=1).fillna(0)
        if self.check_file_exists(file_name, file_dir):
            indu_names = self.load_file_attr(file_name, file_dir, 'indu_names').split(",")
            factor_data = factor_data.reindex(columns=indu_names, copy=False)
        indu_names = ",".join(factor_data.columns)
        dummy_pack = np.packbits(factor_data.values.astype(dtype='uint8', copy=False), axis=1)
        new_dummy = pd.DataFrame(dummy_pack, index=factor_data.index, columns=[str(x) for x in range(dummy_pack.shape[1])])
        self.save_factor(new_dummy, file_name, file_dir, dtypes={str(x): INTEGER_ENCODING for x in new_dummy.columns},
                         if_exists=if_exists)
        self.add_file_attr(file_name, file_dir, {'indu_names': indu_names})

    def load_as_dummy(self, file_name, file_dir, dates=None, ids=None, idx=None, drop_first=False):
        """加载哑变量数据"""
        dummy = self.load_factor(file_name, file_dir, ids=ids, dates=dates, idx=idx).dropna(how='all').sort_index(axis=1)
        industry_names = self.load_file_attr(file_name, file_dir, 'indu_names').split(",")
        dummy_value = np.unpackbits(dummy.values.astype('uint8'), axis=1)[:, :len(industry_names)]
        new_dummy = pd.DataFrame(dummy_value, index=dummy.index, columns=industry_names)
        if drop_first:
            return new_dummy.iloc[:, 1:]
        return new_dummy

    def add_factor_attr(self, file_name, file_dir, attr_dict):
        """添加因子属性
        """
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            for k, v in attr_dict.items():
                file[k].attrs.update(**v)
            available_name = self.get_available_factor_name(file_name, file_dir)
            file.to_netcdf(self.abs_factor_path(file_dir, available_name), "w", engine="netcdf4", group="data")
        self.delete_factor(file_name, file_dir)
        self.rename_factor(available_name, file_name, file_dir)

    def add_file_attr(self, file_name, file_dir, attr_dict):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            new_dset = file.assign_attrs(**attr_dict)
            available_name = self.get_available_factor_name(file_name, file_dir)
            new_dset.to_netcdf(self.abs_factor_path(file_dir, available_name), "w", engine="netcdf4", group="data")
        self.delete_factor(file_name, file_dir)
        self.rename_factor(available_name, file_name, file_dir)

    def load_factor_attr(self, file_name, file_dir, factor, key):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            return file[factor].attrs.get(key, None)

    def load_file_attr(self, file_name, file_dir, key):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, engine="netcdf4") as file:
            return file.attrs.get(key, None)

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + path.join(factor_path, factor_name + '.nc')

    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while path.isfile(self.abs_factor_path(factor_path, factor_name + str(i))):
            i += 1
        return factor_name + str(i)
