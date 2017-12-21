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
from .converter import parse_nc_encoding


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
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            t = list(file.data_vars)
            t.sort()
            return t

    # 列出文件的日期
    def list_file_dates(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, "data") as file:
            return file.indexes['date']

    # 列出文件的IDs
    def list_file_ids(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            return file.indexes['IDs']

    # 重命名文件
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        os.rename(factor_path, temp_factor_path)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        if not path.isdir(self.data_path + factor_dir):
            os.makedirs(self.data_path + factor_dir)

    # 因子的时间区间
    def get_date_range(self, file_name, file_path):
        dates = self.list_file_dates(file_name, file_path)
        return min(dates), max(dates)

    # --------------------------数据管理-------------------------------------------
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
        df: bool
            是否已DataFrame格式返回. True by default.
        ret: str
            数据返回的格式，选项包括df,ndarray,xarray
        """

        if factor_names is not None:
            all_names = self.list_file_factors(file_name, file_dir)
            factor_names_drop = [x for x in all_names if x not in factor_names]
        else:
            factor_names_drop = None
        if dates is None:
            dates = self.list_file_dates(file_name, file_dir)
        else:
            dates = pd.DatetimeIndex(dates)
        if ids is None:
            ids = self.list_file_ids(file_name, file_dir)
        idx1 = pd.MultiIndex.from_product([dates, ids], names=['date', 'IDs'])
        if idx is not None:
            idx1 = idx1.intersection(idx.index)
        file_path = self.abs_factor_path(file_dir, file_name)
        file = xr.open_dataset(file_path, "data", engine="netcdf4", drop_variables=factor_names_drop)
        factor_data = file.sel(date=idx1.get_level_values(0).unique(), IDs=idx1.get_level_values(1).unique())
        if ret == 'ndarry':
            length = len(ids)
            width = len(dates)
            height = len(factor_names)
            return np.array(dates), np.array(ids).astype('uint32'), factor_data.values.reshape((width, length, height))
        elif ret == 'xarray':
            return factor_data
        elif reset_index:
            return factor_data.to_dataframe().reset_index()
        else:
            df = factor_data.to_dataframe()
        return df

    def save_factor(self, factor_data, file_name, file_dir, if_exists='append', dtypes=None):
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
        # factor_data.sort_index(inplace=True)
        new_dtypes = dict(date=parse_nc_encoding(np.datetime64), IDs=parse_nc_encoding(np.object))
        dtypes = {} if dtypes is None else dtypes
        for k, v in factor_data.dtypes.iteritems():
            if str(k) in dtypes:
                new_dtypes[str(k)] = dtypes[str(k)]
            else:
                new_dtypes[str(k)] = parse_nc_encoding(v)

        self.create_factor_dir(file_dir)
        new_dset = factor_data.to_xarray()
        file_path = self.abs_factor_path(file_dir, file_name)
        if not self.check_file_exists(file_name, file_dir):
            new_dset.to_netcdf(file_path, "w", engine="netcdf4", encoding=new_dtypes, group="data")
        elif if_exists == 'append':
            factors = self.list_file_factors(file_name, file_dir)
            with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
                for factor in factors:
                    old_dtype = file[factor].encoding
                    new_dtypes[factor] = {k: v for k, v in old_dtype.items() if k
                                          in ['_FillValue', 'dtype', 'scale_factor', 'units', 'zlib', 'complevel']}
                new_data = new_dset.combine_first(file)
            available_name = self.get_available_factor_name(file_name, file_dir)
            new_data.to_netcdf(self.abs_factor_path(file_dir, available_name),
                               "w", engine="netcdf4", encoding=new_dtypes, group="data")
            self.delete_factor(file_name, file_dir)
            self.rename_factor(available_name, file_name, file_dir)
        elif if_exists == 'replace':
            self.delete_factor(file_name, file_dir)
            new_dset.to_netcdf(file_path, "w", engine="netcdf4", encoding=new_dtypes, group="data")
        else:
            self._update_info()
            raise KeyError("please make sure if_exists is valide")
        self._update_info()

    def save_as_dummy(self, factor_data, file_name, file_dir, if_exists='append'):
        """把数据当成哑变量保存，节省空间
        factor_data： DataFrame(index:[date, IDs], columns:[industry_1, industry_2, ...])
        """
        from .converter import INTEGER_ENCODING
        factor_data = factor_data.drop('T00018', axis=0, level=1).fillna(0)
        dummy_pack = np.packbits(factor_data.values.astype(dtype='uint8', copy=False), axis=1)
        new_dummy = pd.DataFrame(dummy_pack, index=factor_data.index, columns=[str(x) for x in range(dummy_pack.shape[1])])
        indu_names = ",".join(factor_data.columns)
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
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            for k, v in attr_dict.items():
                file[k].attrs.update(**v)
            available_name = self.get_available_factor_name(file_name, file_dir)
            file.to_netcdf(self.abs_factor_path(file_dir, available_name), "w", engine="netcdf4", group="data")
        self.delete_factor(file_name, file_dir)
        self.rename_factor(available_name, file_name, file_dir)

    def add_file_attr(self, file_name, file_dir, attr_dict):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            new_dset = file.assign_attrs(**attr_dict)
            available_name = self.get_available_factor_name(file_name, file_dir)
            new_dset.to_netcdf(self.abs_factor_path(file_dir, available_name), "w", engine="netcdf4", group="data")
        self.delete_factor(file_name, file_dir)
        self.rename_factor(available_name, file_name, file_dir)

    def load_factor_attr(self, file_name, file_dir, factor, key):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            return file[factor].attrs.get(key, None)

    def load_file_attr(self, file_name, file_dir, key):
        file_path = self.abs_factor_path(file_dir, file_name)
        with xr.open_dataset(file_path, "data", engine="netcdf4") as file:
            return file.attrs.get(key, None)

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + path.join(factor_path, factor_name + '.nc')

    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while path.isfile(self.abs_factor_path(factor_path, factor_name + str(i))):
            i += 1
        return factor_name + str(i)
