# coding: utf-8
"""基于HDF文件的数据库"""

import pandas as pd
import numpy as np
import os
import shutil
import warnings
from multiprocessing import Lock
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime
from ..utils.tool_funcs import ensure_dir_exists
from ..utils.disk_persist_provider import DiskPersistProvider
from .helpers import handle_ids, FIFODict

lock = Lock()
warnings.simplefilter('ignore', category=FutureWarning)


class H5DB(object):
    def __init__(self, data_path, max_cached_files=30):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path+'/../feather')
        self.csv_data_path = os.path.abspath(data_path+'/../csv')
        self.data_dict = None
        self.cached_data = FIFODict(max_cached_files)
        # self._update_info()
    
    def _update_info(self):
        factor_list = []
        
        for root, subdirs, files in os.walk(self.data_path):
            relpath = "/%s/"%os.path.relpath(root, self.data_path).replace("\\", "/")
            for file in files:
                if file.endswith(".h5"):
                    factor_list.append([relpath, file[:-3]])
        self.data_dict = pd.DataFrame(
            factor_list, columns=['path', 'name'])

    def _read_h5file(self, file_path, key):
        if file_path in self.cached_data:
            return self.cached_data[file_path]

        lock.acquire()
        try:
            data = pd.read_hdf(file_path, key)
            self.cached_data[file_path] = data
            lock.release()
            return self.cached_data[file_path]
        except Exception as e:
            lock.release()
            raise e

    def _save_h5file(self, data, file_path, key,
                     complib='blosc', complevel=9,
                     **kwargs):
        try:
            lock.acquire()
            data.to_hdf(file_path, key=key, complib=complib,
                        complevel=complevel, **kwargs)
            if file_path in self.cached_data:
                self.cached_data.update(file_path=data)
            lock.release()
        except Exception as e:
            lock.release()
            raise e

    def _read_pklfile(self, file_path):
        if file_path in self.cached_data:
            return self.cached_data[file_path]
        lock.acquire()
        try:
            d = pd.read_pickle(file_path)
            self.cached_data[file_path] = d
            lock.release()
        except Exception as e:
            lock.release()
            raise e
        return d

    def _save_pklfile(self, data, file_dir, name, protocol=-1):
        dumper = DiskPersistProvider(
            os.path.join(self.data_path, file_dir.strip('/')))
        file_path = os.path.join(
            self.data_path, file_dir.strip('/'), name+'.pkl'
        )
        lock.acquire()
        try:
            dumper.dump(data, name, protocol)
            if file_path in self.cached_data:
                self.cached_data[file_path] = data
        except Exception as e:
            lock.release()
            raise e
        lock.release()

    def _read_feather(self, file_path):
        if file_path in self.cached_data:
            return self.cached_data[file_path]
        lock.acquire()
        try:
            d = pd.read_feather(file_path)
            self.cached_data[file_path] = d
            lock.release()
        except Exception as e:
            lock.release()
            raise e
        return d

    def _save_feather(self, data, file_path):
        lock.acquire()
        try:
            data.to_feather(file_path)
            if file_path in self.cached_data:
                self.cached_data.update(file_path=data)
        except Exception as e:
            lock.release()
            raise e
        lock.release()

    def _delete_cached_factor(self, file_path):
        if file_path in self.cached_data:
            del self.cached_data[file_path]
    
    def set_data_path(self, path):
        self.data_path = path
        # self._update_info()
    
    # ---------------------------因子管理---------------------------------------
    # 查看因子是否存在
    def check_factor_exists(self, factor_name, factor_dir='/'):
        file_path = self.abs_factor_path(factor_dir, factor_name)
        return os.path.isfile(file_path)

    # 删除因子
    def delete_factor(self, factor_name, factor_dir='/'):
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        try:
            os.remove(factor_path)
            self._delete_cached_factor(factor_path)
        except Exception as e:
            print(e)
            pass
        self._update_info()
    
    # 列出因子名称
    def list_factors(self, factor_dir):
        dir_path = os.path.join(self.data_path, factor_dir)
        factors = [x for x in os.listdir(dir_path) if x.endswith('.h5')]
        return factors
    
    # 重命名因子
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        if factor_path in self.cached_data:
            factor_data = self.cached_data.pop(factor_path)
            self.cached_data[temp_factor_path] = factor_data
        else:
            factor_data = self._read_h5file(factor_path, old_name). \
                to_frame().rename(columns={old_name: new_name}).to_panel()
            del self.cached_data[factor_path]
        self._save_h5file(factor_data, temp_factor_path, factor_data.columns[0])
        self.delete_factor(old_name, factor_dir)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        if not os.path.isdir(self.data_path+factor_dir):
            os.makedirs(self.data_path+factor_dir)
    
    # 因子的时间区间
    def get_date_range(self, factor_name, factor_path):
        try:
            max_date = self.read_h5file_attr(factor_name, factor_path, 'max_date')
            min_date = self.read_h5file_attr(factor_name, factor_path, 'min_date')
        except Exception:
            panel = self._read_h5file(
                self.abs_factor_path(factor_path, factor_name), key='data')
            if isinstance(panel, pd.Panel):
                min_date = Datetime2DateStr(panel.major_axis.min())
                max_date = Datetime2DateStr(panel.major_axis.max())
            else:
                min_date = panel.index.get_level_values('date').min()
                max_date = panel.index.get_level_values('date').max()
        return min_date, max_date

    # 读取多列因子的属性
    def read_h5file_attr(self, factor_name, factor_path, attr_name):
        attr_file_path = self.abs_factor_attr_path(factor_path, factor_name)
        if os.path.isfile(attr_file_path):
            return self._read_pklfile(attr_file_path)[attr_name]
        else:
            raise FileNotFoundError('找不到因子属性文件!')

    # --------------------------数据管理-------------------------------------------
    @handle_ids
    def load_factor(self, factor_name, factor_dir=None, dates=None, ids=None, idx=None):
        if idx is not None:
            dates = idx.index.get_level_values('date').unique()
            return self.load_factor(factor_name, factor_dir=factor_dir, dates=dates).reindex(idx.index, copy=False)
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        panel = self._read_h5file(factor_path, factor_name)
        if (ids is not None) and (not isinstance(ids, list)):
            ids = [ids]
        if dates is None and ids is None:
            df = panel.to_frame()
            return df
        elif dates is None:
            df = panel.ix[factor_name, :, ids].stack().to_frame()
            df.index.names = ['date', 'IDs']
            df.columns = [factor_name]            
            return df
        elif ids is None:
            df = panel.ix[factor_name, pd.DatetimeIndex(dates), :].stack().to_frame()
            df.index.names = ['date', 'IDs']
            df.columns = [factor_name]              
            return df
        else:
            df = panel.ix[factor_name, pd.DatetimeIndex(dates), ids].stack().to_frame()
            df.index.names = ['date', 'IDs']
            df.columns = [factor_name]              
            return df

    @handle_ids
    def load_multi_columns(self, file_name, path, group='data', ids=None, dates=None,
                           idx=None, factor_names=None, **kwargs):
        """读取h5File"""
        attr_file_path = self.data_path + path + file_name + '_attr.pkl'
        file_path = self.abs_factor_path(path, file_name)

        # 读取meta data
        try:
            attr = self._read_pklfile(attr_file_path)
            multiplier = attr['multiplier']
        except FileNotFoundError:
            multiplier = kwargs.get('multiplier', 100)

        data = self._read_h5file(file_path, group)
        all_dates = data.index.get_level_values('date').unique()
        all_ids = data.index.get_level_values('IDs').unique()
        if idx is not None:
            data = data.reindex(idx.index)
        else:
            if dates is not None:
                dates = pd.DatetimeIndex(dates).intersection(all_dates)
            else:
                dates = all_dates
                # data = data.loc[pd.DatetimeIndex(dates).values].copy()
            if ids is not None:
                # data = data.loc[pd.IndexSlice[:, list(ids)], :]
                ids = np.intersect1d(ids, all_ids)
            else:
                ids = all_ids
            idx = pd.MultiIndex.from_product([dates, ids], names=['date', 'IDs'])
            data = data.reindex(idx)
        data /= multiplier

        if callable(factor_names):
            return data[[x for x in data.columns if factor_names(x)]]
        if factor_names is not None:
            return data[factor_names]
        return data

    def save_multi_columns(self, data, path, name, group='data',
                           multiplier=100, fill_value=100, **kwargs):
        """保存多列的DataFrame"""
        file_path = self.abs_factor_path(path, name)
        attr_file_path = self.data_path + path + name + '_attr.pkl'

        if os.path.isfile(attr_file_path):
            attr = self._read_pklfile
            multiplier = attr['multiplier']
            fill_value = attr['fill_value']

        data *= multiplier
        ensure_dir_exists(os.path.dirname(file_path))
        if not self.check_factor_exists(name, path):
            self._save_h5file(data, file_path, group)
            new_df = data
        else:
            df = self._read_h5file(file_path, group)
            new_df = df.append(data)
            new_df = new_df[~new_df.index.duplicated(keep='last')].sort_index()
            self._save_h5file(data, file_path, group)
        attr = {'multiplier': multiplier,
                'fill_value': fill_value,
                'factors': new_df.columns.tolist(),
                'max_date': new_df.index.get_level_values('date').max(),
                'min_date': new_df.index.get_level_values('date').min()
                }
        self._save_pklfile(attr, path, name+'_attr', protocol=2)

    def read_h5file(self, file_name, path, group='data'):
        file_path = self.abs_factor_path(path, file_name)
        return self._read_h5file(file_path, key=group)

    def save_h5file(self, data, name, path, group='data', mode='a'):
        file_path = self.abs_factor_path(path, name)
        if self.check_factor_exists(name, path) and mode != 'w':
            df = self.read_h5file(name, path, group=group)
            data = df.append(data).drop_duplicates()
        self._save_h5file(data, file_path, group)

    def list_h5file_factors(self, file_name, file_pth):
        """"提取h5File的所有列名"""
        attr_file_path = self.data_path + file_pth + file_name + '_attr.pkl'
        file_path = self.abs_factor_path(file_pth, file_name)
        if os.path.isfile(attr_file_path):
            attr = pd.read_pickle(attr_file_path)
            return attr['factors']
        attr_file_path = self.data_path + file_pth + file_name + '_mapping.pkl'
        try:
            attr = pd.read_pickle(attr_file_path)
            return attr
        except FileNotFoundError:
            df = self._read_h5file(file_path, "data")
            return df.columns.tolist()

    def load_latest_period(self, factor_name, factor_dir=None, ids=None, idx=None):
        max_date = self.get_date_range(factor_name, factor_dir)[1]
        return self.load_factor(
            factor_name, factor_dir, dates=[max_date], ids=ids, idx=idx).reset_index(level=0, drop=True)

    def load_factors(self, factor_names_dict, dates=None, ids=None):
        _l = []
        for factor_path, factor_names in factor_names_dict.items():
            for factor_name in factor_names:
                df = self.load_factor(factor_name, factor_dir=factor_path, dates=dates, ids=ids)
                _l.append(df)
        return pd.concat(_l, axis=1)
    
    def save_factor(self, factor_data, factor_dir, if_exists='append'):
        """往数据库中写数据
        数据格式：DataFrame(index=[date,IDs],columns=data)
        """
        if factor_data.index.nlevels == 1:
            if isinstance(factor_data.index, pd.DatetimeIndex):
                factor_data['IDs'] = '111111'
                factor_data.set_index('IDs', append=True, inplace=True)
            else:
                factor_data['date'] = DateStr2Datetime('19000101')
                factor_data.set_index('date', append=True, inplace=True)

        self.create_factor_dir(factor_dir)
        for column in factor_data.columns:
            factor_path = self.abs_factor_path(factor_dir, column)
            if not self.check_factor_exists(column, factor_dir):
                self._save_h5file(factor_data[[column]].dropna().to_panel(),
                                  factor_path, column)
            elif if_exists == 'append':
                old_panel = self._read_h5file(factor_path, column)
                new_frame = old_panel.to_frame().append(factor_data[[column]].dropna())
                new_panel = new_frame[~new_frame.index.duplicated(keep='last')].to_panel()
                available_name = self.get_available_factor_name(column, factor_dir)
                self._save_h5file(new_panel,
                                  self.abs_factor_path(factor_dir, available_name), available_name)
                self.rename_factor(available_name, column, factor_dir)
            elif if_exists == 'replace':
                self._save_h5file(factor_data[[column]].dropna().to_panel(),
                                  factor_path, column)
            else:
                raise KeyError("please make sure if_exists is valide")

    def save_as_dummy(self, factor_data, factor_dir, indu_name=None, if_exists='append'):
        """往数据库中存入哑变量数据
        factor_data: pd.Series or pd.DataFrame
        当factor_data是Series时，首先调用pd.get_dummy()转成行业哑变量
        """
        if isinstance(factor_data, pd.Series):
            assert factor_data.name is not None or indu_name is not None
            factor_data.dropna(inplace=True)
            indu_name = indu_name if indu_name is not None else factor_data.name
            factor_data = pd.get_dummies(factor_data)
        else:
            assert isinstance(factor_data, pd.DataFrame) and indu_name is not None
        factor_data = factor_data.drop('T00018', axis=0, level='IDs').fillna(0)
        factor_data = factor_data.loc[(factor_data != 0).any(axis=1)]
        file_pth = self.abs_factor_path(factor_dir, indu_name)
        if self.check_factor_exists(indu_name, factor_dir):
            mapping = self._read_pklfile(file_pth.replace('.h5', '_mapping.pkl'))
            factor_data = factor_data.reindex(columns=mapping)
            new_saver = pd.DataFrame(np.argmax(factor_data.values, axis=1), columns=[indu_name],
                                     index=factor_data.index)
        else:
            new_saver = pd.DataFrame(np.argmax(factor_data.values, axis=1), columns=[indu_name],
                                     index=factor_data.index)
            mapping = factor_data.columns.values.tolist()
        self.save_factor(new_saver, factor_dir, if_exists=if_exists)
        self._save_pklfile(mapping, factor_dir, indu_name+'_mapping', protocol=2)

    def load_as_dummy(self, factor_name, factor_dir, dates=None, ids=None, idx=None):
        """读取行业哑变量"""
        mapping_pth = self.data_path + factor_dir + factor_name + '_mapping.pkl'
        mapping = self._read_pklfile(mapping_pth)
        data = self.load_factor(factor_name, factor_dir, dates=dates, ids=ids, idx=idx).dropna()
        dummy = np.zeros((len(data), len(mapping)))
        dummy[np.arange(len(data)), data[factor_name].values.astype('int')] = 1
        return pd.DataFrame(dummy, index=data.index, columns=mapping, dtype='int8')
    
    def to_feather(self, factor_name, factor_dir):
        """将某一个因子转换成feather格式，便于跨平台使用"""
        target_dir = self.feather_data_path + factor_dir
        ensure_dir_exists(target_dir)
        if factor_name is None:
            factor_name = self.list_factors(factor_dir)
        elif isinstance(factor_name, str):
            factor_name = [factor_name]
        for f in factor_name:
            data = self.load_factor(f, factor_dir).reset_index()
            self._save_feather(data, target_dir+f+'.feather')

    def to_csv(self, factor_name, factor_dir):
        target_dir = self.csv_data_path + factor_dir
        ensure_dir_exists(target_dir)
        if factor_name is None:
            factor_name = self.list_factors(factor_dir)
        elif isinstance(factor_name, str):
            factor_name = [factor_name]
        for f in factor_name:
            data = self.load_factor(f, factor_dir).reset_index()
            data.to_csv(self.csv_data_path + factor_dir + f + '.csv')

    def combine_factor(self, left_name, left_dir, right_name, right_dir, drop_right=True):
        """把两个因子合并，并删除右边的因子"""
        right_data = self.load_factor(right_name, right_dir).rename(columns={right_name: left_name})
        self.save_factor(right_data, left_dir)
        if drop_right:
            self.delete_factor(right_name, right_dir)

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + os.path.join(factor_path, factor_name+'.h5')

    def abs_factor_attr_path(self, factor_path, factor_name):
        return self.data_path + factor_path + factor_name + '_attr.pkl'
    
    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while os.path.isfile(self.abs_factor_path(factor_path, factor_name+str(i))):
            i += 1
        return factor_name + str(i)
