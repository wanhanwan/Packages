# -*- coding: utf-8 -*-
#
# pkldb.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2018-8-3 16:58:30
import pandas as pd
import pickle
import os
from os import path
from multiprocess import Lock
from .helpers import handle_ids
from ..utils.tool_funcs import ensure_dir_exists

lock = Lock()


class PickleDB(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_dict = None
        self._update_info()

    def _update_info(self):
        pass

    def set_data_path(self, path):
        self.data_path = path
        self._update_info()

    def load_file(self, path):
        try:
            lock.acquire()
            data = pd.read_pickle(path)
            lock.release()
            return data
        except Exception as e:
            lock.release()
            raise e
    
    def dump_file(self, obj, path):
        try:
            lock.acquire()
            if isinstance(obj, (pd.Panel, pd.DataFrame,
                                pd.Series)):
                obj.to_pickle(path, protocol=-1)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(obj, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            lock.release()
        except Exception as e:
            lock.release()
            raise e

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
        data = self.load_file(file_path)
        return list(data.columns)

    # 列出文件的日期
    def list_file_dates(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        data = self.load_file(file_path)
        return data.index.unique(level='date')

    # 列出文件的IDs
    def list_file_ids(self, file_name, file_dir):
        file_path = self.abs_factor_path(file_dir, file_name)
        data = self.load_file(file_path)
        return data.index.unique(level='IDs')

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

    @handle_ids
    def load_factor(self, file_name, file_dir=None, factor_names=None, dates=None, ids=None,
                    idx=None):
        if not self.check_file_exists(file_name, file_dir):
            raise FileNotFoundError("%s in %s not exist"%(file_name, file_dir))
        file_path = self.abs_factor_path(file_dir, file_name)
        data = self.load_file(file_path)

        if factor_names is not None:
            data = data[factor_names]

        if (idx is None) and (ids is None) and (dates is None):
            return data

        if idx is None:
            if not ((dates is None) or (ids is None)):
                idx1 = pd.MultiIndex.from_product([dates, ids], names=['date', 'IDs'])
                data = data.reindex(idx1)
            else:
                if dates is not None:
                    data = data.loc[dates]
                if ids is not None:
                    data = data.loc[pd.IndexSlice[:, ids]]
        else:
            data = data.reindex(idx.index)
        return data
    
    def save_factor(self, data, file_name, file_dir, if_exists='append'):
        file_path = self.abs_factor_path(file_dir, file_name)
        ensure_dir_exists(os.path.dirname(file_path))
        if (not self.check_file_exists(file_name, file_dir)) or (if_exists == 'replace'):
            self.dump_file(data, file_path)
        else:
            old = self.load_file(file_path)
            new = data.append(old[~old.index.isin(data.index)]).sort_index()
            self.dump_file(new, file_path)

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        factor_path = factor_path.strip('/\\')
        return path.join(self.data_path, factor_path, factor_name + '.pkl')

    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while path.isfile(self.abs_factor_path(factor_path, factor_name + str(i))):
            i += 1
        return factor_name + str(i)
