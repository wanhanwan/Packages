"""h5db性能加强版
保留h5db的框架，提升读写性能
"""

import pandas as pd
import numpy as np
import os
import shutil
import h5py
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime, IntDate2Datetime, Datetime2IntDate
from ..utils.tool_funcs import tradecode_to_intcode, windcode_to_intcode, intcode_to_tradecode
from filemanager import zip_dir, unzip_file
from collections import namedtuple

Rule = namedtuple('Rule', ['rule_id', 'read_func', 'write_func'])
# 数值转换
# 0代表浮点数，1代表日期
Converters = {
    0: Rule(0, lambda x: x.astype('float32'), lambda x: x.astype('float32')),
    1: Rule(1, lambda x: x.astype('str'), lambda x: x.dt.strftime('%Y%m%d').astype('int32'))
}
ConvertersDict = {
    np.float32: Converters[0],
    np.dtype('<M8[ns]'): Converters[1]
}


class H5DB2(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path + '/../feather')
        self.snapshots_path = os.path.abspath(data_path + '/../snapshots')
        self.data_dict = None
        self._update_info()

    def _update_info(self):
        factor_list = []

        for root, subdirs, files in os.walk(self.data_path):
            relpath = "/%s/" % os.path.relpath(root, self.data_path).replace("\\", "/")
            for file in files:
                if file.endswith(".hdf5"):
                    factor_list.append([relpath, file[:-5]])
        self.data_dict = pd.DataFrame(
            factor_list, columns=['path', 'name'])

    def set_data_path(self, path):
        self.data_path = path
        self._update_info()

    # ---------------------------因子管理---------------------------------------
    # 查看因子是否存在
    def check_factor_exists(self, factor_name, factor_dir='/'):
        return factor_name in self.data_dict[self.data_dict['path'] == factor_dir]['name'].values

    # 删除因子
    def delete_factor(self, factor_name, factor_dir='/'):
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        try:
            os.remove(factor_path)
        except Exception as e:
            print(e)
            pass
        self._update_info()

    # 列出因子名称
    def list_factors(self, factor_dir):
        factors = self.data_dict[self.data_dict.path == factor_dir]['name']
        return factors.tolist()

    # 重命名因子
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        os.rename(factor_path, temp_factor_path)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        if not os.path.isdir(self.data_path + factor_dir):
            os.makedirs(self.data_path + factor_dir)

    # 因子的时间区间
    def get_date_range(self, factor_name, factor_path):
        panel = pd.read_hdf(self.abs_factor_path(factor_path, factor_name))
        min_date = Datetime2DateStr(panel.major_axis.min())
        max_date = Datetime2DateStr(panel.major_axis.max())
        return min_date, max_date

    # --------------------------数据管理-------------------------------------------

    def _read_whole(self, factor_path, factor_name):
        with h5py.File(factor_path, "r") as file:
            factor_data = np.asarray(file['data'][...])
            all_dates = np.sort(np.asarray(file['date'][...]))
            all_ids = np.sort(np.asarray(file['IDs'][...]))
            data_type = file['data'].attrs['dtype']
        return factor_data, all_dates, all_ids, data_type

    def load_factor(self, factor_name, factor_dir=None, dates=None, ids=None, idx=None,
                    df=True):
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
        if not df and idx is not None:
            raise ValueError("Parameters df must be True if idx is not None")
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        factor_data, all_dates, all_ids, data_type = self._read_whole(factor_path, factor_name)
        dates = all_dates if dates is None else self.save_convert_date(dates)
        ids = all_ids if ids is None else self.save_convert_ids(ids)
        dates.sort()
        ids.sort()
        temp = np.zeros((len(dates), len(ids))) * np.nan
        temp[np.in1d(dates, all_dates), np.in1d(ids, all_ids)] = factor_data[np.in1d(all_dates, dates), np.in1d(all_ids, ids)]

        if df:
            date_index = [IntDate2Datetime(x) for x in dates]
            ids_index = [intcode_to_tradecode(x) for x in ids]
            factor_data = pd.DataFrame(temp, index=date_index, columns=ids_index).stack().to_frame(factor_name)
            factor_data.index.names = ['date', 'IDs']
            if idx is not None:
                return factor_data.align(idx, join='right')[0].apply(Converters[data_type].read_func)
            return factor_data.apply(Converters[data_type].read_func)
        else:
            return temp, dates, ids, data_type

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
        all_dates = np.sort([Datetime2IntDate(x) for x in factor_data.index.get_level_values(0).unique()])
        all_ids = np.sort([tradecode_to_intcode(x) for x in factor_data.index.get_level_values(1).unique()])
        for column in factor_data.columns:
            factor_path = self.abs_factor_path(factor_dir, column)
            temp = factor_data[[column]].dropna().apply(ConvertersDict[factor_data[column].dtype].write_func).unstack().values
            if not self.check_factor_exists(column, factor_dir):
                with h5py.File(factor_path, "w") as file:
                    dset = file.create_dataset("data", temp.shape, dtype=temp.dtype, data=temp)
                    dset.attrs['dtype'] = ConvertersDict[temp.dtype].rule_id
                    file.create_dataset("date", (len(all_dates), ), dtype=all_dates.dtype, data=all_dates)
                    file.create_dataset("IDs", (len(all_ids), ), dtype=all_ids.dtype, data=np.asarray(all_ids))
            elif if_exists == 'append':
                old_frame = self.load_factor(column, factor_dir)
                new_frame = old_frame().append(factor_data[[column]].dropna())
                new_frame = new_frame[~new_frame.index.duplicated(keep='last')]
                new_dates = np.sort([Datetime2IntDate(x) for x in new_frame.index.get_level_values(0).unique()])
                new_ids = np.sort([tradecode_to_intcode(x) for x in new_frame.index.get_level_values(1).unique()])
                new_value = new_frame.apply(ConvertersDict[new_frame[column].dtype].write_func).unstack().values
                available_name = self.get_available_factor_name(column, factor_dir)
                with h5py.File(self.abs_factor_path(factor_dir, available_name), "w") as file:
                    dset = file.create_dataset("data", new_value.shape, dtype=new_value.dtype, data=new_value)
                    dset.attrs['dtype'] = ConvertersDict[new_value.dtype].rule_id
                    file.create_dataset("date", (len(new_dates), ), dtype=new_dates.dtype, data=new_dates)
                    file.create_dataset("IDs", (len(new_ids), ), dtype=new_ids.dtype, data=np.asarray(new_ids))
                self.delete_factor(column, factor_dir)
                self.rename_factor(available_name, column, factor_dir)
            elif if_exists == 'replace':
                self.delete_factor(column, factor_dir)
                with h5py.File(factor_path, "w") as file:
                    dset = file.create_dataset("data", temp.shape, dtype=temp.dtype, data=temp)
                    dset.attrs['dtype'] = ConvertersDict[temp.dtype].rule_id
                    file.create_dataset("date", (len(all_dates), ), dtype=all_dates.dtype, data=all_dates)
                    file.create_dataset("IDs", (len(all_ids), ), dtype=all_ids.dtype, data=np.asarray(all_ids))
            else:
                self._update_info()
                raise KeyError("please make sure if_exists is valide")
        self._update_info()

    def snapshot(self, dates, zipname=None, mail=False):
        """获取数据库快照并保存"""
        self._update_info()
        dates = list(dates)
        date_now = max(dates).strftime("%Y%m%d")
        if os.path.isdir(os.path.join(self.snapshots_path, date_now)):
            shutil.rmtree(os.path.join(self.snapshots_path, date_now), True)
        os.mkdir(os.path.join(self.snapshots_path, date_now))
        target_path = os.path.join(self.snapshots_path, date_now)
        for d in self.data_dict['path'].unique():
            os.makedirs(target_path + d)
        for idx, row in self.data_dict.iterrows():
            data = self.load_factor(row['name'], row['path']).reset_index()
            snapshot = data[data['date'].isin(dates)]
            if snapshot.empty:
                snapshot = data[data['date'] == data['date'].max()]
            snapshot.to_csv(target_path + row['path'] + row['name'] + '.csv', index=False)
        if zipname is not None:
            zip_dir(target_path, os.path.join(self.snapshots_path, '%s_%s.zip' % (date_now, zipname)))
        if mail:
            from mailing.mailmanager import mymail
            mymail.connect()
            mymail.login()
            content = "hello everyone, this is factor data on %s" % date_now
            attachment = os.path.join(self.snapshots_path, '%s_%s.zip' % (date_now, zipname))
            try:
                mymail.send_mail("base factor data on %s" % date_now, content, {attachment})
            except:
                mymail.connect()
                mymail.send_mail("base factor data on %s" % date_now, content, {attachment})
            mymail.quit()

    def read_snapshot(self, name):
        snapshotzip = self.snapshots_path + "/%s" % name
        unzip_file(snapshotzip, snapshotzip.replace('.zip', ''))
        snapshotdir = snapshotzip.replace('.zip', '')
        for dirpath, subdirs, filenames in os.walk(snapshotdir):
            factor_dir = '/%s/' % os.path.relpath(dirpath, snapshotdir).replace('\\', '/')
            for file in filenames:
                print(file)
                if file.endswith(".csv"):
                    try:
                        data = pd.read_csv(os.path.join(dirpath, file), converters={'IDs': str}, parse_dates=['date'])
                    except:
                        data = pd.read_csv(os.path.join(dirpath, file), converters={'IDs': str}, encoding="GBK",
                                           parse_dates=['date'])
                    data['IDs'] = data['IDs'].str.zfill(6)
                    data.set_index(['date', 'IDs'], inplace=True)
                    if data.columns.isin(['list_date', 'backdoordate']).any():
                        data = data.astype('str')
                    self.save_factor(data, factor_dir)

    # -------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + os.path.join(factor_path, factor_name + '.hdf5')

    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while os.path.isfile(self.abs_factor_path(factor_path, factor_name + str(i))):
            i += 1
        return factor_name + str(i)

    def save_convert_ids(self, ids):
        if isinstance(ids[0], str):
            return [tradecode_to_intcode(x) if len(x) > 6 else windcode_to_intcode(x) for x in ids]
        elif isinstance(ids[0], int):
            return ids
        else:
            raise TypeError

    def save_convert_date(self, dates):
        if isinstance(dates, pd.DatetimeIndex):
            return dates.strftime("%Y%m%d").astype('int32').tolist()
        elif isinstance(dates[0], str):
            return [int(x) for x in dates]
        else:
            raise TypeError