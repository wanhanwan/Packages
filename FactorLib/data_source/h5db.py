# coding: utf-8
"""基于HDF文件的数据库"""

import pandas as pd
import numpy as np
import os
import shutil
from multiprocessing import Lock
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime
from ..utils.tool_funcs import ensure_dir_exists
from ..utils.disk_persist_provider import DiskPersistProvider
from filemanager import zip_dir, unzip_file
from .helpers import handle_ids

lock = Lock()


class H5DB(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path+'/../feather')
        self.csv_data_path = os.path.abspath(data_path+'/../csv')
        self.snapshots_path = os.path.abspath(data_path+'/../snapshots')
        self.data_dict = None
        self._update_info()
    
    def _update_info(self):
        factor_list = []
        
        for root, subdirs, files in os.walk(self.data_path):
            relpath = "/%s/"%os.path.relpath(root, self.data_path).replace("\\", "/")
            for file in files:
                if file.endswith(".h5"):
                    factor_list.append([relpath, file[:-3]])
        self.data_dict = pd.DataFrame(
            factor_list, columns=['path', 'name'])
    
    def set_data_path(self, path):
        self.data_path = path
        self._update_info()
    
    #---------------------------因子管理---------------------------------------
    # 查看因子是否存在
    def check_factor_exists(self, factor_name, factor_dir='/'):
        return factor_name in self.data_dict[self.data_dict['path']==factor_dir]['name'].values

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
        factor_data = pd.read_hdf(factor_path, old_name).to_frame().rename(columns={old_name: new_name}).to_panel()
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)
        factor_data.to_hdf(temp_factor_path, new_name, complevel=9, complib='blosc')
        self.delete_factor(old_name, factor_dir)

    # 新建因子文件夹
    def create_factor_dir(self, factor_dir):
        if not os.path.isdir(self.data_path+factor_dir):
            os.makedirs(self.data_path+factor_dir)
    
    # 因子的时间区间
    def get_date_range(self, factor_name, factor_path):
        panel = pd.read_hdf(self.abs_factor_path(factor_path, factor_name))
        min_date = Datetime2DateStr(panel.major_axis.min())
        max_date = Datetime2DateStr(panel.major_axis.max())
        return min_date, max_date

    # --------------------------数据管理-------------------------------------------
    @handle_ids
    def load_factor(self, factor_name, factor_dir=None, dates=None, ids=None, idx=None):
        if idx is not None:
            dates = idx.index.get_level_values('date').unique()
            return self.load_factor(factor_name, factor_dir=factor_dir, dates=dates).reindex(idx.index, copy=False)
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        panel = pd.read_hdf(factor_path, factor_name)
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
    def load_h5file(self, file_name, path, group='data', ids=None, dates=None,
                    idx=None, **kwargs):
        """读取h5File"""
        attr_file_path = self.data_path + path + file_name + '_attr.pkl'
        file_path = self.abs_factor_path(path, file_name)
        try:
            attr = pd.read_pickle(attr_file_path)
            fill_value = attr['fill_value']
            multiplier = attr['multiplier']
        except FileNotFoundError as e:
            fill_value = kwargs.get('fill_value', 100)
            multiplier = kwargs.get('multiplier', 100)
        
        data = pd.read_hdf(file_path, group)
        if idx is not None:
            data = data.reindex(idx.index)
        else:
            if dates is not None:
                data = data.loc[pd.DatetimeIndex(dates).values]
            if ids is not None:
                data = data.loc[pd.IndexSlice[:, list(ids)], :]
        data /= multiplier
        data.replace(fill_value, np.nan, inplace=True)
        return data

    def save_h5file(self, data, path, name, group='data',
                    multiplier=100, fill_value=100, **kwargs):
        """保存多列的DataFrame"""
        file_path = self.abs_factor_path(path, name)
        attr_file_path = self.data_path + path + name + '_attr.pkl'
        if os.path.isfile(attr_file_path):
            attr = pd.read_pickle(attr_file_path)
            multiplier = attr['multiplier']
            fill_value = attr['fill_value']
        data = (data.fillna(fill_value) * multiplier).astype('int')
        with pd.HDFStore(file_path, complib='blosc', complevel=9, mode='a') as store:
            try:
                df = store.select(group)
                store.remove(group)
            except KeyError as e:
                df = pd.DataFrame()
        new_df = df.append(data)
        new_df = new_df[~new_df.index.duplicated(keep='last')]
        ensure_dir_exists(self.data_path+path)
        data.to_hdf(file_path, group, complib='blosc', complevel=9, mode='w')
        persist_provider = DiskPersistProvider(self.data_path+path)
        persist_provider.dump({'multiplier': multiplier, 'fill_value':fill_value},
                              name=name+'_attr', protocol=2)

    def load_latest_period(self, factor_name, factor_dir=None, ids=None, idx=None):
        max_date = self.get_date_range(factor_name, factor_dir)[1]
        return self.load_factor(factor_name, factor_dir, dates=[max_date], ids=ids, idx=idx).reset_index(level=0, drop=True)

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
        try:
            lock.acquire()
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
                    factor_data[[column]].dropna().to_panel().to_hdf(factor_path, column, complevel=9, complib='blosc')
                elif if_exists == 'append':
                    old_panel = pd.read_hdf(factor_path, column)
                    new_frame = old_panel.to_frame().append(factor_data[[column]].dropna())
                    new_panel = new_frame[~new_frame.index.duplicated(keep='last')].to_panel()
                    available_name = self.get_available_factor_name(column, factor_dir)
                    new_panel.to_hdf(
                        self.abs_factor_path(factor_dir, available_name), available_name, complevel=9, complib='blosc')
                    self.delete_factor(column, factor_dir)
                    self.rename_factor(available_name, column, factor_dir)
                elif if_exists == 'replace':
                    self.delete_factor(column, factor_dir)
                    factor_data[[column]].dropna().to_panel().to_hdf(factor_path, column, complevel=9, complib='blosc')
                else:
                    self._update_info()
                    raise KeyError("please make sure if_exists is valide")
            self._update_info()
            lock.release()
        except Exception as e:
            lock.release()

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
        factor_data.drop('T00018', axis=0, level='IDs').fillna(0)
        factor_data = factor_data.loc[(factor_data!=0).any(axis=1)]
        file_pth = self.abs_factor_path(factor_dir, indu_name)
        if self.check_factor_exists(indu_name, factor_dir):
            try:
                lock.acquire()
                mapping = pd.read_pickle(file_pth.replace('.h5', '_mapping.pkl'))
            finally:
                lock.release()
            factor_data = factor_data.reindex(columns=mapping)
            new_saver = pd.DataFrame(np.argmax(factor_data.values, axis=1), columns=[indu_name],
                                     index=factor_data.index)
        else:
            new_saver = pd.DataFrame(np.argmax(factor_data.values, axis=1), columns=[indu_name],
                                     index=factor_data.index)
            mapping = factor_data.columns.values.tolist()
        self.save_factor(new_saver, factor_dir, if_exists=if_exists)
        diskprovider = DiskPersistProvider(self.data_path+factor_dir)
        try:
            lock.acquire()
            diskprovider.dump(mapping, indu_name+'_mapping', protocol=2)
        finally:
            lock.release()

    def load_as_dummy(self, factor_name, factor_dir, dates=None, ids=None, idx=None):
        """读取行业哑变量"""
        file_pth = self.abs_factor_path(factor_dir, factor_name)
        mapping_pth = self.data_path + factor_dir + factor_name + '_mapping.pkl'
        try:
            lock.acquire()
            # mapping = pd.read_hdf(file_pth, "mapping", mode='r')
            mapping = pd.read_pickle(mapping_pth)
        finally:
            lock.release()
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
            data.to_feather(self.feather_data_path+factor_dir+f+'.feather')

    def to_csv(self,factor_name, factor_dir):
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

    def import_factor(self, factor_name, factor_dir, src_file):
        fanme = os.path.split(src_file)[1].replace('.h5', '')
        data = pd.read_hdf(src_file, fanme).to_frame().rename(columns={fanme: factor_name})
        self.save_factor(data, factor_dir)

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
            file_path = self.abs_factor_path(row['path'], row['name'])
            data = pd.read_hdf(file_path, row['name']).to_frame().reset_index()
            snapshot = data[data['date'].isin(dates)]
            if snapshot.empty:
                snapshot = data[data['date']==data['date'].max()]
            snapshot.to_csv(target_path+row['path']+row['name']+'.csv', index=False)
        if zipname is not None:
            zip_dir(target_path, os.path.join(self.snapshots_path, '%s_%s.zip'%(date_now, zipname)))
        if mail:
            from mailing.mailmanager import mymail
            mymail.connect()
            mymail.login()
            content = "hello everyone, this is factor data on %s"%date_now
            attachment = os.path.join(self.snapshots_path, '%s_%s.zip'%(date_now, zipname))
            try:
                mymail.send_mail("base factor data on %s"%date_now, content, {attachment})
            except:
                mymail.connect()
                mymail.send_mail("base factor data on %s" % date_now, content, {attachment})
            mymail.quit()

    def read_snapshot(self, name):
        snapshotzip = self.snapshots_path+"/%s"%name
        unzip_file(snapshotzip, snapshotzip.replace('.zip',''))
        snapshotdir = snapshotzip.replace('.zip','')
        for dirpath, subdirs, filenames in os.walk(snapshotdir):
            factor_dir = '/%s/'%os.path.relpath(dirpath, snapshotdir).replace('\\','/')
            for file in filenames:
                print(file)
                if file.endswith(".csv"):
                    try:
                        data = pd.read_csv(os.path.join(dirpath, file), converters={'IDs':str}, parse_dates=['date'])
                    except:
                        data = pd.read_csv(os.path.join(dirpath, file), converters={'IDs':str}, encoding="GBK", parse_dates=['date'])
                    data['IDs'] = data['IDs'].str.zfill(6)
                    data.set_index(['date', 'IDs'], inplace=True)
                    if data.columns.isin(['list_date', 'backdoordate']).any():
                        data = data.astype('str')
                    self.save_factor(data, factor_dir)


    #-------------------------工具函数-------------------------------------------
    def abs_factor_path(self, factor_path, factor_name):
        return self.data_path + os.path.join(factor_path, factor_name+'.h5')
    
    def get_available_factor_name(self, factor_name, factor_path):
        i = 2
        while os.path.isfile(self.abs_factor_path(factor_path, factor_name+str(i))):
            i += 1
        return factor_name + str(i)