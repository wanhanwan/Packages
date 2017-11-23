# coding: utf-8
"""基于HDF文件的数据库"""

import pandas as pd
import os
import shutil
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime
from filemanager import zip_dir, unzip_file


class H5DB(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path+'/../feather')
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

    #--------------------------数据管理-------------------------------------------

    def load_factor(self, factor_name, factor_dir=None, dates=None, ids=None, idx=None):
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        panel = pd.read_hdf(factor_path, factor_name)
        if idx is not None:
            return panel.to_frame().rename_axis(['date','IDs']).loc[idx,:]
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
    
    def to_feather(self, factor_name, factor_dir):
        """将某一个因子转换成feather格式，便于跨平台使用"""
        target_dir = self.feather_data_path + factor_dir
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        data = self.load_factor(factor_name, factor_dir).reset_index()
        data.to_feather(self.feather_data_path+factor_dir+factor_name+'.feather')

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