#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# edb.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2020/6/24 下午1:47:37
"""
宏观数据库接口

Notes：
------
1、 数据源
主要来自Wind宏观数据库和通联数据宏观数据库。

2、数据字典维护
所有的数据都放在"resource/edb_tableinfo.xlsx"中
进行维护。

3、数据存储形式
父级目录都以文件夹的形式存在，文件夹的名字以数据字典中对应
的中文名称命名。

非父级目录中所有的指标以CSV格式的文件存储。文件以数据源命名，
例如wind/datayes。

数据的日期按照日历日展示，月度数据和季度数据都以当月/季末最后
一天显示。空数据暂时不进行任何填充。

"""
import os
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from opendatatools import datayes
from FactorLib.utils.tool_funcs import ensure_dir_exists
from FactorLib.data_source.wind_addin import WindAddIn

def _load_data_table():
    pth = Path(__file__).parents[1] / 'resource' / 'edb_tableinfo.xlsx'
    table_info = pd.read_excel(pth, header=0)
    return table_info


class EDBDataDict:
    data_dict = _load_data_table()

    @staticmethod
    def _find_an_element(filter_key, filter_value, search_key):
        return EDBDataDict.data_dict.loc[
            EDBDataDict.data_dict[filter_key] == filter_value, search_key
            ].iat[0]

    @staticmethod
    def find_class_pid(key, value):
        return EDBDataDict._find_an_element(key, value, 'class_pid')
    
    @staticmethod
    def find_name_cn(key, value):
        return EDBDataDict._find_an_element(key, value, 'name_cn')
    
    @staticmethod
    def find_class_id(key, value):
        return EDBDataDict._find_an_element(key, value, 'class_id')
    
    @staticmethod
    def find_data_id(key, value, source):
        return EDBDataDict._find_an_element(key, value, f'{source}_id')


class DataYes(object):
    username = '18516759861'
    password = 'Ws1991822929'

    def __init__(self):
        datayes.login(DataYes.username, DataYes.password)

    def download_data(self, datayes_id, **kwargs):
        """
        下载数据

        Parameters:
        ------------
        datayes_id: 指标ID，对应数据字典中indicld字段

        Return:
        -------
        data: DataFrame
            DataFrame(periodDate, dataValue, updateTime, dataYear)
        info: dict
            数据描述
        """
        data, info, flag = datayes.get_series(datayes_id)
        if data is None:
            raise RuntimeError(
                        "Failed to download data from datayes:%s"
                        %flag
                    )
        else:
            data = data.set_index('periodDate')[['dataValue']].astype('float64')
            data.index = pd.DatetimeIndex(data.index)
            return data, info


class Wind(object):
    api = WindAddIn()

    def download_data(self, wind_id, start_date=None, end_date=None):
        """
        下载数据
        """
        start_date = '20000101' if start_date is None else start_date
        end_date = datetime.today().strftime('%Y%m%d') if end_date is None else end_date
        data = Wind.api.edb(wind_id, start_date, end_date)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data


class CSVManager(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _absolute_path(self, rel_path: str):
        return os.path.join(self.root_dir, rel_path.strip('/').replace(':', '-'))
    
    def check_file_existence(self, file_path):
        return os.path.isfile(file_path)
    
    def load_meta(self, file_dir):
        meta_file_path = self._absolute_path(file_dir) + '/meta.json'
        if not self.check_file_existence(meta_file_path):
            return {}
        else:
            with open(meta_file_path, 'r', encoding='GBK') as f:
                meta = json.load(f)
            return meta
    
    def save_meta(self, meta, file_dir):
        meta_file_path = self._absolute_path(file_dir) + '/meta.json'
        with open(meta_file_path, 'w', encoding='GBK') as f:
            json.dump(meta, f, ensure_ascii=False)

    def save_data(self,
                  data: pd.DataFrame,
                  save_dir,
                  save_file_name,
                  save_meta_data=None):
        """
        保存一个DataFrame

        Parameters:
        -----------
        data: DataFrame
            索引是日期，每列是一个数据
        save_dir: str
            相对路径
        save_file_name: str
            文件名称
        save_meta_data: dict
            描述数据
        """
        ensure_dir_exists(self._absolute_path(save_dir))
        file_path = self._absolute_path(save_dir) + '/%s.csv' % save_file_name
        if self.check_file_existence(file_path):
            with open(file_path, 'r') as f:
                raw = pd.read_csv(f, index_col=0, parse_dates=True, header=0, encoding='GBK', dtype='float64')
                new = raw.join(data[[x for x in data.columns if x not in raw.columns]], how='outer')
                new.update(data)
            with open(file_path, 'w') as f:
                new.dropna(how='all').to_csv(f, encoding='GBK')
        else:
            with open(file_path, 'w') as f:
                data.dropna(how='all').to_csv(f, encoding='GBK')
        
        if save_meta_data is not None:
            meta_file_path = self._absolute_path(save_dir) + '/meta.json'
            if self.check_file_existence(meta_file_path):
                meta = self.load_meta(save_dir)
                meta.update(save_meta_data)
                self.save_meta(meta, save_dir)
            else:
                self.save_meta(save_meta_data, save_dir)
    
    def load_data(self, file_dir, file_name):
        file_path = self._absolute_path(file_dir) + '/%s.csv' % file_name
        data = pd.read_csv(file_path, header=0,
                index_col=0, parse_dates=True,
                encoding='GBK', dtype='float64'
            )
        return data


class EDB(object):
    
    # 类成员
    data_dict = EDBDataDict
    data_sources = {
        'datayes': DataYes(),
        'wind': Wind()
    }

    def __init__(self, root_dir):
        self.data_manager = CSVManager(root_dir)

    def _find_classifications(self, data_id, source='datayes'):
        """
        查找一个指标的分类路径。
        
        如：
        
        “GDP:现价:当季值” 归属于 国民经济核算-国内生产总值-国内生产总值(季)
        
        """
        top_class = {
            402273: '中国宏观',
            771263: '行业经济',
            1138921: '国际宏观',
            632815: '特色数据'
        }
        search_key = source + '_id'

        pid = self.data_dict.find_class_pid(search_key, data_id)
        class_loops = []
        while pid not in top_class:
            class_loops.append(self.data_dict.find_name_cn('class_id', pid))
            pid = self.data_dict.find_class_pid('class_id', pid)
        return [top_class[pid]] + class_loops[::-1]
    
    def find_path(self, data_name):
        """
        查找一个指标的存放路径

        Parameters:
        ----------
        data_name: str
            指标中文名称
        """
        data_id = self.data_dict.find_data_id('name_cn', data_name, 'datayes')
        classifications = self._find_classifications(data_id)
        return '/'.join(classifications)

    def read_data(self, data_names, source='datayes'):
        """
        从本地读取宏观数据
        """
        data_bag = {}
        for name in data_names:
            path = self.find_path(name)
            if path in data_bag:
                data_bag[path].append(name)
            else:
                data_bag[path] = [name]
        
        df = []
        for path, names in data_bag.items():
            df.append(
                self.data_manager.load_data(path, source)[names]
            )
        df = pd.concat(df, axis=1)
        return df

    def download_data(self, data_names, source='datayes'):
        """
        下载数据到本地
        """
        ds = EDB.data_sources[source]
        for name in data_names:
            data_id = EDB.data_dict.find_data_id('name_cn', name, source)
            save_dir = self.find_path(name)
            data, meta = ds.download_data(data_id)
            data.columns = [name]
            self.data_manager.save_data(data, save_dir, source, {name: meta})
            print("%s 下载成功， 最新日期%s" % (name, data.index.max().strftime("%Y/%m/%d")))
            time.sleep(1.5)


edb = EDB("D:/data/factors/edb")
        

if __name__ == '__main__':
    data_names = ['制造业PMI']
    edb = EDB("D:/data/factors/edb")
    edb.download_data(data_names)
