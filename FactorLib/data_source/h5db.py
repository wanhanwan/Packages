# coding: utf-8
"""基于HDF文件的数据库"""

import pandas as pd
import numpy as np
import os
import warnings
from multiprocessing import Lock
from ..utils.datetime_func import Datetime2DateStr, DateStr2Datetime
from ..utils.tool_funcs import ensure_dir_exists
from ..utils.disk_persist_provider import DiskPersistProvider
from .financial_data_loader import FinancialDataLoader
from .helpers import handle_ids, FIFODict


lock = Lock()
warnings.simplefilter('ignore', category=FutureWarning)


def append_along_index(df1, df2):
    df1, df2 = df1.align(df2, axis='columns')
    new = pd.DataFrame(np.vstack((df1.values, df2.values)),
                       columns=df1.columns,
                       index=df1.index.append(df2.index))
    new.sort_index(inplace=True)
    return new


class H5DB(object):
    def __init__(self, data_path, max_cached_files=30):
        self.data_path = data_path
        self.feather_data_path = os.path.abspath(data_path+'/../feather')
        self.csv_data_path = os.path.abspath(data_path+'/../csv')
        self.data_dict = None
        self.cached_data = FIFODict(max_cached_files)
        self.max_cached_files = max_cached_files
        self.fin_data_loader = FinancialDataLoader()
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
            if self.max_cached_files > 0:
                self.cached_data[file_path] = data
            lock.release()
            return data
        except Exception as e:
            lock.release()
            raise e

    def _save_h5file(self, data, file_path, key,
                     complib='blosc', complevel=9,
                     mode='w', **kwargs):
        try:
            lock.acquire()
            data.to_hdf(file_path, key=key, complib=complib,
                        mode=mode, complevel=complevel, **kwargs)
            if file_path in self.cached_data:
                self.cached_data.update({file_path: data})
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
            if self.max_cached_files > 0:
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
        dir_path = self.data_path + factor_dir
        factors = [x[:-3] for x in os.listdir(dir_path) if x.endswith('.h5')]
        return factors
    
    # 重命名因子
    def rename_factor(self, old_name, new_name, factor_dir):
        factor_path = self.abs_factor_path(factor_dir, old_name)
        temp_factor_path = self.abs_factor_path(factor_dir, new_name)

        factor_data = self._read_h5file(factor_path, old_name). \
            to_frame().rename(columns={old_name: new_name}).to_panel()
        self._save_h5file(factor_data, temp_factor_path, new_name)
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
            try:
                panel = self._read_h5file(
                    self.abs_factor_path(factor_path, factor_name), key='data')
            except KeyError:
                panel = self._read_h5file(
                    self.abs_factor_path(factor_path, factor_name), key=factor_name)
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

    def clear_cache(self):
        self.cached_data = FIFODict(self.max_cached_files)

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
    def load_factor2(self, factor_name, factor_dir=None, dates=None, ids=None, idx=None,
                     stack=False, check_A=False):
        """加载另外一种类型的因子
        因子的格式是一个二维DataFrame，行索引是DatetimeIndex,列索引是股票代码。

        check_A: 过滤掉非A股股票
        """
        if idx is not None:
            dates = idx.index.get_level_values('date').unique().tolist()
            ids = idx.index.get_level_values('IDs').unique().tolist()
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        if (dates is not None) and (ids is not None):
            where_term = "(index in dates) & (columns in ids)"
        elif dates is not None:
            where_term = "index in dates"
        elif ids is not None:
            where_term = 'columns in ids'
        else:
            where_term = None
        with pd.HDFStore(factor_path, mode='r') as f:
            try:
                data = pd.read_hdf(f, key='data', where=where_term)
            except NotImplementedError:
                data = pd.read_hdf(f, key='data').reindex(index=dates, columns=ids)
        data.name = factor_name
        if check_A:
            data = data.filter(regex='^[6,0,3]', axis=1)
        if stack:
            data = data.stack().to_frame(factor_name)
            data.index.names = ['date', 'IDs']
            if idx is not None:
                data = data.reindex(idx.index)
        return data

    def show_symbol_name(self, factor_data=None, factor_name=None,
                         factor_dir=None, dates=None, data_source=None):
        """返回带有股票简称的因子数据
        Note:
            factor_data应为AST或者SAST数据
        """
        if data_source is None:
            data_source = 'D:/data/factors'
        import pandas as pd
        names = pd.read_csv(os.path.join(data_source,'base','ashare_name.csv'),
                            header=0,
                            converters={'IDs': lambda x: str(x).zfill(6)},
                            encoding='GBK')
        names.set_index('IDs', inplace=True)
        if factor_data is None:
            factor_data = self.load_factor2(factor_name, factor_dir, dates=dates)
            factor_data = factor_data.stack().to_frame(factor_data.name)
        if isinstance(factor_data.index, pd.MultiIndex):
            factor_data = factor_data.reset_index().join(names, on='IDs', how='left')
        else:
            factor_data = factor_data.stack().reset_index().join(names, on='IDs', how='left')
        return factor_data

    def read_h5file(self, file_name, path, group='data', check_A=None):
        file_path = self.abs_factor_path(path, file_name)
        data = self._read_h5file(file_path, key=group)
        if check_A is not None:
            data = data[data[check_A].str.match('^[0,3,6]')]
        return data

    def save_h5file(self, data, name, path, group='data',
                    use_index=True, if_exists='append',
                    sort_by_fields=None, sort_index=False):
        """直接把DataFrame保存成h5文件

        Parameters
        ----------
        use_index: bool
            当文件已存在，去重处理时按照索引去重。
        if_exists: str
            文件已存在时的处理方式：'append', 'replace' or 'update'.
            'append': 直接添加，不做去重处理
            'update': 添加后做去重处理，当'use_index'为TRUE时，按照
                      Index去重。
            'replace': 重写文件
        sort_by_fields: None or list
            写入之前，DataFrame先按照字段排序
        sort_index: bool, 默认为False
            写入之前，是否按照索引排序
        """
        file_path = self.abs_factor_path(path, name)
        if self.check_factor_exists(name, path):
            df = self.read_h5file(name, path, group=group)
            if if_exists == 'append':
                data = df.append(data, ignore_index=True)
            elif if_exists == 'replace':
                pass
            elif use_index and if_exists=='update':
                data = data[~data.index.duplicated(keep='last')]
            else:
                data = df.append(data)
                data.drop_duplicates(inplace=True)
        if sort_by_fields is not None:
            data.sort_values(sort_by_fields, inplace=True)
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

    def load_factors(self, factor_names_dict, dates=None, ids=None):
        _l = []
        for factor_path, factor_names in factor_names_dict.items():
            for factor_name in factor_names:
                df = self.load_factor(factor_name, factor_dir=factor_path, dates=dates, ids=ids)
                _l.append(df)
        return pd.concat(_l, axis=1)
    
    def load_factors2(self, factor_names_dict, dates=None, ids=None, idx=None,
                     merge=True, stack=True):
        assert not (merge is True and stack is False)
        _l = []
        for factor_path, factor_names in factor_names_dict.items():
            for factor_name in factor_names:
                df = self.load_factor2(factor_name, factor_dir=factor_path, dates=dates, ids=ids,
                                       idx=idx, stack=stack)
                _l.append(df)
        if merge:
            return pd.concat(_l, axis=1)
        return tuple(l)
    
    def load_factors3(self, factor_names_dict, dates=None, ids=None,
                      idx=None):
        if (dates is None or ids is None) and (idx is None):
            raise ValueError("idx must not be None, or both date and ids must not be None!")
        l = []
        factor_name_list = []
        for factor_path, factor_names in factor_names_dict.items():
            for factor_name in factor_names:
                factor_name_list.append(factor_name)
                df = self.load_factor2(factor_name, factor_dir=factor_path, dates=dates, ids=ids,
                                       idx=idx, stack=False)
                l.append(df.to_numpy())
        K = len(factor_name_list)
        T, N = l[0].shape
        threeD = np.concatenate(l, axis=0).reshape((K, T*N)).T
        df = pd.DataFrame(threeD,
                          index=pd.MultiIndex.from_product([df.index,df.columns], names=['date', 'IDs']),
                          columns=factor_name_list)
        return df
    
    def save_factor(self, factor_data, factor_dir, if_exists='append'):
        """往数据库中写数据
        数据格式：DataFrame(index=[date,IDs],columns=data)

        Parameters:
        -----------
        factor_data
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

    def save_factor2(self, factor_data, factor_dir, if_exists='append'):
        """往数据库中写数据
        数据格式：DataFrame(index=date, columns=IDs)
        """
        if isinstance(factor_data, pd.Series):
            if isinstance(factor_data.index, pd.MultiIndex):
                factor_name = factor_data.name
                factor_data = factor_data.unstack()
            else:
                raise ValueError("Format of factor_data is invalid.")
        elif isinstance(factor_data, pd.DataFrame):
            if factor_data.shape[1] > 1 and factor_data.index.nlevels > 1:
                raise ValueError("Column of factor_data must be one.")
            elif factor_data.index.nlevels > 1:
                factor_name = factor_data.columns[0]
                factor_data = factor_data[factor_name].unstack()
            else:
                factor_name = factor_data.name
        else:
            raise NotImplementedError
        self.create_factor_dir(factor_dir)
        factor_path = self.abs_factor_path(factor_dir, factor_name)
        if not self.check_factor_exists(factor_name, factor_dir):
            self._save_h5file(factor_data, factor_path, 'data', complevel=0,
                              format='table')
        elif if_exists == 'append':
            raw = pd.read_hdf(factor_path, key='data')
            new = factor_data[~factor_data.index.isin(raw.index)]
            d = append_along_index(raw, new)
            self._save_h5file(d, factor_path, 'data', complevel=0,
                              format='table')
        elif if_exists == 'update':
            raw = pd.read_hdf(factor_path, key='data')
            raw, factor_data = raw.align(factor_data, axis='columns')
            raw.update(factor_data)
            d = append_along_index(raw, factor_data[~factor_data.index.isin(raw.index)])
            self._save_h5file(d, factor_path, 'data', complevel=0,
                              format='table')
        elif if_exists == 'replace':
            self._save_h5file(factor_data, factor_path, 'data', complevel=0,
                              format='table')
        else:
            pass

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

    def save_as_dummy2(self, factor_data, factor_dir, indu_name=None, if_exists='append'):
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
        self.save_factor2(new_saver, factor_dir, if_exists=if_exists)
        self._save_pklfile(mapping, factor_dir, indu_name+'_mapping', protocol=2)

    def load_as_dummy(self, factor_name, factor_dir, dates=None, ids=None, idx=None):
        """读取行业哑变量"""
        mapping_pth = self.data_path + factor_dir + factor_name + '_mapping.pkl'
        mapping = self._read_pklfile(mapping_pth)
        data = self.load_factor(factor_name, factor_dir, dates=dates, ids=ids, idx=idx).dropna()
        dummy = np.zeros((len(data), len(mapping)))
        dummy[np.arange(len(data)), data[factor_name].values.astype('int')] = 1
        return pd.DataFrame(dummy, index=data.index, columns=mapping, dtype='int8')

    def load_as_dummy2(self, factor_name, factor_dir, dates=None, ids=None, idx=None):
        """读取行业哑变量"""
        mapping_pth = self.data_path + factor_dir + factor_name + '_mapping.pkl'
        mapping = self._read_pklfile(mapping_pth)
        data = self.load_factor2(factor_name, factor_dir, dates=dates, ids=ids, idx=idx).stack()
        dummy = np.zeros((len(data), len(mapping)))
        dummy[np.arange(len(data)), data.values.astype('int')] = 1
        return pd.DataFrame(dummy, index=data.index, columns=mapping, dtype='int8')
    
    def to_csv(self, factor_name, factor_dir, dates=None):
        target_dir = self.csv_data_path + factor_dir
        ensure_dir_exists(target_dir)
        if factor_name is None:
            factor_name = self.list_factors(factor_dir)
        elif isinstance(factor_name, str):
            factor_name = [factor_name]
        for f in factor_name:
            data = self.load_factor(f, factor_dir, dates=dates).reset_index()
            data.to_csv(self.csv_data_path + factor_dir + f + '.csv', index=False)

    def to_csv2(self, factor_name, factor_dir, dates=None):
        target_dir = self.csv_data_path + factor_dir
        ensure_dir_exists(target_dir)
        if factor_name is None:
            factor_name = self.list_factors(factor_dir)
        elif isinstance(factor_name, str):
            factor_name = [factor_name]
        for f in factor_name:
            data = self.load_factor2(
                f, factor_dir, dates=dates).stack().to_frame(f).rename_axis(['date', 'IDs']).reset_index()
            data = data[data['IDs'] != 'T00018']
            data.to_csv(self.csv_data_path + factor_dir + f + '.csv', index=False)

    def walk(self, factor_dir):
        """遍历某个文件夹和其子文件夹，依次返回每个因子名称和因子路径"""
        abs_factor_dir = self.data_path + factor_dir
        for dirname, subdirlist, filelist in os.walk(abs_factor_dir):
            sub_factor_dir = '/'+dirname.replace(self.data_path, '').strip('/')+'/'
            for file in filelist:
                if file.endswith('.h5'):
                    yield sub_factor_dir, file[:-3]

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

    def factor_corr(self, factor_dict, dates=None, ids=None, idx=None,
                    method='spearman'):
        """计算因子的两两相关系数"""
        from QuantLib.utils import CalFactorCorr
        data = self.load_factors2(factor_dict, dates=None, ids=None, merge=False,
                                  idx=idx, stack=True)
        corr = CalFactorCorr(*data, dates=dates, ids=ids, idx=idx, method=method)
        return corr
