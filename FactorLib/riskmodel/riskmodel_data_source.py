"""对H5DB进行再一次封装
   1. 增加了进程锁,便于支持多进程运算
   2. 支持提前设置因子列表，设置完成之后可以直接调用prepare_data函数提取数据
"""
from multiprocessing import Lock
from ..data_source.trade_calendar import trade_calendar,as_timestamp
from ..data_source.base_data_source_h5 import H5DB
import pandas as pd
import numpy as np
from ..utils.datetime_func import DateRange2Dates
from ..utils.disk_persist_provider import DiskPersistProvider
from ..utils.tool_funcs import ensure_dir_exists
from os import path
from warnings import warn
tc = trade_calendar()
default_riskds_root = 'D:/data/risk_model'


class RiskModelDataSourceOnH5(object):
    def __init__(self, h5_db, lock=None):
        self.h5_db = h5_db
        self.name = None
        self.set_multiprocess_lock(lock)
        self.all_dates = None
        self.all_ids = None
        self.idx = None
        self.estu_name = None
        self.factor_dict = {}
        self.factor_names = []
        self.cache_data = {}            # 缓存数据
        self.max_cache_num = 0          # 最大缓存数量
        self.cached_factor_num = 0      # 已经缓存的因子数量
        self.factor_read_num = None     # 因子的读取次数

    def set_name(self, name):
        self.name = name

    def set_dimension(self, idx):
        """设置日期和股票"""
        self.idx = idx
        self.all_ids = idx.index.get_level_values(1).unique().tolist()
        self.all_dates = idx.index.get_level_values(0).unique().tolist()

    def set_estu(self, estu_name):
        self.estu_name = estu_name

    def update_estu(self, dates):
        self.multiprocess_lock.acquire()
        stocklist = self.h5_db.load_factor(self.estu_name, '/indexes/', dates=dates)
        self.multiprocess_lock.release()
        stocklist = stocklist[stocklist.iloc[:, 0] == 1]
        return stocklist

    def set_multiprocess_lock(self, lock):
        self.multiprocess_lock = Lock() if lock is None else lock

    def prepare_factor(self, table_factor, start=None, end=None, dates=None,
                       ids=None, prepare_id_date=False):
        """
            准备因子列表生成数据源，若prepare_id_date为TRUE，则更新self.all_dates、
        self.all_ids为所有factors的交集.
            table_factor的原始格式是[(name, path, direction)...]，转换成factor_dict{name:path}
        """
        factor_dict = {}
        for factor in table_factor:
            factor_dict[factor[0]] = factor[1]

        if prepare_id_date:
            for fname, fpath in factor_dict.items():
                self.multiprocess_lock.acquire()
                idx = self.h5_db.load_factor(fname, fpath).index
                d = idx.get_level_values(0).unique()
                s = idx.get_level_values(1).unique()
                if self.all_dates is not None:
                    self.all_dates = set(self.all_dates).intersection(set(d))
                else:
                    self.all_dates = set(d)
                if self.all_ids is not None:
                    self.all_ids = set(self.all_ids).intersection(set(s))
                else:
                    self.all_ids = set(s)
                self.multiprocess_lock.release()

        if dates is not None:
            dates = pd.DatetimeIndex(dates)
        else:
            dates = tc.get_trade_days(start, end, retstr=None)
        if self.all_dates is not None:
            self.all_dates = set(self.all_dates).intersection(set(dates))
        else:
            self.all_dates = set(dates)
        self.all_dates = sorted(list(self.all_dates))

        if self.all_ids is None:
            self.all_ids = list(ids)
        elif ids is not None:
            self.all_ids = set(self.all_ids).intersection(set(ids))
        self.all_ids = sorted(list(self.all_ids))
        self.factor_dict = factor_dict
        self.factor_names = list(factor_dict)
        self.factor_read_num = pd.Series([0]*len(self.factor_names), index=self.factor_names)

    @DateRange2Dates
    def get_factor_data(self, factor_name, start_date=None, end_date=None, ids=None, dates=None, idx=None):
        """获得单因子的数据"""
        if idx is None:
            idx = self.idx[self.idx.index.get_level_values(0).isin(dates)]
        else:
            idx = idx[idx.index.get_level_values(0).isin(dates)]
        if ids is not None:
            idx = idx.loc[pd.IndexSlice[:, ids], :]
        if self.max_cache_num == 0:     # 无缓存机制
            self.multiprocess_lock.acquire()
            data = self.h5_db.load_factor(factor_name, self.factor_dict[factor_name], dates=dates, ids=ids)
            self.multiprocess_lock.release()
            return data.reindex(idx.index)
        factor_data = self.cache_data.get(factor_name)
        self.factor_read_num[factor_name] += 1
        if factor_data is None:     # 因子尚未进入缓存
            if self.cached_factor_num < self.max_cache_num:
                self.cached_factor_num += 1
                self.multiprocess_lock.acquire()
                factor_data = self.h5_db.load_factor(factor_name, self.factor_dict[factor_name])
                self.multiprocess_lock.release()
                self.cache_data[factor_name] = factor_data
            else:   # 当前缓存因子数大于等于最大缓存因子数，那么检查最小读取次数的因子
                cached_factor_read_nums = self.factor_read_num[self.cache_data.keys()]
                min_read_idx = cached_factor_read_nums.argmin()
                self.multiprocess_lock.acquire()
                if cached_factor_read_nums.iloc[min_read_idx] < self.factor_read_num[factor_name]:
                    factor_data = self.h5_db.load_factor(factor_name, self.factor_dict[factor_name])
                    self.cache_data.pop(cached_factor_read_nums.index[min_read_idx])
                    self.cache_data[factor_name] = factor_data
                else:
                    data = self.h5_db.load_factor(factor_name, self.factor_dict[factor_name], dates=dates, ids=ids)
                    self.multiprocess_lock.release()
                    return data.reindex(idx.index)
                self.multiprocess_lock.release()
        return factor_data.reindex(idx.index)

    @DateRange2Dates
    def get_data(self, factor_names, start_date=None, end_date=None, ids=None, dates=None, idx=None):
        """加载因子数据到一个dataframe里"""
        frame = []
        for ifactor in factor_names:
            ifactor_data = self.get_factor_data(ifactor, dates=dates, ids=ids, idx=idx)
            frame.append(ifactor_data)
        data = pd.concat(frame, axis=1)
        return data

    def get_factor_unique_data(self, factor_name, start_date=None, end_date=None, ids=None, dates=None):
        """提取因子unique数据"""
        factor_data = self.get_factor_data(factor_name,start_date, end_date, ids, dates)
        return factor_data[factor_name].unique().tolist()

    def save_info(self, path="D/data/riskmodel/datasources"):
        """保存数据源信息"""
        ensure_dir_exists(path)
        dumper = DiskPersistProvider(persist_path=path)
        info = {
            'name': self.name,
            'all_dates': self.all_dates,
            'all_ids': self.all_ids,
            'idx': self.idx,
            'factor_dict': self.factor_dict,
            'factor_names': self.factor_names,
            'max_cache_num': self.max_cache_num,
            'estu_name': self.estu_name
        }
        dumper.dump(info, name=self.name)

    def load_info(self, name, path="D/data/riskmodel/datasources"):
        """加载数据源信息"""
        dumper = DiskPersistProvider(persist_path=path)
        info = dumper.load(name)
        self.name = info['name']
        self.all_dates = info['all_dates']
        self.all_ids = info['all_ids']
        self.idx = info['idx']
        self.factor_dict = info['factor_dict']
        self.factor_names = info['factor_names']
        self.max_cache_num = info['max_cache_num']
        self.estu_name = info['estu_name']
        self.factor_read_num = pd.Series(np.zeros(len(self.factor_names)), index=self.factor_names)

    def save_factor(self, data, path):
        self.multiprocess_lock.acquire()
        self.h5_db.save_factor(data, path)
        self.multiprocess_lock.release()


class RiskDataSource(object):
    """
    风险数据库数据源，包括风险因子收益率、风险因子收益协方差矩阵、股票特质收益率、股票特质收益协方差矩阵、
    截面回归的统计量、原始的风险因子收益协方差矩阵(调整前)、 原始的特质收益协方差矩阵(调整前)。每一个数据库对应一个文件夹。

    文件夹的结构：
    =============
    RiskDataSource:
        factorRisk:
            factorRisk1.csv
            factorRisk2.csv
            ...
        specificRisk:
            specificRisk1.csv
            specificRisk2.csv
            ...
        rawFactorRisk:
            rawFactorRisk1.csv
            rawFactorRisk2.csv
            ...
        rawSpecificRisk:
            rawSpecificRisk1.csv
            rawSpecificRisk2.csv
            ...
        factor_return.h5
        regress_stats.h5
        resid_return.h5

    文件存储的格式:
    =============
    风险因子收益率：h5
        DataFrame(index:[date IDs], columns:[factor_return])
    风险因子收益协方差矩阵: csv
        每日一个csv文件，以 YYYYMMDD 命名。文件内的数据格式：第一列和第一行是因子名称
    股票特质(残差)收益率: h5
        DataFrame(index:[date IDs], columns:[resid_return])
    股票特质收益协方差矩阵: csv
        每日一个csv文件，以YYYYMMDD命名。文件内数据格式: 第一列和第一行是股票代码
    截面回归统计量: h5
        DataFrame(index:[date stats_name], columns:[regress_stats])
    原始的风险因子收益协方差矩阵(调整前)
        每日一个csv文件，以 YYYYMMDD 命名。文件内的数据格式：第一列和第一行是因子名称
    原始的特质收益协方差矩阵(调整前)
        每日一个csv文件，以YYYYMMDD命名。文件内数据格式: 第一列和第一行是股票代码
    """
    root_dir = default_riskds_root
    h5_db = H5DB(data_path=default_riskds_root)

    def __init__(self, name):
        self._name = name
        self._dspath = path.join(RiskDataSource.root_dir, self._name)
        self._h5_dir = '/%s/'%self._name
        self.h5_db = H5DB(data_path=self._dspath)
        self.persist_helper = DiskPersistProvider(self._h5_dir)
        ensure_dir_exists(self._dspath)
        self.initialize()

    def initialize(self):
        ensure_dir_exists(path.join(self._dspath, 'factorRisk'))
        ensure_dir_exists(path.join(self._dspath, 'specificRisk'))

    def check_file_exists(self, file_name):
        return path.isfile(path.join(self._dspath, file_name))

    def check_dir_exists(self, dir_name):
        return path.isdir(path.join(self._dspath, dir_name))

    @DateRange2Dates
    def load_returns(self, start_date=None, end_date=None, dates=None):
        if not self.check_file_exists('factor_return.h5'):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=dates)
        else:
            print(FileNotFoundError("因子收益率文件不存在！"))
            return pd.DataFrame()
        return ret['factor_return'].unstack()

    @DateRange2Dates
    def load_factor_return(self, factor_name, start_date=None, end_date=None, dates=None):
        if isinstance(factor_name, str):
            factor_name = [factor_name]
        if not self.check_file_exists('factor_return.h5'):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=dates, ids=factor_name)
        else:
            print(FileNotFoundError("因子收益率文件不存在！"))
            return pd.DataFrame()
        return ret['factor_return'].unstack()

    def load_snapshot_return(self, date):
        if not self.check_file_exists('factor_return.h5'):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=[date])
        else:
            raise FileNotFoundError("因子收益率文件不存在！")
        return ret['factor_return'].unstack()

    @DateRange2Dates
    def load_stats(self, start_date=None, end_date=None, dates=None):
        if not self.check_file_exists('regress_stats.h5'):
            stats = self.h5_db.load_factor('regress_stats', self._h5_dir, dates=dates)
        else:
            raise FileNotFoundError("回归诊断文件不存在！")
        return stats['regress_stats'].unstack()

    @DateRange2Dates
    def load_factor_riskmatrix(self, start_date=None, end_date=None, dates=None):
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        for i, date in enumerate(dates_str):
            csv_file = path.join(self._dspath, 'factorRisk/%s.csv'%date)
            if path.isfile(csv_file):
                matrix = pd.read_csv(csv_file, index_col=0, header=0)
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！"%date)
        return matrixes

    @DateRange2Dates
    def load_specific_riskmatrix(self, start_date=None, end_date=None, dates=None):
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        for i, date in enumerate(dates_str):
            csv_file = path.join(self._dspath, 'specificRisk/%s.csv'%date)
            if path.isfile(csv_file):
                matrix = pd.read_csv(csv_file, index_col=0, header=0)
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！"%date)
        return matrixes

    @DateRange2Dates
    def load_resid_factor(self, ids=None, start_date=None, end_date=None, dates=None):
        if not self.check_file_exists('resid_return.h5'):
            ret = self.h5_db.load_factor('resid_return', self._h5_dir, dates=dates, ids=ids)
        else:
            raise FileNotFoundError("因子收益率文件不存在！")
        return ret

    def save_data(self, **kwargs):
        tvalue = pd.DataFrame()
        if 'tvalue' in kwargs:
            if not kwargs['tvalue'].empty:
                tvalue = kwargs['tvalue'].rename(columns=lambda x: 'tstat_' + x).stack().to_frame('regress_stats')
                tvalue.index.names = ['date', 'IDs']
        fvalue = pd.DataFrame()
        if 'fvalue' in kwargs:
            if not kwargs['fvalue'].empty:
                fvalue = kwargs['fvalue'].to_frame().stack().to_frame('regress_stats')
                fvalue.index.names = ['date', 'IDs']
        rsquared = pd.DataFrame()
        if 'rsquared' in kwargs:
            if not kwargs['rsquared'].empty:
                rsquared = kwargs['rsquared'].to_frame().stack().to_frame('regress_stats')
                rsquared.index.names = ['date', 'IDs']
        adjust_rsquared = pd.DataFrame()
        if 'adjust_rsquared' in kwargs:
            if not kwargs['adjust_rsquared'].empty:
                adjust_rsquared = kwargs['adjust_rsquared'].to_frame().stack().to_frame('regress_stats')
                adjust_rsquared.index.names = ['date', 'IDs']
        vifs = pd.DataFrame()
        if 'vifs' in kwargs:
            if not kwargs['vifs'].empty:
                vifs = kwargs['vifs'].stack().to_frame('regress_stats')
                vifs.index.names = ['date', 'IDs']
        stats = pd.concat([tvalue, fvalue, rsquared, adjust_rsquared, vifs]).sort_index()
        self.h5_db.save_factor(stats, self._h5_dir)
        if 'factor_return' in kwargs:
            if not kwargs['factor_return'].empty:
                factor_return = kwargs['factor_return'].stack().to_frame('factor_return')
                factor_return.index.names = ['date', 'IDs']
                self.h5_db.save_factor(factor_return, self._h5_dir)
        if 'resid_factor' in kwargs:
            if not kwargs['resid_return'].empty:
                resid_return = kwargs['resid_return'].stack().toframe('resid_return')
                resid_return.index.names = ['date', 'IDs']
                self.h5_db.save_factor(resid_return, self._h5_dir)
        if 'factor_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'factorRisk')
            for k, v in kwargs['factor_riskmatrix']:
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv"%date))
        if 'specific_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'specificRisk')
            for k, v in kwargs['specific_riskmatrix']:
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv"%date))
        return

    def load_others(self, name):
        """
        使用.pkl文件，加载其他数据

        """
        obj = self.persist_helper.load(name)
        return obj

    def save_other(self, name, **kwargs):
        """使用.pkl文件，序列化对象"""
        self.persist_helper.dump(kwargs, name)
        return


