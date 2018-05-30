# coding: utf-8
"""对H5DB进行再一次封装
   1. 增加了进程锁,便于支持多进程运算
   2. 支持提前设置因子列表，设置完成之后可以直接调用prepare_data函数提取数据
"""
from multiprocessing import Lock
from ..data_source.trade_calendar import trade_calendar
from ..data_source.base_data_source_h5 import H5DB, h5
from ..data_source.ncdb import NCDB
from ..utils.datetime_func import DateRange2Dates
from ..utils.disk_persist_provider import DiskPersistProvider
from ..utils.tool_funcs import ensure_dir_exists, drop_patch
from os import path
from warnings import warn
import pandas as pd
import numpy as np
import os
import xarray as xr
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
    ============
    RiskDataSource:
        factorRisk:
            factorRisk1.csv \n
            factorRisk2.csv \n
            ...
        specificRisk:
            specificRisk1.csv \n
            specificRisk2.csv \n
            ...
        rawFactorRisk:
            rawFactorRisk1.csv \n
            rawFactorRisk2.csv \n
            ...
        rawSpecificRisk:
            rawSpecificRisk1.csv \n
            rawSpecificRisk2.csv \n
            ...
        factorData:
            factor1.h5 \n
            factor2.h5 \n
            ...
        factor_return.h5 \n
        regress_stats.h5 \n
        resid_return.h5 \n

    文件存储的格式:
    ==============
    风险因子收益率：h5
        DataFrame(index:[date factor_names], columns:[factor_return])
    风险因子收益协方差矩阵: csv
        每日一个csv文件，以YYYYMMDD命名。文件内的数据格式：第一列和第一行是因子名称
    股票特质(残差)收益率: h5
        DataFrame(index:[date IDs], columns:[resid_return])
    股票特质收益协方差矩阵: csv
        每日一个csv文件，以YYYYMMDD命名。文件内数据格式: 第一列股票代码
    截面回归统计量: h5
        DataFrame(index:[date stats_name], columns:[regress_stats])
    原始的风险因子收益协方差矩阵(调整前): csv
        每日一个csv文件，以 YYYYMMDD 命名。文件内的数据格式：第一列和第一行是因子名称
    原始的特质收益协方差矩阵(调整前): pkl
        每日一个pkl文件，以YYYYMMDD命名。文件内数据格式: 第一列是股票代码
    风险因子数据: h5
        每个因子是一个h5文件， 以因子名称命名。DataFrame(index:[date IDs], columns:[factor_name])
    """
    root_dir = default_riskds_root
    base_db = h5

    def __init__(self, name):
        self._name = name
        self._dspath = path.join(RiskDataSource.root_dir, self._name)
        self._h5_dir = '/%s/' % self._name
        self.h5_db = H5DB(data_path=self._dspath)
        self.nc_db = NCDB(data_path=self._dspath)
        self.persist_helper = DiskPersistProvider(self._dspath)
        ensure_dir_exists(self._dspath)
        self.initialize()

    def initialize(self):
        ensure_dir_exists(path.join(self._dspath, 'factorRisk'))
        ensure_dir_exists(path.join(self._dspath, 'rawFactorRisk'))
        ensure_dir_exists(path.join(self._dspath, 'specificRisk'))
        ensure_dir_exists(path.join(self._dspath, 'rawSpecificRisk'))
        ensure_dir_exists(path.join(self._dspath, 'others'))
        ensure_dir_exists(path.join(self._dspath, 'factorData'))

    def check_file_exists(self, file_name):
        return path.isfile(path.join(self._dspath, file_name))

    def check_dir_exists(self, dir_name):
        return path.isdir(path.join(self._dspath, dir_name))

    def list_files(self, dir_name, ignores=[]):
        return [drop_patch(x) for x in os.listdir(self._dspath+'/%s'%dir_name) if x not in ignores]

    def list_factor_names(self, file_name, file_dir):
        """列出文件所包含的因子名称"""
        return self.h5_db.list_h5file_factors(file_name, file_dir)

    def list_file_meta(self, file_name, file_dir, attr_name):
        """列出文件的meta data"""
        return self.nc_db.load_file_attr(file_name, file_dir, attr_name)

    @property
    def max_date_of_factor(self):
        min_date, max_date = self.h5_db.get_date_range('risk_factor', '/factorData/')
        return max_date

    @property
    def max_date_of_factor_return(self):
        return self.h5_db.get_date_range('factor_return', '/%s/'%self._name)[1]

    @property
    def min_date_of_factor_return(self):
        return self.h5_db.get_date_range('factor_return', '/%s/' % self._name)[0]

    def load_factors(self, factor_names, ids=None, start_date=None, end_date=None, dates=None):
        """
        加载风险因子数据

        风险因子文件格式：nc
        文件内部数据格式：DataFrame(index:[date IDs], columns:[factor_name])

        Return
        ======
        factor_data: DataFrame
            DataFrame(index:[date IDs], columns:[factor_names])
        """
        raw_dates = pd.DatetimeIndex(dates)
        dates = raw_dates.copy()
        max_date_of_factor = self.max_date_of_factor
        if_reindex = False
        if max(dates) > max_date_of_factor:
            warn("不是所有日期都存在风险数据, 风险数据的最近日期为%s"%max_date_of_factor.strftime("%Y-%m-%d"))
            if_reindex = True
            if len(dates) == 1:
                dates = pd.DatetimeIndex([max_date_of_factor])
            else:
                dates = dates[dates <= max_date_of_factor]
                if max_date_of_factor not in dates:
                    dates = dates.append(pd.DatetimeIndex([max_date_of_factor]))
        if factor_names == 'ALL':
            style = self.h5_db.load_multi_columns('risk_factor', '/factorData/', dates=dates, ids=ids)
            indu = self.nc_db.load_as_dummy('industry', '/factorData/', dates=dates, ids=ids)
            new = style.join(indu, how='left')
        elif factor_names == 'STYLE':
            new = self.h5_db.load_multi_columns('risk_factor', '/factorData/', dates=dates, ids=ids)
        elif 'Estu' not in factor_names:
            new = self.h5_db.load_multi_columns('risk_factor', '/factorData/', dates=dates, ids=ids)
            if any([x.startswith('Indu_') for x in factor_names]):
                indu = self.nc_db.load_as_dummy('industry', '/factorData/', dates=dates, ids=ids)
                new = new.join(indu, how='left')
            new = new[factor_names]
        else:
            new = self.h5_db.load_factor('Estu', '/factorData/', ids=ids, dates=dates)
        if if_reindex:
            new = new.to_xarray()
            new = new.reindex({'date': raw_dates}, method='ffill').to_dataframe().dropna(how='all')
            if new.index.names[0] == 'IDs':
                new.index = new.index.swaplevel('IDs', 'date')
                new.sort_index(inplace=True)
        new.rename(columns=lambda x: x[5:] if x.startswith("Indu_") else x, inplace=True)
        return new

    @DateRange2Dates
    def load_style_factor(self, factor_names, ids=None, start_date=None, end_date=None, dates=None):
        """加载风格因子数据"""
        raw_dates = pd.DatetimeIndex(dates)
        dates = raw_dates.copy()
        ret = 'df'
        max_date_of_factor = self.max_date_of_factor

        if max(dates) > max_date_of_factor:
            warn("不是所有日期都存在风险数据, 风险数据的最近日期为%s"%max_date_of_factor.strftime("%Y-%m-%d"))
            ret = 'xarray'
            if len(dates) == 1:
                dates = pd.DatetimeIndex([max_date_of_factor])
            else:
                dates = dates[dates <= max_date_of_factor]
                if max_date_of_factor not in dates:
                    dates = dates.append(pd.DatetimeIndex([max_date_of_factor]))

        style = self.h5_db.load_multi_columns('risk_factor', '/factorData/', factor_names=factor_names, dates=dates, ids=ids)
        if ret == 'xarray':
            style = style.to_xarray()
            style = style.reindex({'date': raw_dates}, method='ffill').to_dataframe().dropna(how='all')
            if style.index.names[0] == 'IDs':
                style.index = style.index.swaplevel('IDs', 'date')
                style.sort_index(inplace=True)
        return style

    @DateRange2Dates
    def load_industry(self, ids=None, start_date=None, end_date=None, dates=None):
        """
        加载行业哑变量
        行业变量文件以"Indu_"开头， 存储在factorData文件夹中

        Returns:
        ========
        industry: DataFrame
            DataFrame(index:[date IDs], columns:[industry_names])
        """
        raw_dates = pd.DatetimeIndex(dates)
        dates = raw_dates.copy()
        ret = 'df'
        max_date_of_factor = self.max_date_of_factor

        if max(dates) > max_date_of_factor:
            warn("不是所有日期都存在风险数据, 风险数据的最近日期为%s" % max_date_of_factor.strftime("%Y-%m-%d"))
            ret = 'xarray'
            if len(dates) == 1:
                dates = pd.DatetimeIndex([max_date_of_factor])
            else:
                dates = dates[dates <= max_date_of_factor]
        indu = self.nc_db.load_as_dummy('industry', '/factorData/', dates=dates, ids=ids)
        if ret == 'xarray':
            indu = indu.to_xarray().reindex({'date': raw_dates}, method='ffill').to_dataframe().dropna(how='all')
            indu.index = indu.index.swaplevel('IDs', 'date')
            indu.sort_index(inplace=True)
        indu.rename(columns=lambda x: x[5:] if x.startswith("Indu_") else x, inplace=True)
        return indu

    @DateRange2Dates
    def load_returns(self, start_date=None, end_date=None, dates=None):
        """
        加载风险因子收益率数据,一次性加载所有的因子。

        风险因子收益率文件格式：h5
        文件内部数据格式：DataFrame(index:[date factor_names], columns:[factor_return])

        Return
        ======
        ret: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        """
        if self.check_file_exists('%s/factor_return.h5' % self._name):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=dates)
        else:
            raise FileNotFoundError("因子收益率文件不存在！")
        return ret['factor_return'].unstack()

    @DateRange2Dates
    def load_factor_return(self, factor_name, start_date=None, end_date=None, dates=None):
        """
        加载单一风险因子的收益率

        Return
        ======
        ret: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        """
        if isinstance(factor_name, str):
            factor_name = [factor_name]
        if self.check_file_exists('%s/factor_return.h5' % self._name):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=dates, ids=factor_name)
        else:
            print(FileNotFoundError("因子收益率文件不存在！"))
            return pd.DataFrame()
        return ret['factor_return'].unstack().astype('float')

    def load_snapshot_return(self, date):
        """
        加载某一时间点的风险因子收益率数据

        Return
        ======
        ret: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        """
        if self.check_file_exists('%s/factor_return.h5' % self._name):
            ret = self.h5_db.load_factor('factor_return', self._h5_dir, dates=[date])
        else:
            raise FileNotFoundError("因子收益率文件不存在！")
        return ret['factor_return'].unstack()

    @DateRange2Dates
    def load_stats(self, start_date=None, end_date=None, dates=None):
        """
        加载截面回归统计诊断指标，主要包括了：
        1. 每一个风险因子的t统计量
        2. 截面回归的F统计量
        3. 截面回归的R方、调整的R方
        4. 每一个风险因子的方差膨胀系数(VIF)

        Return
        ======
        stats: DataFrame
            DataFrame(index:[date], columns:[stat_names])
        """
        if self.check_file_exists('%s/regress_stats.h5' % self._name):
            stats = self.h5_db.load_factor('regress_stats', self._h5_dir, dates=dates)
        else:
            raise FileNotFoundError("回归诊断文件不存在！")
        return stats['regress_stats'].unstack()

    @DateRange2Dates
    def load_factor_riskmatrix(self, start_date=None, end_date=None, dates=None, raw=False):
        """
        加载风险因子的风险矩阵

        Paramters:
        ==========
        raw: bool
            是否加载原始的风险因子收益协方差矩阵

        Return:
        =======
        riskmatrix: dict
            dict(key:date, value:DataFrame(index:[factor_names], columns:[factor_names])
        """
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        if raw:
            dirpth = path.join(self._dspath, 'rawFactorRisk')
        else:
            dirpth = path.join(self._dspath, 'factorRisk')

        for i, date in enumerate(dates_str):
            csv_file = path.join(dirpth, '%s.csv' % date)
            if path.isfile(csv_file):
                matrix = pd.read_csv(csv_file, index_col=0, header=0)
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！" % date)
        return matrixes

    @DateRange2Dates
    def load_xy_riskmatrix(self, start_date=None, end_date=None, dates=None, raw=False):
        """
        加载兴业金工风险模型风险矩阵
        风险矩阵是月频的总风险
        当指定日期的风险矩阵不存在时，用最近的代替

        Return:
        =======
        riskmatrix: dict
            dict(key:date, value:DataFrame(index:[IDs], columns:[IDs])
        """
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        dirpth = path.join(self._dspath, 'stockRisk')

        for i, date in enumerate(dates_str):
            csv_file = path.join(dirpth, '%s.csv' % date)
            if path.isfile(csv_file):
                matrix = pd.read_csv(csv_file, index_col=0, header=0).rename(
                    index=lambda x: str(x).zfill(6), columns=lambda x: str(x).zfill(6))
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！用最近的风险矩阵代替!" % date)
                csv_file = path.join(dirpth, '%s' % max(os.listdir(dirpth)))
                matrix = pd.read_csv(csv_file, index_col=0, header=0).rename(
                    index=lambda x: str(x).zfill(6), columns=lambda x: str(x).zfill(6))
                matrixes[dates[i]] = matrix
        return matrixes

    @DateRange2Dates
    def load_uqer_riskmatrix(self, start_date=None, end_date=None, dates=None, raw=False):
        """
        加载优矿风险模型风险矩阵
        风险矩阵是short的总风险

        Return:
        =======
        riskmatrix: dict
            dict(key:date, value:DataFrame(index:[IDs], columns:[IDs])
        """
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        factor_risk_pth = path.join(self._dspath, 'factorRisk')
        spec_risk_pth = path.join(self._dspath, 'specificRisk')

        for i, date in enumerate(dates_str):
            factor_risk_file = path.join(factor_risk_pth, '%s.csv' % date)
            spec_risk_file = path.join(spec_risk_pth, '%s.csv' % date)
            if path.isfile(factor_risk_file):
                factor_risk = pd.read_csv(factor_risk_file, index_col=0, header=0) / 10000.0
                factor_expo = self.load_factors('ALL', dates=[date]).reindex(columns=factor_risk.columns).dropna().reset_index(level='date', drop=True)
                spec_risk = (pd.read_csv(spec_risk_file, header=0, converters={'ticker': lambda x: str(x).zfill(6)}).set_index(['ticker']).rename_axis('IDs')
                             ['SRISK'] / 100.0) ** 2
                common_stocks = np.intersect1d(factor_expo.index.values, spec_risk.index.values)
                matrix = factor_expo.reindex(common_stocks).values.dot(factor_risk.values).dot(factor_expo.reindex(common_stocks).values.T) + \
                         np.diag(spec_risk.reindex(common_stocks).values)
                matrix = pd.DataFrame(matrix, index=common_stocks, columns=common_stocks)
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！" % date)
        return matrixes

    @DateRange2Dates
    def load_specific_riskmatrix(self, start_date=None, end_date=None, dates=None, raw=False):
        """
        加载特质风险矩阵

        Paramters:
        ==========
        raw: bool
            是否加载原始的特质收益协方差矩阵

        Return:
        ======
        specific_riskmatrix: dict
            dict(key:date, value:Series(index:[IDs], value:specific_risk)
        """
        dates_str = [x.strftime("%Y%m%d") for x in dates]
        matrixes = {}
        if raw:
            dirpth = path.join(self._dspath, 'rawSpecificRisk')
        else:
            dirpth = path.join(self._dspath, 'specificRisk')

        for i, date in enumerate(dates_str):
            csv_file = path.join(dirpth, '%s.csv' % date)
            if path.isfile(csv_file):
                matrix = pd.read_csv(csv_file, index_col=0, header=None, squeeze=True)
                matrix.index = [str(x).zfill(6) for x in matrix.index]
                matrixes[dates[i]] = matrix
            else:
                warn("%s 风险矩阵不存在！" % date)
        return matrixes

    @DateRange2Dates
    def load_resid_factor(self, ids=None, start_date=None, end_date=None, dates=None):
        """
        加载残差收益率

        Return:
        ======
        resid_factor: DataFrame
            DataFrame(index:[date, IDs], columns:[resid_return])
        """
        if self.check_file_exists('%s/resid_return.h5' % self._name):
            ret = self.h5_db.load_factor('resid_return', self._h5_dir, dates=dates, ids=ids)
        else:
            raise FileNotFoundError("因子收益率文件不存在！")
        return ret

    def save_data(self, **kwargs):
        """
        保存风险模型的数据，保存规则与数据存储规则一致。

        Paramters:
        ==========
        tvalue: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        fvalue: Series
            Series(index:[date], name: fvalue)
        rsquared: Series
            Series(index:[date], name:rsquared)
        adjust_rsquared: Series
            Series(index:[date], name:adjust_rsquared)
        vifs: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        factor_return: DataFrame
            DataFrame(index:[date], columns:[factor_names])
        resid_return: DataFrame
            DataFrame(index;[date], columns:[IDs])
        factor_riskmatrix: dict
            dict(key: date, value:DataFrame(index:[factor_names], columns:[factor_names]))
        raw_factor_riskmatrix: dict
            dict(key: date, value:DataFrame(index:[factor_names], columns:[factor_names]))
        raw_specific_riskmatrix: dict
            dict(key: date, value:Series(index:[IDs], columns:[IDs])
        specific_riskmatrix: dict
            dict(key: date, value:Seriex(index:[IDs], columns:[IDs])
        factor_data: DataFrame
            DataFrame(indedx:[date IDs], columns:[factor_names])
        """
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
                vifs = kwargs['vifs'].rename(columns=lambda x: 'vif_' + x).stack().to_frame('regress_stats')
                vifs.index.names = ['date', 'IDs']
        stats = pd.concat([tvalue, fvalue, rsquared, adjust_rsquared, vifs]).sort_index()
        self.h5_db.save_factor(stats, self._h5_dir)
        if 'factor_return' in kwargs:
            if not kwargs['factor_return'].empty:
                factor_return = kwargs['factor_return'].stack().to_frame('factor_return')
                factor_return.index.names = ['date', 'IDs']
                self.h5_db.save_factor(factor_return, self._h5_dir)
        if 'resid_return' in kwargs:
            if not kwargs['resid_return'].empty:
                resid_return = kwargs['resid_return'].stack().to_frame('resid_return')
                resid_return.index.names = ['date', 'IDs']
                self.h5_db.save_factor(resid_return, self._h5_dir)
        if 'factor_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'factorRisk')
            for k, v in kwargs['factor_riskmatrix'].items():
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv" % date))
        if 'raw_factor_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'rawFactorRisk')
            for k, v in kwargs['raw_factor_riskmatrix'].items():
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv" % date))
        if 'specific_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'specificRisk')
            for k, v in kwargs['specific_riskmatrix'].items():
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv" % date))
        if 'raw_specific_riskmatrix' in kwargs:
            save_dir = path.join(self._dspath, 'rawSpecificRisk')
            for k, v in kwargs['raw_specific_riskmatrix'].items():
                date = k.strftime("%Y%m%d")
                v.to_csv(path.join(save_dir, "%s.csv" % date))
        if 'factor_data' in kwargs:
            self.h5_db.save_factor(kwargs['factor_data'], '/factorData/')
        return

    def load_others(self, name):
        """
        使用.pkl文件，加载其他数据

        """
        name = 'others/' + name
        obj = self.persist_helper.load(name)
        return obj

    def save_other(self, name, **kwargs):
        """
        使用.pkl文件，序列化对象
        """
        name = 'others/' + name
        self.persist_helper.dump(kwargs, name)
        return


