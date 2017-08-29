"""对H5DB进行再一次封装
   1. 增加了进程锁,便于支持多进程运算
   2. 支持提前设置因子列表，设置完成之后可以直接调用prepare_data函数提取数据
"""
from multiprocessing import Lock
from ..data_source.trade_calendar import trade_calendar, as_timestamp
import pandas as pd
from ..utils.datetime_func import DateRange2Dates
tc = trade_calendar()

class RiskModelDataSourceOnH5(object):
    def __init__(self, h5_db, lock=None):
        self.h5_db = h5_db
        self.set_multiprocess_lock(lock)
        self.all_dates = None
        self.all_ids = None
        self.factor_dict = {}
        self.factor_names = []
        self.cache_data = {}            # 缓存数据
        self.max_cache_num = 0          # 最大缓存数量
        self.cached_factor_num = 0      # 已经缓存的因子数量
        self.factor_read_num = None     # 因子的读取次数

    def set_dimension(self, dates, ids):
        """设置日期和股票"""
        self.all_ids = ids
        self.all_dates = pd.DatetimeIndex(dates).tolist()

    def set_multiprocess_lock(self, lock):
        self.multiprocess_lock = Lock() if lock is None else lock

    def prepare_factor(self, table_factor, start=None, end=None, dates=None,
                       ids=None, prepare_id_date=False):
        """
            准备因子列表生成数据源，若prepare_id_date为TRUE，则更新self.all_dates、
        self.all_ids为所有factors的交集.
            table_fator的原始格式是[(name, path, direction)...]，转换成factor_dict{name:path}
        """
        factor_dict = {}
        for factor in table_factor:
            factor_dict[factor(0)] = factor(1)

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

        if (start is None) and (end is None):
            dates = pd.DatetimeIndex(dates)
        else:
            dates = tc.get_trade_days(start, end)
        if self.all_dates is not None:
            self.all_dates = set(self.all_dates).intersection(set(dates))
        else:
            self.all_dates = set(dates)
        self.all_dates = list(self.all_dates)

        if self.all_ids is None:
            self.all_ids = list(ids)
        else:
            self.all_ids = set(self.all_ids).intersection(set(ids))
        self.all_ids = list(self.all_ids)
        self.factor_dict = factor_dict
        self.factor_names = list(factor_dict)
        self.factor_read_num = pd.Series([0]*len(self.factor_names), index=self.factor_names)

    @DateRange2Dates
    def get_factor_data(self, factor_name, start_date=None, end_date=None, ids=None, dates=None):
        """获得单因子的数据"""
        if self.max_cache_num == 0: #无缓存机制
            return self.h5_db.load_factor(factor_name, self.factor_dict[factor_name], dates=self.all_dates,
                                          ids=self.all_ids)
        factor_data = self.cache_data.get(factor_name)
        self.factor_read_num[factor_name] += 1
        if factor_data is None: #因子尚未进入缓存
            if self.cached_factor_num < self.max_cache_num:
                self.cached_factor_num += 1
                factor_data = self.h5_db.load_factor(factor_name, self.factor_dict[factor_name], dates=self.all_dates,
                                                     ids=self.all_ids)
                self.cache_data[factor_name] = factor_data
            else:






