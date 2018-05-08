from .data_source.base_data_source_h5 import data_source, sec, h5, tc
from .utils.disk_persist_provider import DiskPersistProvider
from .factor_performance.ic_analyser import IC_Calculator
from .data_source.stock_universe import from_formula
import pandas as pd
import os

class Environment(object):
    _env = None

    def __init__(self, config):
        Environment._env = self
        self._config = config
        self._stock_universe = config.universe
        self._h5DB = h5
        self._trade_calendar = tc
        self._sector = sec
        self._disk_persist_provider = DiskPersistProvider()
        self._data_source = data_source
        self._benchmark_return = None
        self._stock_return = None
        self._all_trade_dates = None
        self._factor_group_info_dates = None
        self._factors = None
        self._factor_data_process_mods = None
        self._ic_calculator = IC_Calculator()

    @classmethod
    def get_instance(cls):
        return Environment._env
    
    def _initialize(self):
        """初始化"""
        # 加载基准的收益率
        benchmark = self._config.benchmark
        if benchmark.startswith('101'):
            benchmark_return = self._data_source.load_factor('daily_returns_%', '/indexprices/', ids=[benchmark],
                                                             start_date=self._config.start_date, end_date=self._config.end_date)
            self._benchmark_return = benchmark_return.reset_index(level=1, drop=True) / 100
        else:
            benchmark_return = self._data_source.load_factor('daily_returns_%', '/indexprices/', ids=[benchmark],
                                                             start_date=self._config.start_date, end_date=self._config.end_date).\
                reset_index(level=1, drop=True) / 100
            self._benchmark_return = benchmark_return

        # 加载股票的日收益率
        stock_return = self._data_source.load_factor('daily_returns', '/stocks/', start_date=self._config.start_date,
                                                     end_date=self._config.end_date)
        self._stock_return = stock_return
        
        # 确定回测的交易日序列
        dt_start = self._config.start_date
        dt_end = self._config.end_date
        self._all_trade_dates = self._trade_calendar.get_trade_days(
            dt_start, dt_end, retstr=None)
        
        # 确定测试时需要的分组信息的日期
        group_start_date = self._trade_calendar.tradeDayOffset(
            self._config.start_date, 0, self._config.freq
        )
        self._factor_group_info_dates = self._trade_calendar.get_trade_days(
            group_start_date, self._config.end_date, self._config.freq, retstr=None
        )
        if self._config.user_rebalance_dates is not None:
            with open(self._config.user_rebalance_dates) as csv_file:
                dates = pd.DatetimeIndex(pd.read_csv(csv_file, header=0, squeeze=True, parse_dates=['date']).values)
            self._factor_group_info_dates = dates

        # IC计算器初始化
        self._ic_calculator.set_stock_returns(self)
        # 加载持久化工具
        self._disk_persist_provider.set_path(self._config.extra.result_file_dir)
    
    # 为因子进行初始化
    def set_factors(self, factors):
        factor_father_dir = self._config.extra.result_file_dir
        for factor in factors:
            factor.initialize(self)
            if self._config.extra.persist:
                if os.path.isfile(
                    os.path.join(
                        factor_father_dir,"{factor_name}/{factor_name}.pkl".format(factor_name=factor.name))):
                    state = self._disk_persist_provider.load(
                        "{factor_name}/{factor_name}".format(factor_name=factor.name))
                    factor.set_state(state)

        # 为因子加载数据
        sector_name = self._config.universe
        if sector_name == '全A':
            stocks = self._env._sector.get_history_ashare(self._factor_group_info_dates)
        else:
            stocks = from_formula(sector_name).get(dates=list(self._factor_group_info_dates))
        ids = stocks.index.get_level_values(1).unique().tolist()
        for factor in factors:
            if factor.alias_name is not None:
                factor_data = self._env._data_source.load_factor(factor.alias_name, factor.axe, ids=ids, dates=self._factor_group_info_dates)
                factor_data.columns = [factor.name]
            else:
                factor_data = self._env._data_source.load_factor(factor.name, factor.axe, ids=ids, dates=self._factor_group_info_dates)
            if not factor.data.empty:
                old = factor.data[~factor.data.index.isin(factor_data.index)]
            else:
                old = pd.DataFrame()
            factor.data = old.append(factor_data).reindex(stocks.index)
        self._factors = factors