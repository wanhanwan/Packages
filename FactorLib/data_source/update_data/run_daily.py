
from ..base_data_source_h5 import data_source
from ..update_data.value_factor import ValueFuncListDaily
from ..update_data.momentum_factor import MomentumFuncListDaily
from ..update_data.liquidity_factor import LiquidityFuncListDaily
from ..update_data.reverse_fator import ReverseFuncListDaily
from ..update_data.time_series_factor import TimeSeriesFuncListDaily
from ..update_data.alternative_factor import AlternativeFuncListDaily
from ..update_data.growth_factor import GrowthFuncListDaily
from ..update_data.profit_factor import ProfitFuncListDaily
from ..update_data.universe import UniverseFuncListDaily


def dailyfactors(start, end):
    # 更新价值类因子数据
    for func in ValueFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新动量类因子
    for func in MomentumFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新流动性数据
    for func in LiquidityFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新反转因子
    for func in ReverseFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新时间序列类因子
    for func in TimeSeriesFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新另类因子
    for func in AlternativeFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新成长类因子
    for func in GrowthFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新盈利类因子
    for func in ProfitFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新股票池
    for func in UniverseFuncListDaily:
        func(start, end, data_source=data_source)