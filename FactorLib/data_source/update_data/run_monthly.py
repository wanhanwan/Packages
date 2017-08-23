from FactorLib.data_source.base_data_source_h5 import data_source
from .value_factor import ValueFuncListMonthly
from .momentum_factor import MomentumFuncListMonthly
from .liquidity_factor import LiquidityFuncListMonthly
from .reverse_fator import ReverseFuncListMonthly
from .time_series_factor import TimeSeriesFuncListMonthly
from .alternative_factor import AlternativeFuncListMonthly


def monthlyfactors(start, end):
    for func in ValueFuncListMonthly:
        func(start, end, data_source=data_source)

    #更新动量类因子
    for func in MomentumFuncListMonthly:
        func(start, end, data_source=data_source)

    #更新流动性数据
    for func in LiquidityFuncListMonthly:
        func(start, end, data_source=data_source)

    #更新反转因子
    for func in ReverseFuncListMonthly:
        func(start, end, data_source=data_source)

    #更新时间序列类因子
    for func in TimeSeriesFuncListMonthly:
        func(start, end, data_source=data_source)

    #更新另类因子
    for func in AlternativeFuncListMonthly:
        func(start, end, data_source=data_source)