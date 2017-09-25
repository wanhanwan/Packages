"""
股票组合优化器
"""

import numpy as np
import pandas as pd
from FactorLib.data_source.base_data_source_h5 import data_source
from .riskmodel_data_source import RiskDataSource


class Optimizer(object):
    """
    股票组合优化器，封装cplex。
    优化器提供现金、行业、风格的中性的限制以及跟踪误差的限定，得到最优化权重

    Paramters:
    ==========
    signal: Series
        组合的预期收益率，以股票代码为索引
        Series(index:[IDs], value:signal)
    date: datetime
        优化日期
    ds_name: str
        风险数据源名称
    active: bool
        目标函数采用主动权重，但返回结果仍是绝对权重
    benchmark: str
        基准组合代码
    risk_mul: float
        风险厌恶系数lambda
    """

    def __init__(self, signal, date, ds_name, active=False, benchmark='000905', risk_mul=0):
        self._signal = signal
        self._date = date
        self._rskds = RiskDataSource(ds_name)
        self._active = active
        self._benchmark = benchmark
        self._riskmul = risk_mul

        # 初始化优化结果，权重设为等权
        self.asset = pd.DataFrame(np.ones(len(self._signal)) / len(self._signal),
                                  index=pd.MultiIndex.from_product([[date], signal.index], names=['date','IDs']))

    def prepare_portfolio_style(self):
        data = self._rskds.l
