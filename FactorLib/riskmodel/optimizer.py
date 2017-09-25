"""
股票组合优化器
"""

import numpy as np
import pandas as pd
import cplex as cpx
import sys
from FactorLib.data_source.base_data_source_h5 import data_source
from .riskmodel_data_source import RiskDataSource


class Optimizer(object):
    """
    股票组合优化器，底层优化依赖Cplex \n
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
                                  index=pd.MultiIndex.from_product([[date], signal.index], names=['date', 'IDs']))
        self._init_opt_prob()

    def _init_opt_prob(self):
        """
        初始化优化问题
        设置了现金中性和做空限制
        """
        vars = self._signal.index.tolist()
        self._c = cpx.Cplex()
        self._c.variables.add(names=vars)
        # 禁止做空
        self._c.variables.set_lower_bounds([(x, 0) for x in vars])
        # 现金中性
        self._c.linear_constraints.add(lin_expr=[[vars, [1]*len(vars)]],
                                       senses=['E'],
                                       rhs=[1],
                                       names=['csh_ntrl']
                                       )
        self._c.set_log_stream(sys.stdout)
        self._c.set_results_stream(sys.stdout)

    def _add_style_cons(self, style_dict, active=True):
        """
        设置风格限制

        Paramters:
        ==========
        style_dict: dict
            以字典形式进行风格约束，{key: style_name, value: constraint}
        active: bool
            相对于基准的主动暴露，默认为True。
        """
        cons = pd.Series(style_dict)
        if active:
            bchmk_style = self._prepare_benchmark_style(list(style_dict))
            cons = cons + bchmk_style
        portf_style = self._prepare_portfolio_style(list(style_dict))
        if np.any(pd.isnull(portf_style)) or np.any(pd.isnull(cons)):
            raise ValueError("风格因子数据存在缺失值!")
        lin_exprs = []
        rhs = []
        senses = ['E'] * len(portf_style)
        names = []
        for style, value in portf_style.iterrows():
            lin_exprs.append([value.index.tolist(), value.values.tolist()])
            rhs.append(cons[style])
            names.append(style)
        self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)

    def _add_industry_cons(self, industry_dict=None, active=True):
        """
        添加行业约束，默认所有行业都进行行业中性处理

        Paramters:
        ==========
        industry_dict: dict
            以字典形式进行行业约束，{key: industry_name, value: constraint} \n
            注意： 不在字典中进行显式约束的行业自动与基准进行匹配
        active: bool
            相对于基准的主动暴露，默认为True。
        """
        if industry_dict is None or active:
            indu_bchmrk = self._prepare_benchmark_indu()
        cons = pd.Series()

    def _prepare_benchmark_indu(self):
        indu = self._rskds.load_factors()

    def _prepare_portfolio_style(self, styles):
        portfolio = self._signal.index.tolist()
        data = self._rskds.load_factors(styles, ids=portfolio, dates=[self._date]).reset_index(level=0, drop=True)
        return data.reindex(self._signal.index).T

    def _prepare_benchmark_style(self, styles):
        """
        基准的风格因子

        Return:
        ======
        style_benchmark: Series
        基准的风格因子，Series(index:[factor_names], values:style)
        """
        weight = data_source.sector.get_index_weight(ids=self._benchmark, dates=[self._date])
        members = weight.index.get_level_values(1).tolist()
        style_data = self._rskds.load_factors(styles, ids=members, dates=[self._date])
        style_benchmark = style_data.mul(weight.iloc[:, 0], axis='index').sum() / weight.iloc[:, 0].sum()
        return style_benchmark