"""
股票组合优化器
"""

import numpy as np
import pandas as pd
import cplex as cpx
import sys
from FactorLib.data_source.base_data_source_h5 import data_source
from .riskmodel_data_source import RiskDataSource
from itertools import combinations_with_replacement


class Optimizer(object):
    """
    股票组合优化器，底层优化依赖于Cplex \n
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
        self.optimal = False
        self.solution_status = ""
        self.solution_value = None

        # 初始化优化结果，权重设为等权
        self.asset = pd.DataFrame(np.ones(len(self._signal)) / len(self._signal),
                                  index=pd.MultiIndex.from_product([[date], signal.index], names=['date', 'IDs']),
                                  columns=['optimal_weight'])
        self._bchmrk_weight = data_source.sector.get_index_weight(
            ids=self._benchmark, dates=[self._date]).reset_index(level=0, drop=True).iloc[:, 0]
        self._init_opt_prob()

    def _init_opt_prob(self):
        """
        初始化优化问题
        设置了现金中性和做空限制
        """
        nvar = self._signal.index.tolist()
        self._c = cpx.Cplex()
        self._c.variables.add(names=nvar)
        # 禁止做空
        self._c.variables.set_lower_bounds([(x, 0) for x in nvar])
        # 现金中性
        self._c.linear_constraints.add(lin_expr=[[nvar, [1]*len(nvar)]],
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
        for style, value in portf_style.iteritems():
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
        indu_bchmrk = self._prepare_benchmark_indu()
        if industry_dict is None:
            cons = indu_bchmrk
        elif active:
            cons = pd.Series(industry_dict)
            cons = cons + indu_bchmrk
        else:
            cons = pd.Series(industry_dict)
            cons = indu_bchmrk.update(cons)
        portf_indu = self._prepare_portfolio_indu()
        if np.any(pd.isnull(portf_indu)) or np.any(pd.isnull(cons)):
            raise ValueError("行业变量存在缺失值！")
        lin_exprs = []
        rhs = []
        senses = ['E'] * len(cons)
        names = []
        for indu, value in portf_indu.iteritems():
            lin_exprs.append([value.index.tolist(), value.values.tolist()])
            rhs.append(cons[indu])
            names.append(indu)
        self._c.linear_constraints.add(lin_expr=lin_exprs, rhs=rhs, senses=senses, names=names)

    def _add_stock_limit(self, default_min=0.0, default_max=1.0):
        """
        添加个股权重的上下限
        """
        # assert ((default_min < 0) or (default_max > 1))
        nvar = self._signal.index.tolist()
        min_limit = list(zip(nvar, [default_min]*len(nvar)))
        max_limit = list(zip(nvar, [default_max]*len(nvar)))
        self._c.variables.set_lower_bounds(min_limit)
        self._c.variables.set_upper_bounds(max_limit)

    def _prepare_portfolio_indu(self):
        """
        组合的行业因子

        Returns:
        ========
        data: DataFrame
            DataFrame(index:[industry_names], columns:[IDs])
        """
        portfolio = self._signal.index.tolist()
        data = self._rskds.load_industry(ids=portfolio, dates=[self._date]).reset_index(level=0, drop=True)
        return data.reindex(self._signal.index)

    def _prepare_benchmark_indu(self):
        """
        基准的行业因子
        """
        weight = self._bchmrk_weight
        indu = self._rskds.load_industry(ids=weight.index.tolist(), dates=[self._date]).reset_index(level=0, drop=True)
        indu = indu.mul(weight.iloc[:, 0], axis='index').sum() / weight.iloc[:, 0].sum()
        return indu

    def _prepare_portfolio_style(self, styles):
        """
        组合的风格因子

        Returns:
        ========
        data: DataFrame
            DataFrame(index:[factor_names], columns:[IDs])
        """
        portfolio = self._signal.index.tolist()
        data = self._rskds.load_factors(styles, ids=portfolio, dates=[self._date]).reset_index(level=0, drop=True)
        return data.reindex(self._signal.index)

    def _prepare_benchmark_style(self, styles):
        """
        基准的风格因子

        Return:
        ======
        style_benchmark: Series
        基准的风格因子，Series(index:[factor_names], values:style)
        """
        weight = self._bchmrk_weight
        members = weight.index.tolist()
        style_data = self._rskds.load_factors(styles, ids=members, dates=[self._date])
        style_benchmark = style_data.mul(weight.iloc[:, 0], axis='index').sum() / weight.iloc[:, 0].sum()
        return style_benchmark

    def _add_trckerr(self, target_err, **kwargs):
        if self._active:
            raise ValueError("跟踪误差与风险嫌恶项重复设置！")
        bchmrk_weight = self._bchmrk_weight.reindex(self._signal.index, fill_value=0)
        sigma = self._load_portfolio_risk()
        bchmrk_risk = np.dot(np.dot(bchmrk_weight.values, sigma), bchmrk_weight.values)
        qlin = [self._signal.index.tolist(), (-2.0 * np.dot(bchmrk_weight.values, sigma)).tolist()]
        qvar_expr = [list(x) for x in zip(*combinations_with_replacement(self._signal.index.tolist(), 2))]
        qvar_mul = np.triu(sigma) + np.tril(sigma, -1).T
        qvar_mul = qvar_mul.ravel()[np.flatnonzero(qvar_mul)]
        quad = qvar_expr + [qvar_mul.tolist()]

        self._c.quadratic_constraints.add(lin_expr=qlin, quad_expr=quad, sense='L', rhs=-bchmrk_risk+target_err,
                                          name='tracking_error')

    def _load_portfolio_risk(self):
        """
        加载股票组合风险矩阵
        """
        factor_risk = self._rskds.load_factor_riskmatrix(dates=[self._date], raw=False)[self._date]
        portf_style = self._prepare_portfolio_style(styles='ALL').reindex(columns=factor_risk.columns)
        specidic_risk = self._rskds.load_specific_riskmatrix(dates=[self._date], raw=False)[self._date].reindex(
            self._signal.index)
        if np.any(pd.isnull(portf_style)):
            raise ValueError("股票组合的风格因子存在缺失值！")
        sigma = np.dot(np.dot(portf_style.values, factor_risk.values), portf_style.values.T) + np.diag(
            specidic_risk.values)
        return sigma

    def add_constraint(self, key, *args, **kwargs):
        """
        设置限制条件
        目前支持:
            1. 风格敞口限制，key='Style'.敞口限定值以字典的形式传入，字典的key为因子名称，value为限定值。
            2. 行业敞口限制，key='Indu'.默认所有行业都限定为中性，把需要显式设定的行业敞口按照字典形式传入。
            3. 跟踪误差限制，key='TrackingError', value= target_tracking_error.
            4. 个股权重限制, key='StockLimit'. 个股上下限以tuple形式传入。
        Paramters:
        ==========
        key: str
            限制条件类型，可选选项：'Style'\'Indu'\'TrackingError'
        value: object
            key对应的限定目标值
        kwargs: object
            其他传入self._add_*()的参数

        Examples:
        =========
        >>> add_constraint('Style', {'SIZE': -0.3}, active=True)
        >>> add_constraint('Indu', {'Indu_Media': -0.5}, active=True)
        >>> add_constraint('TrackingError', 0.0025)
        """
        if key == 'Style':
            self._add_style_cons(*args, **kwargs)
        elif key == 'Indu':
            self._add_industry_cons(*args, **kwargs)
        elif key == 'TrackingError':
            self._add_trckerr(*args, **kwargs)
        elif key == 'StockLimit':
            self._add_stock_limit(*args, **kwargs)
        else:
            raise NotImplementedError("不支持的限制类型:%s" % key)

    def _create_obj(self):
        """
        生成目标函数
        """
        if np.any(pd.isnull(self._signal)):
            raise ValueError("组合信号存在缺失值！")
        self._c.objective.set_sense(self._c.objective.sense.maximize)
        if self._riskmul != 0:
            portfolio_risk = self._load_portfolio_risk()
            quad_obj = []
            ind = np.arange(portfolio_risk.shape[0]).tolist()
            for i in ind:
                quad_obj.append([ind, (-1.0 * portfolio_risk[i, :]).tolist()])
            self._c.objective.set_quadratic(quad_obj)
            if self._active:
                bchmrk_weight = self._bchmrk_weight.reindex(self._signal.index, fill_value=0)
                lin_coeffs = self._signal.values + 2.0 * np.dot(bchmrk_weight.values, portfolio_risk)
            else:
                lin_coeffs = self._signal.values
            self._c.objective.set_linear(list(zip(self._signal.index.tolist(), lin_coeffs)))
        else:
            lin_coeffs = self._signal.values
            self._c.objective.set_linear(list(zip(self._signal.index.tolist(), lin_coeffs)))

    def solve(self):
        self._create_obj()
        self._c.solve()

        self.optimal = self._c.solution.get_status() == 1
        self.solution_status = self._c.solution.status[self._c.solution.get_status()]
        self.solution_value = self._c.solution.get_objective_value()

        if self.optimal:
            self.asset['optimal_weight'] = self._c.solution.get_values()
