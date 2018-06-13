# coding: utf-8

"""
股票组合优化器, 使用兴业金工数据
"""

import numpy as np
import pandas as pd
import cplex as cpx
import os
import copy
import sys
import warnings
from FactorLib.data_source.base_data_source_h5 import data_source
from FactorLib.utils.tool_funcs import get_available_names
from QuantLib.stockFilter import suspendtrading, typical, _intersection
from .riskmodel_data_source import RiskDataSource
from itertools import combinations_with_replacement
from QuantLib.utils import StandardByQT


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

    def __init__(self, signal, target_ids, date, ds_name, asset=None, active=False, benchmark='000905', risk_mul=0,
                 **kwargs):
        """初始化优化器"""
        self.target_ids = [x.encode('utf8') for x in list(set(signal.index.values).intersection(set(target_ids)))]
        self.target_ids.sort()
        self._date = date

        if isinstance(ds_name, str):
            self._rskds = RiskDataSource(ds_name)
        else:
            self._rskds = ds_name
        self._active = active
        self._benchmark = benchmark
        self._asset = asset
        self._riskmul = risk_mul
        self.optimal = False
        self.solution_status = ""
        self.solution_value = None
        self._internal_limit = None
        self._signalrisk = None
        self._nontrad_pre = None
        self._nontrad_target = None
        self._bchmrk_notin_tar = None
        self._preid_notin_tar = None
        self._signal = None
        self.estu = self._rskds.load_factors(['Estu'], dates=[self._date]).reset_index(level='date', drop=True)

        self._create_signal(signal, asset)
        # self._allids = self._signal.index.tolist()
        self._init_opt_result(asset)
        self._init_opt_prob()
        self.names_used = []
        self.auxiliary_vars = []
        self.diversion_limit_num = 0

    def _create_signal(self, signal, asset):
        """重新整合参数，确定需要优化的股票池以及对应的信号
        这里需要划分几个概念：
            1. 目标股票池，初始权重设置不为零的股票，是待优化的全样本
            2. 基准指数池，需要用户传入，包括成分股代码和权重
            3. 上期股票池，也需要用户传入，通常是为了设定换手率限制
            4. 全部股票池，通常是以上所有股票的并集
            需要优化的股票池应该设定为目标股票和基准指数的并集，不在目标股票池的基准指数
        成分股设定的权重是零。
            另外，上期持仓中停牌的股票、目标股票池中停牌、ST的股票都需要事先设定权重是零
        或者期初权重
        """
        # 基准指数的成分股和权重，如果是NULL代表基准成分股的权重都是零。
        if self._benchmark == 'NULL':
            self._bchmrk_weight = pd.Series(np.zeros(len(self.target_ids)), index=self.target_ids, name='NULL')
        else:
            if isinstance(self._benchmark, pd.Series):    # 自定义的基准，格式是pd.Series(IDs:weight)
                bchmrk_weight = self._benchmark
            else:
                bchmrk_weight = data_source.sector.get_index_weight(
                    ids=self._benchmark, dates=[self._date]).reset_index(level='date', drop=True).iloc[:, 0]
            # bchmrk_weight = bchmrk_weight[bchmrk_weight.index.intersection(self.estu.index)]
            bchmrk_weight = bchmrk_weight / bchmrk_weight.sum()
            self._bchmrk_weight = bchmrk_weight

        # 找出上期持仓中停牌的股票
        preids = []
        nontrad_pre = []
        if asset is not None:
            preids = [x.encode('utf8') for x in asset[asset != 0.0].index.tolist()]
            if preids:
                nontrad_pre = suspendtrading(preids, self._date)
        
        # 找出不在target_ids中的基准成分股
        bchmrk_notin_tar = list(set(self._bchmrk_weight.index.tolist()).difference(set(self.target_ids)))
        bchmrk_notin_tar = [x.encode('utf8') for x in bchmrk_notin_tar]
        
        # 找出target_ids中停牌的股票
        nontrad_target = suspendtrading(self.target_ids, self._date)
        nontrad_target = [x.encode('utf8') for x in nontrad_target]
        
        # 全部股票池包含了三部分的股票：目标股票池、基准指数以及上期股票池中停牌的股票
        allids = list(set(self.target_ids+self._bchmrk_weight.index.tolist()+nontrad_pre+preids))
        allids = [x.encode('utf8') for x in allids]
        allids.sort()

        self._allids = allids
        self._bchmrk_weight = self._bchmrk_weight.reindex(allids, fill_value=0.0)
        self._signal = signal.reindex(allids, fill_value=0.0)
        if self._asset is not None:
            self._asset = self._asset.reindex(allids, fill_value=0.0)
        self._nontrad_pre = nontrad_pre
        self._nontrad_target = nontrad_target
        self._bchmrk_notin_tar = bchmrk_notin_tar
        self._preid_notin_tar = [x.encode('utf8') if isinstance(x, unicode) else x
                                 for x in list(set(preids) - set(self.target_ids))]
        # return nontrad_pre, nontrad_target, bchmrk_notin_tar, signal.reindex(allids, fill_value=.0)

    def _init_opt_prob(self):
        """
        初始化优化问题
        设置了现金中性、禁止做空、停牌无法交易等限制.此函数还应该生成一个signal序列， 并且包含如下的股票池：
        1. target_ids
        2. benchmark_ids
        3. 上一期持仓中停牌的股票，因为这部分股票无法交易，肯定会留在组合当中。

        在限制条件中，需要把上期持仓停牌的股票权重设置为上期的权重；不在target_ids中的基准指数成分股设置为零；
        target_ids中停牌的股票设置为零。
        """

        nvar = self._allids
        self._c = cpx.Cplex()
        self._c.variables.add(names=nvar)

        # 禁止做空
        self._c.variables.set_lower_bounds([(x, 0) for x in nvar])

        # 加入线性限制
        # 上期持仓中停牌股票的权重保持不变；不在目标股票池中的股票权重设为零；
        # 目标持仓中停牌的股票权重亦为零
        limit_ids = list(set(self._nontrad_pre+self._bchmrk_notin_tar+self._nontrad_target+self._preid_notin_tar))
        if limit_ids:
            limit_values = pd.Series(np.zeros(len(limit_ids)), index=limit_ids)
            if self._nontrad_pre:
                limit_values.loc[self._nontrad_pre] = self._asset.loc[self._nontrad_pre]
            if self._preid_notin_tar:
                limit_values.loc[self._preid_notin_tar] = self._asset.loc[self._preid_notin_tar]
            lin_exprs = []
            rhs = []
            names = []
            senses = ['E']*len(limit_values)
            for i, x in limit_values.iteritems():
                if isinstance(i, unicode):
                    i = i.encode('utf8')
                lin_exprs.append([[i], [1]])
                rhs.append(x)
                names.append('notrading_%s' % i)
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)
            self._internal_limit = limit_values
        # 上期持仓中涨停股票不能买入，跌停股票不能卖出。
        pre_down = []
        pre_up = []
        if self._asset is not None:
            uplimit = data_source.sector.get_uplimit(dates=[self._date])
            downlimit = data_source.sector.get_downlimit(dates=[self._date])
            pre_up = list(set(uplimit.index.get_level_values('IDs'))\
                          .intersection(self._asset[self._asset != 0.0].index.get_level_values('IDs')))
            pre_down = list(set(downlimit.index.get_level_values('IDs'))\
                          .intersection(self._asset[self._asset != 0.0].index.get_level_values('IDs')))
            lin_exprs = []
            rhs = []
            names = []
            senses = []
            for i, x, v in zip(pre_up+pre_down, ['L']*len(pre_up)+['G']*len(pre_down),
                               self._asset[pre_up+pre_down].values):
                if isinstance(i, unicode):
                    i = i.encode('utf8')
                lin_exprs.append([[i], [1]])
                senses.append(x)
                rhs.append(v)
                names.append('tradelimit_%s' % i)
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)

        # 现金中性, target_ids和nontrade_pre中的股票权重之和是1
        trading_stocks = list(set(self.target_ids).union(set(self._preid_notin_tar)))
        trading_stocks = [x.encode('utf8') if isinstance(x, unicode) else x for x in trading_stocks]
        self._c.linear_constraints.add(lin_expr=[[trading_stocks, [1]*len(trading_stocks)]],
                                       senses=['E'],
                                       rhs=[1],
                                       names=['csh_ntrl']
                                       )

        self._c.set_warning_stream(sys.stdout)
        self._c.set_log_stream(None)
        self._c.set_error_stream(None)
        self._c.set_results_stream(None)

    def _init_opt_result(self, asset):
        # 初始化优化结果，在上一期的权重里添加一列optimal_weight。若求解成功，则该列为优化后的权重，否则，仍然返回上期
        # 权重。
        self.opt_rslt = pd.DataFrame(np.ones(len(self._signal)) / len(self._signal),
                                     index=pd.MultiIndex.from_product([[self._date], self._signal.index], names=['date', 'IDs']),
                                     columns=['optimal_weight'])
        if self._asset is None:
            self.opt_rslt['previous_weight'] = 0.0
        else:
            self.opt_rslt['previous_weight'] = self._asset.reindex(self._signal.index, fill_value=0.0).values

    def _add_style_cons(self, style_dict, active=True, sense='E', std_qt=False,
                        limit_type='value', **kwargs):
        """
        设置风格限制

        Parameters:
        ==========
        style_dict: dict
            以字典形式进行风格约束，{key: style_name, value: constraint}
            constraint允许两种形式, 单值传入和列表传入。列表传入代表的是上下限,
            单值传入与sense一致
        active: bool
            相对于基准的主动暴露，默认为True。
        std_qt: bool
            事先将风格因子分位数标准化。
        limit_type: str
            value or std value代表绝对值的偏离；std代表与基准偏离的标准差。
        
        Examples:
        ==========
        >>> _add_style_cons({'Size':[0.0, 0.2]}, active=True)
        >>> _add_style_cons({'Size': 0.2}, active=True, sense='G')
        """
        style_data = self._get_style_factors(list(style_dict), std_qt=std_qt)
        style_dict2 = {}
        style_dict = style_dict.copy()
        for k, v in style_dict.items():
            if isinstance(v, list):
                assert v[0] <= v[1]
                style_dict2[k] = style_dict.pop(k)
        if style_dict2:
            cons2 = pd.DataFrame(style_dict2).T.rename(columns={0: 'min', 1: 'max'})
            if active:
                bchmk_style = self._prepare_benchmark_style(list(style_dict2), data=style_data)
                if limit_type == 'std':
                    std = self._prepare_benchmark_style_std(
                        list(style_dict2), data=style_data)
                    cons2 = cons2.mul(std, axis='index')
                cons2 = cons2.add(bchmk_style, axis=0)
            portf_style = self._prepare_portfolio_style(list(style_dict2), data=style_data)
            if np.any(pd.isnull(portf_style)) or np.any(pd.isnull(cons2)):
                warnings.warn("设置风格敞口时, 风格因子数据存在缺失值!")
                na_stocks = portf_style[portf_style.isnull().any(axis=1)]
                limit_values = pd.Series(np.zeros(len(na_stocks)), index=na_stocks.index)
                lin_exprs = []
                rhs = []
                names = []
                senses = ['E'] * len(na_stocks)
                for i in na_stocks.index.values:
                    lin_exprs.append([[i.encode('utf8')], [1]])
                    rhs.append(0.0)
                    available_name = get_available_names('notrading_%s' % i, self.names_used)
                    names.append(available_name)
                    self.names_used += [available_name]
                self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)
                self._internal_limit = self._internal_limit.append(limit_values)
                portf_style.fillna(0.0, inplace=True)
            lin_exprs = []
            rhs_min = []
            rhs_max = []
            names = []
            for style, value in portf_style.iteritems():
                lin_exprs.append([[x.encode('utf8') for x in value.index.tolist()], value.values.tolist()])
                rhs_min.append(cons2.loc[style, 'min'])
                rhs_max.append(cons2.loc[style, 'max'])
                available_name = get_available_names(style, self.names_used)
                names.append(available_name)
                self.names_used += [available_name]
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=['G']*len(cons2), rhs=rhs_min, names=[x+'1' for x in names])
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=['L']*len(cons2), rhs=rhs_max, names=[x+'2' for x in names])
        if style_dict:
            cons = pd.Series(style_dict)
            if active:
                bchmk_style = self._prepare_benchmark_style(list(style_dict), data=style_data)
                if limit_type == 'std':
                    std = self._prepare_benchmark_style_std(
                        list(style_dict2), data=style_data)
                    cons = cons.mul(std, axis='index')
                cons = cons + bchmk_style
            portf_style = self._prepare_portfolio_style(list(style_dict), data=style_data)
            if np.any(pd.isnull(portf_style)) or np.any(pd.isnull(cons)):
                warnings.warn("设置风格敞口时, 风格因子数据存在缺失值!")
                na_stocks = portf_style[portf_style.isnull().any(axis=1)]
                limit_values = pd.Series(np.zeros(len(na_stocks)), index=na_stocks.index)
                lin_exprs = []
                rhs = []
                names = []
                senses = ['E'] * len(na_stocks)
                for i in na_stocks.index.values:
                    lin_exprs.append([[i.encode('utf8')], [1]])
                    rhs.append(0.0)
                    available_name = get_available_names('notrading_%s' % i, self.names_used)
                    names.append(available_name)
                    self.names_used += [available_name]
                self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)
                self._internal_limit = self._internal_limit.append(limit_values)
                portf_style.fillna(0.0, inplace=True)
            lin_exprs = []
            rhs = []
            senses = [sense] * len(cons)
            names = []
            for style, value in portf_style.iteritems():
                lin_exprs.append([[x.encode('utf8') for x in value.index.tolist()], value.values.tolist()])
                rhs.append(cons[style])
                available_name = get_available_names(style, self.names_used)
                names.append(available_name)
                self.names_used += [available_name]
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)

    def _add_industry_cons(self, industry_dict=None, active=True, sense='E'):
        """
        添加行业约束

        Parameters:
        ==========
        industry_dict: dict
            以字典形式进行行业约束，{key: industry_name, value: constraint} \n
        active: bool
            相对于基准的主动暴露，默认为True。

        Examples:
        >>> self._add_industry_cons({'Banks': [0.0, 0.1]})
        >>> self._add_industry_cons({'Banks': 0.2}, sense='L')
        """

        industry_dict2 = {}
        industry_dict = industry_dict.copy()
        indu_bchmrk = self._prepare_benchmark_indu()
        if industry_dict is None:
            cons = indu_bchmrk
            portf_indu = self._prepare_portfolio_indu().reindex(columns=cons.index)
            if np.any(pd.isnull(portf_indu)) or np.any(pd.isnull(cons)):
                raise ValueError("行业变量存在缺失值！")
            lin_exprs = []
            rhs = []
            senses = ['E'] * len(cons)
            names = []
            for indu, value in portf_indu.iteritems():
                lin_exprs.append([[x.encode('utf8') for x in value.index.tolist()], value.values.tolist()])
                rhs.append(cons[indu])
                available_name = get_available_names(indu, self.names_used)
                names.append(available_name)
                self.names_used += [available_name]
            self._c.linear_constraints.add(lin_expr=lin_exprs, rhs=rhs, senses=senses, names=names)
        else:
            for k, v in industry_dict.items():
                if isinstance(v, list):
                    industry_dict2[k] = industry_dict.pop(k)
            if industry_dict:
                cons = pd.Series(industry_dict)
                if active:
                    cons = cons + indu_bchmrk.reindex(cons.index)
                # cons = indu_bchmrk.update(cons)
                portf_indu = self._prepare_portfolio_indu().reindex(columns=cons.index)
                if np.any(pd.isnull(portf_indu)) or np.any(pd.isnull(cons)):
                    raise ValueError("行业变量存在缺失值！")
                lin_exprs = []
                rhs = []
                senses = [sense] * len(cons)
                names = []
                for indu, value in portf_indu.iteritems():
                    lin_exprs.append([[x.encode('utf8') for x in value.index.tolist()], value.values.tolist()])
                    rhs.append(cons[indu])
                    available_name = get_available_names(indu, self.names_used)
                    names.append(available_name)
                    self.names_used += [available_name]
                self._c.linear_constraints.add(lin_expr=lin_exprs, rhs=rhs, senses=senses, names=names)
            if industry_dict2:
                cons2 = pd.DataFrame(industry_dict2).T.rename(columns={0: 'min', 1: 'max'})
                if active:
                    cons2 = cons2.add(indu_bchmrk.reindex(cons2.index), axis=0)
                portf_indu = self._prepare_portfolio_indu().reindex(columns=cons2.index)
                if np.any(pd.isnull(portf_indu)) or np.any(pd.isnull(cons2)):
                    raise ValueError("行业变量存在缺失值！")
                lin_exprs = []
                rhs_min = []
                rhs_max = []
                names = []
                for indu, value in portf_indu.iteritems():
                    lin_exprs.append([[x.encode('utf8') for x in value.index.tolist()], value.values.tolist()])
                    rhs_min.append(cons2.loc[indu, 'min'])
                    rhs_max.append(cons2.loc[indu, 'max'])
                    available_name = get_available_names(indu, self.names_used)
                    names.append(available_name)
                    self.names_used += [available_name]
                self._c.linear_constraints.add(lin_expr=lin_exprs, rhs=rhs_min, senses=['G']*len(cons2),
                                               names=[x+'1' for x in names])
                self._c.linear_constraints.add(lin_expr=lin_exprs, rhs=rhs_max, senses=['L']*len(cons2),
                                               names=[x+'2' for x in names])

    def _add_stock_limit(self, default_min=0.0, default_max=1.0, active=False,
                         abs_max=0.07, multiple=None):
        """
        添加个股权重的上下限

        Parameters:
        =================
        default_min : float or list of tuples
            个股权重上限 。 float格式会广播到all_ids
            list的格式 : [(ID, min_limit)...]
        default_max : same as default_min
        active : bool
            是否相对偏离。选择相对偏离时，default_min和default_max就代表了相对基准指数
            的超额权重。
        abs_max : float
            在选择相对偏离模式时，可以给个股设置一个绝对权重上限。
            即个股权重上限是min(default_max+weight_in_benchmark, abs_max)
        multiple : float
            个股的权重上限是min(default_max+weight_in_benchmark, weight_in_benchmark*multiple)
        """

        # assert ((default_min < 0) or (default_max > 1))
        nvar = self._signal.index.tolist()
        if self._internal_limit is not None:
            nvar = list(set(nvar).difference(set(self._internal_limit.index.tolist())))
        if active:
            benchmark_weight = self._bchmrk_weight.loc[nvar]
            if isinstance(default_min, list):
                default_min = pd.Series(default_min) + benchmark_weight
            else:
                default_min = default_min + benchmark_weight
            default_min.loc[default_min < 0] = 0
            min_limit = list(zip(default_min.index, default_min.values))
            if isinstance(default_max, list):
                default_max = pd.Series(default_max) + benchmark_weight
            else:
                default_max = default_max + benchmark_weight
            if multiple is not None:
                temp_max = benchmark_weight * multiple
                default_max = default_max.where(default_max < temp_max, temp_max)
            default_max.loc[default_max < 0] = 0
            default_max.loc[default_max > abs_max] = abs_max
            max_limit = list(zip(default_max.index, default_max.values))
        else:
            if isinstance(default_min, list):
                min_limit = [(k, v) for k, v in default_min if k in nvar]
            else:
                min_limit = list(zip(nvar, [default_min]*len(nvar)))
            if isinstance(default_max, list):
                max_limit = [(k, v) for k, v in default_max if k in nvar]
            else:
                max_limit = list(zip(nvar, [default_max]*len(nvar)))
        self._c.variables.set_lower_bounds(min_limit)
        self._c.variables.set_upper_bounds(max_limit)

    def _prepare_portfolio_indu(self):
        """
        组合的行业因子

        Returns:
        ========
        data: DataFrame
            DataFrame(index:[IDs], columns:[industries])
        """
        portfolio = self._signal.index.tolist()
        data = self._rskds.load_industry(ids=portfolio, dates=[self._date]).reset_index(level='date', drop=True)
        return data.reindex(self._signal.index)

    def _prepare_benchmark_indu(self):
        """
        基准的行业因子
        """
        weight = self._bchmrk_weight
        indu = self._rskds.load_industry(ids=weight.index.tolist(), dates=[self._date]).reset_index(level='date', drop=True)
        if weight.sum() > 0:
            indu = indu.mul(weight, axis='index').sum() / weight.sum()
        else:
            indu = indu.mul(weight, axis='index').sum()
        return indu

    def _get_style_factors(self, styles, std_qt=False):
        """提取风格因子数据

        Return
        -----------------
        data: pd.DataFrame(index=[date, IDs], values=style_factor)
        """
        data = self._rskds.load_style_factor(styles, dates=[self._date])
        if std_qt:
            data = data.apply(lambda x: StandardByQT(x.to_frame(), x.name))
        return data

    def _prepare_portfolio_style(self, styles, data=None, **kwargs):
        """
        组合的风格因子

        Returns:
        ========
        data: DataFrame
            DataFrame(index:[factor_names], columns:[IDs])
        """
        if data is None:
            portfolio = self._signal.index.tolist()
            data = self._rskds.load_style_factor(styles, ids=portfolio, dates=[self._date]) \
                              .reset_index(level='date', drop=True)
            if kwargs.get('std_qt', False):
                data = data.apply(lambda x: StandardByQT(x.to_frame(), x.name))
        else:
            data = data.reindex(columns=styles).reset_index(level='date', drop=True)
        return data.reindex(self._signal.index)

    def _prepare_benchmark_style(self, styles, data=None, **kwargs):
        """
        基准的风格因子

        Return:
        ======
        style_benchmark: Series
        基准的风格因子，Series(index:[factor_names], values:style_value)
        """
        weight = self._bchmrk_weight
        members = weight.index.tolist()
        if data is None:
            style_data = self._rskds.load_style_factor(styles, ids=members, dates=[self._date])\
                .reset_index(level='date', drop=True)
            if kwargs.get('std_qt', False):
                style_data = style_data.apply(lambda x: StandardByQT(x.to_frame(), x.name))
        else:
            style_data = data.reindex(columns=styles).reset_index(level='date', drop=True)

        if weight.sum() > 0:
            style_benchmark = style_data.mul(weight, axis='index').sum() / weight.sum()
        else:
            style_benchmark = style_data.mul(weight, axis='index').sum()
        return style_benchmark
    
    def _prepare_benchmark_style_std(self, styles, data=None, **kwargs):
        """
        基准风格因子的标准差
        """
        weight = self._bchmrk_weight / self._bchmrk_weight.sum()
        members = weight.index.tolist()
        if data is None:
            style_data = self._rskds.load_style_factor(styles, ids=members, dates=[self._date])\
                .reset_index(level='date', drop=True)
            if kwargs.get('std_qt', False):
                style_data = style_data.apply(lambda x: StandardByQT(x.to_frame(), x.name))
        else:
            style_data = data.reindex(columns=styles).reset_index(level='date', drop=True)
        style_data = style_data.fillna(style_data.mean())
        std = ((style_data - style_data.mean()) ** 2).mul(weight, axis='index').sum()
        return np.sqrt(std)

    def _add_trckerr(self, target_err, **kwargs):
        if self._active:
            raise ValueError("跟踪误差与风险嫌恶项重复设置！")
        bchmrk_weight = self._bchmrk_weight
        sigma = self._load_portfolio_risk()
        bchmrk_risk = np.dot(np.dot(bchmrk_weight.values, sigma), bchmrk_weight.values)
        qlin = [self._signal.index.tolist(), (-2.0 * np.dot(bchmrk_weight.values, sigma)).tolist()]
        qvar_expr = [list(x) for x in zip(*combinations_with_replacement(self._allids, 2))]
        qvar_mul = np.triu(sigma) + np.tril(sigma, -1).T
        # qvar_mul = qvar_mul.ravel()[np.flatnonzero(qvar_mul)]
        qvar_mul = qvar_mul[np.triu_indices_from(qvar_mul)]
        quad = qvar_expr + [qvar_mul.tolist()]

        self._c.quadratic_constraints.add(lin_expr=qlin, quad_expr=quad, sense='L', rhs=-bchmrk_risk+target_err,
                                          name=get_available_names('tracking_error', self.names_used))
        self.names_used += [get_available_names('tracking_error', self.names_used)]
        self._signalrisk = sigma

    def _add_userlimit(self, user_conf, **kwargs):
        """添加用户自定义的因子限制
        user_conf: dict
        自定义限制条件 :
            factor_data : pd.DataFrame
            自定义风险因子的数据, 每一列是一个因子，[date, IDs]为索引

            factor_name : str
            若factor_data为None, factor_name和factor_dir必须非空, h5db
            会从中提取数据

            factor_dir : str
            若factor_data为None, factor_name和factor_dir必须非空, h5db
            会从中提取数据

            limit : float or list of floats or dict
            每个风险因子的限制值, 若limit是列表，其长度必须与因子个数相同
            若limit是字典型, key值为factor_data中的列名, value是列表或者
            scalar

            standard : bool
            在加入到优化器之前是否对输入的因子进行QT标准化

        kwargs: dict
            active : bool
            限制条件是否是相对行业的限制，默认为True

            sense : str
            限制类型： 'E': equal / 'G': greater than / 'L': lower than
        """
        if user_conf.get('factor_data', pd.DataFrame()).empty:
            factor_name = user_conf.get('factor_name')
            factor_dir = user_conf.get('factor_dir')
            factor_data = data_source.load_factor(factor_name, factor_dir, dates=[self._date])
        else:
            factor_data = user_conf.get('factor_data')
            factor_name = factor_data.columns
        limit = user_conf.get('limit')
        is_standard = user_conf.get('standard', False)
        is_active = kwargs.get('active', False)
        sense = kwargs.get('sense', 'E')
        limit_type = kwargs.get('limit_type', 'value')

        if isinstance(factor_data, pd.Series):
            factor_data = factor_data.to_frame(factor_name)
            factor_name = [factor_name]

        limit_min = {}
        limit_max = {}
        limit_sense = {}
        if isinstance(limit, dict):
            for k, v in limit.items():
                if isinstance(v, list):
                    assert v[0] <= v[1]
                    limit_min[k] = v[0]
                    limit_max[k] = v[1]
                elif isinstance(v, (int, float)):
                    limit_sense[k] = float(v)
                else:
                    raise ValueError("自定义因子敞口限定值不合法!")
        if isinstance(limit, (int, float)):
            limit_sense = {x: limit for x in factor_name}
        if isinstance(limit, list):
            if len(limit) != len(factor_name):
                raise ValueError("limit dimension dose not match factor dimension")
            limit_sense = {x: y for x, y in zip(factor_name, limit)}

        if isinstance(sense, str):
            sense = [sense] * len(limit_sense)
        else:
            if len(sense) != len(limit_sense):
                raise ValueError("sense dimension dose not match factor dimension")

        for f, l in limit_min.items():
            if is_standard:
                factor_data2 = StandardByQT(factor_data, f).loc[self._date].reindex(self._allids, fill_value=0.0)
            else:
                factor_data2 = factor_data.loc[self._date, f].reindex(self._allids, fill_value=0.0)
            if is_active:
                if limit_type == 'std':
                    l *= self._prepare_benchmark_userexpo_std(factor_data2)
                l += self._prepare_benchmark_userexpo(factor_data2)
            portfolio_factor = factor_data2.loc[self._allids]
            if np.any(np.isnan(portfolio_factor.values)):
                raise ValueError("自定义因子因子数据存在缺失值!")
            lin_expr = []
            sense = ['G']
            rhs = [l]
            name = [get_available_names(x, self.names_used) for x in ['user_%s' % f]]
            lin_expr.append([portfolio_factor.index.tolist(), portfolio_factor.values.tolist()])
            self._c.linear_constraints.add(lin_expr=lin_expr, senses=sense, rhs=rhs, names=name)
            self.names_used += name

        for f, l in limit_max.items():
            if is_standard:
                factor_data2 = StandardByQT(factor_data, f).loc[self._date].reindex(self._allids, fill_value=0.0)
            else:
                factor_data2 = factor_data.loc[self._date, f].reindex(self._allids, fill_value=0.0)
            if is_active:
                if limit_type == 'std':
                    l *= self._prepare_benchmark_userexpo_std(factor_data2)
                l += self._prepare_benchmark_userexpo(factor_data2)
            portfolio_factor = factor_data2.loc[self._allids]
            if np.any(np.isnan(portfolio_factor.values)):
                raise ValueError("自定义因子因子数据存在缺失值!")
            lin_expr = []
            sense = ['L']
            rhs = [l]
            name = [get_available_names(x, self.names_used) for x in ['user_%s' % f]]
            lin_expr.append([portfolio_factor.index.tolist(), portfolio_factor.values.tolist()])
            self._c.linear_constraints.add(lin_expr=lin_expr, senses=sense, rhs=rhs, names=name)
            self.names_used += name

        for f, s in zip(limit_sense, sense):
            l = limit_sense[f]
            if is_standard:
                factor_data2 = StandardByQT(factor_data, f).loc[self._date].reindex(self._allids, fill_value=0.0)
            else:
                factor_data2 = factor_data.loc[self._date, f].reindex(self._allids, fill_value=0.0)
            if is_active:
                if limit_type == 'std':
                    l *= self._prepare_benchmark_userexpo_std(factor_data2)
                l += self._prepare_benchmark_userexpo(factor_data2)
            portfolio_factor = factor_data2.loc[self._allids]
            if np.any(np.isnan(portfolio_factor.values)):
                raise ValueError("自定义因子因子数据存在缺失值!")
            lin_expr = []
            sense = [s]
            rhs = [l]
            name = [get_available_names(x, self.names_used) for x in ['user_%s' % f]]
            lin_expr.append([portfolio_factor.index.tolist(), portfolio_factor.values.tolist()])
            self._c.linear_constraints.add(lin_expr=lin_expr, senses=sense, rhs=rhs, names=name)
            self.names_used += name

    def _prepare_benchmark_userexpo(self, data):
        """计算基准指数的用户因子暴露"""
        weight = self._bchmrk_weight
        members = weight.index.tolist()
        userexpo_benchmark = data.loc[members].mul(weight, axis='index').sum() / weight.sum()
        return userexpo_benchmark
    
    def _prepare_benchmark_userexpo_std(self, data):
        """计算基准指数的用户因子暴露的标准差"""
        weight = self._bchmrk_weight / self._bchmrk_weight.sum()
        members = weight.index.tolist()
        userexpo_benchmark = data.reindex(members)
        userexpo_benchmark = userexpo_benchmark.fillna(
            userexpo_benchmark.mean())
        std = ((userexpo_benchmark - userexpo_benchmark.mean())**2).mul(
            weight, axis='index').sum()
        return np.sqrt(std)

    def _load_portfolio_risk(self):
        """
        加载股票组合风险矩阵
        """
        func_name = 'load_%s_riskmatrix' % self._rskds._name
        sigma = getattr(self._rskds, func_name)(dates=[self._date])[self._date].loc[self._allids, self._allids]
        if np.any(pd.isnull(sigma)):
            warnings.warn("股票组合的风险存在缺失值！")
            na_stocks = sigma[sigma.isnull().all(axis=1)]
            limit_values = pd.Series(np.zeros(len(na_stocks)), index=na_stocks.index)
            lin_exprs = []
            rhs = []
            names = []
            senses = ['E'] * len(na_stocks)
            for i in na_stocks.index.values:
                lin_exprs.append([[i.encode('utf8')], [1]])
                rhs.append(0.0)
                names.append('notrading_%s' % i)
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)
            self._internal_limit = self._internal_limit.append(limit_values)
            sigma.fillna(0.0, inplace=True)
        return sigma.values

    def _add_diversion_limit(self, base, limit):
        """添加与基准偏离的约束
        
        Paramters:
        ===========
        base : str/list/pd.Series
            计算偏离度的基准
        limit : float
            设定的偏离度上限
        """
        if isinstance(base, str):
            assert base == 'Benchmark'
            base_weight = self._bchmrk_weight[~self._bchmrk_weight.index.isin(self._nontrad_target)]
        elif isinstance(base, pd.Series):
            base_weight = base[~base.index.isin(self._nontrad_target)]
        elif isinstance(base, (list, tuple, np.array)):
            base_weight = self._bchmrk_weight[
                list((set(self.target_ids)-set(self._nontrad_target)).intersection(set(base)))]
        else:
            raise ValueError("不支持的base设定")
        limit_ids = list(base_weight.index.values)
        if isinstance(limit_ids[0], unicode):
            limit_ids = [x.encode('utf8') for x in limit_ids]
        base_weight_values = base_weight.values.tolist()

        auxiliary_vars = []
        for i in limit_ids:
            a = get_available_names('a_%s' % i, self.auxiliary_vars)
            auxiliary_vars.append(a)
            self.auxiliary_vars.append(a)

        # 辅助变量的绝对值之和小于等于limit
        if isinstance(auxiliary_vars[0], unicode):
            auxiliary_vars = [x.encode('utf8') for x in auxiliary_vars]
        self._c.variables.add(names=auxiliary_vars)
        self._c.linear_constraints.add(lin_expr=[[auxiliary_vars, [1]*len(auxiliary_vars)]], senses=['L'], rhs=[limit],
                                       names=[get_available_names('diversion', self.names_used)]
                                       )
        self.names_used.append(get_available_names('diversion', self.names_used))

        # for a, b, v in zip(auxiliary_vars, limit_ids, base_weight_values):
        #     name1 = get_available_names('diversion_abs', self.names_used)
        #     self.names_used.append(name1)
        #     name2 = get_available_names('diversion_abs', self.names_used)
        #     self.names_used.append(name2)
        #     self._c.linear_constraints.add(lin_expr=[[[a, b], [1, -1]],[[a, b], [1, 1]]], senses=['G','G'], rhs=[-v, v],
        #         names=[name1, name2])

        names = np.arange(self.diversion_limit_num+1,
                          self.diversion_limit_num+1+len(limit_ids)*2).astype('str').tolist()
        v = pd.DataFrame({'auxiliary_vars': auxiliary_vars, 'limit_ids': limit_ids})[['auxiliary_vars', 'limit_ids']]
        v['v1'] = 1
        v['v2'] = -1
        lin_expr = v.values.reshape((-1, 2, 2)).tolist()
        senses = ['G'] * len(limit_ids)
        rhs = -np.asarray(base_weight_values)
        self._c.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs.tolist(),
                                       names=names[:len(limit_ids)])
        v['v2'] = 1
        lin_expr = v.values.reshape((-1, 2, 2)).tolist()
        rhs *= -1
        self._c.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs.tolist(),
                                       names=names[len(limit_ids):])
        self.diversion_limit_num += (len(limit_ids) * 2)

    def _add_turnover_limit(self, limit):
        """添加换手率限制"""
        if self._asset is None:
            raise ValueError("设置换手率时，上期资产权重不能为空！")
        self._add_diversion_limit(self._asset, limit)

    def add_constraint(self, key, *args, **kwargs):
        """
        设置限制条件
        目前支持:
            1. 风格敞口限制，key='Style'.敞口限定值以字典的形式传入，字典的key为因子名称，value为限定值。
            2. 行业敞口限制，key='Indu'.默认所有行业都限定为中性，把需要显式设定的行业敞口按照字典形式传入。
            3. 跟踪误差限制，key='TrackingError', value= target_tracking_error.
            4. 个股权重限制, key='StockLimit'. 个股上下限以tuple形式传入。
            5. 基准偏离限制, key='Diversion'. Base代表基准权重，limit代表上限
            6. 换手率限制, key='Tunover'

        Parameters:
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
        try:
            key, num = key.split('_')
        except ValueError as e:
            pass

        if key == 'Style':
            self._add_style_cons(*args, **kwargs)
        elif key == 'Indu':
            self._add_industry_cons(*args, **kwargs)
        elif key == 'TrackingError':
            self._add_trckerr(*args, **kwargs)
        elif key == 'StockLimit':
            self._add_stock_limit(*args, **kwargs)
        elif key == 'UserLimit':
            self._add_userlimit(*args, **kwargs)
        elif key == 'Diversion':
            self._add_diversion_limit(*args, **kwargs)
        elif key == 'Turnover':
            self._add_turnover_limit(*args, **kwargs)
        else:
            raise NotImplementedError("不支持的限制类型:%s" % key)

    def _create_obj(self):
        """
        生成目标函数
        """
        if isinstance(self._signal, pd.DataFrame):
            self._signal = self._signal.iloc[:, 0]
        if np.any(pd.isnull(self._signal)):
            warnings.warn("组合信号存在缺失值！")
            # 把信号缺失的股票的权重设为零
            na_stocks = self._signal[self._signal.isnull()]
            limit_values = pd.Series(np.zeros(len(na_stocks)), index=na_stocks.index)
            lin_exprs = []
            rhs = []
            names = []
            senses = ['E'] * len(na_stocks)
            for i in na_stocks.index.values:
                lin_exprs.append([[i.encode('utf8')], [1]])
                rhs.append(0.0)
                names.append('notrading_%s' % i)
            self._c.linear_constraints.add(lin_expr=lin_exprs, senses=senses, rhs=rhs, names=names)
            self._internal_limit = self._internal_limit.append(limit_values)
            self._signal.fillna(0, inplace=True)

        self._c.objective.set_sense(self._c.objective.sense.maximize)
        if self._riskmul != 0:
            bchmrk_weight = self._bchmrk_weight
            sigma = self._load_portfolio_risk() * self._riskmul * 0.5
            qlin = 2.0 * np.dot(bchmrk_weight.values, sigma)
            quad_obj = []
            ind = np.arange(sigma.shape[0]).tolist()
            for i in ind:
                quad_obj.append(cpx.SparsePair(ind, np.around(-1.0 * sigma[i, :], 5).tolist()))
            self._c.objective.set_quadratic(quad_obj)
            lin_coeffs = self._signal.values
            lin_coeffs += qlin
            self._c.objective.set_linear(list(zip(self._allids, lin_coeffs)))
        else:
            lin_coeffs = self._signal.values
            self._c.objective.set_linear(list(zip(self._allids, lin_coeffs)))

    def solve(self):
        """求解"""
        self._create_obj()
        self._c.solve()

        self.optimal = self._c.solution.get_status() == self._c.solution.status.optimal
        self.solution_status = self._c.solution.status[self._c.solution.get_status()]

        if self.optimal:
            self.solution_value = self._c.solution.get_objective_value()
            ans = pd.Series(self._c.solution.get_values(), index=self._c.variables.get_names())
            self.opt_rslt['optimal_weight'] = ans.reindex(self.opt_rslt.index.get_level_values('IDs')).values

    def check_ktt(self):
        """
        检验风格敞口和行业敞口以及跟踪误差是否满足条件
        """
        style_expo = None
        indu_expo = None
        terr_expo = None
        if self.optimal:
            optimal_weight = self.opt_rslt['optimal_weight'].reset_index(level=0, drop=True)
            # 计算风格敞口
            bchmrk_style = self._prepare_benchmark_style('STYLE')
            portf_style = self._prepare_portfolio_style('STYLE').mul(optimal_weight, axis='index').sum()
            style_expo = pd.concat([portf_style, bchmrk_style], axis=1, ignore_index=True).rename(
                columns={0: 'style_portfolio', 1: 'style_benchmark'})
            style_expo['expo'] = style_expo['style_portfolio'] - style_expo['style_benchmark']
            # 计算行业敞口
            bckmrk_indu = self._prepare_benchmark_indu()
            portf_indu = self._prepare_portfolio_indu().mul(optimal_weight, axis='index').sum()
            indu_expo = pd.concat([portf_indu, bckmrk_indu], axis=1, ignore_index=True).rename(
                columns={0: 'indu_portfolio', 1: 'indu_benchmark'})
            indu_expo['expo'] = indu_expo['indu_portfolio'] - indu_expo['indu_benchmark']
            # 跟踪误差敞口
            if self._signalrisk is not None:
                active_weight = optimal_weight - self._bchmrk_weight
                terr_expo = np.dot(np.dot(active_weight.values, self._signalrisk), active_weight.values)
            # 计算用户因子敞口

        return style_expo, indu_expo, terr_expo


class PortfolioOptimizer(object):
    """股票组合优化器"""

    def __init__(self, signal, stock_pool, benchmark, constraints, dates=None, ds_name='xy',
                 boostrap=None, **kwargs):
        self.ds = RiskDataSource(ds_name)
        self.signal, self.stock_pool = self._create_signal_and_stockpool(signal, stock_pool, dates)
        self.benchmark = benchmark
        self.constraints = constraints
        self.result = None
        self.log = {}
        self.boostrap = boostrap
        self.kwargs = kwargs

    def _create_signal_and_stockpool(self, signal, stock_pool, dates):
        from collections import Iterable
        if isinstance(signal, dict):
            signal = data_source.load_factor(signal['factor_name'], signal['factor_dir'], dates=dates).iloc[:, 0]
        elif dates is not None:
            if isinstance(dates, Iterable):
                dates = list(dates)
            signal = signal.loc[dates]
        else:
            raise KeyError("Incorrect Parameters!")
        signal.dropna(inplace=True)

        if isinstance(stock_pool, pd.Series):
            stock_pool_valid = typical(stock_pool.to_frame())
        elif isinstance(stock_pool, str):
            stock_pool = data_source.sector.get_index_members(ids=stock_pool, dates=dates)
            stock_pool_valid = typical(stock_pool)
        else:
            stock_pool_valid = typical(stock_pool)
        stock_pool_valid = _intersection(signal, stock_pool_valid)
        estu = self.ds.load_factors(['Estu'], dates=dates)
        stock_pool_valid = _intersection(estu[estu['Estu'] == 1], stock_pool_valid)
        return signal, stock_pool_valid.reset_index(level=1)['IDs']

    def optimize_weights(self):
        print("optimizing...")
        dates = self.signal.index.get_level_values(0).unique()
        optimal_assets = []
        for i, idate in enumerate(dates):
            print("current date is:%s, progress:%d/%d" % (idate.strftime("%Y-%m-%d"), i + 1, len(dates)))
            signal = self.signal.loc[idate]
            stock_pool = self.stock_pool.loc[idate].tolist()
            if 'TrackingError' in self.constraints and self.ds._name == 'xy':
                ids = self.ds.load_xy_riskmatrix(dates=[idate])[idate].index.tolist()
                stock_pool = list(set(stock_pool).intersection(set(ids)))
            optimizer = Optimizer(signal, stock_pool, idate, self.ds, benchmark=self.benchmark, **self.kwargs)

            for k, v in self.constraints.items():
                optimizer.add_constraint(k, **v)
            optimizer.solve()

            if optimizer.optimal:
                print("%s succeed" % idate.strftime("%Y-%m-%d"))
                opt_weight = optimizer.opt_rslt.loc[optimizer.opt_rslt['optimal_weight'] > 0.001, 'optimal_weight']
                opt_weight = opt_weight / opt_weight.sum()
                optimal_assets.append(opt_weight)
            else:
                print("%s failed:%s" % (idate.strftime("%Y-%m-%d"), optimizer.solution_status))
                self.log[idate.strftime("%Y-%m-%d")] = optimizer.solution_status
                if self.boostrap is not None:
                    constraints = copy.deepcopy(self.constraints)
                    while 1:
                        constraints, flag = self.boostrap(constraints)
                        if not flag:
                            opt_weight = optimizer.opt_rslt.loc[
                                optimizer.opt_rslt['previous_weight'] > 0.001, 'previous_weight']
                            break
                        optimizer = Optimizer(signal, stock_pool, idate, self.ds, benchmark=self.benchmark,
                                              **self.kwargs)
                        for k, v in constraints.items():
                            optimizer.add_constraint(k, **v)
                        optimizer.solve()
                        if optimizer.optimal:
                            print("%s succeed" % idate.strftime("%Y-%m-%d"))
                            opt_weight = optimizer.opt_rslt.loc[
                                optimizer.opt_rslt['optimal_weight'] > 0.001, 'optimal_weight']
                            break
                    if not optimizer.optimal:
                        opt_weight = optimizer.opt_rslt.loc[
                            optimizer.opt_rslt['previous_weight'] > 0.001, 'previous_weight']
                else:
                    opt_weight = optimizer.opt_rslt.loc[optimizer.opt_rslt['previous_weight'] > 0.001, 'previous_weight']
                if not opt_weight.empty:
                    opt_weight = opt_weight / opt_weight.sum()
                    opt_weight.name = 'optimal_weight'
                    optimal_assets.append(opt_weight)
        print("finished...")
        self.result = pd.concat(optimal_assets).to_frame()

    def save_results(self, name, path=None):
        import os
        from FactorLib.utils.tool_funcs import tradecode_to_windcode
        path = os.getcwd() if path is None else path
        weight_data = self.result.reset_index().rename(columns={'optimal_weight': 'Weight'})
        weight_data['IDs'] = weight_data['IDs'].apply(tradecode_to_windcode)
        weight_data.to_csv(os.path.join(path, name), index=False)

