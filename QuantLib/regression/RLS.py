#!python
# -*- coding: utf-8 -*-
#
# RLS.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/4/22 上午11:20:44
import numpy as np
import cvxpy as cvx
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLS, RegressionResults


class RLS(GLS):
    """
    带限制的最小二乘法
    """
    def __init__(self, endog, exog,
                 A=None, Aeq=None,
                 B=None, Bmin=None,
                 Bmax=None, lb=None,
                 ub=None, sigma=None,
                 ):
        N, Q = exog.shape
        self.endog = endog
        self.exog = exog

        self.beta = cvx.Variable(Q)
        cons = []
        if A is not None:
            cons.append(A @ self.beta == Aeq)
        if B is not None:
            if Bmin is not None:
                cons.append(B @ self.beta >= Bmin)
            if Bmax is not None:
                cons.append(B @ self.beta <= Bmax)
        if lb is not None:
            cons.append(self.beta >= lb)
        if ub is not None:
            cons.append(self.beta <= ub)
        self.constraints = cons

        if sigma is not None:
            self.exog *= sigma
            self.endog *= sigma
        self.problem = cvx.Problem(
            cvx.Minimize(
                cvx.sum_squares(
                    self.exog@self.beta - self.endog
                    )
                ),
                self.constraints
            )
        super(RLS, self).__init__(self.endog, self.exog)
    
    def fit(self, **kwargs):
        self.problem.solve(**kwargs)
        if self.problem.status == cvx.OPTIMAL:
            return RegressionResults(self, self.beta.value)
        else:
            raise RuntimeError("Fit Failed.")
