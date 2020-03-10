#!python
# -*- coding: utf-8 -*-
#
# optimize.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/2/17 下午1:08:17
import cvxpy as cvx
import numpy as np
from .popt import absolute_portfolio_optimization, portfolio_optimization


def portfolio_optimization2(r, cov=None, Xf=None, F=None, D=None, wb=None, w0=None, c_=None,
                            lambda_=None, sigma=None, delta_=None, wmin=None, wmax=None,A=None,
                            bmin=None,bmax=None,B=None,beq=None,
                            fmin=None, fmax=None,cpct=None, **kwargs):
    """
    组合优化器

    目标函数：
    --------
    maximize x'r - c'|w-w0| - lambda x'Vx
    subject to 
        w = wb + x
        x'Vx <= sigma^2
        ||w-w0|| <= delta * 2
        wmin <= w <= wmax
        fmin <= Xf'x <= fmax
        w'(wb>0) >= cpct
        bmin <= A'x <= bmax
        B'x == beq
    其中 , w 是绝对权重 , x 是主动权重 , 股票协方差 V 由结构化因子模型确定.
    如果传入cov, V=cov; 否则，V=V = Xf'(F)Xf + D^2

    r numpy.array(n,) 股票预期收益
    cov numpy.array(n,n)股票协方差矩阵
    Xf numpy.array(n,m) 风险因子取值
    F numpy.array(m,m) 风险因子收益率协方差矩阵
    D numpy.array(n,) 股票残差风险矩阵
    w0 numpy.array(n,) 组合初始权重 , None 表示首次建仓 初始权重为 0
    wb numpy.array(n,) 基准指数权重
    c_ float 或者 numpy.array(n,) 换手惩罚参数
    lambda_ float 风险惩罚参数
    sigma_ float 跟踪误差约束 , 0.05 表示 5%
    delta_ float 换手约束参数 单边
    wmin float 或者 numpy.array(n,) 绝对权重最小值
    wmax float 或者 numpy.array(n,) 绝对权重最大值
    fmin float list tuple 或者 numpy.array(m,) 因子暴露最小值
    fmax float list tuple 或者 numpy.array(m,) 因子暴露最大值
    cpct float 0 到 1 之间 成分股内股票权重占比
    A numpy.array(n,p) 其他线性约束矩阵
    """
    n = len(r) # 资产数量
    x = cvx.Variable(n)

    # 设定目标函数
    obj = x @ r
    if wb is None:
        wb = np.zeros(n, dtype='float')
    if w0 is None:
        w0 = np.zeros(n, dtype='float')
    w = wb + x
    if isinstance(c_, float):
        obj = obj - cvx.sum(c_ * cvx.abs(w - w0))
    elif isinstance(c_, np.ndarray):
        obj = obj - c_ @ cvx.abs(w - w0)
    if cov is not None:
        risk = cvx.quad_form(x, cov)
    else:
        risk = cvx.quad_form(x, Xf.T @ F @ Xf) + cvx.quad_form(x, np.diag(D**2))
    if lambda_:
        obj = obj - lambda_ * risk
    
    # 限制条件
    constraints = []
    if wmin is not None:
        constraints.append(wmin<=w)
    if wmax is not None:
        constraints.append(w<=wmax)
    if A is not None:
        ineq = A.T @ x
        if bmin is not None:
            constraints.append(ineq>=bmin)
        if bmax is not None:
            constraints.append(ineq<=bmax)
    if B is not None:
        eq = B.T @ x
        constraints.append(beq==eq)
        # constraints.append(beq>=eq)
    
    # 优化
    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(**kwargs)
    if prob.status == 'optimal':
        result = {
            'abswt': x.value+wb,
            'relwt': x.value,
            'er': x.value @ r,
            'sigmal': risk.value
        }
    else:
        result = {}
    return result

