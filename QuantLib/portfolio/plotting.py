#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# plotting.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2019/10/18 下午4:03:44
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from QuantLib.tools import return2nav

sns.set(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_nav_year_by_year(returns, benchmark_returns=None,
                          hide_tick_labels=False):
    """分年的净值图"""
    if (benchmark_returns is not None) and (not np.all(benchmark_returns.to_numpy()==0.0)):
        returns = returns.to_frame('组合').join(benchmark_returns.to_frame('基准'))
    else:
        returns = returns.to_frame('组合')
    years = returns.index.year.unique()
    n1 = int(len(years)**0.5)
    if len(years) % n1 == 0:
        n2 = int(len(years) / n1)
    else:
        n2 = int(len(years) / n1 + 1)
    fig, axes = plt.subplots(n1,n2)
    if n1 == n2 == 1:
        axes = np.array([[axes]])
    for i, (r, c) in enumerate(product(range(n1), range(n2))):
        if i+1 > len(years):
            break
        ax = axes[r, c]
        nav = return2nav(returns[returns.index.year==years[i]])
        nav.plot(ax=ax)
        if hide_tick_labels:
            ax.xaxis.set_ticklabels([])
        ax.set_title(str(years[i]))
    return fig, axes
