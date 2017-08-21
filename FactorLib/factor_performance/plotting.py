import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import ticker
from empyrical import stats


def plot_ic(factor):
    """绘制因子IC柱状图
    """
    fig, ax = plt.subplots(1,1)
    ic = factor.ic_series.ic_series

    # IC柱状图
    my_color = ['r' if x > 0 else 'g' for x in ic]
    ic.plot(kind='bar', ax=ax, color=my_color)
    _adjust_date_axis(ax, ic.index)
    ax.set_title("IC Series", fontsize=9)

    return fig, ax


def plot_long_short_return(factor):
    fig, axes = plt.subplots(2,1)

    # 多空月度收益率柱状图
    long_short_returns = factor.long_short_return.to_frame()
    long_short_m_returns = long_short_returns.apply(stats.aggregate_returns,args=('monthly',))
    new_index = [x * 100 + y for x, y in long_short_m_returns.index]

    long_short_m_returns.plot(kind='bar',ax=axes[0])
    _adjust_date_axis(axes[0], new_index)
    axes[0].set_title("Monthly Return of Long-Short Portfolio", fontsize=9)

    # 多空净值曲线
    long_short_cum_returns = long_short_returns.apply(stats.cum_returns,args=(1,))
    long_short_cum_returns.plot(linewidth=2, ax=axes[1])
    axes[1].set_title("Net Value of Long-Short Portfolio", fontsize=9)

    return fig, axes


def plot_bar_group_return(factor):
    """绘制因子分组收益柱状图"""
    group_return = factor.group_return.to_frame()
    group_annual_return = group_return.apply(stats.annual_return)

    average_return = group_return.groupby(axis=1, level=0).mean()
    average_annual_return = average_return.apply(stats.annual_return)

    excess_return = group_annual_return.subtract(average_annual_return, axis=0, level=0)
    excess_return = excess_return.swaplevel().unstack()

    ax = excess_return.plot(kind='bar', title="Excess Return")
    return ax.figure, ax

def plot_group_cum_return(factor):
    """绘制分组的累计收益图"""
    fig, axes = plt.subplots(len(factor.group_return.group_methods), 1)
    group_return = factor.group_return.to_frame().sort_index(axis=1)
    group_ids = np.linspace(1, factor.group_return.n_groups, 3, dtype=np.int).tolist()
    for i, method in enumerate(factor.group_return.group_methods):
        try:
            ax = (1 + group_return.loc[:, (method, group_ids)]).cumprod().copy().plot(ax=axes[i], linewidth=2)
        except:
            ax = (1 + group_return.loc[:, (method, group_ids)]).cumprod().copy().plot(ax=axes, linewidth=2)
        ax.set_title(method, fontsize=9)
    return fig, axes

def plot_first_group_return(factor):
    """绘制第一组收益率"""
    fig, axes = plt.subplots(2, 1)
    first_group_return = factor.group_return.get_group_return(1)
    benchmark = factor.group_return.get_benchmark_return().iloc[:, 0]
    
    m_returns = first_group_return.apply(stats.aggregate_returns, args=('monthly',))
    new_index = [x * 100 + y for x, y in m_returns.index]

    m_returns.plot(kind='bar',ax=axes[0])
    _adjust_date_axis(axes[0], new_index)
    axes[0].set_title("Monthly Return of First Group Portfolio", fontsize=9)

    # 多空净值曲线
    first_group_cum_return = first_group_return.apply(stats.cum_returns,args=(1,))
    first_group_cum_return['benchmark'] = stats.cum_returns(benchmark, 1)

    first_group_cum_return.plot(linewidth=2, ax=axes[1])
    axes[1].set_title("Net Value of First Group Portfolio", fontsize=9)

    return fig, axes

def monthly_active_return_heat_map(factor):
    """绘制月收益率热力图"""
    from pyfolio import plot_monthly_returns_heatmap
    n = len(factor.first_group_active_return.long_short_returns)
    fig, axes = plt.subplots(1, n)
    for i , method in enumerate(factor.first_group_active_return.long_short_returns):
        actice_return = factor.first_group_active_return.long_short_returns[method].activeRet
        try:
            ax = plot_monthly_returns_heatmap(actice_return, ax=axes[i])
        except:
            ax = plot_monthly_returns_heatmap(actice_return, ax=axes)
        ax.title.set_text("Monthly Return(%s)" % method)
    return fig, axes

def _adjust_date_axis(ax, date_seq):
    tick_labels = [""] * len(date_seq)
    show = 6
    step = int(len(date_seq) / show)
    while step == 0:
        show -= 1
        step = int(len(date_seq) / show)
    try:
        tick_labels[::step] = [x.to_pydatetime().strftime("%Y%m") for x in date_seq[::step]]
    except:
        tick_labels[::step] = list(map(str, date_seq[::step]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, horizontalalignment='right', fontsize=7)
    ax.set_xlabel("")

FuncList = [plot_ic, plot_bar_group_return, plot_long_short_return, plot_first_group_return,
            plot_group_cum_return, monthly_active_return_heat_map]