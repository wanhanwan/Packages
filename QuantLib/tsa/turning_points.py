#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# turningpoint_detection.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2020/12/7 下午1:12:22
import pandas as pd
import numpy as np
from cif import cif
from QuantLib.utils import align_dataframes

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def detect_tp_of_time_series(series: pd.Series, minimal_cycle_length=15, minimal_phase_length=5, log_file=None):
    """
    Bry-Boschan算法识别时间序列拐点

    Parameters:
    ------
    series: Series
        时间序列
    minimal_cycle_length: int
        一个周期最小持续长度要求
    minimal_phase_length: int
        一个波峰或者波谷的最小长度要求
    log_file: str
        日志文件名称

    Returns:
    -------
    indicator: series
        拐点标志，1代表顶点，-1代表底点，0代表其它
    """
    df = series.to_frame()

    if isinstance(log_file, str):
        log_file = open(log_file, 'w')

    # a) Looking for local minimal/maximal
    col_ind_local = cif.getLocalExtremes(df, showPlots=False)

    # b) Check the turning points alterations
    col_ind_neigh = cif.checkNeighbourhood(df=df, indicator=col_ind_local, showPlots=False, saveLogs=log_file)
    col_ind_alter = cif.checkAlterations(df=df, indicator=col_ind_neigh, keepFirst=False, showPlots=False, saveLogs=log_file)

    # c) Check minimal lenth of cycles
    col_ind_cyclelength = cif.checkCycleLength(df=df, indicator=col_ind_alter, cycleLength=minimal_cycle_length,
                                               showPlots=False, saveLogs=log_file)

    # d) Check the turning points alterations again
    col_ind_neigh_again = cif.checkNeighbourhood(df=df, indicator=col_ind_cyclelength, showPlots=False, saveLogs=log_file)
    col_ind_alter_again = cif.checkAlterations(df=df, indicator=col_ind_neigh_again, keepFirst=False, showPlots=False,
                                               saveLogs=log_file)

    # e) Check minimal length of phase
    col_ind_phaselength = cif.checkPhaseLength(df=df, indicator=col_ind_alter_again, keepFirst=False,
                                               phaseLength=minimal_phase_length, showPlots=False,
                                               saveLogs=log_file)

    # f) Check the turning points alterations for the last time
    col_ind_neigh_last = cif.checkNeighbourhood(df=df, indicator=col_ind_phaselength, showPlots=False, saveLogs=log_file)
    col_ind_turning_points = cif.checkAlterations(df=df, indicator=col_ind_neigh_last, keepFirst=False, showPlots=False,
                                                  saveLogs=log_file)

    return col_ind_turning_points[col_ind_turning_points.columns[0]]


def match_tp_of_two_series(series1=None, series2=None, indicator1=None, indicator2=None,
                           freq_scaler=1, minimal_cycle_length1=15, minimal_cycle_length2=15,
                           minimal_phase_length1=5, minimal_phase_length2=5, log_file=None,
                           look_back_cache=None, lookforward_cache=0, return_details=False):
    """两个序列的拐点匹配"""

    if isinstance(log_file, str):
        log_file = open(log_file, 'w')

    if series1:
        indicator1 = detect_tp_of_time_series(series1, minimal_cycle_length1, minimal_phase_length1, log_file)
    indicator1 = indicator1[indicator1 != 0.0]
    if series2:
        indicator2 = detect_tp_of_time_series(series2, minimal_cycle_length2, minimal_phase_length2, log_file)
    indicator2 = indicator2[indicator2 != 0.0]

    if indicator2.index.min() < indicator1.index.min():
        indicator2 = indicator2.iloc[np.where(indicator2.index < indicator1.index.min())[0][-1]:]
    if indicator2.index.max() > indicator1.index.max():
        indicator2 = indicator2.iloc[:np.where(indicator2.index > indicator1.index.max())[0][0]+1]

    matched = 0
    no_data = len(indicator1[indicator1.index < indicator2.index.min()])
    leading_periods = []
    matched_periods_of_series1 = []
    matched_periods_of_series2 = []
    for dt1, tp1 in indicator1.iloc[no_data:].iteritems():

        # 匹配规则：寻找series1拐点前后相临近的series2拐点,若是向前匹配，则还要判断向前时段是否符合条件。
        idx = np.where(indicator2.index <= dt1)[0][-1]
        latest_tp2 = indicator2.iat[idx]
        lastet_dt2 = indicator2.index[idx]

        # Check if the turning point of series2 matches the one of series1
        if tp1 == latest_tp2:
            if ((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') > look_back_cache * freq_scaler) or \
               (len(indicator1.loc[lastet_dt2:dt1]) > 1):
                continue
            matched += 1
            leading_periods.append(int((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') / freq_scaler))
            matched_periods_of_series1.append(dt1)
            matched_periods_of_series2.append(lastet_dt2)
        else:
            idx = np.where(
                indicator2.index <= dt1+pd.Timedelta(int(lookforward_cache*freq_scaler), 'D')
            )[0][-1]
            latest_tp2 = indicator2.iat[idx]
            lastet_dt2 = indicator2.index[idx]
            if tp1 == latest_tp2:
                matched += 1
                leading_periods.append(int((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') / freq_scaler))
                matched_periods_of_series1.append(dt1)
                matched_periods_of_series2.append(lastet_dt2)

    # 匹配率：序列1的匹配数 / (序列1的总拐点数 - 无数据拐点数)
    mached_ratio = matched / (len(indicator1) - no_data)

    # 多余率: 指标多余拐点 / 指标总拐点
    redundant_ratio = (len(indicator2) - matched) / len(indicator2)

    # 平均领先阶数
    if leading_periods:
        avg_leading_periods = np.mean(leading_periods)
        std_leading_periods = np.std(leading_periods)
    else:
        avg_leading_periods = np.nan
        std_leading_periods = np.nan

    result_dict = {
        '无数据拐点': no_data,
        '可匹配拐点': matched,
        '有效拐点总数': len(indicator1) - no_data,
        '匹配率': mached_ratio,
        '多余率': redundant_ratio,
        '平均领先阶数': avg_leading_periods,
        '领先阶数标准差': std_leading_periods
    }
    mached_periods = pd.DataFrame(
        [matched_periods_of_series1, matched_periods_of_series2],
        index=[indicator1.name, indicator2.name]).T

    if return_details:
        return result_dict, {'匹配时点': mached_periods}
    return result_dict


def compare_two_series(series1, series2, ind1, ind2, save_plots=False, show_plots=True,
                       matched_periods=None):
    """比较两个时间序列的拐点"""
    series1, series2, ind1, ind2 = align_dataframes(series1, series2, ind1, ind2, axis='index', join='inner')
    times = np.arange(len(series1))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax2 = ax1.twinx()

    # 设置x坐标, 6个ticks
    xtick_nums = 6
    xtick_values = np.linspace(0, len(times)-1, xtick_nums, dtype='int32', endpoint=True)
    xtick_labels = list(series1.index.strftime("%Y%m%d")[xtick_values])
    plt.xticks(xtick_values, xtick_labels, rotation=45)

    # 划曲线，高点和底点用几何图形标注
    peak_marker = '^'
    peak_xticks1 = times[ind1.values == 1]
    peak_yticks1 = series1.values[peak_xticks1]

    trough_marker = 'v'
    trough_xticks1 = times[ind1.values == -1]
    trough_yticks1 = series1.values[trough_xticks1]

    color1 = 'gold'
    color2 = 'deepskyblue'
    line1 = ax1.plot(times, series1, linestyle='-', color=color1, label=series1.name)[0]
    line2 = ax2.plot(times, series2, linestyle='-', color=color2, label=series2.name)[0]

    import matplotlib.ticker
    nticks = 6
    ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    ax2.grid(None)
    ax1.grid(True, linestyle='--', linewidth=1.0)
    scatter1 = ax1.scatter(peak_xticks1, peak_yticks1, marker=peak_marker, color='red', s=100)
    scatter2 = ax1.scatter(trough_xticks1, trough_yticks1, marker=trough_marker, color='green', s=100)
    ax1.legend([line1, line2, scatter1, scatter2], [series1.name, series2.name, '极大值', '极小值'], fontsize='x-small')

    peak_xticks2 = times[ind2.values == 1]
    peak_yticks2 = series2.values[peak_xticks2]
    trough_xticks2 = times[ind2.values == -1]
    trough_yticks2 = series2.values[trough_xticks2]

    ax2.scatter(peak_xticks2, peak_yticks2, marker=peak_marker, color='red', s=100)
    ax2.scatter(trough_xticks2, trough_yticks2, marker=trough_marker, color='green', s=100)

    # 匹配的拐点用箭头表示
    from matplotlib.patches import ConnectionPatch
    if isinstance(matched_periods, pd.DataFrame):
        for dt1, dt2 in matched_periods[[ind1.name, ind2.name]].values:

            xtick_ax1 = times[np.where(ind1.index == dt1)[0][0]]
            ytick_ax1 = series1.loc[dt1]
            xtick_ax2 = times[np.where(ind2.index == dt2)[0][0]]
            ytick_ax2 = series2.loc[dt2]

            con = ConnectionPatch(xyA=(xtick_ax1, ytick_ax1), xyB=(xtick_ax2, ytick_ax2),
                                  coordsA="data", coordsB="data", arrowstyle="<->",
                                  axesA=ax1, axesB=ax2, shrinkA=0, shrinkB=0)
            con.set_color('black')
            con.set_linewidth(1)
            con.set_linestyle('--')
            ax2.add_artist(con)

    plt.subplots_adjust(hspace=0.0)
    fig.tight_layout()

    if show_plots:
        plt.show()

    if save_plots:
        plt.savefig(save_plots, dpi=300)

    plt.close(fig)


if __name__ == '__main__':
    data = pd.read_clipboard(header=0, index_col=0, parse_dates=True)
    tp = detect_tp_of_time_series(data['tips_smoothed'], 200, 70)
    tp = data['tips_smoothed'].where(tp['tips_smoothed'] != 0, other=np.nan)
    tp.to_clipboard()
