# coding: utf-8
# date: 2020/12/9
# email: wanshuai_shufe@163.com
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib as mpl
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ts_plot(ts_data, second_y=None, show_plot=True, save_plot=None):
    """时间序列折线图"""
    num_colors = ts_data.shape[1]
    cmap = mpl.cm.get_cmap(name='rainbow')
    currentColors = [cmap(1.*i/num_colors) for i in range(num_colors)]

    second_y = second_y or []
    first_y = [x for x in ts_data.columns if x not in second_y]

    xtick_locator = mdates.AutoDateLocator()
    xtick_formater = mdates.AutoDateFormatter(xtick_locator)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    times = ts_data.index.to_numpy(dtype='datetime64')
    lines = []
    for i, col_name in enumerate(first_y):
        line = ax.plot(times, ts_data[col_name].to_numpy(),
                       color=currentColors[i], label=col_name)[0]
        lines.append(line)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formater)
    ax.grid(True, linestyle='--', linewidth=0.8)
    ax.yaxis.set_major_locator(mticker.LinearLocator(6))

    if second_y:
        ax2 = ax.twinx()
        ax2.grid(None)
        ax2.yaxis.set_major_locator(mticker.LinearLocator(6))
        for i, col_name in enumerate(second_y):
            line = ax2.plot(times, ts_data[col_name].to_numpy(),
                            color=currentColors[len(first_y)+i],
                            label=col_name)[0]
            lines.append(line)

    ax.legend(lines, list(ts_data.columns), fontsize='x-small')

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(save_plot, dpi=300)

