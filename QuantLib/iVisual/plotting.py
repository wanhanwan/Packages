# coding = utf-8
# 数据可视化工具包

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# 函数1
# 横截面蜡烛图，判断离群值
def cross_section_cndl(data, factor_name):
    '''在每一个时间截面上画出蜡烛图，
    标记最大值、最小值以及均值等
    
    输入
    ------------------------------
    data:DataFrame(index:[Date,IDs],factor1,factor2,...)

    factor_name:str
    '''
    data = data.reset_index()
    sns.set(style='ticks')

    ax = sns.boxplot(x='Date', y=factor_name, data=data, palette='PRGn')
    sns.despine(offset=10, trim=True)

    return ax

# 函数2
# 频率分布图, 因子在横截面上的频率分布图
def cross_section_hist(data, factor_name, date):
    '''画出因子在某个时间截面的分布图，并画出拟合曲线
    
    输入
    --------------------------------
    data:DataFrame(index:[Date,IDs],factor1,factor2,...)
    factor_name:str
    date：str
    '''
    plot_data = data.ix[(date,), factor_name].reset_index(drop=True)

    ax = sns.distplot(plot_data)

    return ax

# 函数3
# Quantile-Quantile，判断是否符合某个已知分布
def cross_section_qqplot(data, factor_name, date):
    '''
    输入
    --------------------------------
    data:DataFrame(index:[Date,IDs],factor1,factor2,...)
    factor_name:str
    date：str
    '''
    ax = plt.gca()
    plot_data = data.ix[(date,), factor_name].values
    fig = sm.qqplot(plot_data, line='45', fit=True,ax=ax)
    plt.show()

    return ax

# 函数4
# ic 分布图
def ic_bar_plot(data):
    '''
    输入
    ---------------------------------
    data: Series(index:Date)
    '''
    my_color = ['r' if x > 0 else 'g' for x in data]
    ax = data.plot(kind='bar', color=my_color)
    return ax

# 函数5
# ic 衰变图
def ic_decay_plot(data):
    '''
    输入
    ---------------------------------
    data: Series(index:delay days)
    '''
    ax = data.plot(linewidth=2)
    return x

# 函数6
# 分组收益率柱状图
def monthly_return_bar_plot(data):
    '''
    输入
    ---------------------------------
    data:DateFrame(index:Date,columns:groupID,benchmark)
    '''
    excess_return = data.iloc[:, :-1].sub(data.iloc[:, -1], axis=0)
    excess_monthly_return = excess_return.mean() * 20

    ax = excess_monthly_return.plot(kind='bar')
    labels = ["%.3f"%i for i in excess_monthly_return.values]

    rects = ax.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        if label[0]=='-':
            text_y = - height - 0.001
        else:
            text_y = height + 0.001
        ax.text(rect.get_x()+rect.get_width()/2, text_y,label,ha='center',
                va='bottom')
    return ax
