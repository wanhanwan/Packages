#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# x13.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2019/11/24 上午9:53:41
"""
X13季节性调整。

注：
cny.csv文件记录中国历年农历春节的日期，目前截止到2020年春节。
x13as.exe是X13主程序目录。
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.x13 import x13_arima_analysis
from pandas import DataFrame, Series, Timestamp
from pandas.tseries.frequencies import to_offset
from functools import lru_cache
from QuantLib.tools import RollingResultWrapper


curr_path = Path(__file__).parent

@lru_cache()
def get_spring_val(before=10, after=7): ###用于生成移动假日参数，移动假日放在cny.dat中
    data = pd.read_csv(curr_path/'cny.csv', index_col='rank')
    x1=Series()
    x2=Series()
    x3=Series()
    xx=Series(0,index=pd.date_range('1960-01-01','2030-01-01'))
    for i in range(len(data)):
        s=data.iloc[i]
        d=s[1]
        m=s[0]
        y=s[2]
        t=Timestamp(y,m,d)
        start=t-to_offset('%dd'%before)
        end=t+to_offset('%dd'%after)
        xx[pd.date_range(start,end)]=1
        
        d1=xx['%d-%d'%(y,1)].sum()/31
        d2=xx['%d-%d'%(y,2)].sum()/28
        d3=xx['%d-%d'%(y,3)].sum()/31
        x1[Timestamp(y,1,31)]=d1
        x2[Timestamp(y,2,1)+to_offset('M')]=d2
        x3[Timestamp(y,3,31)]=d3
         
    x1=x1-x1.mean()
    x2=x2-x2.mean()
    x3=x3-x3.mean()
    xx=Series(0,index=pd.date_range('1960-01-01','2030-01-01',freq='M'))
    xx[x1.index]=x1
    xx[x2.index]=x2
    xx[x3.index]=x3
    return xx.sort_index()
a=get_spring_val()

def merge_spring_element(data):
    """
    删除1、2月份的数据，用这两
    个月的均值填充为新的2月份数据。
    """
    if not isinstance(data, (Series, DataFrame)):
        raise ValueError('非常规时间序列')
    start = data.index[0]
    gap=Timestamp(data.index[1])-Timestamp(start)
    if gap.days<=5:
        raise ValueError('时间序列不能实现季调')
    elif gap.days>=62:
        return data
    else:
        adj = data[np.in1d(data.index.month, [1, 2])].groupby(
            pd.Grouper(freq='Y')
        ).mean()
        adj.index = pd.DatetimeIndex([pd.Timestamp(x.year, 2, 1) for x in adj.index]) + to_offset('1m')
        d = data[data.index.month > 2].append(adj).sort_index()
    return d


def remove_spring(data):
    new_data = merge_spring_element(data)
    new_data = new_data.reindex(data.index, method='ffill')
    return new_data


def seas_adj(series, holidayfile='chn', remove_spring=False,
             qoq=False, before=10, after=7):
    '''
    对数据进行季节性调整(X13-arima-seats)
    1. start:开始时点
    2. end:结束时点
    3. holidayfile:移动假日的参数信息文件路径
    4. remove_spring:是否直接合并1-2月信息
    5. qoq:是否对调整后数据求当月环比
    6. before:节假日中前before天异常
    7. after:节假日中后after天异常
    '''
    if holidayfile is None:
        d=x13_arima_analysis(series).seasadj
    elif holidayfile == 'chn':
        name = f'chn_{before}_{after}.dat'
        if not os.path.isfile(curr_path/name):
            val=get_spring_val(before,after)
            val=DataFrame(val,columns=['val'])
            val['year'] =val.index.year
            val['month']=val.index.month
            val=val[['year','month','val']]['1970':]
            val.to_csv(curr_path/name, index=False, header=False, sep=' ')
        d=x13_arima_analysis(series, holidayfile=curr_path/name).seasadj
    else:
        d=x13_arima_analysis(series, holidayfile=holidayfile).seasadj
    if remove_spring:
        d=merge_spring_element(d)
    if qoq:
        d=np.log(d).diff()
    d.name = series.name
    return d.dropna().reindex(series.index, method='ffill')
    

def rolling_seas_adj(data, window, holidayfile='chn',
                     qoq=False, before=10, after=7,
                     remove_spring=False):
    """滚动季调"""
    @RollingResultWrapper
    def func(x, **kwargs):
        return seas_adj(x, **kwargs).tolist()

    data = data.dropna()
    if isinstance(data, pd.Series):
        data = data.to_frame()
    try:
        data.rolling(window).apply(
            func,
            raw=False,
            kwargs={'holidayfile': holidayfile,
                    'qoq': qoq,
                    'before': before,
                    'after': after,
                    'remove_spring': remove_spring
                    })
        idx = pd.MultiIndex.from_product([data.index[window-1:], data.columns],
                                         names=['date', 'factors'])
        idx = idx.sortlevel(level='factors')[0]
        if qoq:
            cols = np.arange(1, window, dtype='int')
        else:
            cols = np.arange(1, window+1, dtype='int')
        df = pd.DataFrame(func.rlist, index=idx, columns=cols).sort_index()
    finally:
        func.reset()
    return df

