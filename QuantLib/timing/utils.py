#!python
# -*- coding: utf-8 -*-
#
# utils.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/5/7 下午3:29:15
import numpy as np


def is_discrete_signal(df_signal):
    """
    判断是不是离散的信号

    离散信号具体指信号值只有0，1，-1三种。
    """
    is_discrete = df_signal.apply(
        lambda x: np.all(x.dropna().isin([1, -1, 0]))
    )
    return is_discrete

