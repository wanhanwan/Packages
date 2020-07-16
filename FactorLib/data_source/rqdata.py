#!python
# -*- coding: utf-8 -*-
#
# rqdata.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/3/27 上午10:25:53
"""rqdata API"""
import rqdatac
import pandas as pd

rqdatac.init()


class RQDataAPI(object):

    rq_api = rqdatac

    def __getattr__(self, attr):
        return getattr(rqdatac, attr)
    

id_convert = rqdatac.id_convert
rq_data = RQDataAPI

import rqalpha