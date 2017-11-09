# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:33:59 2017
解析天软数据格式
@author: ws
"""
import pandas as pd

_max_iter_stocks = 100


def _int2date(int_date):
    if int_date < 10000000:
        return pd.NaT
    return pd.datetime(int_date//10000, int_date%10000//100, int_date%100)


def parseByStock(TSData, date_parse=None):
    """ 
    按照股票为单位，逐个解析数据。
    数据格式为两列的Array,第一列是股票代码，第二列是对应的子Array。
    不同的股票的每个子Array中的列名是相同的，以此保证可以把所有股票的数据按行连接起来。
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    iter_stock = 0
    table = pd.DataFrame()
    temp_table = []
    for idata in TSData[1]:
        stockID = idata[0].decode('utf8')[2:]
        stockData = []
        iter_stock += 1
        for itable in idata[1]:
            new_dict = {k.decode('gbk'): v for k, v in itable.items()}
            new_data = pd.DataFrame(new_dict, index=pd.Index([stockID], name='IDs'))
            stockData.append(new_data)
        if stockData:
            stockData = pd.concat(stockData)
        else:
            continue
        if iter_stock < _max_iter_stocks:
            temp_table.append(stockData)
        else:
            _ = pd.concat(temp_table)
            table = pd.concat([table, _])
            temp_table = []
            iter_stock = 0
    if date_parse:
        table[date_parse] = table[date_parse].applymap(_int2date)
    return table


def parseCrossSection2DArray(TSData, date):
    """
    解析横截面上的二维数组, 行索引是股票代码，
    列索引是因子名称。
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    iter_stock = 0
    table = pd.DataFrame()
    temp_table = []
    date = pd.to_datetime(date)
    for stock_id, factors in TSData[1].items():
        stock_id = stock_id.decode('utf8')[2:]
        factors = {k.decode('GBK'): v for k, v in factors.items()}
        stockData = pd.DataFrame(factors, index=pd.Index([stock_id], name='IDs'))

        iter_stock += 1
        if iter_stock < _max_iter_stocks:
            temp_table.append(stockData)
        else:
            _ = pd.concat(temp_table)
            table = pd.concat([table, _])
            temp_table = []
            iter_stock = 0
    table.index = pd.MultiIndex.from_product([[date], table.index], names=['date', 'IDs'])
    return table
