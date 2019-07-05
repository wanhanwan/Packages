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

    Return:
    =======
    DataFrame(index=IDs, columns=data)
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    iter_stock = 0
    table = pd.DataFrame()
    temp_table = []
    for idata in TSData[1]:
        stockID = idata[b'IDs'].decode('utf8')[2:]
        stockData = []
        iter_stock += 1
        for itable in idata[b'data']:
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
    if temp_table:
        _ = pd.concat(temp_table)
        table = pd.concat([table, _])
    if date_parse:
        table[date_parse] = table[date_parse].applymap(_int2date)
    return table.sort_index()


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
    return table.sort_index()


def parse2DArray(TSData, column_decode=None, encoding='utf8'):
    """解析天软二维数组
    二维数组的第一列是股票代码(字符串)，行是指标名称
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    data = pd.DataFrame(TSData[1])
    data.rename(columns=lambda x: x.decode(encoding),
                inplace=True)
    if column_decode:
        for column in column_decode:
            data[column] = data[column].str.decode(encoding)
    return data


def parse1DArray(TSData, col_name, encoding='utf8'):
    """解析天软的一维数组
    一维数组的索引是股票代码(字符串), 例如：
    arr := array();
    arr['SZ000001'] := 0.01;
    arr['SZ000002'] := 0.02;
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    data = pd.Series(TSData[1], name=col_name)
    try:
        data.rename(index=lambda x: x.decode(encoding), inplace=True)
    except AttributeError:
        pass
    return data


def parse2DArrayWithIDIndex(TSData, column_decode=None, encoding='utf8'):
    """解析天软二维数组
    二维数组的索引是股票代码(字符串)，行是指标名称
    """
    if TSData[0] != 0:
        raise ValueError("天软数据提取失败！")
    data = pd.DataFrame(TSData[1]).T
    data.rename(columns=lambda x: x.decode(encoding),
                index=lambda x: x.decode(encoding),
                inplace=True)
    if column_decode:
        for column in column_decode:
            data[column] = data[column].str.decode(encoding)
    return data
