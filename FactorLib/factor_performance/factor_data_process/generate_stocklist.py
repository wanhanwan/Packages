"""生成股票列表"""
from generate_stocks import stocklist
import pandas as pd


def __add_stocks_to_factor(factor, stocks, method='typical'):
    if method not in factor.stock_list:
        factor.stock_list[method] = pd.DataFrame()
    factor_stock_list = factor.stock_list[method].append(stocks[['Weight']])
    factor_stock_list = factor_stock_list[~factor_stock_list.index.duplicated(keep='last')]
    factor.stock_list[method] = factor_stock_list


def typical(factor, industry_neutral=True, industry_name='中信一级', prc=0.05, **kwargs):
    benchmark = kwargs['env']._config.benchmark
    stocks = stocklist.typical(factor.data, factor.name, factor.direction, industry_neutral, benchmark, industry_name,
                                 prc, **kwargs)
    __add_stocks_to_factor(factor, stocks, 'typical')


FuncList = {'typical': typical}