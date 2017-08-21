from utils.tool_funcs import import_module
from data_source.base_data_source_h5 import data_source
from utils.excel_io import write_xlsx
import funcs
import stocklist
import pandas as pd


def run(config_name, config_path, **kwargs):
    config = import_module(config_name, config_path)
    for k in kwargs:
        setattr(config, k, kwargs[k])
    factors, direction = funcs._to_factordict(config.factors)
    dates = data_source.trade_calendar.get_trade_days(config.start, config.end, config.rebalance_frequence)
    stockpool = funcs._stockpool(config.stockpool, dates, config.stocks_unable_trade)

    factor_data = funcs._load_factors(factors, stockpool)
    factor_data = funcs._drop_outlier(factor_data, config.drop_outlier_method)
    factor_data = funcs._standard(factor_data)
    total_score = funcs._total_score(factor_data, direction, config.weight)
    factor_data = factor_data.merge(total_score, left_index=True, right_index=True, how='left')
    direction['total_score'] = 1

    stocks = []
    for i in [x for x in factor_data.columns if x != 'total_score']:
        stocks.append(stocklist.typical(factor_data, i, direction[i], config.industry_neutral, config.benchmark,
                                        config.industry, prc=config.prc))
    stocks = pd.concat(stocks, axis=1, ignore_index=True).fillna(0)
    stocks.columns = [x+'_weight' for x in factor_data.columns if x != 'total_score']
    stocks['Weight'] = stocks.mean(axis=1)
    stocks = pd.concat([stocks, factor_data.reindex(stocks.index)], axis=1)

    stocks2 = stocklist.typical(factor_data, 'total_score', 1, config.industry_neutral, config.benchmark,
                                config.industry, prc=config.prc)
    stocks2 = pd.concat([stocks2, factor_data.reindex(stocks2.index)], axis=1)
    write_xlsx("D:/data/stocklists/%s.xlsx"%config.name, stocks1=stocks, stocks2=stocks2)


if __name__ == '__main__':
    run('config', 'D:/FactorLib/generate_stocks/config.py')
