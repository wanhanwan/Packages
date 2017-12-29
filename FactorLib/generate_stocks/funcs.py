from ..data_source.base_data_source_h5 import data_source
from ..riskmodel import stockpool
from QuantLib.utils import DropOutlier, Standard, ScoringFactors
from ..utils.tool_funcs import parse_industry, tradecode_to_windcode
import numpy as np
import pandas as pd


def _load_factors(factor_dict, stocklist):
    dates = stocklist.index.get_level_values(0).unique()
    data = data_source.h5DB.load_factors(factor_dict, dates=list(dates))
    return data.reindex(stocklist.index).dropna()


def _load_latest_factors(factor_dict, stocklist):
    l = []
    ids = stocklist.index.tolist()
    for factor_dir in factor_dict:
        for factor in factor_dict[factor_dir]:
            d = data_source.h5DB.load_latest_period(factor, factor_dir, ids=ids)
            l.append(d)
    data = pd.concat(l, axis=1).dropna()
    return data

def _rawstockpool(sectorid, dates):
    stockpool = data_source.sector.get_index_members(sectorid, dates=dates)
    return stockpool


def _stockpool(sectorid, dates, qualify_method):
    raw = _rawstockpool(sectorid, dates)
    return stockpool._qualify_stocks(raw, qualify_method)


def score_by_industry(factor_data, industry_name, factor_names=None, **kwargs):
    factor_names = factor_names if factor_names is not None else list(factor_data.columns)
    industry_str = parse_industry(industry_name)
    all_ids = factor_data.index.get_level_values(1).unique().tolist()
    all_dates = factor_data.index.get_level_values(0).unique().tolist()

    # 个股的行业信息与因子数据匹配
    industry_info = data_source.sector.get_stock_industry_info(
        all_ids, industry=industry_name, dates=all_dates).reset_index()
    factor_data = factor_data.reset_index()
    factor_data = pd.merge(factor_data, industry_info, how='left')
    score = factor_data.set_index(['date', 'IDs']).groupby(
        industry_str, group_keys=False).apply(ScoringFactors, factors=factor_names, **kwargs)
    return score


def score_typical(factor_data, factor_names=None, **kwargs):
    factor_names = factor_names if factor_names is not None else list(factor_data.columns)
    score = ScoringFactors(factor_data, factor_names, **kwargs)
    return score


def _total_score(factors, directions, weight=None):
    n_factors = factors.shape[1]
    if weight is None:
        weight = np.array([1/n_factors]*n_factors)[:, np.newaxis]
    else:
        weight = np.array(weight)[:, np.newaxis]
    directions = np.array([directions[x] for x in directions])
    total_score = np.dot(factors.values*directions, weight)
    return pd.DataFrame(total_score, index=factors.index, columns=['total_score'])


def _to_factordict(factors):
    _dict = {}
    direction = {}
    for factor in factors:
        if factor[1] not in _dict:
            _dict[factor[1]] = [factor[0]]
        else:
            _dict[factor[1]].append(factor[0])
        direction[factor[0]] = factor[2]
    return _dict, direction


def generate_wind_pms_template(stocklist, save_path):
    """
    根据股票列表生成wind调仓模板

    wind调仓模板的格式如下：
        column1：证券代码 column2：持仓权重 column3: 成本价格 column4：调整日期 column5：证券类型
    :param stocklist: pandas.dataframe
        股票列表的格式 index[date IDs] column[Weight]
    :param save_path: str
        excel保存路径
    :return: excel

    """
    columnmap = {'date':'调整日期', 'IDs':'证券代码', 'close':'成本价格', 'Weight':'持仓权重'}
    close_price = data_source.load_factor('close', '/stocks/', idx=stocklist.index)
    temp = pd.concat([stocklist, close_price], axis=1).reset_index().rename(columns=columnmap)
    temp['证券类型'] = '股票'
    temp['证券代码'] = temp['证券代码'].apply(tradecode_to_windcode)
    temp = temp[['证券代码', '持仓权重', '成本价格', '调整日期', '证券类型']]
    writer = pd.ExcelWriter(save_path, datetime_format='yyyy/mm/dd')
    temp.to_excel(writer, "Sheet1", index=False)


def generate_stocklist_txt(stocklist, save_path):
    """把股票列表中的股票导入到txt文件"""

    IDs = stocklist.index.get_level_values(1).unique().tolist()
    IDs = [tradecode_to_windcode(x)+'\n' for x in IDs]

    with open(save_path, 'w') as f:
        f.writelines(IDs)


if __name__ == "__main__":
    _stockpool('全A', ['20070131'], 'typical')




