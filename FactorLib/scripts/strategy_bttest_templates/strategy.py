'''
从csv中读取股票列表,然后导入到rqalpha进行回测。
'''


import pandas as pd
from rqalpha.api import *


# 初始化策略的逻辑
def init(context):
    context.stockList = get_stocklist_as_df()

# 交易之前触发该函数，确定一天的交易列表
def before_trading(context):
    try:
        context.tradingListNow = context.stockList.ix[context.now]
    except:
        context.tradingListNow = pd.DataFrame()

def handle_bar(context, bar_dict):
    """
        执行配置再平衡策略，买卖现有股票，以达到目标权重。
        1. 确定需要卖出的股票，卖出并获得现金
        2. 确定需要买入的股票，买入
    """
    if not context.tradingListNow.empty:
        pctPositions = {x:context.portfolio.positions[x].value_percent for x in context.portfolio.positions}
        pctPositions = pd.Series(pctPositions, name='current_weight')
        targetPostions = pd.concat([context.tradingListNow, pctPositions], axis=1, join='outer').fillna(0)
        targetPostions['Weight_DIFF'] = targetPostions['Weight'] - targetPostions['current_weight']
        # 先卖出股票获得现金
        stocksSell = targetPostions[targetPostions['Weight_DIFF'] < 0]
        for order_book_id, weight in stocksSell['Weight'].iteritems():
            order_target_percent(order_book_id, weight)
        # 再买入股票
        stocksBuy = targetPostions[targetPostions['Weight_DIFF'] > 0]
        for order_book_id, weight in stocksBuy['Weight'].iteritems():
            order_target_percent(order_book_id, weight)
    else:
        pass
