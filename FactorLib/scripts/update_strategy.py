from FactorLib.utils.strategy_manager import StrategyManager

start = '20170901'
end = '20170929'

sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
for strategy_id in sm._strategy_dict['id'].values:
    sm.update_stocks(start, end, strategy_id=strategy_id)
    latest_rebalance_date = sm.get_attribute('latest_rebalance_date', strategy_id=strategy_id)
    print("策略【%s】持仓更新完毕, 最新调仓日期【%s】..."%(sm.strategy_name(strategy_id), latest_rebalance_date))
