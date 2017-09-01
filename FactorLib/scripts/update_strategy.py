from FactorLib.utils.strategy_manager import StrategyManager

start = '20170801'
end = '20170831'

sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
for strategy_id in sm._strategy_dict['id'].values:
    sm.update_stocks(start, end, strategy_id=strategy_id)
