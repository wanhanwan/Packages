import os
import shutil
from FactorLib.utils.strategy_manager import StrategyManager
sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')

# for d in os.listdir(r"D:\data\factor_investment_temp_strategies"):
#     if os.path.isdir(os.path.join(r"D:\data\factor_investment_temp_strategies", d)):
#         sm.create_from_directory(os.path.join(r"D:\data\factor_investment_temp_strategies", d))

# for d in os.listdir(r"D:\data\factor_investment_strategies"):
#     if os.path.isdir(os.path.join(r"D:\data\factor_investment_strategies", d)):
#         print(d)
#         sm.analyze_return(strategy_name=d)
sm.generate_tradeorder(strategy_id=7, capital=200000000)
sm.generate_tradeorder(strategy_id=8, capital=200000000)
# sm.update_stocks('20070131', '20170930', strategy_id=8)
# sm.run_backtest('20070131','20171011', strategy_id=7)