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
sm.generate_stocklist_txt(strategy_name='兴基VG_逆向_800', date='20170831')