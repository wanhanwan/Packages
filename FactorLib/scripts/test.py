import os
import pandas as pd
from utils.strategy_manager import StrategyManager, update_nav, collect_nav
from utils.excel_io import write_xlsx


sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
