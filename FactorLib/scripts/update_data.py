from FactorLib.data_source.update_data.update_h5db_base_data import (onlist,
                                                           update_price,
                                                           update_sector,
                                                           update_trade_status,
                                                           update_idx_weight,
                                                           update_industry_name,
                                                           update_industry_index_prices,
                                                           update_stock_constest,
                                                           stockname
                                                                    )
from FactorLib.data_source.update_data.run_daily import dailyfactors
from FactorLib.data_source.wind_financial_data_api import update
from FactorLib.data_source.base_data_source_h5 import h5
from datetime import datetime, time
from FactorLib.utils.strategy_manager import update_nav, collect_nav
from FactorLib.utils.tool_funcs import import_module
from FactorLib.single_factor_test.main import run
import pandas as pd
import os
from FactorLib.data_source.base_data_source_h5 import tc

latest_update_date_0 = '20170906'
latest_update_date_1 = '20170906'

UpdateFuncs = [
               onlist,
               update_price,
               update_sector,
               update_trade_status,
               update_idx_weight,
               update_industry_name,
               update_industry_index_prices,
               update_stock_constest,
               stockname
               ]

flag0 = 0
flag1 = 0
while 1:
    if datetime.today().date() > datetime.strptime(latest_update_date_0, '%Y%m%d').date():
        flag0 = 1
    if datetime.today().date() > datetime.strptime(latest_update_date_1, '%Y%m%d').date():
        flag1 = 1
    if datetime.now().time() > time(17, 0, 0) and flag0:
        print("即将更新因子数据...")
        start = tc.tradeDayOffset(latest_update_date_0, 1)
        end = datetime.today().strftime('%Y%m%d')
        # end = '20170831'
        for iFunc in UpdateFuncs:
            iFunc(start, end)
        update.update_all(start, end)
        dailyfactors(start, end)
        h5.snapshot(pd.date_range(start, end), 'base_factor', mail=True)
        flag0 = 0
        latest_update_date_0 = datetime.today().strftime("%Y%m%d")
        run(r"D:\factors\全市场_中证500基准_2010年以来\update\config.yml",
            {'start_date': start, 'end_date': end})
        collect = import_module('collect', r"D:\factors\全市场_中证500基准_2010年以来\update\collect_nav.py")
        collect.collect(mailling=True)
        print("单因子回测更新完成...")
    if datetime.now().time() > time(20, 0, 0) and flag1:
        print('即将更新回测数据...')
        os.system("rqalpha update_bundle")
        flag1 = 0
        start = tc.tradeDayOffset(latest_update_date_1, 1)
        latest_update_date_1 = datetime.now().strftime("%Y%m%d")
        update_nav(start=start, end=latest_update_date_1)
        collect_nav(mailling=True)
