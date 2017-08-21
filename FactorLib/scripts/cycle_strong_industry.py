"""
把周期性强的股票合并，创建一个新的指数，指数的代码命名为101002。周期性强的行业划分为：
1. 化工
2. 煤炭\石油石化
3. 钢铁
4. 有色金属
5. 交通运输
6. 房地产
7. 非银行金融
"""

from data_source import data_source
import pandas as pd


start = '20070101'
end = '20170709'
industry_codes = ['CI005006', 'CI005001', 'CI005002', 'CI005005', 'CI005003', 'CI005024',
                  'CI005023', 'CI000522']
new_code = '101002'

# 合并行业成分股
cycle_strong = []
for industry in industry_codes:
    d = data_source.sector.get_index_members(industry, start_date=start, end_date=end)
    cycle_strong.append(d)
cycle_strong = pd.concat(cycle_strong, axis=1)
cycle_strong['_'+new_code] =  cycle_strong.any(axis=1)

data_source.h5DB.save_factor(cycle_strong[['_'+new_code]].astype('int'), '/indexes/')

# 构造指数收益率
pct = data_source.h5DB.load_factor('changeper', '/indexprices/cs_level_1/', ids=industry_codes,
                                   dates=data_source.trade_calendar.get_trade_days(start, end))
cycle_strong_pct = pct.groupby(level=0).mean().rename(columns={'changeper':'daily_returns_%'})
cycle_strong_pct.index = pd.MultiIndex.from_product([cycle_strong_pct.index, [new_code]], names=['date', 'IDs'])
data_source.h5DB.save_factor(cycle_strong_pct, '/indexprices/')