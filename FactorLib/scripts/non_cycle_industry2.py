"""     把次非周期性行业的股票合并，创建一个新的的指数，指数的代码命名为101004。
非周期的行业分为：
        1. 建筑材料
        2. 建筑装饰
        3. 电力设备
        4. 机械设备
        5. 国防军工
        6. 汽车
        7. 农林牧渔
        指数的收益率计算方法为：先计算中信行业指数的日收益率，在对所有行业的收益率作简单算术平均。
"""


from data_source import data_source
import pandas as pd


start = '20070101'
end = '20170709'
industry_codes = ['CI005007', 'CI005008', 'CI005011', 'CI005010', 'CI005012', 'CI005013', 'CI005020']
new_code = '101004'


# 合并行业成分股
non_cycle = []
for industry in industry_codes:
    d = data_source.sector.get_index_members(industry, start_date=start, end_date=end)
    non_cycle.append(d)
non_cycle = pd.concat(non_cycle, axis=1)
non_cycle['_'+new_code] =  non_cycle.any(axis=1)

data_source.h5DB.save_factor(non_cycle[['_'+new_code]].astype('int'), '/indexes/')

# 构造指数收益率
pct = data_source.h5DB.load_factor('changeper', '/indexprices/cs_level_1/', ids=industry_codes,
                                   dates=data_source.trade_calendar.get_trade_days(start, end))
non_cycle_pct = pct.groupby(level=0).mean().rename(columns={'changeper':'daily_returns_%'})
non_cycle_pct.index = pd.MultiIndex.from_product([non_cycle_pct.index, [new_code]], names=['date', 'IDs'])
data_source.h5DB.save_factor(non_cycle_pct, '/indexprices/')