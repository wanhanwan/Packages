
from ..base_data_source_h5 import data_source
from ..update_data.value_factor import ValueFuncListDaily
from ..update_data.momentum_factor import MomentumFuncListDaily
from ..update_data.liquidity_factor import LiquidityFuncListDaily
from ..update_data.reverse_fator import ReverseFuncListDaily
from ..update_data.time_series_factor import TimeSeriesFuncListDaily
from ..update_data.alternative_factor import AlternativeFuncListDaily
from ..update_data.growth_factor import GrowthFuncListDaily
from ..update_data.profit_factor import ProfitFuncListDaily
from ..update_data.universe import UniverseFuncListDaily
from .other_factors import merge_accquisition, msci_china
from .other_factors import barra_factor_return, gtja_risky_stocks, dividends
from .dummies.class_define import classUpdateFuncsDaily


def change_indexmembers():
    from FactorLib.data_source.wind_financial_data_api import windchangecode, aindexmembers, aindexmemberswind
    from datetime import timedelta, datetime

    change_code = windchangecode.all_data
    for table in [aindexmembers, aindexmemberswind]:
        raw_data = table.load_h5(table.table_id)
        for old_id, new_id, change_dt in zip(
                change_code['IDs'].values, change_code['new_id'].values, change_code['change_dt'].values):
            if (new_id in raw_data['IDs'].values) and (old_id not in raw_data['IDs'].values):
                old = raw_data.query("IDs == @new_id").copy()
                new2add = old.copy()

                old = old[old['out_date'] >= change_dt]
                old['in_date'] = int(
                    (datetime.strptime(str(change_dt), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d'))

                new2add['IDs'] = old_id
                new2add['cur_sign'] = 0
                new2add.loc[new2add['out_date'] >= change_dt, 'out_date'] = change_dt

                new_data = raw_data[raw_data['IDs'] != new_id].append(old).append(new2add)
                table.save_data(new_data, if_exists='replace')


def dailyfactors(start, end):
    # 更新价值类因子数据
    for func in ValueFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新动量类因子
    for func in MomentumFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新流动性数据
    for func in LiquidityFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新反转因子
    for func in ReverseFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新时间序列类因子
    for func in TimeSeriesFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新另类因子
    for func in AlternativeFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新成长类因子
    for func in GrowthFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新盈利类因子
    for func in ProfitFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新股票池
    gtja_risky_stocks.get_risky_stocks(start, end, data_source=data_source)
    for func in UniverseFuncListDaily:
        func(start, end, data_source=data_source)

    # 更新哑变量
    for func in classUpdateFuncsDaily:
        func(start, end)

    # 其他函数
    change_indexmembers()
    merge_accquisition.update_raw_from_uqer(start, end, data_source=data_source)
    barra_factor_return.cal_barra_factor_return(start, end)
    dividends.get_dividends(start, end)
    msci_china.save_msci_members(start_date=start, end_date=end, data_source=data_source)