# coding: utf-8

"""
优化器测试文件
使用兴业风格VGS因子作为选股信号
"""
from datetime import datetime
from FactorLib.riskmodel.optimizer_xyquant import Optimizer
from FactorLib.data_source.base_data_source_h5 import data_source, tc
import pandas as pd

# 待优化股票池
secID = '全A'
# 基准指数
benchmark = '000905'
# 待优化的信号
signal_name = 'StyleFactor_VG'
signal_dir = '/XYData/StyleFactor/'

optimal_assets = []
for date in tc.get_trade_days('20170801', '20171130', freq='1m', retstr=None):
    # date = datetime(2017, 2, 24)
    stockIDs = data_source.sector.get_index_members(ids=secID, dates=[date])
    # stockIDs = stockIDs[stockIDs.iloc[:, 0] == 1.0]
    # stockIDs = drop_suspendtrading(stockIDs)    # 剔除停牌股票
    stockIDs = stockIDs[stockIDs.iloc[:, 0] == 1.0].index.get_level_values(1).tolist()
    signal = data_source.load_factor(signal_name, signal_dir,
                                     dates=[date])[signal_name].reset_index(level=0, drop=True)

    opt = Optimizer(signal, stockIDs, date, ds_name='xy', benchmark=benchmark)
    opt.add_constraint('StockLimit', default_max=0.08)
    opt.add_constraint('Style', {'Size': 0.0})
    opt.add_constraint('TrackingError', 0.01/12)
    # opt.add_constraint('Indu')
    opt.add_constraint('UserLimit', {'factor_name': 'turn_60d', 'factor_dir': '/stock_liquidity/',
                                     'standard': True, 'limit': -0.1}, active=True)
    opt.solve()

    if opt.optimal:
        print("%s 权重优化成功" % date.strftime("%Y%m%d"))
        optimal_assets.append(opt.asset)
        style_expo, indu_expo, terr = opt.check_ktt()
    else:
        print("%s 权重优化失败：%s"%(date.strftime("%Y%m%d"), opt.solution_status))
optimal_assets = pd.concat(optimal_assets)
optimal_assets.to_csv(r"D:\spyder\guoqigaige_tr0020.csv")
