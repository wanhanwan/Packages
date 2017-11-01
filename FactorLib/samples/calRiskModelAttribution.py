"""风险模型的业绩归因示例
"""

from FactorLib.data_source.base_data_source_h5 import tc
from FactorLib.factor_performance.analyzer import Analyzer

# 时间区间
start_date = '20140101'
end_date = '20170930'

analyzer = Analyzer(r"D:\data\factor_investment_strategies\兴业风格_价值成长等权\backtest\BTresult.pkl",
                    benchmark_name='000905')

# 计算风险敞口
dates = tc.get_trade_days(start_date, end_date)
style_expo, indu_expo, risk_expo = analyzer.portfolio_risk_expo('xy', dates)

# 业绩归因
attribution = analyzer.range_attribute(start_date, end_date)