from single_factor_test.factor_list import *
from single_factor_test.selfDefinedFactors import *

# 研究员
researcher = 'wanshuai'

# 策略名称
name = '价值动量因子'

# 定义因子
factors = [StyleFactor_MomentumFactor, StyleFactor_ValueFactor]

# 起始日期
start = '20170101'
end = '20170731'

# 换仓频率
rebalance_frequence = '1m'

# 股票池, 默认为全部A股
stockpool = '全A'

# 定义不交易股票，默认为一字涨跌停、停牌、ST、新股
stocks_unable_trade = 'typical'

# 因子变量去除异常值方法, 方法包括: FixedRatio\Mean-Variance\MAD\BoxPlot
drop_outlier_method = 'MAD'

# 各因子的权重, 默认为None(等权)
weight = None

# 是否行业中性
industry_neutral = True

# 基准指数
benchmark = '000905'

# 行业分类
industry = '中信一级'

# 分位数入选门槛
prc = 0.05

