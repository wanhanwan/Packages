"""计算Barra因子的因子收益率"""
from FactorLib.riskmodel.barra_modelV2 import BarraFactorReturn


factorret = BarraFactorReturn('xy')
factorret.set_args('开始时间', '20171201')
factorret.set_args('结束时间', '20180112', commit=True)
# factorret.set_args('行业因子', 'cs_level_1', commit=True)
factorret.getFactorReturn()
factorret.save_results()