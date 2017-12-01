"""计算Barra因子的因子收益率"""
from FactorLib.riskmodel.barra_modelV2 import BarraFactorReturn


factorret = BarraFactorReturn('xy')
factorret.set_args('开始时间', '20171118')
factorret.set_args('结束时间', '20171130', commit=True)
factorret.getFactorReturn()
factorret.save_results()