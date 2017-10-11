from FactorLib.riskmodel.barra_model import BarraModel

barra = BarraModel('barra', r'D:\Packages\FactorLib\riskmodel\model1')
barra.setDimension(start_date='20100101', end_date='20161231')
barra.getFactorReturnArgs()
barra.getFactorReturn(start_date='20100101', end_date='20161231')
barra.save_regress_results("D:/data/risk_model/regression_results")