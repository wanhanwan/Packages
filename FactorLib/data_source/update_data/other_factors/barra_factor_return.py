from FactorLib.riskmodel.barra_modelV2 import BarraFactorReturn


def cal_barra_factor_return(start, end, **kwargs):
    factorret = BarraFactorReturn('xy')
    factorret.set_args('开始时间', start)
    factorret.set_args('结束时间', end)
    factorret.set_args('行业因子', 'diversified_finance_cs', commit=True)
    factorret.getFactorReturn()
    factorret.save_results()


if __name__ == '__main__':
    cal_barra_factor_return('20100101', '20180524')