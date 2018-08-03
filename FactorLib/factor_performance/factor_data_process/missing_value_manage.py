"""汇集缺失值的处理函数"""


def delete(factor, **kwargs):
    factor.data = factor.data.dropna()


FuncList = {'delete': delete}

