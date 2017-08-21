"""汇集缺失值的处理函数"""
from environment import Environment

def delete(factor):
    return factor.data.dropna()


missing_value_func_dict = {'delete':delete}