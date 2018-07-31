from QuantLib.utils import DropOutlier
from QuantLib.stockFilter import drop_false_growth


def drop(factor, **kwargs):
    new_data = DropOutlier(factor.data,
                           factor_name=factor.data.columns[0],
                           ret_raw=False,
                           **kwargs)
    factor.data = new_data


def false_growth(factor, **kwargs):
    """去掉伪成长的股票"""
    env = kwargs.pop('env')
    new_data = drop_false_growth(factor.data, **kwargs)
    factor.data = new_data


FuncList = {'drop': drop, 'false_growth': false_growth}