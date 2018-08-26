# coding: utf-8
# from FactorLib.data_source.stock_universe import StockUnivers
from pandas import DataFrame, Series
from functools import wraps
from collections import OrderedDict


# 装饰器，当参数ids是StockUniverse实例时，将其转成具体的股票列表
def handle_ids(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ids = kwargs.get('ids', None)
        if not isinstance(ids, (list, DataFrame, Series)) and ids is not None:
            s = 'start_date'
            if 'start' in kwargs:
                s = 'start'
            e = 'end_date'
            if 'end' in kwargs:
                e = 'end'
            stocks = ids.get(start_date=kwargs.get(s, None), end_date=kwargs.get(e, None),
                             dates=kwargs.get('dates', None))
            if 'idx' in func.__code__.co_varnames:
                return func(*args, idx=stocks, **{k: v for k, v in kwargs.items() if k not in ['ids', 'idx']})
            else:
                tmp = func(*args, **{k: v for k, v in kwargs.items() if k not in ['ids']})
                return tmp.reindex(stocks.index)
        else:
            return func(*args, **kwargs)
    return wrapper


# 使用先进先出法的OrderedDict。
# 当key超过预设数量时，自动删除最先进入字典的字段。
class FIFODict(OrderedDict):
    def __init__(self, max_keys=10):
        super(FIFODict, self).__init__()
        self.max_keys = max_keys

    def __setitem__(self, key, value):
        contain_key = 1 if key in self else 0
        if len(self) - contain_key >= self.max_keys:
            self.popitem(last=False)
        if contain_key:
            del self[key]
        super(FIFODict, self).__setitem__(key, value)


if __name__ == '__main__':
    my_dict = FIFODict(3)
    my_dict['a'] = 1
    my_dict['b'] = 2
    my_dict['c'] = 3
    my_dict['d'] = 4
    my_dict['a'] = 9
    print(my_dict)
