# coding: utf-8
# from FactorLib.data_source.stock_universe import StockUniverse


# 装饰器，当参数ids是StockUniverse实例时，将其转成具体的股票列表
def handle_ids(func):
    def wrapper(*args, **kwargs):
        ids = kwargs.get('ids', None)
        if not isinstance(ids, list) and ids is not None:
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
