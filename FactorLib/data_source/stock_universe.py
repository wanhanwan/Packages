from FactorLib.data_source.base_data_source_h5 import sec
from fastcache import clru_cache
from QuantLib.stockFilter import _intersection, _difference, _union


class StockUniverse(object):
    def __init__(self, base_universe, other=None, algorithm=None):
        self.base = base_universe
        self.other = other
        self.algorithm = algorithm

    def __add__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _union)
        else:
            return self.__class__(self, other, _union)

    def __sub__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _difference)
        else:
            return self.__class__(self, other, _difference)

    def __mul__(self, other):
        if isinstance(other, str):
            return self.__class__(self, self.__class__(other), _intersection)
        else:
            return self.__class__(self, other, _intersection)

    @clru_cache()
    def get(self, start_date=None, end_date=None, dates=None):
        if self.algorithm is None:
            return sec.get_index_members(self.base, dates, start_date, end_date)
        else:
            base = self.base.get(start_date, end_date, dates)
            other = self.other.get(start_date, end_date, dates)
            return self.algorithm(base, other)


if __name__ == '__main__':
    u1 = StockUniverse('000906')
    u2 = StockUniverse('000300')

    u = u1 - u2
    s = u.get('20100104', '20110106')
    print(s)
    print(len(s))
