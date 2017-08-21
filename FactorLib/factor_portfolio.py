# encoding=utf-8
import pandas as pd

class FactorPortfolio(object):
    '''因子组合
    因子分组中将每一组封装成一个类
    '''
    def __init__(self):
        self.start_date = None
        self.positions = pd.DataFrame()

    def update_positions(self, new_positions):
        '''

        :param
            new_positions: DataFrame with index:Date,IDs and column:factor_data
        :return:
        '''
        if self.positions.empty:
            self.positions = new_positions
        else:
            temp = self.positions[~self.positions.index.isin(new_positions.index)]
            self.positions = temp.append(new_positions).sort_index()
        self.start_date = self.positions.index.levels[0].min()

    def get_snapshot_by_date(self, date):
        try:
            return self.positions.ix[date]
        except Exception as e:
            return None

    def set_start_date(self, start_date):
        self.start_date = start_date

    def get_dates(self):
        try:
            return self.positions.index.levels[0].tolist()
        except:
            return []

class GroupInfo(object):
    def __init__(self, n_groups=10):
        self.total_groups = n_groups
        self.portfolios = {x:FactorPortfolio() for x in range(1, n_groups+1)}

    def update_info(self, new_data):
        for i in range(1, self.total_groups+1):
            self.portfolios[i].update_positions(new_data[new_data['group_id']==i][['factor_data']])

    def get_snapshot_by_date(self, date):
        _l = []
        for i, portfolio in self.portfolios.items():
            temp = portfolio.get_snapshot_by_date(date)
            if temp is None:
                return None
            temp['group_id'] = i
            _l.append(temp)
        return pd.concat(_l)

    def to_frame(self):
        _l = []
        for i, portfolio in self.portfolios.items():
            temp = portfolio.positions.copy()
            temp['group_id'] = i
            _l.append(temp)
        return pd.concat(_l)

    def get_state(self):
        return {'total_groups':self.total_groups,'portfolios':self.to_frame()}

    def get_dates(self):
        _l = []
        for i in self.portfolios:
            _l += self.portfolios[i].get_dates()
        return list(set(_l))

    def set_state(self, state):
        self.total_groups = state['total_groups']
        self.update_info(state['portfolios'])

class FactorGroups(object):
    def __init__(self):
        self.group_methods = None
        self.group_info = None

    def initialize(self, methods, n_groups):
        self.group_methods = methods
        self.group_info = {x:GroupInfo(n_groups) for x in methods}

    def to_frame(self):
        _l = []
        for method in self.group_methods:
            group = self.group_info[method].to_frame()
            columns = pd.MultiIndex.from_product([[method], group.columns])
            group.columns = columns
            _l.append(group)
        return pd.concat(_l, axis=1)

    def from_frame(self, frame):
        for method in self.group_methods:
            try:
                temp = frame.ix[:, method]
            except Exception as e:
                continue
            self.group_info[method].update_info(temp)

    def get_state(self):
        _d = {}
        _d['group_methods'] = self.group_methods
        _d['group_info'] = self.to_frame()
        return _d

    def set_state(self, state):
        self.group_methods = state['group_methods']
        self.from_frame(state['group_info'])

    def get_snapshot_by_date(self, date):
        _l = []
        for m, x in self.group_info:
            group_id = x.get_snapshot_by_date(date)
            if group_id is None:
                return None
            group_id.columns = pd.MultiIndex.from_product([[m],group_id.columns])
            _l.append(group_id)
        return pd.concat(_l, axis=1)

    def get_dates(self):
        _l = []
        for i in self.group_info:
            _l += self.group_info[i].get_dates()
        return list(set(_l))

