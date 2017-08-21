from .factor import factor
from factor_portfolio import FactorGroups
from factor_performance.fund_performance import LongShortReturn, FactorGroupReturn
from factor_performance.ic_analyser import IC_Analyzer
import pandas as pd

class panelFactor(factor):
    def __init__(self, name, axe, direction=1):
        super(panelFactor, self).__init__(name, axe)
        self.direction = direction
        self.group_info = FactorGroups()
        self.stock_list = {}
        self.data = pd.DataFrame()
        self.group_return = FactorGroupReturn()
        self.long_short_return = LongShortReturn()
        self.first_group_active_return = LongShortReturn()
        self.ic_series = IC_Analyzer()

    def set_state(self, state):
        self.group_return.set_state(state['group_return'])
        self.group_info.set_state(state['group_info'])
        self.ic_series.set_state(state['ic_series'])
        self.first_group_active_return.set_state(state['first_group_active_return'])
        self.long_short_return.set_state(state['long_short_return'])
        self.stock_list = state['stock_list']
        self.direction = state['direction']

    def get_state(self):
        _d = {}
        _d['name'] = self.name
        _d['axe'] = self.axe
        _d['direction'] = self.direction
        _d['group_info'] = self.group_info.get_state()
        _d['stock_list'] = self.stock_list
        _d['group_return'] = self.group_return.get_state()
        _d['long_short_return'] = self.long_short_return.get_state()
        _d['first_group_active_return'] = self.first_group_active_return.get_state()
        _d['ic_series'] = self.ic_series.get_state()
        return _d
    
    def initialize(self, env):
        self.group_info.initialize(
            methods=env._config.mod.grouping_manage.func,
            n_groups=env._config.mod.grouping_manage.kwargs.total_groups
        )
        self.group_return.initialize(
            methods=env._config.mod.grouping_manage.func,
            n_groups=env._config.mod.grouping_manage.kwargs.total_groups                        
        )
        self.long_short_return.initialize(
            methods=env._config.mod.grouping_manage.func
        )
        self.first_group_active_return.initialize(
            methods=env._config.mod.grouping_manage.func
        )        





