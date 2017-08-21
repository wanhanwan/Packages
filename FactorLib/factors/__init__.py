import pandas as pd
from .panel_factor import panelFactor

def load_factor(factor_path):
    """加载因子"""
    factor_data = pd.read_pickle(factor_path)
    # 初始化因子
    name = factor_data['name']
    axe = factor_data['axe']
    direction = factor_data['direction']
    factor = panelFactor(name, axe, direction)
    
    group_info_methods = factor_data['group_info']['group_methods']
    group_info_groups = factor_data['group_info']['group_info'].loc[:, (group_info_methods[0], 'group_id')].max()
    factor.group_info.initialize(methods=group_info_methods,n_groups=group_info_groups)
    
    group_return_methods = factor_data['group_return']['group_methods']
    group_return_groups = factor_data['group_return']['group_returns'][group_info_methods[0]]['total_groups']
    factor.group_return.initialize(methods=group_return_methods,n_groups=group_return_groups)
    
    long_short_return_methods = factor_data['long_short_return']['methods']
    factor.long_short_return.initialize(methods=long_short_return_methods)
    
    first_group_active_return_methods = factor_data['first_group_active_return']['methods']
    factor.first_group_active_return.initialize(methods=first_group_active_return_methods)
    
    factor.set_state(factor_data)
    return factor