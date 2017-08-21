# 单因子测试主函数

from single_factor_test.config import parse_config
from utils import AttrDict
from utils.tool_funcs import deep_update_dict
from environment import Environment
from factors.panel_factor import panelFactor
from utils.mod_handler import (FactorDataProcessModHandler,
                               FactorPerformanceModHandler,
                               FactorStoreModHandler
                               )

import importlib.util as ilu


def run(config_path, user_config=None):
    # 加载设置文件
    config = parse_config(config_path)
    if user_config is None:
        user_config = {}
    deep_update_dict(user_config, config)
    config = AttrDict(config)

    # 初始化运行环境
    print("初始化运行环境...")
    _env = Environment(config)
    _env._initialize()

    # 加载因子
    print("加载因子...")
    factors = load_factors(config)
    factor_list = []
    for factor in factors:
        factor_list.append(panelFactor(*factor))
    _env.set_factors(factor_list)
    
    # 数据提取加工
    print("正在计算...")
    data_handler = FactorDataProcessModHandler()
    data_handler.set_env(_env)
    data_handler.mod_start()
 
    # 计算因子收益
    print("计算收益...")
    performance_handler = FactorPerformanceModHandler()
    performance_handler.set_env(_env)
    performance_handler.mod_start()

    # 因子存储
    print("存储因子...")
    store_handler = FactorStoreModHandler()
    store_handler.set_env(_env)
    store_handler.mod_start()


def load_factors(config):
    spec = ilu.spec_from_file_location("factor_list", config.factor_list_file)
    factor_list = ilu.module_from_spec(spec)
    spec.loader.exec_module(factor_list)
    return factor_list.factor_list

if __name__ == '__main__':
    run("D:/FactorLib/single_factor_test/config.yml")