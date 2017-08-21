# 单因子测试主函数

from fama_macbeth.config import parse_config
from utils import AttrDict
from fama_macbeth.fmreg import run_fama_macbeth, save_result
import importlib.util as ilu


def run(config_path):
    # 加载设置文件
    config = parse_config(config_path)
    config = AttrDict(config)

    print("加载因子...")
    factors = load_factors(config)

    print("模型启动...")
    params, params_st = run_fama_macbeth(factors, config)

    print("存储结果...")
    save_result(params, params_st, config.result_file_dir)


def load_factors(config):
    spec = ilu.spec_from_file_location("factor_list", config.factor_list_file)
    factor_list = ilu.module_from_spec(spec)
    spec.loader.exec_module(factor_list)
    return factor_list.factor_list


run("D:/FactorLib/fama_macbeth/config.yml")