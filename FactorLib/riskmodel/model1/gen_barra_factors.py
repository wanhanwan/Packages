from FactorLib.single_factor_test.factor_list import get_factors
from FactorLib.riskmodel.stockpool import get_estu, _parse_estuconfig
from FactorLib.riskmodel import riskmodel_data_source, cross_section_operate
from FactorLib.data_source.base_data_source_h5 import tc, h5
from FactorLib.utils.tool_funcs import distribute_equal, import_module
from multiprocessing import cpu_count, Process
import os
import gc
import pandas as pd


def prepare_factors(config_file):
    descriptors = []
    descriptors += get_factors(config_file.risk_descriptors)
    descriptors += get_factors(config_file.others)
    return descriptors


def prepare_save_info(config_file):
    _args = dict()
    _args['factor2save'] = config_file.save_info['factor_to_save']
    _args['run_type'] = config_file.save_info['run_type']
    _args['factor_save_path'] = config_file.save_info['factor_save_path']
    _args['split_length'] = config_file.save_info['split_length']
    _args['industry_factor'] = config_file.save_info['industry_factor']
    _args['regress_weight_factor'] = config_file.save_info['regress_weight_factor']
    if config_file.save_info['cpu_use'] is not None:
        _args['cpu_use'] = config_file.save_info['cpu_use']
    else:
        _args['cpu_use'] = cpu_count()
    return _args


def parallel_func(args):
    _data_source = args['data_source']
    _all_dates = args['all_dates']
    _n_dates = len(args['all_dates'])
    n_funcmodules = len(args['func_modules'])
    estu = args['estu']
    loop = 0

    while loop < _n_dates:
        print("进程%d 执行%d/%d"%(args['PID'], loop/args['save_info']['split_length']+1, _n_dates//args['save_info']['split_length']+1))
        factor_gen = cross_section_operate.CSOperator(_data_source, _data_source.h5_db)
        gc.collect()
        _idx = pd.IndexSlice
        iestu = estu.loc[_idx[_all_dates[loop:loop+args['save_info']['split_length']], :], :]
        factor_gen.set_dimension(iestu)
        factor_gen.prepare_data()

        for i in range(n_funcmodules):
            cross_section_operate.FunctionModule[args['func_modules'][i]['func_name']](factor_gen, **args['func_modules'][i]['arg'])
        factor_gen.save_factor(args['save_info'])
        loop += args['save_info']['split_length']
    return


def generate_barra_datasource(arg):
    factor_dict = {x: arg['save_info']['factor_save_path'] for x in arg['save_info']['factor2save']}
    factor_dict[arg['save_info']['industry_factor']] = get_factors([arg['save_info']['industry_factor']])[0][1]
    factor_dict[arg['save_info']['regress_weight_factor']] = get_factors([arg['save_info']['regress_weight_factor']])[0][1]
    factor_names = arg['save_info']['factor2save']
    factor_names.append(arg['save_info']['industry_factor'])
    factor_names.append(arg['save_info']['regress_weight_factor'])
    data_source = riskmodel_data_source.RiskModelDataSourceOnH5(h5_db=h5)
    data_source.factor_names = factor_names
    data_source.factor_dict = factor_dict
    data_source.set_estu(arg['estu_config']['ESTU'])
    data_source.name = 'DS-Barra'
    data_source.save_info(path=os.path.dirname(__file__))


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    import config
    Args = {}
    start = '20170101'
    end = '20170906'
    config_file_path = "config.py"

    # config = import_module('config', config_file_path)
    all_dates = tc.get_trade_days(start, end, retstr=None)
    Args['all_dates'] = all_dates
    estu_config = _parse_estuconfig(config)
    estu = get_estu(all_dates, estu_config)
    Args['estu_config'] = estu_config
    Args['estu'] = estu
    Args['descriptors'] = prepare_factors(config)
    Args['func_modules'] = config.funcs
    Args['save_info'] = prepare_save_info(config)
    data_source = riskmodel_data_source.RiskModelDataSourceOnH5(h5_db=h5)
    data_source.set_dimension(Args['estu'])
    data_source.prepare_factor(table_factor=Args['descriptors'])
    Args['data_source'] = data_source

    if Args['save_info']['run_type'] == 'S':
        Args['PID'] = 0
        parallel_func(Args)
    else:
        PIDs = range(1, Args['save_info']['cpu_use']+1)
        n_dates = len(all_dates)
        n_subdates = distribute_equal(n_dates, Args['save_info']['cpu_use'])
        procs = []

        for pid in PIDs:
            Args['PID'] = pid
            start_ind = sum(n_subdates[:pid-1])
            Args['all_dates'] = all_dates[start_ind:start_ind+n_subdates[pid-1]]
            procs.append(Process(target=parallel_func, args=(Args,)))
            procs[-1].start()
        for iproc in procs:
            iproc.join()
    generate_barra_datasource(Args)
