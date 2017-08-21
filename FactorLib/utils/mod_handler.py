from utils.tool_funcs import import_mod, ensure_dir_exists
from utils.excel_io import save_details_to_excel, save_summary_to_excel, save_stock_list
from utils import AttrDict
from factor_performance.plotting import FuncList
import pandas as pd

class FactorDataProcessModHandler(object):
    def __init__(self):
        self._env = None
        self._mod_list = list()
        self._mod_args = dict()
        self._mod_funcs = dict()
        
    def set_env(self, environment):
        self._env = environment
        
        config = environment._config
        
        for mod_name in config.mod.__dict__:
            mod_config = getattr(config.mod, mod_name)
            if not mod_config.enabled:
                continue
            self._mod_list.append((mod_name, mod_config))
        for idx, (mod_name, mod_config) in enumerate(self._mod_list):
            if hasattr(mod_config, 'lib'):
                lib_name = mod_config.lib
            else:
                lib_name = 'FactorLib.' + mod_name
            mod_module = import_mod(lib_name)
            if mod_module is None:
                del self._mod_list[idx]
                continue
            func_list = mod_module.FuncList
            if mod_config.func is not None:
                for ifunc in mod_config.func:
                    self._mod_funcs[(mod_name, ifunc)] = func_list[ifunc]
            if mod_config.kwargs is not None:
                self._mod_args[mod_name] = mod_config.kwargs
            else:
                self._mod_args[mod_name] = AttrDict()
        self._mod_list.sort(key=lambda item: getattr(item[1], 'priority', 100))

    def mod_start(self):
        for factor in self._env._factors:
            if not self._env._factor_group_info_dates.isin(factor.group_info.get_dates()).all():
                for mod_name, mod_config in self._mod_list:
                    for imod_name, func in self._mod_funcs:
                        if mod_name == imod_name:
                            self._mod_funcs[(imod_name,func)](
                                factor, env=self._env, **self._mod_args[mod_name].__dict__)


class FactorPerformanceModHandler(object):
    def __init__(self):
        self._env = None

    def set_env(self, env):
        self._env = env

    def mod_start(self):
        for factor in self._env._factors:
            group_id = factor.group_info.to_frame().ix[self._env._factor_group_info_dates].xs('group_id',level=1,axis=1)
            group_id = group_id.unstack().reindex(self._env._all_trade_dates, method='ffill').stack()
            group_id = group_id.stack().reset_index().rename(columns={'level_0':'date','level_2':'methods',0:'group_id'})
            common = pd.merge(group_id, self._env._stock_return.reset_index(), how='left')

            # 计算分组收益率
            factor_returns = common.groupby(['date','methods','group_id'])['daily_returns'].mean()
            factor_returns = factor_returns.reset_index().fillna(0).pivot_table(
                index='date',columns=['methods','group_id'],values='daily_returns')
            factor.group_return.from_frame(factor_returns, self._safe_convert_series(self._env._benchmark_return))

            # 计算多空收益率
            total_groups = factor_returns.columns.levels[1].max()
            long = factor_returns.xs(1, level=1, axis=1)
            short = factor_returns.xs(total_groups, level=1, axis=1)
            factor.long_short_return.update_info(long, short)

            # 计算第一组的超额收益率
            benchmark_return = pd.DataFrame(
                {x:self._safe_convert_series(self._env._benchmark_return) for x in factor.group_return.group_returns}
            )
            factor.first_group_active_return.update_info(long, benchmark_return)

            #计算因子IC
            self._env._ic_calculator.set_factor(factor)
            ic_series = self._env._ic_calculator.calculate(freq = self._env._config.freq)
            factor.ic_series.update_info(*ic_series)

    def _safe_convert_series(self, data):
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        else:
            return data

class FactorStoreModHandler(object):
    def __init__(self):
        self._env = None

    def set_env(self, env):
        self._env = env

    def mod_start(self):
        import os
        for factor in self._env._factors:
            factor_path = ensure_dir_exists(
                os.path.join(self._env._config.extra.result_file_dir, factor.name))

            # 因子持久化存储
            state = factor.get_state()
            self._env._disk_persist_provider.dump(state, os.sep.join([factor.name]*2))

            # 绘图存储
            for func in FuncList:
                fig, ax = func(factor)
                fig.savefig(os.path.join(factor_path, func.__name__+'.png'))
                fig.clf()

            # excel存储
            save_details_to_excel(factor, factor_path, self._env)
            if factor.stock_list:
                save_stock_list(factor, factor_path, self._env)
        save_summary_to_excel(self._env._factors, self._env._config.extra.result_file_dir, self._env)
