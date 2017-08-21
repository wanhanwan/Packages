from utils import AttrDict
from single_factor_test.config import parse_config
from data_source import data_source
from single_factor_test.factor_list import *
from single_factor_test.selfDefinedFactors import *
from utils.disk_persist_provider import DiskPersistProvider
from datetime import datetime
from utils.tool_funcs import tradecode_to_windcode
import os
from generate_stocks import funcs, stocklist
import pandas as pd


class AbstractStockGenerator(object):
    def __init__(self):
        self.config = None
        self.temp_path = os.path.join(os.getcwd(), 'temp')
        self.persist_provider = DiskPersistProvider(self.temp_path)

    def _prepare_config(self, **kwargs):
        config_dict = parse_config('config.yml')
        for k, v in kwargs:
            if k in config_dict:
                config_dict[k] = v
        if isinstance(config_dict['factors'][0], str):
            factors = []
            for f in config_dict['factors']:
                factors.append(globals()[f])
            config_dict['factors'] = factors
        self.config = AttrDict(config_dict)

    def generate_stocks(self, start, end):
        raise NotImplementedError

    def generate_tempdata(self, start, end, **kwargs):
        pass

    def _update_stocks(self, start, end):
        stocks = self.generate_stocks(start, end).reset_index(level=1)
        stocks['IDs'] = stocks['IDs'].apply(tradecode_to_windcode)
        if os.path.isfile(self.config.stocklist.output):
            raw = pd.read_csv(self.config.stocklist.output).set_index('date')
            new = raw.append(stocks)
            new = new[~new.index.duplicated(keep='last')].reset_index()
            new.reset_index().to_csv(self.config.stocklist.output, float_format="%.4f", index=False)
        else:
            stocks.reset_index().to_csv(self.config.stocklist.output, float_format="%.4f", index=False)

    def _update_tempdata(self, start, end, **kwargs):
        temp = self.generate_tempdata(start, end, **kwargs)
        for k, v in temp.items():
            name = "%s_%s"%(k, datetime.now().strftime("%Y%m%d%H%M"))
            self.persist_provider.dump(v, name)

    def update(self, start, end, **kwargs):
        self._prepare_config(**kwargs)
        self._update_stocks(start, end)
        self._update_tempdata(start, end)


class FactorInvestmentStocksGenerator(AbstractStockGenerator):
    def __init__(self):
        super(FactorInvestmentStocksGenerator, self).__init__()
        self.factors = None
        self.direction = None
        self.factor_data = None

    def _set_factors(self):
        self.factors, self.direction = funcs._to_factordict(self.config.factors)

    def _prepare_data(self, start, end):
        dates = data_source.trade_calendar.get_trade_days(start, end, self.config.rebalance_frequence)
        stockpool = funcs._stockpool(self.config.stockpool, dates, self.config.stocks_unable_trade)
        factor_data = funcs._load_factors(self.factors, stockpool)
        score = getattr(funcs, self.config.scoring_mode.function)(factor_data, industry_name=self.config.stocklist.industry,
                                                                  method=self.config.scoring_mode.drop_outlier_method)
        total_score = funcs._total_score(score, self.direction, self.config.weight)
        factor_data = factor_data.merge(total_score, left_index=True, right_index=True, how='left')
        self.direction['total_score'] = 1
        self.factor_data = factor_data

    def generate_stocks(self, start, end):
        self._set_factors()
        self._prepare_data(start, end)
        stocks = getattr(stocklist,self.config.stocklist.function)(self.factor_data, 'total_score', 1,
                                                                   self.config.stocklist.industry_neutral,
                                                                   self.config.stocklist.benchmark,
                                                                   self.config.stocklist.industry,
                                                                   prc=self.config.stocklist.prc)
        return stocks

    def generate_tempdata(self, start, end, **kwargs):
        temp = []
        for i in [x for x in self.factor_data.columns if x != 'total_score']:
            temp.append(
                getattr(stocklist, self.config.stocklist.function)(self.factor_data,
                                                                   i,
                                                                   self.direction[i],
                                                                   self.config.stocklist.industry_neutral,
                                                                   self.config.stocklist.benchmark,
                                                                   self.config.stocklist.industry,
                                                                   prc=self.config.stocklist.prc))
        temp = pd.concat(temp, axis=1, ignore_index=True).fillna(0)
        temp.columns = [x + '_weight' for x in self.factor_data.columns if x != 'total_score']
        temp['Weight'] = temp.mean(axis=1)
        temp = pd.concat([temp, self.factor_data.reindex(temp.index)], axis=1)
        return {'score_details': temp}
