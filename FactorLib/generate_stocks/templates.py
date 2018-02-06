from ..utils import AttrDict
from ..single_factor_test.config import parse_config
from ..data_source.base_data_source_h5 import data_source
from ..data_source.converter import IndustryConverter
from ..data_source.wind_plugin import realtime_quote
from ..single_factor_test.factor_list import *
from ..single_factor_test.selfDefinedFactors import *
from ..utils.disk_persist_provider import DiskPersistProvider
from datetime import datetime
from ..utils.tool_funcs import tradecode_to_windcode, ensure_dir_exists
import os
from ..generate_stocks import funcs, stocklist
import pandas as pd


class AbstractStockGenerator(object):
    def __init__(self):
        self.config = None
        self.temp_path = os.path.join(os.getcwd(), 'temp')
        ensure_dir_exists(self.temp_path)
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
        """
        注：新生成的股票列表只会增量更新，不会对旧的股票列表进行修改。
        """
        stocks = self.generate_stocks(start, end).reset_index(level=1)
        stocks['IDs'] = stocks['IDs'].apply(tradecode_to_windcode)
        if os.path.isfile(self.config.stocklist.output):
            csvf = open(self.config.stocklist.output)
            raw = pd.read_csv(csvf, parse_dates=['date']).set_index('date')
            csvf.close()
            stocks = stocks[~stocks.index.isin(raw.index)]
            new = raw.append(stocks)
            new.sort_index().reset_index().to_csv(self.config.stocklist.output, float_format="%.6f", index=False)
        else:
            stocks.sort_index().reset_index().to_csv(self.config.stocklist.output, float_format="%.6f", index=False)

    def _update_tempdata(self, start, end, **kwargs):
        temp = self.generate_tempdata(start, end, **kwargs)
        for k, v in temp.items():
            name = "%s_%s"%(k, datetime.now().strftime("%Y%m%d%H%M"))
            self.persist_provider.dump(v, name, protocol=2)

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
                                                                   prc=self.config.stocklist.prc,
                                                                   top=self.config.stocklist.__dict__.get('top', None)
                                                                   )
        return stocks

    def generate_tempdata(self, start, end, **kwargs):
        dates = self.factor_data.index.get_level_values(0).unique().tolist()
        ids = self.factor_data.index.get_level_values(1).unique().tolist()
        indu = data_source.sector.get_stock_industry_info(ids, industry=self.config.stocklist.industry, dates=dates)
        temp = self.factor_data.join(indu, how='left')
        return {'score_details': temp}


class FactorTradesListGenerator(AbstractStockGenerator):
    """实盘模拟的因子组合生成器"""
    def __init__(self):
        super(FactorTradesListGenerator, self).__init__()
        self.factors = None
        self.direction = None
        self.factor_data = None

    def _set_factors(self):
        self.factors, self.direction = funcs._to_factordict(self.config.factors)

    def _prepare_data(self):
        date = data_source.trade_calendar.get_latest_trade_days(datetime.today().date())
        yesterday = data_source.trade_calendar.tradeDayOffset(date, -1)
        stockpool = funcs._stockpool(self.config.stockpool, [yesterday], self.config.stocks_unable_trade)
        factor_data = funcs._load_latest_factors(self.factors, stockpool.xs(yesterday, level=0))
        factor_data.index = pd.MultiIndex.from_product([[pd.to_datetime(yesterday)], factor_data.index], names=['date', 'IDs'])

        score = getattr(funcs, self.config.scoring_mode.function)(factor_data, industry_name=self.config.stocklist.industry,
                                                                  method=self.config.scoring_mode.drop_outlier_method)
        total_score = funcs._total_score(score, self.direction, self.config.weight)
        factor_data = factor_data.merge(total_score, left_index=True, right_index=True, how='left')
        self.direction['total_score'] = 1
        self.factor_data = factor_data

    def generate_stocks(self, start, end):
        self._set_factors()
        self._prepare_data()
        stocks = getattr(stocklist,self.config.stocklist.function)(self.factor_data, 'total_score', 1,
                                                                   self.config.stocklist.industry_neutral,
                                                                   self.config.stocklist.benchmark,
                                                                   self.config.stocklist.industry,
                                                                   prc=self.config.stocklist.prc,
                                                                   top=self.config.stocklist.top,
                                                                   indu_weight=self.config.stocklist.indu_weight)
        return stocks

    def generate_tempdata(self, start, end, **kwargs):
        dates = self.factor_data.index.get_level_values(0).unique().tolist()
        ids = self.factor_data.index.get_level_values(1).unique().tolist()
        indu = data_source.sector.get_stock_industry_info(ids, industry=self.config.stocklist.industry, dates=dates)
        temp = self.factor_data.join(indu, how='left')
        return {'_'.join(self.factor_data.columns): temp}

    def _update_stocks(self, start, end):
        stocks = self.generate_stocks(start, end).reset_index(level=0, drop=True)
        stocks.rename_axis('股票代码', inplace=True)

        capital = self.config.stocklist.cash
        stock_ids = stocks.index.tolist()
        tradeprice = realtime_quote(['rt_last'], ids=stock_ids)['rt_last']
        tradeorders = (stocks['Weight'] * capital / tradeprice / 100).to_frame('手数').rename_axis('股票代码').reset_index()
        tradeorders = tradeorders.join(stocks, on=['股票代码']).rename(columns={'Weight': '权重'})
        writer = pd.ExcelWriter(self.config.stocklist.output, engine='xlsxwriter')

        tradeorders[['股票代码', '权重', '手数']].to_excel(writer, index=False, float_format='%.6f', sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.0000'})
        worksheet.set_column('C:C', None, format1)
        writer.save()


class OptimizedPortfolioGenerator(AbstractStockGenerator):
    """使用风险优化模型生成股票组合模板
    """
    def generate_stocks(self, start, end):
        risk_datasource = self.config.stocklist.data_source
        benchmark = 'NULL' if self.config.benchmark is None else self.config.benchmark


    def _set_factors(self):
        self.factors, self.direction = funcs._to_factordict(self.config.factors)

    def _parse_optimize_config(self):
        constraints = self.config.stocklist.constraints


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


class _OptimizeConfig(object):
    def __init__(self, attr_config, dates):
        self._config = attr_config
        self._dates = dates

    @property
    def constraints(self):
        from FactorLib.utils.tool_funcs import parse_industry
        d = {}
        if 'Style' in self._config:
            d['Style'] = self._config.Style.__dict__
        if 'Indu' in self._config:
            i = self._config.Indu.__dict__.copy()
            if self._config.Indu.industry_dict is not None:
                if self._config.Indu.industry is None:
                    i.pop('industry')
                    d['Indu'] = i
                else:
                    indu_name = i['industry']
                    dummy = data_source.sector.get_industry_dummy(industry=indu_name, dates=self._dates)
                    c = [x.encode('utf8') for x in dummy.columns]
                    dummy.columns = [str(x) for x in IndustryConverter.name2id(parse_industry(indu_name), c)]
                    d['UserLimit'] = {}
        return