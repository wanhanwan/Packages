# coding: utf-8
"""组合归因分析"""

import pandas as pd
import numpy as np
from fastcache import clru_cache
from .riskmodel_data_source import RiskDataSource
from ..data_source.base_data_source_h5 import data_source


class RiskExposureAnalyzer(object):
    """
    风险因子的暴露分析框架,提供时间截面上风险暴露计算、图示等。
    行业分类可以自定义，通过h5数据源提取；风险因子除了BARRA风险因子之外也提供自定义功能。
    """
    def __init__(self, barra_datasource='xy', industry=None, risk_factors=None, stocks=None, benchmark=None):
        """
        Paramters:
        ==========
        barra_datasource: str
            BARRA风险数据源， 默认为'xy'
        industry: str
            行业分类，默认为None, 以BARRA模型中的行业因子替代。
        risk_factors: list-like
            风险因子，默认为None
        stocks: DataFrame
            股票持仓 DataFrame(index:[date, IDs], columns:[Weight])
        """
        self.barra_name = barra_datasource
        self.barra_ds = RiskDataSource(self.barra_name)
        self.industry = industry
        self.risk_factors = risk_factors
        self.stock_positions = stocks
        self.benchmark = benchmark
        self.barra_data = {}
        self.industry_data = {}
        self.risk_factors_data = {}

    @classmethod
    def from_df(cls, df, **kwargs):
        """
        从DataFrame导入股票持仓，返回类实例
        """
        return cls(stocks=df, **kwargs)

    @classmethod
    def from_csv(cls, csv_path, **kwargs):
        """
        从csv文件中导入股票持仓，返回类实例 \n
        csv文件的格式：日期  代码(wind格式)  权重
        """
        with open(csv_path) as f:
            stocks = pd.read_csv(f, header=0, index_col=None, parse_dates=['date'],
                                 converters={'IDs': lambda x: x[:6]})
        stocks = stocks.set_index(['date', 'IDs'])
        return cls(stocks=stocks, **kwargs)

    def _load_data(self, dates):
        """
        加载风险数据,风险数据包括BARRA风险因子，行业因子，自定义风险因子
        """
        old_dates = list(set(dates).intersection(set(self.barra_data.keys())))
        new_dates = list(set(dates).difference(set(old_dates)))
        old_barra_data = new_barra_data = pd.DataFrame()
        old_industry_data = new_industry_data = pd.DataFrame()
        old_riskfactors_data = new_risk_data = pd.DataFrame()
        if old_dates:
            old_barra_data = pd.concat([self.barra_data[x] for x in old_dates])
            old_industry_data = pd.concat([self.industry_data[x] for x in old_dates])
            if self.risk_factors is not None:
                old_riskfactors_data = pd.concat([self.risk_factors_data[x] for x in old_dates])
        if new_dates:
            new_barra_data = self.barra_ds.load_factors(factor_names='STYLE', dates=new_dates)
            if self.industry is not None:
                new_industry_data = data_source.sector.get_industry_dummy(ids=None, dates=new_dates, drop_first=False)
            else:
                new_industry_data = self.barra_ds.load_industry(ids=None, dates=new_dates)
            if self.risk_factors is not None:
                new_risk_data = data_source.h5DB.load_factors(self.risk_factors, dates=new_dates)
            for d in new_dates:
                self.barra_data[d] = new_barra_data.loc[[pd.to_datetime(d)]]
                self.industry_data[d] = new_industry_data.loc[[pd.to_datetime(d)]]
                if self.risk_factors is not None:
                    self.risk_factors_data[d] = new_risk_data.loc[[pd.to_datetime(d)]]
        barra_data = pd.concat([old_barra_data, new_barra_data]).sort_index()
        industry_data = pd.concat([old_industry_data, new_industry_data]).sort_index()
        if self.risk_factors is not None:
            risk_data = pd.concat([old_riskfactors_data, new_risk_data]).sort_index()
        else:
            risk_data = None
        return barra_data, industry_data, risk_data

    def _cal_risk_of_bchmrk(self, dates):
        """
        计算基准指数的风险因子值，若无基准指数则以零
        """
        barra, indus, risk_factor = self._load_data(dates)
        barra_b = pd.DataFrame(np.zeros((len(dates), len(barra.columns))), index=dates, columns=barra.columns)
        indus_b = pd.DataFrame(np.zeros((len(dates), len(indus.columns))), index=dates, columns=indus.columns)
        if risk_factor is not None:
            riskfactor_b = pd.DataFrame(np.zeros((len(dates), len(risk_factor.columns))), index=dates, columns=risk_factor.columns)
        else:
            riskfactor_b = None
        if self.benchmark is not None:
            weight_bchmrk = data_source.sector.get_index_weight(ids=self.benchmark, dates=dates)
            for d in dates:
                idata, iweight = barra.loc[d].align(weight_bchmrk.loc[d], join='right', axis=0)
                barra_b.loc[d] = idata.mul(iweight.iloc[:, 0], axis='index').sum()

                iindu, iweight = indus.loc[d].align(weight_bchmrk.loc[d], join='right', axis=0)
                indus_b.loc[d] = iindu.mul(iweight.iloc[:, 0], axis='index').sum()

                if risk_factor is not None:
                    irisk, iweight = indus.loc[d].align(weight_bchmrk.loc[d], join='right', axis=0)
                    riskfactor_b.loc[d] = irisk.mul(iweight.iloc[:, 0], axis='index').sum()
        return barra_b, indus_b, riskfactor_b

    def cal_risk_of_portfolio(self, dates):
        """
        计算组合的风险因子值
        """
        barra, indus, risk_factor = self._load_data(dates)
        barra_p = pd.DataFrame(np.zeros((len(dates), len(barra.columns))), index=dates, columns=barra.columns)
        indus_p = pd.DataFrame(np.zeros((len(dates), len(indus.columns))), index=dates, columns=indus.columns)
        if risk_factor is not None:
            riskfactor_p = pd.DataFrame(np.zeros((len(dates), len(risk_factor.columns))), index=dates, columns=risk_factor.columns)
        else:
            riskfactor_p = None
        for d in dates:
            idata, iweight = barra.loc[d].align(self.stock_positions.loc[pd.to_datetime(d)], join='right', axis=0)
            barra_p.loc[d] = idata.mul(iweight.iloc[:, 0], axis='index').sum()

            iindu, iweight = indus.loc[d].align(self.stock_positions.loc[pd.to_datetime(d)], join='right', axis=0)
            indus_p.loc[d] = iindu.mul(iweight.iloc[:, 0], axis='index').sum()

            if risk_factor is not None:
                irisk, iweight = indus.loc[d].align(self.stock_positions.loc[pd.to_datetime(d)], join='right', axis=0)
                riskfactor_p.loc[d] = irisk.mul(iweight.iloc[:, 0], axis='index').sum()
        return barra_p, indus_p, riskfactor_p

    @clru_cache()
    def cal_singledate_expo(self, date):
        """
        计算单期因子暴露分析
        """
        date = [date]
        return self.cal_multidates_expo(date)

    def cal_multidates_expo(self, dates):
        """
        计算多期风险暴露。\n

        Returns:
        ========
        因子暴露数据结构：
        barra_expo: DataFrame
            DataFrame(index:[date style_name], columns:[portfolio benchmark])
        indus_expo: DataFrame
            DataFrame(index:[date industry_name], columns:[portfolio benchmark])
        risk_expo: DataFrame
            DataFrame(index:[date  riskfactor_name], columns:[portfolio benchmark])
        """
        risk_b = self._cal_risk_of_bchmrk(dates)
        risk_p = self.cal_risk_of_portfolio(dates)

        # BARRA风险
        barra_expo = pd.concat([risk_p[0].stack(), risk_b[0].stack()], axis=1, ignore_index=True)
        barra_expo.index.names = ['date', 'barra_style']
        barra_expo.columns = ['portfolio', 'benchmark']
        barra_expo['expo'] = barra_expo['portfolio'] - barra_expo['benchmark']

        # 行业风险
        indus_expo = pd.concat([risk_p[1].stack(), risk_b[1].stack()], axis=1, ignore_index=True)
        indus_expo.index.names = ['date', 'industry']
        indus_expo.columns = ['portfolio', 'benchmark']
        indus_expo['expo'] = indus_expo['portfolio'] - indus_expo['benchmark']

        # 自定义风险
        if risk_p[2] is not None:
            risk_expo = pd.concat([risk_p[2].stack(), risk_b[2].stack()], axis=1, ignore_index=True)
            risk_expo.index.names = ['date', 'risk_factor']
            risk_expo.columns = ['portfolio', 'benchmark']
            risk_expo['expo'] = risk_expo['portfolio'] - risk_expo['benchmark']
        else:
            risk_expo = None
        return barra_expo, indus_expo, risk_expo