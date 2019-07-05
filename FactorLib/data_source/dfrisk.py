# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:33:49 2018
东方金工风险模型数据库
@author: wanshuai
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from functools import lru_cache


# 数据连接
class DFQuantRiskModelDataSource(object):
    mysql_url = "mysql+pymysql://wanshuai:dfquant@139.196.77.199:81/dfrisk?charset=gbk&local_infile=1"

    def __init__(self):
        self.conn = None
        self.database_engine = None

    def connect(self):
        if self.conn is None:
            self.database_engine = create_engine(self.mysql_url)
            self.conn = self.database_engine.connect()
        else:
            self.conn = self.database_engine.connect()

    def disconnect(self):
        self.conn.close()
        self.database_engine.dispose()

    @property
    @lru_cache()
    def factor_names(self):
        indunamedict = {'银行': 'Bank', '非银行金融': 'NonBankFinance', '医药': 'Medicine',
                        '电子元器件': 'ElectronicComponents', '食品饮料': 'FoodBeverage',
                        '房地产': 'RealEstate', '机械': 'Machinery', '基础化工': 'BasicChemical',
                        '电力及公用事业': 'Utilities', '汽车': 'Cars', '计算机': 'Computers',
                        '家电': 'HomeAppliance', '有色金属': 'Metal', '建筑': 'Architecture',
                        '交通运输': 'Transportation', '电力设备': 'ElectricalEquipment',
                        '传媒': 'Media', '通信': 'Commutation', '商贸零售': 'CommercialRetail',
                        '农林牧渔': 'Agriculture', '建材': 'BuildingMaterials', '钢铁': 'Iron',
                        '国防军工': 'Military', '石油石化': 'Petroleum', '轻工制造': 'LightManufacturing',
                        '煤炭': 'Coal', '纺织服装': 'Clothing', '餐饮旅游': 'CateringTourism',
                        '综合': 'Comprehensive'}
        res = pd.DataFrame(self.conn.execute("SELECT * FROM dfrisk.riskfactor_info").fetchall())
        res.columns = ['id', 'name']
        res.set_index('id', inplace=True)
        res['name'].replace(indunamedict, inplace=True)
        return res

    def read_risk_huber(self, dt, uni='000000'):
        try:
            # 日期
            dt = pd.to_datetime(dt).strftime('%Y%m%d')

            # 风险因子暴露
            res = self.conn.execute(
                "SELECT fvjson FROM dfrisk.factorexposure WHERE tradingdate='{0}' and universe='{1}'".format(dt, uni))
            risk_factors = pd.read_json(res.fetchone()[0], orient='split', convert_axes=False)
            risk_factors.rename(columns=self.factor_names['name'], inplace=True)

            # 因子收益率
            res = self.conn.execute(
                "SELECT fretjson FROM dfrisk.factorreturnhuber WHERE tradingdate='{0}' and universe='{1}'".format(dt, uni)
            )
            factor_returns = pd.read_json(res.fetchone()[0], orient='split', convert_axes=False, typ='series')
            factor_returns.index = self.factor_names['name']

            # 风险因子收益率协方差
            res = self.conn.execute(
                "SELECT fcovjson FROM dfrisk.factorcovhuber WHERE tradingdate ='{0}' and universe='{1}'".format(dt,
                                                                                                                uni))
            factor_covs_huber = pd.read_json(res.fetchone()[0], orient='split', convert_axes=False)
            factor_covs_huber.index = self.factor_names['name']
            factor_covs_huber.columns = self.factor_names['name']
            factor_covs_huber *= (21*12*0.0001)

            # 股票残差风险
            res = self.conn.execute(
                "SELECT sriskjson FROM dfrisk.specificriskhuber WHERE tradingdate ='{0}' and universe='{1}'".format(dt,
                                                                                                                    uni))
            specific_risks_huber = pd.read_json(res.fetchone()[0], orient='split', convert_axes=False, typ='series')
            specific_risks_huber *= (np.sqrt(21*12) *.01)
            specific_risks_huber **= 2

            # 特质收益率
            res = self.conn.execute(
                "SELECT sretjson FROM dfrisk.specificreturnhuber WHERE tradingdate='{0}' and universe='{1}'".format(dt,
                                                                                                                  uni)
            )
            spec_returns = pd.read_json(res.fetchone()[0], orient='split', convert_axes=False, typ='series') / 100.0

            # 月化风险, 1%表示为.01
            return risk_factors, factor_returns, factor_covs_huber, specific_risks_huber, spec_returns
        except Exception as e:
            self.disconnect()
            raise e


if __name__ == '__main__':
    datasource=DFQuantRiskModelDataSource()
    datasource.connect()
    # field_names = datasource.conn.execute("select table_name from information_schema.tables where table_schema='dfrisk'")
    # print(field_names.fetchall())
    risk_factors, factor_returns,factor_risk, spec_risk, spec_returns = datasource.read_risk_huber(dt='20190613')
    datasource.disconnect()
    # print(factor_risk.index.tolist?())
