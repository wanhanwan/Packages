"""
计算股票组合的风险因子暴露
"""

from FactorLib.riskmodel.attribution import RiskExposureAnalyzer

analyzer = RiskExposureAnalyzer.from_csv(r"D:\data\factor_investment_stocklists\兴业风格_价值成长等权.csv",
                                         benchmark='000905')
barra_expo, indus_expo, risk_expo = analyzer.cal_singledate_expo('20170731')
