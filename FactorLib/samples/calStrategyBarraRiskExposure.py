from FactorLib.factor_performance.analyzer import Analyzer
import pandas as pd

analyzer = Analyzer(r"D:\data\factor_investment_strategies\兴业风格_价值\backtest\BTresult.pkl",
                    benchmark_name='000905')
a = analyzer.portfolio_risk_expo('xy', ['20170807','20170808'])