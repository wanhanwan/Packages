from .database import (WindIncomeSheet,
                       WindConsensusDB,
                       WindBalanceSheet,
                       WindSQIncomeSheet,
                       WindProfitExpress,
                       WindProfitNotice,
                       WindAshareCapitalization
                       )

incomesheet = WindIncomeSheet()
sqincomesheet = WindSQIncomeSheet()
balancesheet = WindBalanceSheet()
consensussheet = WindConsensusDB()
profitexpress = WindProfitExpress()
profitnotice = WindProfitNotice()
asharecapitalization = WindAshareCapitalization()


__all__ = ['incomesheet', 'sqincomesheet', 'consensussheet', 'balancesheet', 'profitexpress', 'profitnotice',
           'asharecapitalization']