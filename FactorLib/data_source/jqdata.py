#!python
# -*- coding: utf-8 -*-
#
# jqdata.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Date   : 2020/1/23 上午10:30:35
"""
聚宽数据源

"""
import jqdatasdk
from jqdatasdk import *
import pandas as pd
from FactorLib.utils.tool_funcs import tradecode_to_uqercode

jqdatasdk.auth('18516759861', '1991822929')


class JQDataAPI(object):
    _max_records = 3000
    
    @staticmethod
    def _query(module_name, table_name):
        return query(getattr(globals()[module_name], table_name))
    
    @staticmethod
    def _query_multi_fields(*args):
        return query(
            *(
                getattr(getattr(globals()[module_name], table_name), field_name)
                for module_name, table_name, field_name in args
              )
            )
    
    @staticmethod
    def _filter(_query, *filter_strings):
        return _query.filter(*(eval(s) for s in filter_strings))
    
    def opt_contract_adjsment_get(self, contract_code=None, start_date=None,
                                  end_date=None, fields='*'):
        """
        期权合约调整记录(一次最大返回3000行记录)
        记录ETF期权因分红除息所带来的期权交易代码，合约简称，合约单位，行权价格的变化

        Parameters:
        -----------
        contract_code: str 
            8位数字的期权合约代码(需添加后缀)。期权合约变更前后，这个期权代码是不变的。
        start_date: str
            YYYY-MM-DD 调整的起始日期
        end_date: str
            YYYY-MM-DD 调整的结束日期
        fields: str
            返回字段
        
        Returns:
        --------
        DataFrame: code合约代码、adj_date调整日期、contract_type合约类型(CO-认购/PO认沽)
                   ex_trading_code原交易代码、ex_name原合约简称、ex_exercise_price原行权价、
                   ex_contract_unit原合约单位、new_trading_code新交易代码、new_name新合约简称
                   new_exercise_price新行权价、adj_reason调整原因、expire_date到期日、
                   last_trade_date最后交易日、exercise_date行权日
        """
        module = 'opt'
        table = 'OPT_ADJUSTMENT'

        if contract_code is not None:
            q = self._filter(self._query(module, table), f"{module}.{table}.code=='{contract_code}'")
        else:
            assert (start_date is not None and end_date is not None)
            q = self._filter(self._query(module, table),
                             f"{module}.{table}.adj_date>='{start_date}'",
                             f"{module}.{table}.adj_date<='{end_date}'"
                             )
        data = pd.DataFrame(opt.run_query(q)).drop(columns=['id'], errors='ignore')

        if fields == '*':
            return data
        return data[fields.strip().split(',')]
    
    def opt_contract_info_get(self, contract_code=None, underlying_symbol=None,
                              trade_date=None, fields='*'):
        """
        期权合约基本信息(每次最多返回3000行数据)

        Parameters:
        -----------
        contract_code: str
            8位数字的期权合约代码(需添加后缀)。期权合约变更前后，这个期权代码是不变的。
        underlying_symbol: str
            标的代码。例，510050.XSEG上证50ETF
        trade_date: str
            YYYY-MM-DD 交易日期。若非空，代表在当日进行交易的合约。
        fields: str
            返回字段  code合约代码(带后缀)  trading_code合约交易代码(合约调整会产生新的交易代码)
            contract_type合约类型(CO-认购、PO-认沽)  underlying_symbol标的代码  underlying_name标的简称
            exercise_price行权价格  contract_unit合约单位  list_date挂牌日期  list_reason挂牌原因
            expire_date到期日  last_trade_date最后交易日  exercise_date行权日  delist_date摘牌日期(最后交易日T+1)
            is_adjust是否调整
        """
        module = 'opt'
        table = 'OPT_CONTRACT_INFO'

        q = self._query(module, table)
        filter_str = []
        if contract_code is not None:
            filter_str.append(f"{module}.{table}.code=='{contract_code}'")
        if underlying_symbol is not None:
            filter_str.append(f"{module}.{table}.underlying_symbol=='{underlying_symbol}'")
        if trade_date is not None:
            filter_str.append(f"{module}.{table}.list_date<='{trade_date}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(opt.run_query(q)).drop(columns=['id'], errors='ignore')
        if trade_date is not None:
            data = data[~(data['delist_date']<pd.to_datetime(trade_date))]
        if fields == '*':
            return data
        return data[fields.strip().split(',')]
    
    def opt_daily_preopen_get(self, contract_code=None, trade_date=None, contract_trade_code=None,
                          fields='*'):
        """
        ETF期权每日交易的基本参数，包含合约单位、行权价格，持仓量，涨跌停价格数据等。

        Parameters:
        -----------
        contract_code: str
            8位数字的期权合约代码(需添加后缀)。期权合约变更前后，这个期权代码是不变的。
        trade_date: str
            交易日期 YYYY-MM-DD
        contract_trade_code: str
            17位数字的期权交易代码
        fileds: str
            交易日期、合约代码、涨跌停价、合约单位、行权价格等等。

            详见https://www.joinquant.com/help/api/help?name=Option#%E8%8E%B7%E5%8F%96%E6%9C%9F%E6%9D%83%E6%AF%8F%E6%97%A5%E7%9B%98%E5%89%8D%E9%9D%99%E6%80%81%E6%96%87%E4%BB%B6
        
        """
        module = 'opt'
        table = 'OPT_DAILY_PREOPEN'
        if fields == '*':
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        filter_str = []
        if contract_code:
            filter_str.append(f"{module}.{table}.code=='{contract_code}'")
        if trade_date:
            filter_str.append(f"{module}.{table}.date=='{trade_date}'")
        if contract_trade_code:
            filter_str.append(f"{module}.{table}.trading_code=='{contract_trade_code}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(opt.run_query(q)).drop(columns=['id'], errors='ignore')
        return data
    
    def income_statement_get(self, ticker=None, period=None, start_period=None,
                             end_period=None, report_type=None, fields=None):
        """
        A股利润表

        Parameters:
        -----------
        ticker: str
            股票代码
        period: str
            报告期 YYYY-MM-DD
        start_period: str
            起始报告期 YYYY-MM-DD
        end_period: str
            结束报告期 YYYY-MM-DD
        report_type: str
            报表类型 0本期 1上期
        fields: str
            字段 参考：https://www.joinquant.com/help/api/help?name=JQData#%E5%90%88%E5%B9%B6%E5%88%A9%E6%B6%A6%E8%A1%A8
        """
        module = 'finance'
        table = 'STK_INCOME_STATEMENT'
        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        # filter_str = [f"{module}.{table}.code=={module}.{table}.a_code"] #只返回A股数据
        filter_str = []
        if ticker:
            ticker = tradecode_to_uqercode(ticker)
            filter_str.append(f"{module}.{table}.code=='{ticker}'")
        if period is not None:
            filter_str.append(f"{module}.{table}.report_date=='{period}'")
        else:
            filter_str.append(f"{module}.{table}.report_date>='{start_period}'")
            filter_str.append(f"{module}.{table}.report_date<='{end_period}'")
        if report_type is not None:
            filter_str.append(f"{module}.{table}.report_type=={report_type}")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(finance.run_query(q)).drop(columns=['id'], errors='ignore')
        return data

    def balance_sheet_get(self, ticker=None, period=None, start_period=None,
                          end_period=None, report_type=None, fields=None):
        """
        A股利润表

        Parameters:
        -----------
        ticker: str
            股票代码
        period: str
            报告期 YYYY-MM-DD
        start_period: str
            起始报告期 YYYY-MM-DD
        end_period: str
            结束报告期 YYYY-MM-DD
        report_type: str
            报表类型 0本期 1上期
        fields: str
            字段 参考：https://www.joinquant.com/help/api/help?name=JQData#%E5%90%88%E5%B9%B6%E5%88%A9%E6%B6%A6%E8%A1%A8
        """
        module = 'finance'
        table = 'STK_BALANCE_SHEET'
        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        # filter_str = [f"{module}.{table}.code=={module}.{table}.a_code"]
        filter_str = []
        if ticker:
            ticker = tradecode_to_uqercode(ticker)
            filter_str.append(f"{module}.{table}.code=='{ticker}'")
        if period is not None:
            filter_str.append(f"{module}.{table}.report_date=='{period}'")
        else:
            filter_str.append(f"{module}.{table}.report_date>='{start_period}'")
            filter_str.append(f"{module}.{table}.report_date<='{end_period}'")
        if report_type is not None:
            filter_str.append(f"{module}.{table}.report_type=={report_type}")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(finance.run_query(q)).drop(columns=['id'], errors='ignore')
        return data
    
    def opt_daily_prices_get(self, contract_code=None, trade_date=None, start_trade_date=None,
                             end_trade_date=None, exchange=None, fields=None):
        """
        期权日行情数据(每次最多返回3000行数据)

        Parameters:
        -----------
        contract_code: str
            8位数字的期权合约代码(需添加后缀)。期权合约变更前后，这个期权代码是不变的。
        trade_date: str
            交易日期 YYYY-MM-DD
        start_trade_date: str
            起始交易日期
        end_trade_date: str
            终止交易日期
        exchange: str
            交易所代码 XSHG上交所、XSHE深交所、CCFX中金所
        fields: str
            返回字段 详见：https://www.joinquant.com/help/api/help?name=Option#%E8%8E%B7%E5%8F%96%E6%9C%9F%E6%9D%83%E6%97%A5%E8%A1%8C%E6%83%85%E6%95%B0%E6%8D%AE
        """
        module = 'opt'
        table = 'OPT_DAILY_PRICE'

        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        # filter_str = [f"{module}.{table}.code=={module}.{table}.a_code"]
        filter_str = []
        if contract_code:
            filter_str.append(f"{module}.{table}.code=='{contract_code}'")
        if trade_date:
            filter_str.append(f"{module}.{table}.date=='{trade_date}'")
        if start_trade_date:
            filter_str.append(f"{module}.{table}.date>='{start_trade_date}'")
        if end_trade_date:
            filter_str.append(f"{module}.{table}.date<='{end_trade_date}'")
        if exchange:
            filter_str.append(f"{module}.{table}.exchange_code=='{exchange}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(opt.run_query(q)).drop(columns=['id'], errors='ignore')
        return data

    def opt_daily_risk_indicators_get(self, contract_code=None, trade_date=None, start_trade_date=None,
                                         end_trade_date=None, exchange=None, fields=None):
        """
        期权日频风险指标

        Parameters:
        -----------
        contract_code: str
            8位数字的期权合约代码(需添加后缀)。期权合约变更前后，这个期权代码是不变的。
            代码后缀：XSHG上交所、XSHE深交所、CCFX中金所
        trade_date: str
            交易日期 YYYY-MM-DD
        exchange: str
            交易所代码 XSHG上交所、XSHE深交所、CCFX中金所
        start_trade_date: str
            起始交易日期
        end_trade_date: str
            终止交易日期
        fields: str
            返回字段 详见：https://www.joinquant.com/help/api/help?name=Option#%E8%8E%B7%E5%8F%96%E6%9C%9F%E6%9D%83%E9%A3%8E%E9%99%A9%E6%8C%87%E6%A0%87%E6%95%B0%E6%8D%AE
        """
        module = 'opt'
        table = 'OPT_RISK_INDICATOR'

        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))

        filter_str = []
        if contract_code:
            filter_str.append(f"{module}.{table}.code=='{contract_code}'")
        if trade_date:
            filter_str.append(f"{module}.{table}.date=='{trade_date}'")
        if start_trade_date:
            filter_str.append(f"{module}.{table}.date>='{start_trade_date}'")
        if end_trade_date:
            filter_str.append(f"{module}.{table}.date<='{end_trade_date}'")
        if exchange:
            filter_str.append(f"{module}.{table}.exchange_code=='{exchange}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(opt.run_query(q)).drop(columns=['id'], errors='ignore')
        return data
    
    def stock_dividend_get(self, ticker=None, start_report_date=None, end_report_date=None, fields=None):
        """
        A股分红送股信息

        Parameters:
        -----------
        ticker: str
            股票代码
        start_report_date: str
            起始报告日期 YYYY-MM-DD
        end_report_date: str
            结束报告日期 YYYY-MM-DD
        fields: str
            返回字段，详见：https://www.joinquant.com/help/api/help?name=Stock#%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E5%88%86%E7%BA%A2%E9%80%81%E8%82%A1%EF%BC%88%E9%99%A4%E6%9D%83%E9%99%A4%E6%81%AF%EF%BC%89%E6%95%B0%E6%8D%AE
        """
        module = 'finance'
        table = 'STK_XR_XD'

        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        
        filter_str = []
        if ticker:
            code = tradecode_to_uqercode(ticker)
            filter_str.append(f"{module}.{table}.code=='{code}'")
        if start_report_date:
            filter_str.append(f"{module}.{table}.report_date>='{start_report_date}'")
        if end_report_date:
            filter_str.append(f"{module}.{table}.report_date<='{end_report_date}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(finance.run_query(q)).drop(columns=['id'], errors='ignore')
        return data

    def fund_dividend_get(self, ticker=None, start_ann_date=None, end_ann_date=None,
                          fields=None):
        """
        基金分红信息

        Parameter:
        ---------
        ticker: str
            基金代码
        start_ann_date: str
            起始公告日期
        end_ann_date: str
            终止公告日期
        fields: str
            返回字段
            详见：https://www.joinquant.com/help/api/help?name=JQData#%E8%8E%B7%E5%8F%96%E5%9F%BA%E9%87%91%E5%88%86%E7%BA%A2%E6%8B%86%E5%88%86%E5%90%88%E5%B9%B6%E4%BF%A1%E6%81%AF
        """
        module = 'finance'
        table = 'FUND_DIVIDEND'
        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        filter_str = []
        if ticker:
            filter_str.append(f"{module}.{table}.code=='{ticker}'")
        if start_ann_date:
            filter_str.append(f"{module}.{table}.pub_date>='{start_ann_date}'")
        if end_ann_date:
            filter_str.append(f"{module}.{table}.pub_date<='{end_ann_date}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(finance.run_query(q)).drop(columns=['id'], errors='ignore')
        return data
    
    def fund_portfolio_get(self, ticker=None, start_period=None, end_period=None, fields=None):
        """
        基金持仓信息

        Parameters:
        -----------
        ticker: str
            基金代码
        start_period: str
            起始财报日期YYYY-MM-DD
        end_period: str
            终止财报日期YYYY-MM-DD
        fields: str
            返回字段
            详见：https://www.joinquant.com/help/api/help?name=JQData#%E8%8E%B7%E5%8F%96%E5%9F%BA%E9%87%91%E6%8C%81%E8%82%A1%E4%BF%A1%E6%81%AF%EF%BC%88%E6%8C%89%E5%AD%A3%E5%BA%A6%E5%85%AC%E5%B8%83%EF%BC%89
        """
        module = 'finance'
        table = 'FUND_PORTFOLIO_STOCK'
        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        filter_str = []
        if ticker:
            filter_str.append(f"{module}.{table}.code=='{ticker}'")
        if start_period:
            filter_str.append(f"{module}.{table}.period_end>='{start_period}'")
        if end_period:
            filter_str.append(f"{module}.{table}.period_end<='{end_period}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(finance.run_query(q)).drop(columns=['id'], errors='ignore')
        for c in ['period_start', 'period_end', 'pub_date']:
            if c in data.columns:
                data = data.astype({c: 'datetime64'})
        return data

    def fund_nav_get(self, ticker=None, start_date=None, end_date=None, fields=None):
        """
        基金净值

        Parameters:
        -----------
        ticker: str
            基金代码(无后缀)
        start_date: str
            起始日期YYYY-MM-DD
        end_date: str
            终止日期YYYY-MM-DD
        fields: str
            返回字段 code、day、net_value、sum_value、factor、acc_factor、refactor_net_value
        """

        module = 'finance'
        table = 'FUND_NET_VALUE'
        if fields is None:
            q = self._query(module, table)
        else:
            q = self._query_multi_fields(*([module, table, x] for x in fields.strip().split(',')))
        filter_str = []
        if ticker:
            filter_str.append(f"{module}.{table}.code=='{ticker}'")
        if start_date:
            filter_str.append(f"{module}.{table}.day>='{start_date}'")
        if end_date:
            filter_str.append(f"{module}.{table}.day<='{end_date}'")
        q = self._filter(q, *filter_str)
        data = pd.DataFrame(
            finance.run_query(q)).drop(columns=['id'],
            errors='ignore',
        )
        return data

jq_api = JQDataAPI()


if __name__ == '__main__':
    jq_api = JQDataAPI()
    data = jq_api.fund_portfolio_get('000311','2020-03-31','2020-03-31')
    print(data.dtypes)


import pandas as pd
df = pd.Series(['2020-01-01', '2020-01-02'])
df.astype('datetime64')
pd.DatetimeIndex(df)