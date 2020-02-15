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
            data = data[~data['delist_date']<trade_date]
        if fields == '*':
            return data
        return data[fields.strip().split(',')]
    
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


jq_api = JQDataAPI()



if __name__ == '__main__':
    jq_api = JQDataAPI()
    data = jq_api.opt_contract_info_get()
    print(data)
