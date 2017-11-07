import pandas as pd
import TSLPy3 as tsl
from ..base_data_source_h5 import h5
from ...utils.TSDataParser import parseByStock


def getDividendInfo(start_date, end_date):
    """
    从天软客户端提取股票分红信息
    """
    start_date = int(start_date)
    end_date = int(end_date)
    data = tsl.RemoteCallFunc('getBKDividendInfo', ('A股', end_date, start_date, end_date), {})
    divd_table = parseByStock(data, date_parse=['截止日','股权登记日','除权除息日','预案公布日'])
    divd_table.rename(columns={
        '截止日': 'date', '股权登记日': 'record_date', '除权除息日':'ex_divd_date', '预案公布日': 'ann_date',
        '红利比': 'cash_per_share', '实得比': 'real_cash_per_share', '送股比': 'total_share', '红股比': 'bonus_share',
        '转增比': 'inc_share'}, inplace=True)
    divd_table = divd_table.reset_index().set_index(['date', 'IDs']).sort_index()
    h5.save_factor(divd_table, '/stock_financial_data/dividends/')