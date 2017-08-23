from WindPy import *
from ..base_data_source_h5 import sec, tc, h5
from ...utils.tool_funcs import ReportDateAvailable, windcode_to_tradecode, tradecode_to_windcode
from ..update_data.update_h5db_base_data import get_ashare
import pandas as pd

w.start()


def update_data(field, start, end, params=None, history=False):
    if history:
        ashare = sec.get_history_ashare(tc.get_trade_days(start, end))
        stocks = ashare.index.get_level_values(1).unique().tolist()
        stocks = [tradecode_to_windcode(x) for x in stocks]
    else:
        stocks = get_ashare(end)
    report_dates = ReportDateAvailable(start, end)
    l = []
    if params is not None:
        param = params + ";Period=Q;Days=Alldays"
    else:
        param = "Period=Q;Days=Alldays"
    for date in report_dates:
        d = w.wsd(stocks, field, date, date, param)
        data = d.Data[0]
        l.append(data)
    tradecodes = [windcode_to_tradecode(x) for x in stocks]
    dates = pd.DatetimeIndex(report_dates, name='date')
    data = pd.DataFrame(l, index=dates, columns=tradecodes).stack().to_frame().rename(columns={0:field})
    data.index.names = ['date', 'IDs']
    h5.save_factor(data, '/stock_financial_data/')

def update_report_ann_dt(start, end):
    """更新报告期和公告期"""
    stocks = get_ashare(end)
    report_dates = ReportDateAvailable(start, end)
    ann_dates = []
    for date in report_dates:
        d = w.wsd(stocks, "stm_issuingdate", date, date, "Period=Q;Days=Alldays")
        iann_dates = d.Data[0]
        iann_dates = [x.strftime("%Y%m%d") if x is not None else x for x in iann_dates]
        ann_dates.append(iann_dates)
    tradecodes = [windcode_to_tradecode(x) for x in stocks]
    dates = pd.DatetimeIndex(report_dates, name='date')
    report_ann_dates = pd.DataFrame(ann_dates, index=dates, columns=tradecodes)
    report_ann_dates = report_ann_dates.stack().to_frame().rename(columns={0:'ann_dt'})
    report_ann_dates.index.names = ['date', 'IDs']
    h5.save_factor(report_ann_dates, '/stock_financial_data/')
    return


def update_all(start, end):
    from .wind_api_const import FINANCIAL_BASE_FIELD, FINANCIAL_DERIVATIVE_FIELD

    # 更新报告期、公告期
    update_report_ann_dt(start, end)

    # 三张表原始数据, 原始数据在参数中需要加上“unit=1;rptType=1”
    for x in FINANCIAL_BASE_FIELD:
        print(x)
        update_data(FINANCIAL_BASE_FIELD[x], start, end,
                    params="unit=1;rptType=1", history=True)

    # update_data(FINANCIAL_BASE_FIELD['营业利润'], start, end, history=True, params='unit=1;rptType=1')

    # 财务衍生指标数据
    for x in FINANCIAL_DERIVATIVE_FIELD:
        print(x)
        update_data(FINANCIAL_DERIVATIVE_FIELD[x], start, end, history=True)


if __name__ == '__main__':
    # update_all('20170101', '20170723')
    update_report_ann_dt('20170101', '20170723')