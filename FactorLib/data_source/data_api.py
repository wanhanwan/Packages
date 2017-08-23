from .base_data_source_h5 import sec, tc

# 返回历史A股成分股
def get_history_ashare(date):
    return sec.get_history_ashare(date)

#返回交易日期
def get_trade_days(start_date=None,end_date=None,freq='1d'):
    return tc.get_trade_days(start_date, end_date, freq)

def trade_day_offset(today, n, freq='1d'):
    return tc.tradeDayOffset(today,n,freq)