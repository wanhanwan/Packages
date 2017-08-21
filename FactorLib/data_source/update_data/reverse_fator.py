"""反转类因子"""
from utils.datetime_func import DateStr2Datetime

# 过去6个月高点下来的收益
def six_month_return_from_high(start, end, **kwargs):
    start_date_pricedata = kwargs['data_source'].trade_calendar.tradeDayOffset(start, -150, incl_on_offset_today=True)
    dates = kwargs['data_source'].trade_calendar.get_trade_days(start_date_pricedata,end)
    close = kwargs['data_source'].load_factor("adj_close", '/stocks/', dates=dates)
    
    close = close.unstack()
    history_max = close.rolling(window=6 * 20).max()
    
    returns = (close / history_max - 1).ix[DateStr2Datetime(start):DateStr2Datetime(end)]
    
    reverse = returns.stack()
    reverse.index.names = ['date', 'IDs']
    reverse.columns = ['six_month_highest_returns']
    # 存储因子
    kwargs['data_source'].h5DB.save_factor(reverse, '/stock_reversal/')

ReverseFuncListDaily = [six_month_return_from_high]
ReverseFuncListMonthly = []


if __name__ == '__main__':
    from data_source import data_source
    six_month_return_from_high('20170701', '20170731', data_source=data_source)
