from FactorLib.data_source.base_data_source_h5 import data_source


def get_risky_stocks(start, end, **kwargs):
    dates = data_source.trade_calendar.get_trade_days(start, end, retstr=None)
    subscore = data_source.load_factor('sub_score_of_risky_stocks', '/indexes/').iloc[:, 0].unstack().\
        reindex(dates, method='nearest')
    totalscore = data_source.load_factor('total_score_of_risky_stocks', '/indexes/').iloc[:, 0].unstack().\
        reindex(dates, method='nearest')
    totalscorev2 = data_source.load_factor('total_score_of_risky_stocks_v2', '/indexes/').iloc[:, 0].unstack().\
        reindex(dates, method='nearest')
    subscore, totalscore = subscore.align(totalscore, fill_value=0)
    risky_stocks = ((subscore + totalscore)>=1).stack().to_frame('risky_stocks').astype('int')
    risky_stocks.index.names = ['date', 'IDs']
    kwargs['data_source'].h5DB.save_factor(risky_stocks, '/indexes/')
    kwargs['data_source'].h5DB.save_factor(
        subscore.stack().to_frame('risky_stocks_subscore').rename_axis(['date', 'IDs']), '/indexes/')
    kwargs['data_source'].h5DB.save_factor(
        totalscorev2.stack().to_frame('risky_stocks_totalscore').rename_axis(['date', 'IDs']), '/indexes/')

if __name__  == '__main__':
    get_risky_stocks('20180528', '20180606', data_source=data_source)
