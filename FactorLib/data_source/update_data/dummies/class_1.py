from FactorLib.data_source.base_data_source_h5 import sec


def update(start_date, end_date):
    indu_info = sec.get_stock_industry_info(None, '中信一级', start_date=start_date,
                                            end_date=end_date)
