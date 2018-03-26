from FactorLib.data_source.base_data_source_h5 import sec, ncdb
from . import file_path
import pandas as pd


def update(start_date, end_date):
    mapping = pd.read_excel(file_path, header=0, sheet_name='class_1')
    indu_info = sec.get_stock_industry_info(None, '中信一级', start_date=start_date,
                                            end_date=end_date)
    indu_info['tagID'] = indu_info['cs_level_1'].map(mapping)
    ncdb.save_as_dummy()