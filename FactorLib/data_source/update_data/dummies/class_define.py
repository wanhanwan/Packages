from FactorLib.data_source.base_data_source_h5 import sec, ncdb
from FactorLib.data_source.update_data.dummies import file_path
import pandas as pd


def update1(start_date, end_date):
    mapping = pd.read_excel(file_path, header=0, sheet_name='class_1', index_col=0)
    indu_info = sec.get_stock_industry_info(None, '中信一级', start_date=start_date,
                                            end_date=end_date)
    indu_info['tagID'] = indu_info['cs_level_1'].map(mapping['TagID'])
    dummy = pd.get_dummies(indu_info['tagID'])
    ncdb.save_as_dummy(dummy, 'user_dummy_class_1', '/dummy/')


classUpdateFuncsDaily = [update1]


if __name__ == '__main__':
    update('20100101', '20180326')