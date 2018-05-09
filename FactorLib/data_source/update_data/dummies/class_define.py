from FactorLib.data_source.base_data_source_h5 import sec, data_source
from FactorLib.data_source.update_data.dummies import file_path
import pandas as pd


def update1(start_date, end_date):
    mapping = pd.read_excel(file_path, header=0, sheet_name='class_1', index_col=0)
    indu_info = sec.get_stock_industry_info(None, '中信一级', start_date=start_date,
                                            end_date=end_date)
    indu_info['tagID'] = indu_info['cs_level_1'].map(mapping['TagID'])
    dummy = pd.get_dummies(indu_info['tagID'])
    data_source.h5DB.save_as_dummy(dummy, '/dummy/', indu_name='user_dummy_class_1')


classUpdateFuncsDaily = [update1]


if __name__ == '__main__':
    update1('20100101', '20180502')