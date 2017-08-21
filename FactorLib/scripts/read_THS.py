from data_source import h5
import pandas as pd

data = pd.read_table("D:/data/ths(20170227-20170531).txt",
                         header=0, sep='|', dtype={'代码': 'str'}, parse_dates=['日期'])
rslt = data.drop_duplicates(subset=['日期', '代码'])[['日期','代码','比例']]
rslt.columns = ['date', 'IDs', 'ths_click_ratio']
rslt = rslt.set_index(['date', 'IDs']).sort_index()
h5.save_factor(rslt, '/stock_alternative/')
    
    