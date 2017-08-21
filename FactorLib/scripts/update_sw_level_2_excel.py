import pandas as pd
import os
from sqlalchemy import create_engine
oracle_engine = create_engine("oracle://windfile:windfile@172.20.65.11:1521/wind")

sql = """select substr(a.s_info_windcode,1,6), a.entry_dt,a.remove_dt,b.Industriesname
  from AShareSWIndustriesClass a,
          AShareIndustriesCode b
 where substr(a.sw_ind_code, 1, 6) = substr(b.IndustriesCode, 1, 6)
   and b.levelnum = '3'
 order by 1
"""
excel_file_path = os.path.abspath("..") + os.sep + 'resource' + os.sep + 'sw_level_2.xlsx'
data = pd.read_sql(sql, oracle_engine)
data.columns = ['IDs', 'entry_dt', 'remove_dt', 'sw_level_2']
data.loc[pd.isnull(data['remove_dt']), 'remove_dt'] = '21000101'

data.to_excel(excel_file_path, index=False)