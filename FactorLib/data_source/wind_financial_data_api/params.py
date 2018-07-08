# coding: utf-8
import platform

# Wind数据库
WIND_USER = 'Filedb'
WIND_PASSWORD = 'Filedb'
WIND_IP = '172.20.65.27'
WIND_PORT = 1521
WIND_DBNAME = 'cibfund'
WIND_DBTYPE = 'oracle'

# 本地财务数据库
if platform.platform().startswith('Windows'):
	LOCAL_FINDB_PATH = 'D:/data/finance'
else:
	LOCAL_FINDB_PATH = '/Users/wanshuai/Data/finance'
