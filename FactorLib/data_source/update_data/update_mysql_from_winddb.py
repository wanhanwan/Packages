from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
sys.path.append("..")
from sqlalchemy import create_engine

# oracle引擎
oracle_engine = create_engine("oracle://windfile:windfile@172.20.65.11:1521/wind")
mysql_engine = create_engine('mysql+pymysql://root:123456@localhost/barrafactors')

def save_insert_to_mysql(df, table_name, engine):
    col = df.columns
    value = df.values
    conn = engine.connect()
    sql = "insert into " + table_name + "(" + ','.join(col) + ") values"
    numRows = len(value)

    for i in range(numRows):
        if len(sql) >= 10485760:
            sql = sql[:-1] + "on duplicate key update id=id"
            sql = sql.replace("'null'",'null')
            conn.execute(sql)
            sql = "insert into " + table_name + "(" + ','.join(col) + ") values"
        valuesStr = map(str, value[i])
        valuesStr = [i for i in valuesStr]
        if i < numRows - 1:
            sql = sql + "('" + "','".join(valuesStr) + "'),"
        else:
            sql = sql + "('" + "','".join(valuesStr) + "') on duplicate key update id=id"
    sql = sql.replace("'null'",'null')
    conn.execute(sql)

def update_hqdata(start, end):
    print("正在更新历史行情数据...")
    # maxdate = local.getMaxDate('hq_data')
    # if maxdate is None:
    #     maxdate = '20170301'
    # if maxdate >= end:
    #     return 0
    # start = maxdate, 1)
    sql = """SELECT
        SUBSTR (T .s_info_windcode, 1, 6),
        T .trade_dt,
        T .s_dq_close,
        T .s_dq_adjclose,
        T .s_dq_pctchange / 100, 
        E .s_val_mv
    FROM
        ASHAREEODPRICES T
    LEFT JOIN ASHAREEODDERIVATIVEINDICATOR E ON T .s_info_windcode = E .s_info_windcode
    AND T .trade_dt = E .trade_dt
    WHERE
        T .trade_dt BETWEEN '%s'
    AND '%s'
    ORDER BY
        T .TRADE_DT,
        T .S_INFO_WINDCODE
    """%(start, end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','spj','adjclose','rsyl','zsz']
    if not data.empty:
        #data.to_sql('hq_data', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'hq_data', mysql_engine)
    print("历史行情数据更新成功....")
    return 1

def update_jyzt(start, end):
    print("正在更新交易状态数据...")
    # maxdate = local.getMaxDate('jyzt')
    # if maxdate is None:
    #     maxdate = '20170301'
    # if maxdate >= end:
    #     return 0
    # start = maxdate, 1)
    sql = """
        SELECT substr(s_info_windcode,1,6),
                     trade_dt,
                     s_dq_volume,
                     s_dq_pctchange,
               s_dq_high,
               s_dq_low
        from ASHAREEODPRICES
        WHERE TRADE_DT BETWEEN '%s' and '%s'
        ORDER BY S_INFO_WINDCODE,TRADE_DT
    """%(start, end)

    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','cjl','zdf','zgj','zdj']
    
    if not data.empty:
        #data.to_sql('jyzt', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'jyzt', mysql_engine)
    print("交易状态更新成功....")
    return 1


def idx_weight_hs300(start, end):
    print("正在更新沪深300指数权重...")
    sql = """
        SELECT SUBSTR(S_CON_WINDCODE,1,6),
               TRADE_DT,
               I_WEIGHT/100 
        from AINDEXHS300FREEWEIGHT t WHERE t.S_INFO_WINDCODE='399300.SZ' 
        AND TRADE_DT BETWEEN '%s' and '%s' ORDER BY TRADE_DT
    """%(start, end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','Weight']

    
    if not data.empty:
        data.to_sql('idx_weight_hs300', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'idx_weight_hs300', mysql_engine)        
    print("沪深300指数权重更新成功....")
    return 1

def idx_weight_zz500(start, end):
    print("正在更新中证500指数权重...")
    sql = """
        SELECT SUBSTR(S_CON_WINDCODE,1,6),
               TRADE_DT,
               I_WEIGHT/100 
        from AINDEXHS300FREEWEIGHT t WHERE t.S_INFO_WINDCODE='000905.SH' 
        AND TRADE_DT BETWEEN '%s' and '%s' ORDER BY TRADE_DT
    """%(start, end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','Weight']

    if not data.empty:
        #data.to_sql('idx_weight_zz500', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'idx_weight_zz500', mysql_engine)
    print("中证500指数权重更新成功....")
    return 1

def idx_weight_000991(start, end):
    print("正在更新000991指数权重...")
    sql = """
        SELECT SUBSTR(S_CON_WINDCODE,1,6),
               TRADE_DT,
               I_WEIGHT/100 
        from AINDEXHS300FREEWEIGHT t WHERE t.S_INFO_WINDCODE='000901.SH' 
        AND TRADE_DT BETWEEN '%s' and '%s' ORDER BY TRADE_DT
    """%(start, end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','Weight']

    if not data.empty:
        #data.to_sql('idx_weight_zz500', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'idx_weight_000991', mysql_engine)
    print("000991指数权重更新成功....")
    return 1


def float_mv(start, end):
    print("正在更新流通市值数据...")
    sql = """
        SELECT
            SUBSTR (s_info_windcode, 1, 6),
            trade_dt,
            s_dq_mv
        FROM
            ASHAREEODDERIVATIVEINDICATOR
        WHERE
            TRADE_DT BETWEEN '%s'
        AND '%s'
        ORDER BY
            S_INFO_WINDCODE,
            TRADE_DT
    """%(start, end)

    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','float_mv']

    if not data.empty:
        #data.to_sql('float_mkt_value', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'float_mkt_value', mysql_engine)
    print("流通市值更新成功....")
    return 1


def pb_pe_pcf(start, end):
    print("正在更新估值类数据...")
    sql = """
        SELECT substr(s_info_windcode,1,6),
                     trade_dt,
                     s_val_pb_new,
                   s_val_pe,
               s_val_pcf_ocf,
               s_val_pe_ttm
        FROM ASHAREEODDERIVATIVEINDICATOR
        WHERE TRADE_DT BETWEEN %s and %s ORDER BY S_INFO_WINDCODE,TRADE_DT
    """%(start,end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','pb','pe','pcf','pe_ttm']
    
    if not data.empty:
        data = data.where(pd.notnull(data), other='null')
        #data.to_sql('pe_pb_pcf', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'pe_pb_pcf', mysql_engine)
    print("估值类数据更新成功...")
    return 1


def index_price(start, end):
    print("正在更新指数行情数据...")
    Index_WindCode = ['000001.SH','399985.SZ','000300.SH','000905.SH','000906.SH','399102.SZ',
    '000991.SH','000808.SH']
    
    sql = """select substr(s_info_windcode,1,6),
                    trade_dt,
                    s_dq_open,
                    s_dq_high,
                    s_dq_low,
                    s_dq_close,
                    s_dq_volume,
                    s_dq_amount 
             from aindexeodprices 
             where s_info_windcode in ('"""
    sql += "','".join(Index_WindCode)+"') and trade_dt between "+start+" and "+end
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','open','high','low','close','vol','amt']
    
    if not data.empty:
        data = data.where(pd.notnull(data), other='null')
        save_insert_to_mysql(data, 'index_price', mysql_engine)
    print("指数行情更新成功...")
    return 1

def turnover(start,end):
    print("正在更新换手率数据...")
    sql = """
        SELECT substr(s_info_windcode,1,6),
                     trade_dt,
                     s_dq_turn,
                   s_dq_freeturnover
        FROM ASHAREEODDERIVATIVEINDICATOR
        WHERE TRADE_DT BETWEEN %s and %s ORDER BY S_INFO_WINDCODE,TRADE_DT
    """%(start,end)
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','Dates','turn','free_turn']
    
    if not data.empty:
        data = data.where(pd.notnull(data), other='NULL')
        #data.to_sql('turnover', mysql_engine, if_exists='append', index=False)
        save_insert_to_mysql(data, 'turnover', mysql_engine)
    print("换手率更新成功...")
    return 1

def swindustryclass(start, end):
    print("正在更新申万一级行业")
    sql = """select substr(s_info_windcode,1,6),a.entry_dt,a.remove_dt,b.industriesName
             from ashareswindustriesclass a, ashareindustriescode b
             where substr(a.sw_ind_code, 1, 4) = substr(b.IndustriesCode, 1, 4)
                   and b.levelnum = '2' order by a.entry_dt"""
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','entry_date','remove_dt','sw_level_1']
    
    if not data.empty:
        data = data.where(pd.notnull(data), other=np.nan)
        excel_path = os.sep.join(
            os.path.abspath('..').split(os.sep)[:2] + ['resource', 'sw_level_1.xlsx'])
        data.to_excel(excel_path, index=False)
    print("申万一级行业更新成功...")
    return 1

def csindustryclass(start, end):
    print("正在更新中信一级行业")
    sql = """select substr(s_info_windcode,1,6),a.entry_dt,a.remove_dt,b.industriesName
             from AShareIndustriesClassCITICS a, ashareindustriescode b
             where substr(a.citics_ind_code, 1, 4) = substr(b.IndustriesCode, 1, 4)
                   and b.levelnum = '2' order by a.entry_dt"""
    data = pd.read_sql(sql, oracle_engine)
    data.columns = ['IDs','entry_date','remove_dt','cs_level_1']
    
    if not data.empty:
        data = data.where(pd.notnull(data), other=np.nan)
        excel_path = os.sep.join(
            os.path.abspath('..').split(os.sep)[:2] + ['resource', 'cs_level_1.xlsx'])
        data.to_excel(excel_path, index=False)
    print("中信一级行业更新成功...")
    return 1

def update_balancesheet(*args):
    print("正在更新资产负债表数据...")
    sql = "select * from asharebalancesheet where statement_type in ('408001000','408005000')"
    data = pd.read_sql(sql, oracle_engine)
    data['year'] = data['report_period'].str[:4].apply(int)
    data['season'] = data['report_period'].str[4:6].apply(int) / 3
    data.to_csv("D:/data/h5/balancesheet.csv")
    print("资产负债表更新成功...")

def update_incomesheet(*args):
    print("正在更新利润表数据...")
    sql = "select * from ashareincome where statement_type in ('408001000','408005000')"
    data = pd.read_sql(sql, oracle_engine)
    data['year'] = data['report_period'].str[:4].apply(int)
    data['season'] = data['report_period'].str[4:6].apply(int) / 3
    data.to_csv("D:/data/h5/incomesheet.csv")
    print("利润表更新成功...")

def update_cashflowsheet(*args):
    print("正在更新现金流量表数据...")
    sql = "select * from asharecashflow where statement_type in ('408001000','408005000')"
    data = pd.read_sql(sql, oracle_engine)
    data['year'] = data['report_period'].str[:4].apply(int)
    data['season'] = data['report_period'].str[4:6].apply(int) / 3
    data.to_csv("D:/data/h5/cashflowsheet.csv")
    print("现金流量表更新成功...")

updateFuncs = [update_balancesheet,update_incomesheet, update_cashflowsheet]
#updateFuncs = [update_balancesheet, update_incomesheet, update_cashflowsheet]
for iFunc in updateFuncs:
    iFunc('20170503', '20170505')
