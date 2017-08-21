import pandas as pd


# mysql引擎
# mysql_engine = create_engine('mysql+pymysql://root:123456@localhost/barrafactors')

# oracle引擎
#oracle_engine = create_engine("oracle://windfile:windfile@172.20.65.11:1521/wind")


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


def save_factor(factor, path):
    h5.save_factor(factor, path)


def get_latest_report_date(dates, report_type=None):
    """特定日期序列的最大报告期"""
    ann_report_dates = _get_report_ann_dt(report_type)
    _ = []
    if isinstance(dates, str):
        dates = [dates]
    for date in dates:
        temp = ann_report_dates[ann_report_dates['ann_dt']<=date]
        max_date = temp.groupby('IDs')['date'].max()
        max_date.index = pd.MultiIndex.from_product([[date], max_date.index], names=['date', 'IDs'])
        _.append(max_date)
    return pd.concat(_).rename('max_report_date').to_frame()


def get_nlatest_report_dates(dates, n, report_type=None):
    """特定日期序列的前n个最大报告期"""
    ann_report_dates = _get_report_ann_dt(report_type)
    _ = []
    if isinstance(dates, str):
        dates = [dates]
    for date in dates:
        temp = ann_report_dates[ann_report_dates['ann_dt']<=date]
        max_date = temp.groupby('IDs')['date'].nlargest(n).reset_index(level=1, drop=True)
        max_date.index = pd.MultiIndex.from_product([[date], max_date.index], names=['date', 'IDs'])
        _.append(max_date)
    return pd.concat(_).rename('max_report_date').to_frame()   


def _get_report_ann_dt(report_type):
    ann_report_dates = h5.load_factor('ann_dt', '/stock_financial_data/').reset_index(level=0)
    if report_type is None:
        return ann_report_dates
    elif report_type == 'Q1':
        return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 3]
    elif report_type == 'Q2':
        return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 6]
    elif report_type == 'Q3':
        return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 9]
    else:
        return ann_report_dates[pd.DatetimeIndex(ann_report_dates['date']).month == 12]

