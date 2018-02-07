# coding: utf-8

# mysql引擎
# mysql_engine = create_engine('mysql+pymysql://root:123456@localhost/barrafactors')

# oracle引擎
#oracle_engine = create_engine("oracle://Filedb:Filedb@172.20.65.27:1521/cibfund")


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


