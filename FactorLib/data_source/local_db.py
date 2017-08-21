class LocalDB(object):
    """Wind Old Data Base"""

    def __init__(self, user="root", password="123456", host='LocalHost', dbname="barrafactors"):
        self.user = user
        self.password = password
        self.dbname = dbname
        self.host = host

    def connectDB(self):
        import pymysql
        try:
            self.Connection = pymysql.connect(user=self.user, passwd=self.password, host=self.host, db=self.dbname)
            return 1
        except pymysql.Error as e:
            print(e)
            return -1

    def disconnectDB(self):
        try:
            if self.Connection is not None:
                self.Connection.close()
                self.Connection = None
            return 1
        except:
            return -1

    # 获取数据库的连接
    def getConnection(self):
        return self.Connection

    # 检查数据库是否可用
    def isDBAvailabe(self):
        if self.Connection is None:
            return False
        else:
            return True

    # 执行SQL查询语句, 返回游标
    def execSQLQuery(self, sql_str, multi=False):
        if self.Connection is not None:
            Cursor = self.Connection.cursor()
            if multi:
                Cursor.execute(sql_str, multi=True)
            else:
                Cursor.execute(sql_str)
            return Cursor
        else:
            return None

    #获取某张表的因子数据
    def loadFactor(self,table,factors=None,
                   ids=None,dates=None,start_date=None,
                   end_date=None,date_col='trade_dt'):

        if factors is None:
            sql_query_cols = 'show columns from ' + table
            cur = self.execSQLQuery(sql_query_cols)
            factors = [x[0] for x in cur.fetchall()]
            cur.close()   
        sql_query="SELECT "+",".join(factors)+" FROM "+table
        if ids is not None:
            ids = CodeConvert(CodeConvertWind(ids))            
            sql_query+=" WHERE windcode IN ('"+"','".join(ids)+"')"
        if dates is not None:
            if ids is None:
                sql_query+=" WHERE %s IN ("%date_col+",".join(dates)+")"
            else:
                sql_query+=" AND %s IN ("%date_col+",".join(dates)+")"
        elif (start_date is not None) and (end_date is not None):
            if ids is not None:
                sql_query+=" AND %s BETWEEN %s AND %s"%(date_col,start_date,end_date)
            else:
                sql_query+=" WHERE %s BETWEEN %s AND %s"%(date_col,start_date,end_date)
        elif (start_date is None) and (end_date is not None):
            if ids is not None:
                sql_query+=" AND %s <= %s"%(date_col,end_date)
            else:
                sql_query+=" WHERE %s <= %s"%(date_col,end_date)
        elif (start_date is not None) and (end_date is None):
            if ids is not None:
                sql_query+=" AND %s >= %s"%(date_col,start_date)
            else:
                sql_query+=" WHERE %s >= %s"%(date_col,start_date)
        cursor = self.execSQLQuery(sql_query)
        rslt = pd.DataFrame(list(cursor.fetchall()), columns=factors)
        cursor.close()
        rslt = format_column_names(rslt)
        rslt = rslt[rslt['IDs'] < 900000000]
        rslt['IDs'] = rslt['IDs'].astype('str').str[3:]
        rslt['Date'] = rslt['Date'].astype('str')
        return rslt

    def insertRow(self, tablename, col, value, transaction=0):
        valuesStr = map(str, value)
        valuesStr = [i for i in valuesStr]

        sql = "INSERT INTO " + tablename + " (" + ','.join(col) + ") VALUES (" + ','.join(valuesStr) + ");"
        # print(sql)
        if self.Connection is not None:
            cursor = self.Connection.cursor()
            cursor.execute(sql)
            cursor.close()
            if transaction == 0:
                self.Connection.commit()
            return 1

    # 批量插入记录
    def insertRows(self, tablename, col, value, transaction=0):
        # value 为二维列表
        sql = "insert into " + tablename + "(" + ','.join(col) + ") values"
        numRows = len(value)
        if self.Connection is not None:
            cursor = self.Connection.cursor()
            for i in range(numRows):
                if len(sql) >= 10485760:
                    sql = sql[:-1]
                    cursor.execute(sql)
                    sql = "insert into " + tablename + "(" + ','.join(col) + ") values"
                valuesStr = map(str, value[i])
                valuesStr = [i for i in valuesStr]
                #                    valuesStr[0] = "'"+valuesStr[0]+"'"
                if i < numRows - 1:
                    sql = sql + "(" + ','.join(valuesStr) + "),"
                else:
                    sql = sql + "(" + ','.join(valuesStr) + ")"
            cursor.execute(sql)
            cursor.close()
            if transaction == 0:
                self.Connection.commit()
            return 1

    # 带自增id的批量插入记录
    def insertRowsWithID(self, tablename, col, value, transaction=0):
        # value 为二维列表
        sql = "insert into " + tablename + "(" + ','.join(col) + ") values"
        numRows = len(value)
        if self.Connection is not None:
            cursor = self.Connection.cursor()
            for i in range(numRows):
                if len(sql) >= 10485760:
                    sql = sql[:-1] + "on duplicate key update id=id"
                    cursor.execute(sql)
                    sql = "insert into " + tablename + "(" + ','.join(col) + ") values"
                valuesStr = map(str, value[i])
                valuesStr = [i for i in valuesStr]
                #                    valuesStr[0] = "'"+valuesStr[0]+"'"
                if i < numRows - 1:
                    sql = sql + "(" + ','.join(valuesStr) + "),"
                else:
                    sql = sql + "(" + ','.join(valuesStr) + ") on duplicate key update id=id"
            cursor.execute(sql)
            cursor.close()
            if transaction == 0:
                self.Connection.commit()
            return 1

    # --------------------表的管理------------------------
    # 获取barrafactors数据库中的表名
    def getTableName(self, dbname='barrafactors', ignore=[]):
        SQLStr = 'SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=\'' + dbname + '\''
        Cursor = self.Connection.cursor()
        Cursor.execute(SQLStr)
        Result = Cursor.fetchall()
        Cursor.close()
        return [iRes[0] for iRes in Result if iRes[0] not in ignore]

    # 检查表是否存在

    def checkTableExistence(self, table_name, dbname='barrafactors'):
        SQLStr = 'SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=\'' + dbname + '\' AND TABLE_NAME=\'' + table_name + '\''
        Cursor = self.execSQLQuery(SQLStr)
        Result = Cursor.fetchall()
        Cursor.close()
        if len(Result) > 0:
            return True
        else:
            return False

    # 创建表,field_name_types字典{字段名:数据类型}，key=字段名，value=字段的数据类型,if_exists='cancel'：取消，'replace'：删除后重建,'error'：报错
    def createTable(self, table_name, field_name_types, if_exists='cancel'):
        SQLStr = 'CREATE TABLE %s (' % (table_name)
        for FieldName in field_name_types:
            SQLStr = SQLStr + '`%s` %s, ' % (FieldName, field_name_types[FieldName])
        SQLStr = SQLStr[:-2] + ')'
        Cursor = self.Connection.cursor()
        try:
            Cursor.execute(SQLStr)
            self.Connection.commit()
        except Exception as e:
            if if_exists == 'cancel':
                pass
            elif if_exists == 'replace':
                self.deleteTable(table_name)
                Cursor.execute(SQLStr)
                self.Connection.commit()
            elif if_exists == 'error':
                raise e
        Cursor.close()
        return 1

    # 重命名表
    def renameTable(self, old_table_name, new_table_name, dbname='barrafactors'):
        SQLStr = 'ALTER TABLE ' + dbname + '.' + old_table_name + ' RENAME TO  ' + dbname + '.' + new_table_name
        Cursor = self.Connection.cursor()
        Cursor.execute(SQLStr)
        self.Connection.commit()
        Cursor.close()
        return 1

    # 删除表
    def deleteTable(self, table_name):
        SQLStr = 'DROP TABLE %s' % (table_name)
        Cursor = self.Connection.cursor()
        try:
            Cursor.execute(SQLStr)
            self.Connection.commit()
        except:
            pass
        Cursor.close()
        return 1

    # 增加新字段
    def addNewFeilds(self, table_name, field_name_types, if_exists='cancel'):
        SQLStr = 'ALTER TABLE ' + table_name + ' ADD COLUMN '
        for FieldName in field_name_types:
            SQLStr = SQLStr + '%s %s,' % (FieldName, field_name_types[FieldName])
        SQLStr = SQLStr[:-2]

        Cursor = self.Connection.cursor()
        try:
            Cursor.execute(SQLStr)
            self.Connection.commit()
        except Exception as e:
            if if_exists == 'cancel':
                pass
            elif if_exists == 'replace':
                self.deleteFields(tablename, [FieldName for FieldName in field_name_types])
                Cursor.execute(SQLStr)
                self.Connection.commit()
            else:
                raise e
        Cursor.close()
        return 1

    # 删除字段
    def deleteFields(self, tablename, FieldName_List):
        SQLStr = 'ALTER TABLE %s DROP COLUMN ' % tablename
        if False in map(lambda x: isinstance(x, basestring), FieldName_List):
            raise Exception('The type of the elements in FieldName_List must be String!!')
        SQLStr = SQLStr + ','.join(FieldName_List)
        Cursor = self.Connection.cursor()
        try:
            Cursor.execute(SQLStr)
            self.Connection.commit()
        except:
            pass
        Cursor.close()
        return 1
