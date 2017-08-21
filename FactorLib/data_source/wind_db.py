from multiprocessing import Lock

class WindDB(object):
    """Wind Old Data Base"""

    def __init__(self, user="windfile", password="windfile", host="172.20.65.11", port="1521", dbname="wind",
                 data_lock=Lock(), wind_dict_path=None,
                 db_type='Oracle', lib_path=None, table_prefix='wind.'):
        self.User = user
        self.Password = password
        self.Host = host
        self.Port = port
        self.DBName = dbname
        self.Connection = None
        self.WindDictPath = wind_dict_path
        self.LibPath = lib_path
        self.Data = data_lock
        self.DBType = db_type
        self.TablePrefix = table_prefix
        if self.WindDictPath is not None:
            self.DataLock.acquire()
            DictFile = shelve.open(self.WindDictPath)
            self.TableInfo = DictFile['TableInfo']
            self.FactorInfo = DictFile['FactorInfo']
            DictFile.close()
            self.DataLock.release()
        return

    # -------------------------------------------数据库的连接与查询------------------------------------
    def getConnInfo(self):
        return {'User': self.User, 'Password': self.Password, 'Host': self.Host, 'Port': self.Port,
                'DBName': self.DBName, 'DBType': self.DBType}

    # 连接数据库
    def connectDB(self):
        if self.DBType == 'Oracle':
            try:
                import cx_Oracle
                self.Connection = cx_Oracle.connect(self.User, self.Password,
                                                    cx_Oracle.makedsn(self.Host, self.Port, self.DBName))
                return 1
            except Exception as e:
                print(e)
                pass
        elif self.DBType == 'SQL Server':
            try:
                import pymssql
                self.Connection = pymssql.connect(server=self.Host, user=self.User, password=self.Password,
                                                  database=self.DBName, charset='UTF-8', port=self.Port)
                return 1
            except:
                pass
        try:
            import pyodbc
            ConnInfo = 'DRIVER={' + self.DBType + '};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s' % (
            self.DBName, self.Host, self.User, self.Password)
            self.Connection = pyodbc.connect(ConnInfo)
            return 1
        except:
            pass
        return -1

    # 断开数据库
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
    def isDBAvailable(self):
        if self.Connection is None:
            return False
        else:
            return True

    # 执行SQL查询语句
    def execSQLQuery(self, sql_str):
        if self.Connection is not None:
            Cursor = self.Connection.cursor()
            Cursor.execute(sql_str)
            return Cursor
        else:
            return None

    def __str__(self):
        ObjectStr = 'Host : %s \n Port : %s \n Data base name : %s \n User ID : %s \n Password : %s' % (
        self.Host, self.Port, self.DBName, self.User, self.Password)
        return ObjectStr