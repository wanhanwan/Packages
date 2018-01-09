from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sqlalchemy as sa

class BaseDB(object):
    def __init__(self, user_name, pass_word, db_name, ip_address, db_type='oracle',
                 port=1521):
        self.db_type = db_type
        self.user_name = user_name
        self.db_name = db_name
        self.ip_address = ip_address
        self.db_engine = create_engine(self.create_conn_string(user_name, pass_word, db_name, ip_address, port, db_type))

    @staticmethod
    def create_conn_string(user_name, pass_word, db_name, ip_address, port, db_type):
        return "{db_type}://{user_name}:{pass_word}@{ip_address}:{port}/{db_name}".format(
            db_type=db_type, user_name=user_name, pass_word=pass_word, ip_address=ip_address, port=port, db_name=db_name)

    def exec_query(self, query_str, **kwargs):
        with self.db_engine.connect() as conn, conn.begin():
            data = pd.read_sql(query_str, conn, **kwargs)
        return data

    def list_columns_of_table(self, table_name):
