from ..wind_financial_data_api.database import BaseDB
from .params import *


class GoGoalDB(BaseDB):
    def __init__(self, user_name=GOGOAL_USER, pass_word=GOGOAL_PASSWORD, db_name=GOGOAL_DBNAME,
        ip_address=GOGOAL_IP, db_type=GOGOAL_DBTYPE, port=GOGOAL_PORT):
        super(BaseDB, self).__init__(user_name, pass_word, db_name, ip_address, db_type, port)