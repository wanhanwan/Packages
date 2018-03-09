"""优矿数据库"""
from uqer import Client, DataAPI


class UqerDB(object):
    """优矿Online数据库封装"""
    _instance = None

    def __init__(self, token="cc064ffebf0f0ad781f6127c7d84cade916f373946f481ea12d80006be7f4d64"):
        self.token = token
        self.client = None

    def connect(self):
        if not self.is_connected():
            self.client = Client(token=self.token)

    def is_connected(self):
        return self.client is not None

    @classmethod
    def get_instance(cls):
        if UqerDB._instance is None:
            UqerDB._instance = UqerDB()
            return UqerDB._instance
        else:
            return UqerDB._instance

    def run_api(self, api_name, *args, **kwargs):
        self.connect()
        return getattr(DataAPI, api_name)(*args, **kwargs)


if __name__ == '__main__':
    uqer_ins = UqerDB.get_instance()
    uqer_ins.connect()
    data = uqer_ins.run_api('EquRestructuringGet', ticker=['000002'],
                            field=['ticker', 'program', 'restructuringType',
                                   'iniPublishDate', 'isSucceed', 'institNameSub', 'publishDate'],
                            beginDate='20180101', endDate='20181231')
    data.to_hdf("D:/data/ma_2018.h5", "data")