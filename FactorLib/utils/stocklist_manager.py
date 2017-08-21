import pandas as pd
import os
import shutil


class StockListManager(object):

    def __init__(self, path):
        self._stocklist_path = path

    def min_rebalance_date(self, name):
        f = open(os.path.join(self._stocklist_path, name+'.csv'))
        return pd.read_csv(f)['date'].min()

    def max_rebalance_date(self, name):
        f = open(os.path.join(self._stocklist_path, name+'.csv'))
        return pd.read_csv(f)['date'].max()

    def check_file_exists(self, name):
        return name+'.csv' in os.listdir(self._stocklist_path)

    def add_new_one(self, src):
        if self.check_file_exists(os.path.split(src)[1].replace('.csv', '')):
            raise FileExistsError
        shutil.copy(src, self._stocklist_path)

    def delete_stocklist(self, info):
        if not self.check_file_exists(info['stocklist_name']):
            raise FileNotFoundError
        os.remove(info['stocklist_name']+'.csv')

    # 获取历史股票持仓
    def get_position(self, name, date):
        f = open(os.path.join(self._stocklist_path, name+'.csv'))
        return pd.read_csv(f).set_index(['date','IDs']).loc[date]

    # 获取最近的持仓
    def get_latest_position(self, name):
        max_date = self.max_rebalance_date(name)
        f = open(os.path.join(self._stocklist_path, name+'.csv'))
        return pd.read_csv(f).loc[max_date]

    # 股票列表的绝对路径
    def get_path(self, name):
        name = name.replace('.csv', '')
        if self.check_file_exists(name):
            return os.path.join(self._stocklist_path, name+'.csv')
        else:
            return
