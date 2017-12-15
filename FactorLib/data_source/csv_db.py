"""CSV数据库
每个文件夹代表若干因子的集合，每个文件夹下的csv文件以日期命名，前两列是Date、IDs
"""

import pandas as pd
import os
from PkgConstVars import CSV_PATH
from FactorLib.utils.tool_funcs import ensure_dir_exists


class CsvDB(object):
    def __init__(self, path=CSV_PATH):
        self.root_path = path

    def list_dirs(self):
        """列出所有文件夹"""
        return os.listdir(self.root_path)

    def list_dates(self, sub_dir):
        """列出所有日期"""
        dir_name = os.path.join(self.root_path, sub_dir)
        return [x[:8] for x in os.listdir(dir_name)]

    def date_range(self, sub_dir):
        """列出某个文件夹的日期区间"""
        all_dates = self.list_dates(sub_dir)
        return min(all_dates), max(all_dates)

    def iter_by_dates(self, sub_dir, dates, factors=None):
        """按照日期，逐一迭代csv文件"""
        dir_name = os.path.join(self.root_path, sub_dir)

        if factors is not None:
            factors = ['date', 'IDs'] + list(factors)

        for idate in dates:
            file_name = os.path.join(dir_name, "%s.csv" % idate)
            data = pd.read_csv(file_name, header=0, usecols=factors, parse_dates=['date'], dtype={'IDs': 'str'},
                               encoding='GBK')
            yield idate, data.set_index(['date', 'IDs'])

    def save_factors(self, data, sub_dir):
        """存储文件"""
        tar_dir = os.path.join(self.root_path, sub_dir)
        ensure_dir_exists(tar_dir)
        groups = data.groupby(level=0)

        cur_dir = os.getcwd()
        os.chdir(tar_dir)
        for date, group in groups:
            group.reset_index().to_csv("%s.csv" % date.strftime("%Y%m%d"), index=False)
        os.chdir(cur_dir)


if __name__ == '__main__':
    csv = CsvDB()
    data_iter = csv.iter_by_dates('tf_six_barra', ['20150108', '20150112'])
    print(next(data_iter))
