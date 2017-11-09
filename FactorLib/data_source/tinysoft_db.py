"""天软数据库
把从上天软下载的数据进行封装，统一存取
"""
from ..utils.tool_funcs import ensure_dir_exists
from ..utils.datetime_func import DateRange2Dates
import pandas as pd
import os

_root_dir = r"D:\data\TinySoft"


class TinyDB(object):
    def __init__(self, dir_path=_root_dir):
        """初始化数据库"""
        self._rootdir = dir_path
        ensure_dir_exists(self._rootdir)
        self._all_sub_dirs = os.listdir(self._rootdir)

    def subdir_exists(self, subdir):
        """子文件夹是否存在"""
        return os.path.isdir(os.path.join(self._rootdir, subdir))

    def load_info(self, sub_dir):
        """加载子文件夹的描述信息"""
        if self.subdir_exists(sub_dir):
            dscrp_file = os.path.join(self._rootdir, sub_dir, "description.txt")
            if os.path.isfile(dscrp_file):
                return pd.read_table(dscrp_file)
            else:
                return
        else:
            raise FileNotFoundError

    @DateRange2Dates
    def read_subdir(self, sub_dir, start_date=None, end_dates=None, dates=None):
        """读取子文件夹中的数据"""
        abs_subdir = os.path.join(self._rootdir, sub_dir)
        data = []
        for idate in dates:
            if os.path.isfile(os.path.join(abs_subdir, "%s.csv"%idate)):
                with open(os.path.join(abs_subdir, "%s.csv"%idate)) as f:
                    idata = pd.read_csv(f,parse_dates=0, header=0,
                                        converters={'IDs': lambda x: str(x).zfill(6)})
                    data.append(idata)
        table = pd.concat(data)
        return table.sort_index()

    def save_subdir(self, data, sub_dir):
        """存数据"""
        abs_subdir = os.path.join(self._rootdir, sub_dir)
        ensure_dir_exists(abs_subdir)
        all_dates = data.index.get_level_values(0).unique()
        for idate in all_dates:
            datestr = idate.strftime("%Y%m%d")
            idata = data.loc[[idate]]
            idata.to_csv(os.path.join(abs_subdir, "%s.csv"%datestr))
