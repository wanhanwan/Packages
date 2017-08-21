"""因子定义
因子定义规范如下:
1. 因子必须有因子名称(可以是中文)、其在数据库中的字段以及因子类型(panel,series,section)
2. 每一个因子只能有一个字段
3. 因子名称和因子类型一旦设定无法更改
"""
import pandas as pd

class factor(object):

    def __init__(self, name, axe):
        self.name = name
        self.axe=axe
        self.type = 'panel'
        self.data = None
        self.Date = None
        self.IDs = None
        self.update_info()

    def update_info(self):
        if self.type == 'series':
            self.IDs = ['000000']
        elif self.type == 'section':
            self.Date = ['11111111']
        elif self.data is not None:
            self.Date = self.data.index.levels[0].unique()
            self.IDs = self.data.index.levels[1].unique()
        self.date_range = self.get_date_range()

    def print_info(self):
        info_dict = {
            'name':self.name,
            'type':self.type,
            'axe':self.axe,
            'data_min_date':self.date_range[0] if self.date_range else 0,
            'data_max_date':self.date_range[1] if self.date_range else 0,
            'data_rows':len(self.data) if not self.data.empty else 0
        }
        print(pd.Series(info_dict))

    def set_axe(self, axe):
        """重新给定字段"""
        self.axe = axe

    def get_date_range(self):
        if self.data is not None:
            max_date = self.data.index.levels[0].max()
            min_date = self.data.index.levels[0].min()
            return (min_date,max_date)
        else:
            return None

    def load_data(self):
        factor.operator.load_factor(self)