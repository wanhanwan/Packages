# coding: utf-8
# 更新成分股的指数
index_members = ['000300.SH', '000905.SH', '399102.SZ', '000906.SH', '399974.SZ', '000016.SH']

# 需要更新的板块
sector_members = {'is_st':'1000006526000000', 'ashare':'a001010100000000'}

# 更新成分股权重的指数
index_weights = ['000905.SH', '000300.SH', '000991.SH', '000906.SH', '399974.SZ']

# 更新自定义指数成分股
slfdef_index = [{'func': 'typical_add_latest_st', 'func_args': {'st_months': 12}, 'name': 'barramodel1'}]

# 更新行业分类
industry_classes = {'sw_level_1':('indexcode_sw', 1),
                    'cs_level_1':('indexcode_citic', 1),
                    'sw_level_2':('indexcode_sw', 2),
                    'cs_level_2':('indexcode_citic', 2),
                    'wind_level_1':('indexcode_wind', 1)
                    }
