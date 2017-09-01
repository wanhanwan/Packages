# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %%
import shelve
import pandas as pd
from contextlib import closing
from FactorLib.data_source.base_data_source_h5 import h5

fields = ['BETA', 'BLEV', 'BTOP', 'CETOP', 'CMRA', 'DASTD', 'DTOA',
          'EGRLF', 'EGRO', 'EGRSF', 'EPFWD', 'ETOP', 'HSIGMA', 'LNCAP',
          'MLEV', 'RSTR', 'SGRO', 'STOA', 'STOM', 'STOQ']

file_name = r"D:\data\barra\barra_%d"

with closing(shelve.open(file_name)) as f:
    for key in f.keys():
        if key in fields:
            data = f[key]
            data.index = pd.DatetimeIndex(data.index, name='date')
            data = data.stack().to_frame(key).astype('float32')
            data.index.names = ['date','IDs']
        h5.save_factor(data, '/barra/')
        