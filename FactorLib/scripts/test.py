import os
import pandas as pd
from FactorLib.data_source.base_data_source_h5 import data_source

data_source.sector.get_industry_dummy(ids=None, start_date='20170701', end_date='20170831')