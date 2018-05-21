import pandas as pd
from QuantLib.utils import NeutralizeByRiskFactors
from FactorLib.data_source.base_data_source_h5 import data_source


dates = data_source.trade_calendar.get_trade_days('20180301', '20180331', '1m')
data = data_source.load_factor("StyleFactor_VGS", "/XYData/StyleFactor/", dates=dates)
data_neu = NeutralizeByRiskFactors(data, factor_name=data.columns[0])