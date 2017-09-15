from ..data_source.base_data_source_h5 import tc
import pandas as pd
import numpy as np


class RiskMatrixGenerator(object):
    def __init__(self, model):
        self._model = model
        self._db = model.riskdb

    def get_factor_risk(self, start_date, end_date, **kwargs):
        """
        计算因子收益率的协方差矩阵
        :param start_date: start date
        :param end_date: str
        :param kwargs: params to risk matrix function
        :type self: barra_model.BarraModel
        :type start_date: str
        :type end_date: str
        :return: risk_matrix
        """

