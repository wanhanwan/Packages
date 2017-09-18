import pandas as pd
import numpy as np
from ..data_source.base_data_source_h5 import tc
from ..data_source.trade_calendar import as_timestamp
from fastcache import clru_cache
from scipy.linalg import eig


@clru_cache()
def getExpWeight(window, half_life):
    """指数权重"""
    base = .5 ** (1 / half_life)
    exp_w = np.power(base, np.arange(window))
    return exp_w / exp_w.sum()


def calCovariance(fj, fk, nlen, weight):
    """计算两列数据的协方差"""
    fjMean = np.nanmean(fj)
    fkMean = np.nanmean(fk)
    Res = (fj-fjMean)*(fk-fkMean)
    TotalWeight = np.sum(weight[~np.isnan(Res)])
    return np.nansum((Res*weight).astype('float32'))/TotalWeight


def calCovMat(factor_returns, date_ind, predict_window, ac_widow, half_life,
              rollback_len=360):
    if predict_window > ac_widow + 1:
        N = ac_widow
    else:
        N = predict_window - 1
    nfactor, factor_len = factor_returns.shape
    varmatrix = np.zeros((factor_len, factor_len)) + np.nan
    ret = factor_returns.iloc[max(0, date_ind-rollback_len+1):date_ind+1].values
    nret = ret.shape[0]
    weight = np.flipud(getExpWeight(nret, half_life[0]))
    weight2 = np.flipud(getExpWeight(nret, half_life[1]))
    for i in np.arange(factor_len):
        for j in np.arange(i, factor_len):
            iret = ret[:, i]
            jret = ret[:, j]
            var = np.zeros(N*2+1)
            coef = np.zeros(N*2+1)
            for delta in np.arange(-N, N+1):
                coef[delta+N] = N+1-np.abs(delta)
                if delta<0:
                    s = max(0, date_ind-rollback_len+1)
                    if s == 0:
                        zeros = np.zeros(min(-delta, nret))
                        iiret = np.hstack((zeros, iret[:delta]))
                    elif s >= -delta:
                        iiret = factor_returns.iloc[s+delta:date_ind+delta+1, i].values
                    else:
                        zeros = np.zeros(-delta-s)
                        iiret = np.hstack((zeros,factor_returns.iloc[:nret+delta+s, i].values))
                    tmp_cov = calCovariance(iiret, jret, nret, weight)
                elif delta>0:
                    s = max(0, date_ind-rollback_len+1)
                    if s == 0:
                        zeros = np.zeros(min(delta, nret))
                        jjret = np.hstack((zeros, jret[:delta]))
                    elif s >= delta:
                        jjret = factor_returns.iloc[s-delta:date_ind-delta+1, j].values
                    else:
                        zeros = np.zeros(delta-s)
                        jjret = np.hstack((zeros,factor_returns.iloc[:nret-delta+s, j].values))
                    tmp_cov = calCovariance(iret, jjret, nret, weight)
                else:
                    tmp_cov = calCovariance(iret, jret, nret, weight)
                iicov = calCovariance(iret,iret,nret,weight)
                jjcov = calCovariance(jret, jret, nret, weight)
                corrij = tmp_cov / (iicov * jjcov) ** .5
                iicov2 = calCovariance(iret,iret,nret,weight2)
                jjcov2 = calCovariance(jret, jret, nret, weight2)
                var[delta+N] = iicov2 ** .5 * corrij * jjcov2 ** .5
            varmatrix[i, j] = np.sum(var * coef)
    varmatrix = np.triu(varmatrix, 0) + np.tril(varmatrix.T, -1)
    if predict_window > ac_widow + 1:
        varmatrix = varmatrix * predict_window / (N+1)
    eigs, y = eig(varmatrix)
    if (eigs < 0).any():
        eigs[eigs < 0] = 0.0000001
        varmatrix = np.dot(y, np.dot(np.diag(eig, np.linalg.inv(y))))
    return varmatrix


def volatility_regime_adjust(factor_returns, old_matrix, matrix, predict_window):
    """
    波动率截面调整
    
    @param 
    """
    


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
        _default_startdate = '20100101'
        factor_returns = self._db.load_returns(start=_default_startdate, end_date=end_date)
        factor_matrix = self._db.load_factor_riskmatrix(start_date=_default_startdate, end_date=end_date)
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        if not factor_returns.empty:
            nfactors = factor_returns.shape[0]
            new_factor_matrix = pd.DataFrame(data=np.empty((nfactors*len(dates), nfactors)),
                                             index=pd.MultiIndex.from_product([dates, factor_returns.columns]))
            for date in dates:
                date_ind = factor_returns.index.get_loc(start_date)
                _matrix = calCovMat(factor_returns, date_ind, **kwargs)
                new_factor_matrix.loc[date] = _matrix
            


        
