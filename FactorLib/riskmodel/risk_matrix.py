import pandas as pd
import numpy as np
from ..data_source.base_data_source_h5 import tc
from ..data_source.tseries import move_dtindex
from fastcache import clru_cache
from numba import jit
import numpy.linalg as la


@clru_cache()
def getExpWeight(window, half_life):
    """指数权重"""
    base = .5 ** (1 / half_life)
    exp_w = np.power(base, np.arange(window))
    return exp_w / exp_w.sum()


def calCovMatrix(mat1, mat2=None, weight=None):
    """计算协方差矩阵
    Patameters:
    -----------
    mat: np.array
        每一列是一个变量
    """
    if mat2 is None:
        mat2 = mat1
    mat1_demean = mat1 - mat1.mean(0)[None, :]
    mat2_demean = mat2 - mat2.mean(0)[None, :]
    mat2_demean *= weight[:, None]
    m = np.dot(mat1_demean.T, mat2_demean)
    return m


@jit()
def _newweyAdjust(ret_mat, k, n, weight):
    """计算Newwey-West中的调整项"""
    cov = np.zeros((ret_mat.shape[1], ret_mat.shape[1]))
    for i in range(1, k+1):
        mat1 = ret_mat[k-i:n+k-i, :]
        mat2 = ret_mat[k:n+k, :]
        cov_i = calCovMatrix(mat1, mat2, weight)
        cov += ((cov_i + cov_i.T) * (1-i/(1+k)))
    return cov


def calRiskByNewwey(ret_mat, date, predict_window, ac_window, half_life,
                    rollback_len=360):
    """使用Newwey-West调整计算协方差矩阵

    Parameters:
    -----------
    ret_mat : pd.DataFrame
        收益率矩阵，每一列代表一个变量
    date : str or datetime
        截止日期
    predict_window : int
        向前预测天数。如果是月频，通常取21。
    ac_window : int
        假设自相关的周期频率。
    half_life : list with size of two
        第一个半衰期用于计算相关系数，第二个半衰期用于
        计算协方差
    rollback_len : int
        滚动窗口期
    """
    ret = ret_mat[:date].iloc[-(rollback_len+ac_window):, :].values
    weight1 = np.flipud(getExpWeight(rollback_len, half_life[0]))
    weight2 = np.flipud(getExpWeight(rollback_len, half_life[1]))

    cov = calCovMatrix(ret[ac_window:, :], weight=weight1)
    std = np.sqrt(np.diag(cov))[:, np.newaxis].dot(
        np.sqrt(np.diag(cov))[np.newaxis, :])

    cov2 = calCovMatrix(ret[ac_window:, :], weight=weight2)
    std2 = np.sqrt(np.diag(cov2))[:, np.newaxis].dot(
        np.sqrt(np.diag(cov2))[np.newaxis, :])

    cov_newwey = cov + _newweyAdjust(ret, ac_window, rollback_len, weight1)
    cov_newwey /= std
    cov_newwey *= std2
    cov_newwey *= predict_window

    eig, vec = la.eig(cov_newwey)
    eig[eig < 0] = 1e-6
    cov_newwey = vec.dot(np.diag(eig)).dot(la.inv(vec))

    return cov_newwey


def calBlendingParam(factor_returns, date_ind, rollback_len=360):
    nfactor = factor_returns.shape[1]
    ret = factor_returns.iloc[max(0, date_ind-rollback_len+1):date_ind+1].values
    gamma = np.zeros(nfactor)
    for i in np.arange(nfactor):
        iret = ret[:, i]
        if np.isnan(iret).sum() < 180:
            gamma[i] = 0
            continue
        else:
            origin_len = len(iret)
            iret = iret[~np.isnan(iret)]
            h = len(iret)
        robust_std = 1 / 1.35 * (np.percentile(iret, 75) - np.percentile(iret, 25))
        iret[iret > 10 * robust_std] = 10 * robust_std
        iret[iret < -10 * robust_std] = -10 * robust_std
        std = np.std(iret)
        zval = np.abs((std - robust_std) / robust_std)
        gamma[i] = min((1, h / origin_len + 0.1)) * min((1, max((0, np.exp(1 - zval)))))
    return pd.Series(gamma, index=factor_returns.columns)


def volatility_regime_adjust(factor_returns, bf_old, matrix, predict_window):
    """
    波动率截面调整. 采用全历史样本估计。
    
    :param factor_returns: 因子收益率时间序列，N*K
    :param bf_old: 已经估算的bf序列
    :param matrix: 待估算的因子标准差
    :param predict_window: 预测区间长度
    """
    half_life = 90
    # 把日频的因子收益率序列调整成与协方差矩阵齐频
    ret_sample = factor_returns[::-1].rolling(window=predict_window, min_periods=predict_window).sum()[::-1]
    ret_sample = move_dtindex(ret_sample.dropna(how='all'), -1)
    if (ret_sample.index.max() > bf_old.index.max()) or (bf_old.empty):   # 若因子收益率序列的最大日期大于已有的bf序列，更新bf序列
        new_bf = pd.Series(np.zeros(len(ret_sample.index[ret_sample.index > bf_old.index.max()])),
                           index=ret_sample.index[ret_sample.index > bf_old.index.max()])
        for idx in new_bf.index:
            ret = ret_sample.loc[idx, 'factor_return'].values
            sigma = np.diag(matrix[idx])
            new_bf[idx] = np.nanmean((ret * sigma) ** 2)
        bf_old = bf_old.append(new_bf)
    # 计算new_matrix
    new_matrix = {}
    for k, v in matrix.items():
        bf = bf_old.loc[:k]
        weight = np.flipud(getExpWeight(len(bf), half_life))
        lambda_f = np.sum(weight * bf**2)
        new_matrix[k] = v * lambda_f
    return new_matrix, bf_old
    

class RiskMatrixGenerator(object):
    def __init__(self, model):
        self._model = model
        self._db = model.riskdb
        self._ds = model.data_source

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
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        try:
            bf = self._db.load_others('bf')['bf']
        except:
            bf = pd.Series()
        if not factor_returns.empty:
            factors = factor_returns.columns
            new_factor_matrix = {}
            for date in dates:
                date_ind = factor_returns.index.get_loc(date)
                _matrix = calCovMat(factor_returns, date_ind, **kwargs)
                new_factor_matrix[date] = pd.DataFrame(_matrix, index=factors, columns=factors)
            new_matrix, bf = volatility_regime_adjust(factor_returns, bf, new_factor_matrix, kwargs['predict_window'])
        else:
            new_matrix = {}
        self._db.save_others('bf', bf=bf)
        return new_matrix

    def get_specific_risk(self, start_date, end_date, strstd_stylefactor, **kwargs):
        _default_startdate = '20010101'
        resid_return = self._db.load_resid_factor(start_date=_default_startdate, end_date=end_date)
        resid_return = resid_return['resid_return'].unstack()
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        sqrt_weight = self._model.getRegressWeight(dates=dates)
        factor_data = self._model.getStyleFactorData(dates=dates)[strstd_stylefactor]
        industry_dummy = self._model.getIndustryDummy(dates=dates)
        if not resid_return.empty:
            for date in dates:
                date_ind  = resid_return.index.get_loc(date)
                ts_std = calSpecificCovMat(resid_return, date_ind=date_ind, **kwargs)
                gamma = calBlendingParam(resid_return, date_ind, kwargs['rollback_len'])



