import numpy.linalg as la
import numpy as np
import pandas as pd
from fastcache import clru_cache
from numba import jit

from ..data_source.tseries import move_dtindex
from ..data_source.base_data_source_h5 import tc
from .riskmodel_data_source import RiskDataSource


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


def calRiskByNewwey(ret_mat, predict_window, ac_window, half_life,
                    rollback_len=360):
    """使用Newwey-West调整计算协方差矩阵

    Parameters:
    -----------
    ret_mat : 2D-Array
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
    ret = ret_mat[-(rollback_len+ac_window):, :]
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
    cov_newwey = vec.dot(np.diag(eig)).dot(vec.T)

    return cov_newwey


@jit()
def _monteCarlo(cov, m, k, d, u, predict_window, ac_window,
                half_life, rollback_len=360):
    """Monte Carlo Simulation when eigen-adjust

    Parameters:
    -----------
    cov : 2D-array
        原始的协方差矩阵
    m : int
        模拟的次数
    k : int
        因子数量
    d : 1D-array
        对cov正交分解后的特征向量
    u : 2D-array
        对cov正交分解后的特征矩阵
    predict_window : int
        向前预测天数。如果是月频，通常取21。
        取值与Newwey-west函数的参数相同。
    ac_window : int
        假设自相关的周期频率。
        取值与Newwey-west函数的参数相同。
    half_life : list with size of two
        第一个半衰期用于计算相关系数，第二个半衰期用于
        计算协方差。
        取值与Newwey-west函数的参数相同。
    rollback_len : int
        滚动窗口期。取值与Newwey-west函数的参数相同.
    """
    ret_len = rollback_len + ac_window
    d_list = np.zeros((m, k))
    d_list_e = np.zeros((m, k))
    for i in range(m):
        b_m = np.random.randn(k, ret_len) * np.sqrt(d)[:, None]  # K*T
        f = u.dot(b_m)
        fcov = calRiskByNewwey(f.T, predict_window, ac_window, half_life,
                               rollback_len)
        eig_m, u_m = la.eig(fcov)
        d_m = u_m.T.dot(cov).dot(u_m)
        d_list[i, :] = np.diag(d_m)
        d_list_e[i, :] = eig_m
    v = (d_list / d_list_e).mean(0)
    v = (1.4 * (v - 1) + 1) ** 2
    d *= v
    return u.dot(np.diag(d)).dot(u.T)


def eigenFactorAdjust(cov, m, predict_window, ac_window,
                      half_life, rollback_len=360):
    """Eigenfactor Risk Adjustment

    Parameters:
    -----------
    cov : 2D-array
        原始的协方差矩阵
    m : int
        模拟的次数
    predict_window : int
        向前预测天数。如果是月频，通常取21。
        取值与Newwey-west函数的参数相同。
    ac_window : int
        假设自相关的周期频率。
        取值与Newwey-west函数的参数相同。
    half_life : list with size of two
        第一个半衰期用于计算相关系数，第二个半衰期用于
        计算协方差。
        取值与Newwey-west函数的参数相同。
    rollback_len : int
        滚动窗口期。取值与Newwey-west函数的参数相同.
    """
    k = cov.shape[0]
    eig, u = la.eig(cov)
    return _monteCarlo(cov, m, k, eig, u,
                       predict_window,
                       ac_window,
                       half_life,
                       rollback_len)


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


def _cal_volatility_bias(ret_df, vol_df):
    """计算横截面上波动率偏差统计量B"""
    sigma = np.sqrt(
        vol_df.groupby('date').apply(lambda x: pd.Series(np.diag(x.values), index=x.columns))
    )
    b = ((ret_df.values / sigma)**2).mean(1) ** 0.5
    return pd.Series(b, index=ret_df.index)


def volatility_regime_adjust(cov, bias_series, half_life, rollback_len):
    """
    波动率截面调整. 采用全历史样本估计。
    
    :param factor_returns: 因子收益率时间序列，N*K
    :param bf_old: 已经估算的bf序列
    :param matrix: 待估算的因子标准差
    :param predict_window: 预测区间长度
    """
    exp_weight = getExpWeight(rollback_len, half_life)
    lambda_f = exp_weight[None, :].dot((bias_series.values ** 2)[:, None])
    return cov * lambda_f


class RiskMatrixGenerator(object):
    """风险矩阵生成器
    在这里可以计算因子的协方差矩阵和股票发特质风险矩阵
    """
    def __init__(self, risk_ds):
        """
        Parameters:
        -----------
        risk_ds : str
            风险数据库名称
        """
        self.ds = RiskDataSource(risk_ds)

    def get_fctrsk_before_voladj(self, start_date, end_date, **kwargs):
        """
        计算因子收益率的协方差矩阵(在volatility regime adjust之前)
        协方差矩阵在因子收益率之后计算，每日更新

        Parameters:
        -----------
        start_date : str
            初始日期 YYYYmmdd
        end_date : str
            截止日期 YYYYmmdd
        """
        _default_startdate = '20100101'
        factor_returns = self.ds.load_returns(
            start_date=_default_startdate, end_date=end_date)
        factor_return_arr = np.asarray(factor_returns.values, dtype='float')
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        len_dates = len(dates)
        k_factors = factor_returns.shape[1]
        cov_list = np.zeros((len_dates*k_factors, k_factors))
        for i, d in enumerate(dates):
            print("calculating... current date: %s"%d.strftime('%Y%m%d'))
            date_ind = factor_returns.index.get_loc(d)
            cov_raw = calRiskByNewwey(ret_mat=factor_return_arr[:date_ind],
                                      predict_window=kwargs['predict_window'],
                                      ac_window=kwargs['ac_window'],
                                      half_life=kwargs['half_life'],
                                      rollback_len=kwargs['rollback_len'])
            cov = eigenFactorAdjust(cov_raw,
                                    kwargs['m'],
                                    kwargs['predict_window'],
                                    ac_window=kwargs['ac_window'],
                                    half_life=kwargs['half_life'],
                                    rollback_len=kwargs['rollback_len'])
            cov_list[i*k_factors: (i+1)*k_factors, :] = cov
        idx = pd.MultiIndex.from_product([dates, factor_returns.columns],
                                         names=['date', 'IDs'])
        cov_df = pd.DataFrame(cov_list, columns=factor_returns.columns,
                              index=idx)
        return cov_df
    
    def save_fctrsk_before_voladj(self, cov_df):
        """保存风险矩阵(波动率调整前)"""
        name = 'factor_cov_before_voladj'
        if self.ds.persist_helper.check_file_existence(name):
            raw = self.ds.persist_helper.load_other(name)
            new = raw.append(cov_df)
            new = new[~new.index.duplicated(keep='last')]
        else:
            new = cov_df
        self.ds.persist_helper.save_other(name, new)

    def get_bias_stat_of_factor_risk(self, start_date, end_date, **kwargs):
        """计算因子收益率协方差矩阵偏差统计量"""
        _default_startdate = '20100101'
        predict_window = kwargs['predict_window']
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        factor_returns = self.ds.load_returns(
            start_date=_default_startdate, end_date=end_date)
        factor_cov = self.ds.load_other('factor_cov_before_voladj')

        start_date_ind = factor_returns.index.get_loc(dates[0])
        end_date_ind = factor_returns.index.get_loc(dates[-1])
        real_ret = factor_returns.iloc[start_date_ind-predict_window:end_date_ind+1, :].rolling(
            window=predict_window, min_periods=predict_window).sum().shift(-predict_window).dropna()

        factor_cov = factor_cov.reindex(real_ret.index, level='date')
        b_series = _cal_volatility_bias(real_ret, factor_cov)
        b_series.index = pd.DatetimeIndex(dates, name='date')
        return b_series