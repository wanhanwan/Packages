import pandas as pd
import numpy as np
import statsmodels.api as sm
from ..data_source.base_data_source_h5 import tc
from ..data_source.trade_calendar import as_timestamp
from ..data_source.tseries import move_dtindex
from fastcache import clru_cache
from scipy import linalg



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
                iicov2 = calCovariance(iret, iret, nret, weight2)
                jjcov2 = calCovariance(jret, jret, nret, weight2)
                var[delta+N] = iicov2 ** .5 * corrij * jjcov2 ** .5
            varmatrix[i, j] = np.sum(var * coef)
    varmatrix = np.triu(varmatrix, 0) + np.tril(varmatrix.T, -1)
    if predict_window > ac_widow + 1:
        varmatrix = varmatrix * predict_window / (N+1)
    eigs, y = linalg.eig(varmatrix)
    if (eigs < 0).any():
        eigs[eigs < 0] = 0.0000001
        varmatrix = np.dot(y, np.dot(np.diag(eigs), np.linalg.inv(y)))
    return varmatrix


def calSpecificCovMat(factor_returns, date_ind, predict_window, ac_widow, half_life,
              rollback_len=360):
    if predict_window > ac_widow + 1:
        N = ac_widow
    else:
        N = predict_window - 1
    nfactor, factor_len = factor_returns.shape
    varmatrix = np.zeros(factor_len) + np.nan
    ret = factor_returns.iloc[max(0, date_ind-rollback_len+1):date_ind+1].values
    for i in np.arange(factor_len):
        iret = ret[:, i]
        if np.isnan(iret).sum() < 180:
            varmatrix[i] = 0
            continue
        else:
            iret = iret[~np.isnan(iret)]
            nret = len(iret)
            weight = np.flipud(getExpWeight(nret, half_life[0]))
            weight2 = np.flipud(getExpWeight(nret, half_life[1]))
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
                tmp_cov = calCovariance(iiret, iret, nret, weight)
            elif delta>0:
                s = max(0, date_ind-rollback_len+1)
                if s == 0:
                    zeros = np.zeros(min(delta, nret))
                    iiret = np.hstack((zeros, iret[:delta]))
                elif s >= delta:
                    iiret = factor_returns.iloc[s-delta:date_ind-delta+1, i].values
                else:
                    zeros = np.zeros(delta-s)
                    iiret = np.hstack((zeros,factor_returns.iloc[:nret-delta+s, i].values))
                tmp_cov = calCovariance(iret, iiret, nret, weight)
            else:
                tmp_cov = calCovariance(iret, iret, nret, weight)
            iicov = calCovariance(iret,iret,nret,weight)
            corrij = tmp_cov / iicov
            iicov2 = calCovariance(iret, iret, nret, weight2)
            var[delta+N] = iicov2 * corrij
            varmatrix[i] = np.sum(var * coef)
    if predict_window > ac_widow + 1:
        varmatrix = varmatrix * predict_window / (N+1)
    return pd.Series(varmatrix, index=factor_returns.columns)


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


def calStrucStd(gamma, ts_std, date, factor_data, industry_data, weight_data):
    """
    建立特异性风险结构化模型
    :param gamma: blending paramters
    :param ts_std: time-series std
    :param date: current date
    :param factor_data: style factor data
    :param industry_data: industry dummies
    :param weight_data: regression weight
    :return:
    """
    ids = gamma[gamma == 1].index.tolist()
    ifactor_data = factor_data.loc[date].reindex(ids).values
    iindustry_data = industry_data.loc[date].reindex(ids).values
    iweight = weight_data.loc[date].reindex(ids).values
    iweight = np.sqrt(iweight / iweight.sum())
    y = np.log(ts_std[ids].values) * iweight
    x = np.vstack((ifactor_data, iindustry_data))
    x = sm.add_constant(x)
    result = sm.OLS(y, x[:, :-1]).fit()







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



