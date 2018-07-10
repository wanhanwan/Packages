import numpy.linalg as la
import statsmodels.api as sm
import numpy as np
import pandas as pd
from fastcache import clru_cache
from numba import jit
from statsmodels.sandbox.rls import RLS
from alphalens.utils import quantize_factor


from ..data_source.tseries import move_dtindex
from ..data_source.base_data_source_h5 import tc, data_source, h5
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


def _calBlendingParam(ret_mat, date_ind, rollback_len=360):
    """计算协调参数gamma"""
    len_ret, len_stock = ret_mat.shape
    gamma = np.zeros(((len_ret-date_ind), len_stock))
    for ii, i in enumerate(range(date_ind, len_ret)):
        imat = ret_mat[i+1-rollback_len:i+1, :]
        robust_std = 1.0 / 1.35 * (np.percentile(imat, 75, axis=0) -
                                   np.percentile(imat, 25, axis=0))
        imat = np.where(imat > 10 * robust_std[None, :],
                        10*np.tile(robust_std[None, :], (rollback_len, 1)),
                        imat)
        imat = np.where(imat < -10 * robust_std[None, :],
                        -10*np.tile(robust_std[None, :], (rollback_len, 1)),
                        imat)
        # imat[imat > 10 * robust_std] = 10 * robust_std
        # imat[imat < -10 * robust_std] = -10 * robust_std
        std = np.std(imat, axis=0)
        zval = np.abs((std - robust_std) / robust_std)
        gamma[ii, :] = min(1.0, max(.0, rollback_len / 120 - 0.5)) * \
                      np.where(np.exp(1.0-zval) > 1.0, np.ones(len(zval)), np.exp(1-zval))
    return gamma


def _cal_volatility_bias(ret_df, vol_df):
    """计算横截面上波动率偏差统计量B"""
    sigma = np.sqrt(
        vol_df.groupby('date').apply(lambda x: pd.Series(np.diag(x.values), index=x.columns))
    )
    b = ((ret_df.values / sigma.values)**2).mean(1) ** 0.5
    return pd.Series(b, index=ret_df.index)


def _volatility_regime_adjust(cov, bias_series, half_life, rollback_len):
    """
    波动率截面调整. 采用全历史样本估计。
    
    Parameters:
    -----------
    cov : 2D-array
        原始协方差矩阵
    bias_series : 1D-array
        截面偏差统计量时间序列
    half_life : int
        半衰期
    rollback_len : int
        回溯时间长度
    """
    exp_weight = getExpWeight(rollback_len, half_life)
    lambda_f = exp_weight[None, :].dot((bias_series ** 2)[:, None])
    return cov * lambda_f


def _getResgressWeight(dates, percentile=0.95, idx=None):
    """计算回归权重, 流通市值的平方根"""
    weight = h5.load_factor('float_mkt_value', '/stocks/',
                            dates=list(dates), idx=idx)
    weight = np.sqrt(weight).astype('float32')
    weight_sum_perdate = weight.groupby(level='date').sum()
    weight = weight / weight_sum_perdate
    quantile_weight_perdate = weight.groupby(level='date').quantile(percentile)
    weight, quantile_weight_perdate = weight.align(quantile_weight_perdate, axis=0, level=0)
    weight.loc[quantile_weight_perdate.iloc[:, 0] < weight.iloc[:, 0]] = quantile_weight_perdate
    return weight


def _getIndustryMarketValue(industry_dummy):
    """计算行业市值"""
    float_mkt_mv = h5.load_factor('float_mkt_value', '/stocks/', idx=industry_dummy)
    industry_mv = pd.DataFrame(industry_dummy.values * float_mkt_mv.values, index=industry_dummy.index,
                               columns=industry_dummy.columns)
    industry_mv = industry_mv.sum(level='date').stack()
    return industry_mv


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

    def _save_middle_df(self, data, name):
        """存储中间变量，以dataframe格式保存"""
        if self.ds.persist_helper.check_file_existence(name):
            raw = self.ds.persist_helper.load_other(name)
            new = raw.append(data)
            new = new[~new.index.duplicated(keep='last')]
        else:
            new = data
        self.ds.save_other(name, new)
        return 1
    
    def save_fctrsk_before_voladj(self, cov_df):
        """保存风险矩阵(波动率调整前)"""
        name = 'factor_cov_before_voladj'
        return self._save_middle_df(cov_df, name)

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
        real_ret = factor_returns.iloc[start_date_ind-predict_window+1:end_date_ind+1, :].rolling(
            window=predict_window, min_periods=predict_window).sum().dropna()

        cov_startdate = tc.tradeDayOffset(start_date, -predict_window)
        cov_enddate = tc.tradeDayOffset(end_date, -predict_window)
        sample_cov = move_dtindex(factor_cov.loc[cov_startdate:cov_enddate, :],
                                  predict_window,
                                  freq='1d')
        assert (len(sample_cov) / sample_cov.shape[1]) == len(real_ret)

        b_series = _cal_volatility_bias(real_ret, sample_cov)
        b_series.index = pd.DatetimeIndex(dates, name='date')
        return b_series

    def save_bias_stat_of_factor_risk(self, bias):
        name = 'bias_stat_of_factor_risk'
        self._save_middle_df(bias, name)

    def get_fctrsk_after_voladj(self, start_date, end_date, **kwargs):
        """volatility regime adjust"""
        half_life = kwargs['half_life']
        rollback_len = kwargs['rollback_len']

        factor_cov = self.ds.load_other('factor_cov_before_voladj')
        bias = self.ds.load_other('bias_stat_of_factor_risk')

        n_factors = factor_cov.shape[1]
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        bias_startdates = [tc.tradeDayOffset(x, -rollback_len+1) for x in dates]

        rslt = np.zeros((len(dates)*n_factors, n_factors))
        for i, (d, dd) in enumerate(zip(dates, bias_startdates)):
            cov = factor_cov.loc[d].values
            b_his = bias.loc[dd:d].values
            cov_new = _volatility_regime_adjust(cov, b_his, half_life, rollback_len)
            rslt[i*n_factors: (i+1)*n_factors, :] = cov_new
        idx = pd.MultiIndex.from_product([dates, factor_cov.columns],
                                         names=['date', 'IDs'])
        return pd.DataFrame(rslt, index=idx, columns=factor_cov.columns)

    def save_factor_risk(self, start_date, end_date, **kwargs):
        cov_raw = self.get_fctrsk_before_voladj(start_date, end_date, **kwargs)
        self.save_fctrsk_before_voladj(cov_raw)

        bias = self.get_bias_stat_of_factor_risk(start_date, end_date, **kwargs)
        self.save_bias_stat_of_factor_risk(bias)

        cov_new = self.get_fctrsk_after_voladj(start_date, end_date, **kwargs)
        cov_dict = {x: cov_new.loc[x] for x in cov_new.index.unique(level='date')}

        self.ds.save_data(factor_riskmatrix=cov_dict)
        return 1

    @staticmethod
    def get_nwadj_spec_risk(resid_ret, start_date, end_date, **kwargs):
        """Newey-West Adjusted Specific Risk

        Note:
            由于模型假设特质性风险存在两两不相关性，所以只返回一个对角阵,
            协方差假设为零。
        """
        # _default_startdate = '20100101'
        # resid_ret = self.ds.load_resid_factor(
        #     start_date=_default_startdate, end_date=end_date)
        resid_ret_arr = np.asarray(resid_ret.values, dtype='float')
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        len_dates = len(dates)
        k_factors = resid_ret.shape[1]
        cov_list = np.zeros((len_dates, k_factors))

        for i, d in enumerate(dates):
            print("calculating... current date: %s"%d.strftime('%Y%m%d'))
            date_ind = resid_ret.index.get_loc(d)
            cov_raw = calRiskByNewwey(ret_mat=resid_ret_arr[:date_ind],
                                      predict_window=kwargs['predict_window'],
                                      ac_window=kwargs['ac_window'],
                                      half_life=kwargs['half_life'],
                                      rollback_len=kwargs['rollback_len'])
            cov_list[i, :] = np.diag(cov_raw)
        cov_df = pd.DataFrame(cov_list, columns=resid_ret.columns,
                              index=dates)
        return cov_df

    @staticmethod
    def get_blending_coefficient(resid_ret, start_date, end_date, **kwargs):
        """计算blending coefficient"""
        resid_ret_arr = np.asarray(resid_ret.values, dtype='float')
        dates = tc.get_trade_days(start_date, end_date, retstr=None)
        date_ind = resid_ret.index.get_loc(dates[0])
        coeff = _calBlendingParam(resid_ret_arr, date_ind, kwargs['rollback_len'])
        gamma = pd.DataFrame(coeff, columns=resid_ret.columns,
                             index=dates)
        return gamma

    def structual_risk_model(self, start_date, end_date, **kwargs):
        """特异性结构化风险模型"""
        _dufault_st = max(tc.tradeDayOffset(start_date, -300), '20100101')
        dates = tc.get_trade_days(start_date, end_date, retstr=None)

        # 准备数据
        resid_ret = self.ds.load_resid_factor(start_date=_dufault_st,
                                              end_date=end_date)
        resid_ret = resid_ret.groupby('IDs').filter(
            lambda j: len(j.loc[:dates[0]]) > kwargs['rollback_len']+kwargs['ac_window'])
        resid_ret = resid_ret.iloc[:, 0].unstack().fillna(0.0)
        # indu_names = IndustryConverter.all_values(kwargs['行业因子'])
        style_factors = self.ds.load_style_factor(kwargs['风格因子'],
                                                  start_date=start_date,
                                                  end_date=end_date)
        indu_factors = data_source.sector.get_industry_dummy(ids=None,
                                                             start_date=start_date,
                                                             end_date=end_date,
                                                             industry=kwargs['行业因子'])
        # 权重是流通市值平方根
        weight = _getResgressWeight(dates).iloc[:, 0]
        # Newey-West covriance matrix
        sigma_ts = self.get_nwadj_spec_risk(resid_ret, start_date, end_date, **kwargs)
        sigma_ts = sigma_ts.stack()
        # moving average of abs(resid_ret)
        abs_resid_ma = np.abs(resid_ret).rolling(kwargs['rollback_len']).mean()
        abs_resid_ma = abs_resid_ma.stack().to_frame('abs_resid')
        # blending coefficients
        gamma = self.get_blending_coefficient(resid_ret, start_date, end_date, **kwargs).stack()
        stocks_regress = gamma[gamma == 1.0]
        # total cap of each industry
        indu_cap = _getIndustryMarketValue(indu_factors)

        indep = style_factors.join(abs_resid_ma, how='inner')
        indep = indep.groupby('date').transform(lambda z: z.fillna(z.mean()))
        indep = indep.join(indu_factors.fillna(0), how='inner')
        n_indu = indu_factors.shape[1]
        gamma = gamma.reindex(indep.index, fill_value=0.0)
        sigma_ts = sigma_ts.reindex(indep.index, fill_value=0.0)
        weight = weight.reindex(indep.index, fill_value=0.0)

        sigma = pd.Series(index=indep.index)
        for d in dates:
            stocks = stocks_regress.loc[[d]]
            w = weight.reindex(stocks.index).values
            w /= w.sum()
            y = np.log(sigma_ts.reindex(stocks.index).values)
            y *= w
            x = indep.reindex(stocks.index).values
            x = sm.add_constant(x)
            x *= w[:, None]
            cons = np.zeros(x.shape[1])
            cons[-n_indu:] = indu_cap.loc[d].values

            result = RLS(y, x, cons).fit()
            p = result.params
            e = np.nansum((y / (y-result.resid)) * w)

            tmp = indep.loc[d].values.dot(p[1:, None]) + p[0]
            sigma_str = np.exp(tmp.squeeze()) * e

            igamma = gamma.loc[d].values
            sigma_u = (igamma * sigma_ts.loc[d].values) + (1-igamma) * sigma_str
            sigma.loc[d] = sigma_u
        sigma = self.bayesian_shrink(sigma, **kwargs)
        return sigma
    
    def bayesian_shrink(self, sigma_series, **kwargs):
        """把特异性风险矩阵进行贝叶斯压缩

        把股票按照市值大小分成N组，用每组市值加权的特质风险对
        个股风险进行压缩。
        """
        n_groups = kwargs['n_groups']
        q = 0.1
        mkt_cap = data_source.load_factor('float_mkt_value',
                                          '/stocks/',
                                          idx=sigma_series)
        mkt_cap.fillna(0.0, inplace=True)
        # 按市值分组
        mkt_group = quantize_factor(mkt_cap, quantiles=n_groups)

        mkt_cap = mkt_cap.groupby(mkt_group).transform(lambda x: x/x.sum())
        sigma_avg = (sigma_series * mkt_cap).groupby('date').sum()
        sigma_demean = sigma_series.sub(sigma_avg, axis='index').abs()
        delta_sigma = (sigma_demean ** 2).group('date').mean()
        v = q * sigma_demean / ((q*sigma_demean).add(delta_sigma, axis='index'))

        sigma_adj1 = v.mul(sigma_avg, axis='index')
        sigma_adj2 = (1.0 - v).mul(sigma_series, axis='index')
        return sigma_adj2.add(sigma_adj1, axis='index')
    
    def save_specrsk_before_voladj(self, sigma):
        """保存波动率调整之前的特质风险"""
        name = 'specrisk_before_voladj'
        self._save_middle_df(sigma, name)
    
    def get_bias_stat_of_spec_risk(self, start_date, end_date):
        """计算特质风险截面偏差统计量"""













































