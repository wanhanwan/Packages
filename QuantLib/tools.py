import pandas as pd
import numpy as np


def df_rolling2(df, window, apply_func, raw=False, *args, **kwargs):
    if isinstance(df, pd.DataFrame):
        arr = df.to_numpy()
    elif isinstance(df, np.ndarray):
        arr = df
    else:
        arr = np.asarray(df, dtype='float')
    temp = np.moveaxis(np_rolling_window(arr.T, window),0,-1)
    rslt = np.asarray([apply_func(x, *args, **kwargs) for x in temp], dtype='float')
    if raw:
        return rslt
    if rslt.ndim == 1:
        return pd.Series(rslt, index=df.index[-len(rslt):])
    return pd.DataFrame(rslt, index=df.index[-len(rslt):])


def np_rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


class RollingResultWrapper(object):
    def __init__(self, func):
        self.rollingfunc = func
        self.rlist = []

    def reset(self):
        self.rlist = []

    def __call__(self, *args, **kwargs):
        """pandas.DataFrame.rolling Wrapper
        当func返回多个值的时候，自动返回一个DataFrame
        """
        rslt = self.rollingfunc(*args, **kwargs)
        self.rlist.append(rslt)
        return 1

    @property
    def df(self):
        return pd.DataFrame(self.rlist)


def expweighted(half_window, arr_len=None, arr=None, scale=False):
    """指数衰减权重序列，升序排列"""
    if arr is None:
        arr = np.ones(arr_len)
    w = 0.5 ** (np.arange(len(arr)) / half_window)[::-1]
    # w = w[::-1]
    if scale:
        w /= w.sum()
    if arr is None:
        return w
    return arr * w


def return2nav(ret_data, start_point=None):
    """收益转净值"""
    nav = (1.0 + ret_data).cumprod()
    if start_point is None:
        start_point = ret_data.index.min() - pd.Timedelta(1, unit='D')
    if isinstance(ret_data, pd.Series):
        nav.loc[start_point] = 1.0
    else:
        nav.loc[start_point, :] = 1.0
    nav.sort_index(inplace=True)
    nav.fillna(method='ffill', inplace=True)
    return nav


def calc_cov(mat1, mat2=None, weight=None, use_ewavg=True):
    """计算加权协方差矩阵
    Parameters:
    ----------
    mat1: 2D-Array
        每一列是一个列向量
    mat2: 2D-Array
        同上
    weight: 1D-Array
        权重向量
    use_ewavg: bool
        是否使用加权历史均值计算协方差
    """
    if mat2 is None:
        mat2 = mat1
    if weight is None or not use_ewavg:
        weight = np.ones(mat1.shape[0], dtype='float64') / mat1.shape[0]
    mat1_demean = mat1 - np.average(mat1, axis=0, weights=weight)[None, :]
    mat2_demean = mat2 - np.average(mat2, axis=0, weights=weight)[None, :]
    mat2_demean *= weight[:, None]
    m = np.dot(mat1_demean.T, mat2_demean)
    return m


def calc_corr(mat1, mat2=None, weight=None, use_ewavg=True):
    """计算加权相关系数"""
    cov = calc_cov(mat1, mat2, weight, use_ewavg)
    std = np.sqrt(np.diag(cov))
    corr = cov / std[:, None] / std[None, :]
    return corr


def get_percentile_at(arr, values=None, side='right', value_type=1):
    """value在数组arr中的分位数(升序)
    如果value是空，默认取arr的最后一个元素.
    遇到重复值，尽量靠右排列(返回值尽可能大)

    Parameters:
    ===========
    arr: array or Series
    value: float number or array, default None.
    side: 'left' or 'right'(default)
        如果遇到重复值，'left'会使返回值尽可能小；'right'反之。
    value_type: int 1 or 2
        计算分位数的方法。1代表先把数组排序，计算当前数值在数组中的分位数；2
        代表先计算数组中的最大值和最小值，计算当前数值在最大值和最小值这个区间
        中的分位数。
    """
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    if values is None:
        values = arr[-1]
        arr = arr[:-1]
    if isinstance(values, float) and np.isnan(values):
        return np.nan
    arr = np.sort(arr)
    if value_type == 1:
        if isinstance(values, float):
            return np.searchsorted(arr, values, side='right')/np.isfinite(arr).sum()
        pos = np.zeros_like(values) * np.nan
        idx = np.isfinite(values)
        pos[idx] = np.searchsorted(arr, values[idx], side='right')
        return pos / np.isfinite(arr).sum()
    elif value_type == 2:
        pos = (values - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
        pos = np.where(pos < 0, 0, pos)
        pos = np.where(pos > 1, 1, pos)
        return pos

import statsmodels.api as sm
def forward_regression(X, y,
                       threshold_in,
                       verbose=False):
    """向前回归"""
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]])),
                        missing='drop').fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(X, y, threshold_out, verbose=False):
    """
    向后逐步回归
    每次回归剔除P值最大的变量，直到所有变量的P值均小于阈值。
    """
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included])), missing='drop').fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


from empyrical.stats import sharpe_ratio
from itertools import combinations
from scipy.stats import rankdata
def compute_pbo(ret_df, S, performance_func=sharpe_ratio):
    """
    计算过拟合概率
    """
    ret_values = ret_df.dropna().to_numpy()
    ret_values = np.array_split(ret_values, S)
    N = ret_df.shape[1]

    Cs = list(combinations(range(S), S // 2))
    all_chunks = set(range(S))
    df_result = pd.DataFrame(index=list(range(len(Cs))),
                             columns=['w', 'metric_in', 'metric_out'])
    for i, chunk in enumerate(Cs):
        in_sample = np.concatenate([ret_values[j] for j in chunk], axis=0)
        out_sample = np.concatenate([ret_values[j] for j in (all_chunks - set(chunk))],
                                    axis=0)

        # compute performance metric
        performance_in = performance_func(in_sample)
        performance_out = performance_func(out_sample)

        # rank
        idx_best = performance_in.argmax()
        rank = rankdata(performance_out)[idx_best]
        w = 1.0 - rank / N
        df_result.iloc[i,:] = [w, performance_in[idx_best], performance_out[idx_best]]

    return (df_result['w'] > 0.5).sum()/len(df_result), df_result


def compute_pbo_sampled_randomly(ret_df, S, N, K, performance_func=sharpe_ratio):
    """
    随机抽选样本，计算过拟合概率

    Parameters:
    -----------
    S: int
        将样本按时间序列分成S份。
    N：int
        每次抽样从S中抽N份作为样本内和样本外。
    K：int
        重复实验次数。
    """
    ret_values = ret_df.dropna().to_numpy()
    ret_values = np.array_split(ret_values, S)
    N = ret_df.shape[1]

    all_chunks = set(range(S))
    Cs = [np.random.choice(list(range(S)), N) for x in range(K)]
    Cs2 = [np.random.choice(list(all_chunks-set(x)), N) for x in Cs]
    df_result = pd.DataFrame(index=list(range(len(Cs))),
                             columns = ['w', 'metric_in', 'metric_out'])
    for i, chunk in enumerate(Cs):
        in_sample = np.concatenate([ret_values[j] for j in chunk], axis=0)
        out_sample = np.concatenate([ret_values[j] for j in Cs2[i]], axis=0)

        # compute performance metric
        performance_in = performance_func(in_sample)
        performance_out = performance_func(out_sample)

        # rank
        idx_best = performance_in.argmax()
        rank = rankdata(performance_out)[idx_best]
        w = 1.0 - rank / N
        df_result.iloc[i,:] = [w, performance_in[idx_best], performance_out[idx_best]]
    
    return (df_result['w'] > 0.5).sum() / len(df_result), df_result
