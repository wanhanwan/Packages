# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, zscore


def __DropOutlierFixedRate__(data, **kwargs):
    """以固定比率截断一部分异常值

    参数
    ----------
    data: DataFrame
          [index:IDs,Factor]
    """

    data = data[kwargs['factor_name']]
    fixedRatio = kwargs['drop_ratio']
    quantileMax = data.quantile(1-fixedRatio)
    quantileMin = data.quantile(fixedRatio)

    if kwargs['drop_mode'] == '截断':
        after_dropOutlier = data.apply(lambda x: min(x, quantileMax))
        after_dropOutlier = after_dropOutlier.apply(
            lambda x: max(x, quantileMin))
    elif kwargs['drop_mode'] == '剔除':
        after_dropOutlier = data.apply(
            lambda x: np.nan if not (quantileMin < x < quantileMax) else x)
    after_dropOutlier.rename(data.name+'_after_drop_outlier', inplace=True)

    return pd.concat([data, after_dropOutlier], axis=1)


def __DropOutlierMeanVariance__(data, **kwargs):
    """以均值方差方式截断部分异常值

    参数
    ----------
    data: DataFrame
          [index:IDs,Factor]
    """

    data = data[kwargs['factor_name']]
    mean = data.mean()
    std = data.std()
    quantileMax = mean + kwargs['alpha'] * std
    quantileMin = mean - kwargs['alpha'] * std

    if kwargs['drop_mode'] == '截断':
        after_dropOutlier = data.apply(lambda x: min(x, quantileMax))
        after_dropOutlier = after_dropOutlier.apply(
            lambda x: max(x, quantileMin))
    elif kwargs['drop_mode'] == '剔除':
        after_dropOutlier = data.apply(
            lambda x: np.nan if not (quantileMin < x < quantileMax) else x)
    after_dropOutlier.rename(data.name+'_after_drop_outlier', inplace=True)

    return pd.concat([data, after_dropOutlier], axis=1)

def __DropOutlierBarra__(data, **kwargs):
    """
    以barra方式去掉极端值，使用该方法之前需要先把数据标准化
    :param data: 标准化之后的因子值
    :param kwargs:
    :return: new_data
    """
    data = data[kwargs['factor_name']]
    s_plus = max(0.0, min(1.0, 0.5 / (data.max() - 3)))
    s_minus = max(0.0, min(1.0, 0.5 / (-3 - data.min())))

    def func(x):

        if x > 3.:
            return 3. * (1. - s_plus) + x * s_plus
        elif x < -3.:
            return -3. * (1. - s_minus) + x * s_minus
        else:
            return x
    after_dropOutlier = data.apply(func).rename(data.name+'_after_drop_outlier')
    return pd.concat([data, after_dropOutlier], axis=1)


def __DropOutlierMAD__(data, **kwargs):
    """以MAD方法剔除异常值"""

    data = data[kwargs['factor_name']]
    median = data.median()
    MAD = (data - median).abs().median()
    quantileMax = median + kwargs['alpha'] * MAD
    quantileMin = median - kwargs['alpha'] * MAD

    if kwargs['drop_mode'] == '截断':
        after_dropOutlier = data.apply(lambda x: min(x, quantileMax))
        after_dropOutlier = after_dropOutlier.apply(
            lambda x: max(x, quantileMin))
    elif kwargs['drop_mode'] == '剔除':
        after_dropOutlier = data.apply(
            lambda x: np.nan if not (quantileMin < x < quantileMax) else x)
    after_dropOutlier.rename(data.name+'_after_drop_outlier', inplace=True)

    return pd.concat([data, after_dropOutlier], axis=1)


def __DropOutlierBoxPlot__(data, **kwargs):
    """以BoxPlot方法剔除异常值"""

    data = data[kwargs['factor_name']]
    Q1 = data.dropna().quantile(0.25)
    Q3 = data.dropna().quantile(0.75)
    IQR = Q3 - Q1

    md = data.median()
    xi = data[data > md].values
    xj = data[data < md].values

    mc = np.nanmedian(
        [((x0-md)-(md-x1))/(x0-x1) for x0 in xi for x1 in xj])
    if mc >= 0:
        quantileMax = Q3 + 1.5 * np.exp(4 * mc) * IQR
        quantileMin = Q1 - 1.5 * np.exp(-3.5 * mc) * IQR
    else:
        quantileMin = Q1 - 1.5 * np.exp(-4 * mc) * IQR
        quantileMax = Q3 + 1.5 * np.exp(3.5 * mc) * IQR

    if kwargs['drop_mode'] == '截断':
        after_dropOutlier = data.apply(lambda x: min(x, quantileMax))
        after_dropOutlier = after_dropOutlier.apply(
            lambda x: max(x, quantileMin))
    elif kwargs['drop_mode'] == '剔除':
        after_dropOutlier = data.apply(
            lambda x: np.nan if not (quantileMin < x < quantileMax) else x)
    after_dropOutlier.rename(data.name+'_after_drop_outlier', inplace=True)

    return pd.concat([data, after_dropOutlier], axis=1)


def DropOutlier(data, factor_name=None, method='FixedRatio',
                drop_ratio=0.1, drop_mode='截断', alpha=3,
                ret_raw=True, squeeze=False, **kwargs):
    """ 处理异常值函数

    参数
    ------------
    data: DataFrame
          [index:[date, IDs],Factor1,Factor2...]
    factor_name: str
          which column to be used
    method: str
          drop method:{FixedRatio、Mean-Variance、MAD、BoxPlot}
    drop_mode: str
          截断 or 剔除
    alpha: int
          alpha倍标准差之外的值视为异常值

    输出
    -------------
    DataFrame:[index:[date,IDs],factor1,factor2,...]
    """
    if isinstance(data, pd.Series):
        factor_name = data.name
        tempData = data.to_frame()
    else:
        assert factor_name is not None
        if {'date', 'IDs'}.issubset(data.index.names):
            tempData = data[[factor_name]]
        else:
            tempData = data[['date', 'IDs', factor_name]].set_index(['date', 'IDs'])

    dropFuncs = {'FixedRatio': __DropOutlierFixedRate__,
                 'Mean-Variance': __DropOutlierMeanVariance__,
                 'MAD': __DropOutlierMAD__,
                 'BoxPlot': __DropOutlierBoxPlot__,
                 'Barra':__DropOutlierBarra__}

    params = {'drop_ratio': drop_ratio, 'drop_mode': drop_mode, 'alpha': alpha,
              'factor_name': factor_name}

    afterDropOutlier = tempData.groupby(
        level='date').apply(dropFuncs[method], **params)

    if ret_raw:
        return afterDropOutlier
    elif squeeze:
        return afterDropOutlier[factor_name+'_after_drop_outlier']
    else:
        return afterDropOutlier[[factor_name+'_after_drop_outlier']]


def __StandardFun__(data0, **kwargs):
    """横截面标准化函数"""
    data_to_standard = data0.reset_index().set_index(['IDs'])[[kwargs['factor_name']]]
    IDNums = len(data_to_standard)
    if kwargs['mean_weight'] is not None:
        avgWeight = data0.reset_index().set_index(['IDs'])[kwargs['mean_weight']]
    else:
        avgWeight = pd.Series(
            np.ones(IDNums)/IDNums, index=data_to_standard.index)
    avgWeightInd = pd.notnull(avgWeight)
    if kwargs['std_weight'] is not None:
        stdWeight = data0.reset_index().set_index(['IDs'])[kwargs['std_weight']]
    else:
        stdWeight = pd.Series(
            np.ones(IDNums)/IDNums, index=data_to_standard.index)
    stdWeightInd = pd.notnull(stdWeight)

    # 计算横截面均值
    data_to_standardInd = pd.notnull(data_to_standard[kwargs['factor_name']])
    tempInd = data_to_standardInd & avgWeightInd
    totalWeight = avgWeight.ix[tempInd].sum()
    if totalWeight != 0:
        avg = (data_to_standard[kwargs['factor_name']] * avgWeight).sum() / totalWeight
    else:
        data_to_standard[kwargs['factor_name']+'_after_standard'] = np.nan
        return data_to_standard
    # 计算截面标准差
    tempInd = data_to_standardInd & stdWeightInd
    totalWeight = stdWeight[tempInd].sum()
    if totalWeight != 0:
        factor = data_to_standard[kwargs['factor_name']]
        std = np.sqrt(
            ((factor-factor.mean())**2*stdWeight/totalWeight).sum())
    else:
        data_to_standard[kwargs['factor_name']+'_after_standard'] = np.nan
        return data_to_standard
    if std != 0:
        data_to_standard[kwargs['factor_name'] +
              '_after_standard'] = (data_to_standard[kwargs['factor_name']] - avg) / std
    else:
        data_to_standard[kwargs['factor_name']+'_after_standard'] = 0
    return data_to_standard

def __StandardFun2__(data0, **kwargs):
    """横截面标准化函数"""
    IDNums = data0.shape[0]
    if kwargs['mean_weight'] is not None:
        avgWeight = data0[kwargs['mean_weight']].to_numpy()
    else:
        avgWeight = np.ones(IDNums, dtype='float32') / IDNums
    avgWeightInd = np.isfinite(avgWeight)
    if kwargs['std_weight'] is not None:
        stdWeight = data0[kwargs['std_weight']].to_numpy()
    else:
        stdWeight = np.ones(IDNums, dtype='float32') / IDNums
    stdWeightInd = np.isfinite(stdWeight)

    # 计算横截面均值
    factor_values = data0[kwargs['factor_name']].to_numpy()
    data_to_standardInd = np.isfinite(factor_values)
    tempInd = np.logical_and(data_to_standardInd, avgWeightInd)
    totalWeight = avgWeight[tempInd].sum()
    if totalWeight != 0:
        avg = (factor_values[tempInd] * avgWeight[tempInd]).sum() / totalWeight
    else:
        df = data0.copy()
        df[kwargs['factor_name']+'_after_standard'] = np.nan
        return df
    # 计算截面标准差
    tempInd = np.logical_and(data_to_standardInd, stdWeightInd)
    totalWeight = stdWeight[tempInd].sum()
    if totalWeight != 0:
        std = np.sqrt(
            ((factor_values[tempInd]-factor_values[tempInd].mean())**2*stdWeight[tempInd]/totalWeight).sum())
    else:
        df = data0.copy()
        df[kwargs['factor_name']+'_after_standard'] = np.nan
        return df
    if std != 0:
        factor_std = np.concatenate((factor_values[:,None], ((factor_values-avg)/std)[:,None]), axis=1)
        data_to_standard = pd.DataFrame(factor_std, columns=[kwargs['factor_name'], kwargs['factor_name']+'_after_standard'],
                                        index=data0.index)
    else:
        df = data0.copy()
        df[kwargs['factor_name']+'_after_standard'] = 0.0
        return df
    return data_to_standard


def __StandardQTFun2__(data):
    # data0 = data.reset_index(level=0, drop=True)
    data_after_standard = np.ones_like(data.values) * np.nan
    NotNAN = ~np.isnan(data.values)
    quantile = rankdata(data.values[NotNAN], method='min').astype('float64') / (NotNAN.sum())
    quantile[quantile == 1] = 1 - 10 ** (-6)
    data_after_standard[NotNAN] = norm.ppf(quantile)
    return pd.Series(data_after_standard, index=data.index.get_level_values(1), name=data.name)


def __StandardQTFun__(data):
    # data0 = data.reset_index(level=0, drop=True)
    data_after_standard = np.ones_like(data.values) * np.nan
    values = data.values
    NotNAN = ~np.isnan(data.values)
    unique_values = np.unique(values[NotNAN])
    quantile = np.linspace(10**(-6), 1-10**(-6), len(unique_values), dtype='float32')
    quantile_indices = np.searchsorted(np.sort(unique_values), values[NotNAN])
    quantile_values = quantile[quantile_indices]
    data_after_standard[NotNAN] = zscore(norm.ppf(quantile_values))
    return pd.Series(data_after_standard, index=data.index.get_level_values(1), name=data.name)


def Standard(data, factor_name=None, mean_weight=None, std_weight=None, ret_raw=True,
             squeeze=False, **kwargs):
    """横截面上标准化数据

    参数
    ----------
    data: DataFrame
          Index is [date, IDs]
    factor_name: str
          which column to be used
    mean_weight: str or None
          如果None，均值权重为等权
    std_weight: str or None
          如果None， 标准差权重设为等权
    ret_raw: bool
        返回数据是是否把原始数据也一并返回

    输出
    -----------
    DataFrame:[index:[date,IDs],factor1,factor2,...]
    """
    factor_name_list = [factor_name]
    if mean_weight is not None:
        factor_name_list += [mean_weight]
    if std_weight is not None:
        factor_name_list += [std_weight]
    if isinstance(data, pd.Series):
        factor_name = data.name
        tempData = data.to_frame()
    else:
        tempData = data[factor_name_list]
    params = {'factor_name': factor_name,
              'mean_weight': mean_weight, 'std_weight': std_weight}
    afterStandard = tempData.groupby(level='date').apply(__StandardFun2__, **params)
    if ret_raw:
        return afterStandard
    elif squeeze:
        return afterStandard[factor_name+'_after_standard']
    else:
        return afterStandard[[factor_name+'_after_standard']]


def StandardByQT(data, factor_name=None, groups=('date',), squeeze=False):
    """横截面上分位数标准化

    参数:
    ----------------------
    data: DataFrame
    [index:date,IDs,data:factor1,factor2,...]

    factor_name: str
    """
    if isinstance(data, pd.Series):
        after_standard = data.groupby(list(groups)).transform(__StandardQTFun__)
    else:
        after_standard = data.groupby(list(groups))[factor_name].transform(__StandardQTFun__)
    if squeeze:
        return after_standard
    return after_standard.to_frame()


def _resid_ols(y, x):
    fit_model = sm.OLS(y, x, missing='drop').fit()
    return fit_model.resid


def OLS(df, no_raise=True):
    v = df.to_numpy()
    try:
        resid = _resid_ols(v[:, 0], v[:, 1:])
    except Exception as e:
        if no_raise:
            print(e)
            return pd.DataFrame(np.ones((len(df), 1))*np.nan, index=df.index,
                                columns=['resid'])
        else:
            raise e
    return pd.DataFrame(resid, index=df.index, columns=['resid'])


def Orthogonalize(left_data, right_data, left_name, right_name, add_const=False,
                  new_name=None):
    """因子正交化
       因子数据的格式为：[index(date,IDs),factor1,factor2...]
    参数
    --------
    left_name: str
        因子1对应的列名
    right_name: str or list
        因子2对应的列名，有多个因子时可以使用列表
    industry: str
        是否加入行业哑变量，None表示不加入行业因子
    """

    factor_1 = left_data[[left_name]].dropna().reset_index()
    factor_1['IDs'] = factor_1['IDs'].astype('category', copy=False)
    if not isinstance(right_name, list):
        right_name = [right_name]
    if add_const:
        right_name.append('alpha')
        right_data['alpha'] = 1.0  # 加入常数项
    factor_2 = right_data[right_name].dropna().reset_index()
    factor_2['IDs'] = factor_2['IDs'].astype('category')
    factor = pd.merge(factor_1, factor_2, on=['date', 'IDs'], how='inner')
    new_factor = factor.groupby('date')[[left_name]+right_name].apply(OLS)
    new_factor.index = pd.MultiIndex.from_arrays([factor['date'].values, factor['IDs'].values], names=['date', 'IDs'])
    if new_name is not None:
        new_factor.rename(columns={'resid': new_name}, inplace=True)
    return new_factor

def TransformBySymmetricOrthogonalize(df):
    """因子进行对称正交
    因子的缺失值由横截面的均值填充
    """
    def duichen(sub_df):
        mat = sub_df.to_numpy()
        mean = np.nanmean(mat, axis=0)
        ind = np.where(np.isnan(mat))
        mat[ind] = np.take(mean, ind[1])
        n = mat.shape[0]
        m = np.cov(mat, rowvar=False, bias=False) * (n-1)
        v, u = np.linalg.eig(m)
        d = np.diag(1.0 / np.sqrt(v))
        m_sqrt = u.dot(d).dot(u.T)
        return pd.DataFrame(mat.dot(m_sqrt),index=sub_df.index,columns=sub_df.columns)
    df2 = df.groupby('date').apply(duichen)
    return df2

def Fillna_Barra(factor_data, factor_names, ref_name, classify_name):
    """
    Barra缺失值填充， 具体方法如下：

        1. 按照分类因子(classify_name)将股票分类，分类因子通常是一级行业因子
        2. 在每一类股票里使用参考变量(ref_name)对待填充因子进行回归，缺失值被回归
           拟合值替代。参考变量通常是流通市值。

    :param factor_data: pandas.dataframe

    :param factor_names: list
        带填充因子名称。因子名称必须是factor_data中列的子集
    :param ref_name: 1元素的list
        参考变量，通常是流通市值,必须是factor_data中列的子集
    :param classify_name: 1元素的list
        分类变量，通常是中信一级行业，必须是factor_data中列的子集
    :return: pandas.dataframe

    """
    factor_tofill = factor_data[factor_names].copy()
    class_factor = factor_data[classify_name]
    ref_factor = np.log(factor_data[ref_name])
    all_dates = factor_tofill.index.get_level_values(0).unique()
    for idate in all_dates:
        iclass_data = class_factor.loc[idate, classify_name[0]]
        iref_data = ref_factor.loc[idate, ref_name[0]]
        for ifactor in factor_names:
            ifactor_data = factor_tofill.loc[idate, ifactor]
            not_nan_idx = pd.notnull(ifactor_data)
            not_nan_idx_sum = not_nan_idx.sum()
            if not not_nan_idx.all():   # 存在缺失值
                for ijclass in iclass_data[~not_nan_idx].unique():
                    ij_not_na = not_nan_idx & (iclass_data == ijclass)
                    ij_na = (~not_nan_idx) & (iclass_data == ijclass)
                    x = iref_data[ij_not_na]
                    y = ifactor_data[ij_not_na]
                    x_mean = x.mean()
                    y_mean = y.mean()
                    beta = ((x*y).sum()-not_nan_idx_sum*x_mean*y_mean)/((x**2).sum()-not_nan_idx_sum*x_mean**2)
                    alpha = y_mean - x_mean*beta
                    ifactor_data.loc[ij_na] = alpha+beta*iref_data[ij_na]
    return factor_tofill


def FillnaByMeanOrQuantile(factor_data, factor_names, ref_names=None, industry=None, fill_value='mean',
                           drop_classification=True):
    """以均值或者分位数填充缺失值"""
    def fill_na(df, fill_df):
        idx = df.index.droplevel('IDs')[0]
        df = df.fillna(fill_df.loc[idx])
        return df

    if factor_names is None:
        factor_names = factor_data.columns
    data_len = len(factor_data)
    groups = ['date']
    if ref_names is not None:
        factor_data = factor_data.dropna(subset=ref_names)
        groups += ref_names
    if industry is not None:
        from FactorLib.data_source.base_data_source_h5 import sec
        from FactorLib.utils.tool_funcs import dummy2name
        indu_flag = sec.get_industry_dummy(industry=industry,
                                           idx=factor_data,
                                           drop_first=False)
        indu_group = dummy2name(indu_flag).dropna()
        factor_data = factor_data.join(indu_group.to_frame(industry), how='inner')
        groups.append(industry)
    factor_data.set_index([x for x in groups if x != 'date'], inplace=True, append=True)
    if fill_value == 'mean':
        fills = factor_data.groupby(groups)[factor_names].mean()
    elif isinstance(fill_value, float):
        fills = factor_data.groupby(groups)[factor_names].quantile(fill_value)
    else:
        raise NotImplementedError("fill_value must be 'mean' or a float number.")
    df = factor_data.groupby(groups)[factor_names].apply(fill_na, fill_df=fills)
    if drop_classification:
        df.reset_index([x for x in groups if x!='date'], drop=True, inplace=True)
    loss = (data_len - len(factor_data)) / data_len
    print("data loss: %.2f"%loss)
    return df


def Join_Factors(*factor_data, merge_names=None, new_name=None, weight=None, style='SAST'):
    """合并因子,按照权重进行加总。只将非缺失的因子的权重重新归一合成。

    Parameters:
    ===========
    factor_data: dataframe or tuple of dataframes
    merge_names: list
        待合并因子名称，必须是data_frame中列的子集
    new_name: str
        合成因子名称
    weight: list or None
        待合并因子的权重
    style : str, 'SAST" or 'AST'
        字段、品种、时间三个维度在factor_data中的排布类型。SAST(Stack Attribute-Symbol-Time)是最常用的，
        索引是Time-Symbol的MultiIndex,列是字段;AST(Attribute-Symbol-Time),Index是时间，Columns是Symbol.
    """

    def nansum(a, w):
        nanind = np.isfinite(a)
        return np.sum(a[nanind] * w[nanind]) / np.sum(w[nanind])

    if new_name is None:
        new_name = 'new'
    if isinstance(merge_names, str):
        merge_names = [merge_names]
    if len(factor_data) == 1:
        if merge_names is None:
            factor_values = factor_data[0].values
        else:
            factor_values = factor_data[0][merge_names].values
    elif style == 'SAST':
        factor_data = align_dataframes(*factor_data)
        factor_values = np.hstack((x.values for x in factor_data))
    else:
        factor_data = align_dataframes(*factor_data, axis='both')
        factor_values = np.stack((x.values for x in factor_data))
    nfactors = factor_values.shape[1] if factor_values.ndim == 2 else factor_values.shape[0]
    if weight is None:
        weight = np.asarray([1.0 / nfactors] * nfactors)
    else:
        weight = np.asarray(weight) / np.sum(weight)

    if factor_values.ndim == 2:
        weight_array = np.tile(weight, (factor_values.shape[0],1))
        na_ind = np.isnan(factor_values)
        weight_array[na_ind] = 0.0
        weight_array = weight_array / weight_array.sum(axis=1)[:, np.newaxis]
        new_values = np.nansum(factor_values * weight_array, axis=1)
        return pd.DataFrame(new_values, index=factor_data[0].index, columns=[new_name])
    else:
        new_values = np.apply_along_axis(nansum, 0, factor_values, w=weight)
        return pd.DataFrame(new_values, index=factor_data[0].index, columns=factor_data[0].columns)


def union_axis(*dfs, axis='index'):
    """DataFrames中某个轴的并集"""
    from functools import reduce
    if isinstance(dfs[0], list):
        dfs = dfs[0]
    if axis == 'index':
        axis_list = [df.index for df in dfs]
    else:
        axis_list = [df.columns for df in dfs]
    new_axis = reduce(lambda x, y: x.union(y), axis_list).sort_values()
    return new_axis


def intersection_axis(*dfs, axis='index'):
    """DataFrames中某个轴的交集"""
    from functools import reduce
    if isinstance(dfs[0], list):
        dfs = dfs[0]
    if axis == 'index':
        axis_list = [df.index for df in dfs]
    else:
        axis_list = [df.columns for df in dfs]
    new_axis = reduce(lambda x, y: x.intersection(y), axis_list).sort_values()
    return new_axis


def first_axis(*dfs, axis='index'):
    """DataFrames中第一个元素的轴"""
    if axis == 'index':
        return dfs[0].index
    else:
        return dfs[0].columns


def align_dataframes(*dfs, axis='index', join='outer'):
    if join == 'outer':
        func = union_axis
    elif join == 'inner':
        func = intersection_axis
    elif join == 'first':
        func = first_axis
    else:
        raise NotImplementedError("join method not implemented.")
    index = None
    columns = None
    if axis == 'index':
        index = func(*dfs, axis='index')
    elif axis == 'columns':
        columns = func(*dfs, axis='columns')
    elif axis == 'both':
        index = func(*dfs, axis='index')
        columns = func(*dfs, axis='columns')
    else:
        raise ValueError("axis must be index, columns or both.")
    r = [None] * len(dfs)
    for i, df in enumerate(dfs):
        r[i] = df.reindex(index=index, columns=columns)
    return tuple(r)


def merge_dataframes(*dfs, join='outer'):
    """横向合并多个DataFrame"""
    from functools import reduce
    dfs = align_dataframes(*dfs, join=join)
    values = np.concatenate(tuple((x.to_numpy() for x in dfs)), axis=1)
    columns = reduce(lambda x, y: list(x)+list(y), [x.columns for x in dfs])
    return pd.DataFrame(values, index=dfs[0].index, columns=columns)


def NeutralizeBySizeIndu(factor_data, factor_name, std_qt=True, indu_name='中信一级',
                         drop_first_indu=True, new_name='resid', **kwargs):
    """ 为因子进行市值和行业中性化

    市值采用对数市值(百万为单位);
    
    默认不加入常数项；

    Paramters:
    ==========
    factor_data: pd.DataFrame(index:[date, IDs])
        因子数值
    factor_name: str
        因子名称
    std_qt: bool
        在中性化之前是否进行分位数标准化，默认为True
    indu_name: str
        行业选取
    add_constant: bool
        OLS时时候在自变量中加入常数项，默认为False.
    """
    from FactorLib.data_source.base_data_source_h5 import h5_2, sec
    lncap = np.log(h5_2.load_factor2(
        'float_mkt_value', '/base/', idx=factor_data.dropna(), stack=True)/10000.0
                   ).rename(columns={'float_mkt_value': 'lncap'})
    indu_flag = sec.get_industry_dummy(industry=indu_name,
                                       idx=factor_data.dropna(),
                                       drop_first=drop_first_indu)
    # indu_flag = indu_flag[(indu_flag==1).any(axis=1)]

    if std_qt:
        factor_data = StandardByQT(factor_data, factor_name)
        lncap = StandardByQT(lncap, 'lncap')
    industry_names = list(indu_flag.columns)
    indep_data = lncap.join(indu_flag, how='inner')
    resid = Orthogonalize(factor_data, indep_data, factor_name, industry_names+['lncap'], **kwargs)
    resid.columns = [new_name]
    return resid.reindex(factor_data.index)


def CalFactorCorr(*factor_data, factor_names=None, dates=None, ids=None, idx=None,
                  style='SAST', method='spearman'):
    """计算因子之间的截面相关系数矩阵
    返回一个DataFrame,每个时间节点的相关系数矩阵。

    Parameters:
    ===========
    factor_data: dataframe or tuple of dataframes
        因子数据
    factor_names: list
        如果factor_data是dataframe, 指定某几列数据作为输入。
    dates: list, yyyymmdd like, str or datetime
        计算哪些日期计算相关系数
    ids: list
        使用哪些股票计算相关系数
    idx: dataframe
        将数据直接索引到给定dataframe的Index上，Index是一个二维索引
    style: str SAST or AST
        如果facor是一个tuple,指定tuple中每一个元素是哪种数据结构。
        SAST(Stacked Attribute-Symbol-Time)，是最常用的，索引为
        日期和时间的二维索引;AST(Attribute-Symbol-Time),索引是日期，
        列是股票代码。
    method: str, spearman or pearson
        相关系数的计算方式.spearman指代秩相关系数，pearson是数值相关系数。
    """
    if style == 'SAST':
        if dates is not None:
            factor_data = (x.loc[dates] for x in factor_data)
        if ids is not None:
            factor_data = (x.loc[pd.IndexSlice[:, ids]] for x in factor_data)
        if idx is not None:
            factor_data = (x.reindex(idx.index) for x in factor_data)
    if style == 'AST':
        factor_data = (x.stack().to_frame(x.name) for x in factor_data)
        if dates is not None:
            factor_data = (x.loc[dates] for x in factor_data)
        if ids is not None:
            factor_data = (x.loc[pd.IndexSlice[:, ids]] for x in factor_data)
        if idx is not None:
            factor_data = (x.reindex(idx.index) for x in factor_data)
    factor_data = tuple(factor_data)
    if len(factor_data) == 1:
        if factor_names is not None:
            factor_data = factor_data[factor_names]
    else:
        factor_data = merge_dataframes(*factor_data, join='first')
    factor_data.index.names = ['date', 'IDs']
    corr = factor_data.groupby('date').corr(method=method)
    return corr


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import h5_2, tc
    dates = tc.get_trade_days('20180101', '20180228', freq='1m')
    value = h5_2.load_factor2('StyleFactor_ValueFactor', '/XYData/StyleFactor/', dates=dates, stack=True)
    growth = h5_2.load_factor2('AsnessQualityWithoutGrowth', '/quality/', dates=dates, stack=True)
    data = pd.concat([value, growth], axis=1)
    fillna = FillnaByMeanOrQuantile(data, ['AsnessQualityWithoutGrowth'], ref_names=['StyleFactor_ValueFactor'])
    print(fillna)
