# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


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


def DropOutlier(data, factor_name, method='FixedRatio',
                drop_ratio=0.1, drop_mode='截断', alpha=3, **kwargs):
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
    if ('date' in data.index.names) and ('IDs' in data.index.names):
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
        level=0).apply(dropFuncs[method], **params)

    return afterDropOutlier


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


def __StandardQTFun__(data):
    # data0 = data.reset_index(level=0, drop=True)
    NotNAN = pd.notnull(data)
    quantile = data.rank(method='min') / (NotNAN.sum() + 1)
    # quantile.loc[quantile[data.columns[0]] == 1, :] = 1 - 10 ** (-6)
    data_after_standard = norm.ppf(quantile)
    return pd.DataFrame(data_after_standard, index=data.index.get_level_values(1), columns=data.columns)


def Standard(data, factor_name, mean_weight=None, std_weight=None, **kwargs):
    """横截面上标准化数据

    参数
    ----------
    data: DataFrame
          [IDs,date,Factor1,Factor2...]
    factor_name: str
          which column to be used
    mean_weight: str or None
          如果None，均值权重为等权
    std_weight: str or None
          如果None， 标准差权重设为等权

    输出
    -----------
    DataFrame:[index:[date,IDs],factor1,factor2,...]
    """
    factor_name_list = [factor_name]
    if mean_weight is not None:
        factor_name_list += [mean_weight]
    if std_weight is not None:
        factor_name_list += [std_weight]
    tempData = data[
        ['date', 'IDs']+factor_name_list].set_index(['date', 'IDs'])
    params = {'factor_name': factor_name,
              'mean_weight': mean_weight, 'std_weight': std_weight}
    afterStandard = tempData.groupby(level=0).apply(__StandardFun__, **params)
    return afterStandard


def StandardByQT(data, factor_name):
    """横截面上分位数标准化

    参数:
    ----------------------
    data: DataFrame
    [index:date,IDs,data:factor1,factor2,...]

    factor_name: str
    """
    factor_name = [factor_name]
    after_standard = data[factor_name].groupby(level=0).apply(__StandardQTFun__)
    return after_standard


def Orthogonalize(left_data, right_data, left_name, right_name):
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

    def OLS(data, left_name):
        tempData = data.copy()
        yData = np.array(tempData.pop(left_name))
        xData = np.array(tempData)
        NaNInd = pd.notnull(yData) & pd.notnull(xData).all(axis=1)
        model = sm.OLS(yData, xData, missing='drop')
        res = model.fit()
        data[left_name+'_orthogonalized'] = np.nan
        data.ix[NaNInd, left_name+'_orthogonalized'] = res.resid
        return data

    factor_1 = left_data[[left_name]]
    if not isinstance(right_name, list):
        right_name = [right_name]
    factor_2 = right_data[right_name]
    factor = pd.concat([factor_1, factor_2], axis=1)
    factor['alpha'] = 1  # 加入常数项
    factor = factor.groupby(level=0).apply(OLS, left_name=left_name)
    return factor[[left_name+'_orthogonalized']]


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


def Join_Factors(factor_data, merge_names, new_name, weight=None):
    """
    合并因子,按照权重进行加总。只将非缺失的因子的权重重新归一合成。

    :param factor_data: pandas.dataframe

    :param merge_names: list
        待合并因子名称，必须是data_frame中列的子集
    :param new_name: str
        合成因子名称
    :param weight: list or None
        待合并因子的权重
    :return: new_data

    """
    if isinstance(merge_names, str):
        merge_names = [merge_names]
    if weight is None:
        weight = np.array([1 / len(merge_names)] * len(merge_names))
    else:
        total_weight = sum(weight)
        weight = [iweight/total_weight for iweight in weight]
    weight_array = np.array([weight]*len(factor_data))
    na_ind = factor_data[merge_names].isnull().values
    weight_array[na_ind] = 0
    weight_array = weight_array / weight_array.sum(axis=1)[:, np.newaxis]
    new_values = np.nansum(factor_data[merge_names].values * weight_array, axis=1)
    return pd.DataFrame(new_values, index=factor_data.index, columns=[new_name])


def NonLinearSize(factor_data, factor_name, new_name):
    """
    BARRA 非线性市值因子
    :param factor_data: pandas.dataframe

    :param factor_name: str
        市值因子
    :param new_name: str
        新因子名
    :return: new_data

    """
    nl_size = factor_data[factor_name].copy()
    all_dates = nl_size.index.get_level_values(0).unique()
    for date in all_dates:
        inl_size = nl_size.loc[date]
        inl_size_cube = inl_size ** 3
        temp_ind = pd.notnull(inl_size)
        inl_size_not_na = inl_size[temp_ind]
        inl_size_cube_not_na = inl_size_cube[temp_ind]

        nlen = len(inl_size_not_na)
        x_mean = inl_size_not_na.mean()
        y_mean = inl_size_cube_not_na.mean()
        beta = ((inl_size_not_na*inl_size_cube_not_na).sum()-nlen*x_mean*y_mean)/((inl_size_not_na**2).sum()-nlen*x_mean**2)
        alpha = y_mean - x_mean*beta
        iresi = inl_size_cube_not_na - alpha - beta*inl_size_not_na
        inl_size.loc[temp_ind] = iresi
    nl_size.columns = new_name
    return nl_size


def Generate_Dummy(category_data, drop_first=True):
    """哑变量生成函数"""
    dummy = pd.get_dummies(category_data, drop_first=drop_first)
    return dummy


def ScoringFactors(factor_data, factors, **kwargs):
    """为每一个因子打分
    factor_data: DataFrame(index:[date, IDs], data:factors)
    """
    # 当只有一个因子时，就不需要处理异常值，因为排序都是一样的。
    if len(factors) == 1:
        return factor_data[factors]
    d = factor_data[factors].apply(lambda x: DropOutlier(x.reset_index(), factor_name=x.name, **kwargs)[x.name+'_after_drop_outlier'])
    d = d.rename(columns=lambda x:x.replace('_after_drop_outlier', ''))
    dd = d.apply(lambda x: Standard(x.reset_index(), factor_name=x.name, **kwargs)[x.name+'_after_standard'])
    return dd.rename(columns=lambda x:x.replace('_after_standard', ''))


def NeutralizeBySizeIndu(factor_data, factor_name, std_qt=True, indu_name='中信一级'):
    """ 为因子进行市值和行业中性化

    市值采用对数市值(百万为单位);

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
    """
    from FactorLib.data_source.base_data_source_h5 import data_source
    dates = factor_data.index.get_level_values(0).unique().tolist()
    ids = factor_data.index.get_level_values(1).unique().tolist()
    lncap = np.log(data_source.load_factor('float_mkt_value', '/stocks/', dates=dates, ids=ids) / 10000.0). \
        rename(columns={'float_mkt_value': 'lncap'}).reindex(factor_data.index)
    indu_flag = data_source.sector.get_industry_dummy(None, industry=indu_name, dates=dates).reindex(factor_data.index)

    if std_qt:
        factor_data = StandardByQT(factor_data, factor_name).rename(columns=lambda x: factor_name)
        lncap = StandardByQT(lncap, 'lncap').rename(columns=lambda x: 'lncap')
    industry_names = list(indu_flag.columns)
    indep_data = lncap.join(indu_flag)
    resid = Orthogonalize(factor_data, indep_data, factor_name, industry_names+['lncap'])
    return resid


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    factor_data = data_source.load_factor('ths_click_ratio','/stock_alternative/', start_date='20120409', end_date='20141231')
    r = NeutralizeBySizeIndu(factor_data, 'ths_click_ratio')
    data_source.h5DB.save_factor(r, '/stock_alternative/')