import os.path as path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.rls import RLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ..utils.tool_funcs import searchNameInStrList
from .riskmodel_data_source import RiskModelDataSourceOnH5, RiskDataSource
from ..data_source.base_data_source_h5 import h5, tc
from ..data_source.converter import IndustryConverter
from ..utils.datetime_func import DateRange2Dates
from ..data_source.tseries import date_shift, move_dtindex
from warnings import warn


class BarraModel(object):
    """Barra风险模型"""
    def __init__(self, name, model_path):
        self.name = name
        self.riskdb = RiskDataSource(self.name)
        self._init_results()
        self._init_args()

    def _init_args(self):
        """初始化参数"""
        args = dict()
        args['行业因子'] = 'DEFAULT'
        args['回归权重因子'] = 'float_mkt_value'
        args['风格因子'] = 'DEFAULT'
        args['忽略行业'] = []
        args['开始时间'] = '20100101'
        args['结束时间'] = pd.datetime.today().strftime('%Y%m%d')
        self.args = args

    def _init_results(self):
        """初始化回归结果"""
        self.factor_return = None  # 记录因子收益率
        self.resid_return = None  # 记录残差收益率
        self.rsquared = None  # 记录拟合优度
        self.adjust_rsquared = None  # 记录调整的拟合优度
        self.tvalue = None  # 记录T统计量
        self.fvalue = None  # 记录F统计量
        self.vifvalue = None  # 记录方差膨胀系数
        self.factor_riskmatrix = {}  # 记录因子收益率协方差矩阵
        self.specific_riskmatrix = {}  # 记录特质收益率协方差矩阵

    def refresh_args(self):
        """
        刷新参数 \n
        参数包括：
        股票池(Estu)
        行业因子
        风格因子
        估计时间区间
        """
        self.args['所有时间'] = tc.get_trade_days(self.args['开始时间'], self.args['结束时间'])
        estu = self.riskdb.load_factors(['Estu'], dates=self.args['所有时间'])
        self.all_ids = estu.index.get_level_values(1).unique().tolist()
        self.estu = estu[estu['Estu'] == 1]

        if self.args['行业因子'] == 'DEFAULT':
            all_industries = [x.replace('.h5', '') for x in self.riskdb.list_files('factorData') if x.startswith('Indu_')]
            self.args['行业因子'] = ('DEFAULT', list(set(all_industries).difference(set(self.args['忽略行业']))))
        else:
            all_industries = IndustryConverter.all_values(self.args['行业因子'])
            self.args['行业因子'] = (self.args['行业因子'], [list(set(all_industries).difference(set(self.args['忽略行业'])))])


    def prepareFactorReturnResults(self, dates):
        all_ids = self.data_source.all_ids
        self.factor_return = pd.DataFrame(index=dates, columns=self.all_factors)
        self.tvalue = pd.DataFrame(index=dates, columns=self.all_factors)
        self.resid_return = pd.DataFrame(index=dates, columns=all_ids)
        self.rsquared = pd.Series(index=dates, name='rsquared')
        self.adjust_rsquared = pd.Series(index=dates, name='adjust_rsquared')
        self.fvalue = pd.Series(index=dates, name='fvalue')
        self.vifvalue = pd.DataFrame(index=dates,columns=self.genFactorReturnArgs['风格因子'])

    def setGenFactorReturnArgs(self, attr, value):
        """设置因子收益率生成参数"""
        if attr in self.genFactorReturnArgs:
            self.genFactorReturnArgs[attr] = value

    @DateRange2Dates
    def setDimension(self, start_date=None, end_date=None, dates=None):
        """设置股票维度和时间维度"""
        self.estu = self.data_source.update_estu(dates)
        # 由于在截面回归时，因子数据需要滞后一天，所以维度需要滞后
        self.estu_fw1d = move_dtindex(self.estu, 1, '1d')
        self.data_source.set_dimension(self.estu)

    def setStyleFactors(self, factors):
        """设置风格因子"""
        init_style_factors = self.genFactorReturnArgs['风格因子']
        self.genFactorReturnArgs['风格因子'] = [x for x in factors if x in init_style_factors]
        self.all_factors = ['Market'] +  self.genFactorReturnArgs['风格因子'] + self.all_industries

    def setIgnoredIndusries(self, industries):
        """设置忽略行业"""
        self.genFactorReturnArgs['忽略行业'] = []
        for industry in industries:
            if industry in self.all_industries:
                self.genFactorReturnArgs['忽略行业'].append(industry)
                self.all_industries.remove(industry)
        self.all_factors = ['Market'] + self.genFactorReturnArgs['风格因子'] + self.all_industries

    def setRegressWeightFactor(self, factor):
        """设置回归权重因子"""
        if factor in self.data_source.factor_names:
            self.genFactorReturnArgs['回归权重因子'] = factor

    def setIndustryFactor(self, factor):
        """设置行业因子"""
        if factor in self.data_source.factor_names:
            self.genFactorReturnArgs['行业因子'] = factor
            self.all_industries = self.data_source.get_factor_unique_data(factor)
            if np.nan in self.all_industries:
                self.all_industries.remove(np.nan)
            self.genFactorReturnArgs['忽略行业'] = []
            self.all_factors = ['Market'] + self.genFactorReturnArgs['风格因子'] + self.all_industries

    def setStrucStdFactor(self, factors):
        """
        设置结构化风险模型回归因子
        :param factor: list-like
        :return:
        """
        self.strucstd_factors = [x for x in factors if x in self.genFactorReturnArgs['风格因子']]

    def getStyleFactorData(self, ids=None, start_date=None, end_date=None, dates=None, idx=None):
        """获得风格因子数据"""
        if idx is not None:
            start_date = None
            end_date = None
            data = self.data_source.get_data(self.genFactorReturnArgs['风格因子'], start_date=start_date,
                                             end_date=end_date, dates=dates, ids=ids, idx=idx)
            data = data.reindex(idx)
        else:
            data = self.data_source.get_data(self.genFactorReturnArgs['风格因子'], start_date=start_date,
                                             end_date=end_date, ids=ids, dates=dates)
        if np.any(pd.isnull(data)):
            warn("风格因子数据存在缺失值，请检查！")
            data = data.dropna()
        return data

    def getIndustryDummy(self, ids=None, start_date=None, end_date=None, dates=None, idx=None):
        """获得行业因子哑变量"""
        all_industries = [x for x in self.all_industries if x not in self.genFactorReturnArgs['忽略行业']]
        all_industries_codes = IndustryConverter.name2id(self.genFactorReturnArgs['行业因子'], all_industries)

        if idx is not None:
            start_date = None
            end_date = None
            data = self.data_source.get_factor_data(self.genFactorReturnArgs['行业因子'], start_date=start_date,
                                                    end_date=end_date, dates=dates, ids=ids, idx=idx)
            data = data.reindex(idx)
        else:
            data = self.data_source.get_factor_data(self.genFactorReturnArgs['行业因子'], start_date=start_date,
                                                    end_date=end_date, ids=ids, dates=dates)
        data = data[data.iloc[:, 0].isin(all_industries_codes)]
        if np.any(data.isnull()):
            warn("行业因子存在缺失数据，请检查！")
            data = data.dropna()
        data = data.astype('int32')
        data[self.genFactorReturnArgs['行业因子']] = data[self.genFactorReturnArgs['行业因子']].astype('category')
        dummy = pd.get_dummies(data)
        return dummy

    @DateRange2Dates
    def getStockReturn(self, ids=None, start_date=None, end_date=None, dates=None, idx=None):
        if idx is not None:
            dates = idx.index.get_level_values(0).unique().tolist()
            ids = idx.index.get_level_values(1).unique().tolist()
            data = h5.load_factor('daily_returns', '/stocks/', ids=ids, dates=dates)
            data = data.reindex(idx.index)
        else:
            data = h5.load_factor('daily_returns', '/stocks/', ids=ids, dates=dates)
        if np.any(data.isnull()):
            warn("股票收益率数据存在缺失值，请检查！")
            data = data.dropna()
        return data

    def getPortfolioFactorExposure(self, portfolio):
        """组合的因子暴露"""
        style_factor = self.getStyleFactorData(idx=portfolio.index)
        industry_factor = self.getIndustryDummy(idx=portfolio.index)
        return pd.concat([style_factor, industry_factor], axis=1)

    def getResgressWeight(self, ids=None, start_date=None, end_date=None, dates=None, percentile=0.95, idx=None):
        """计算回归权重"""
        weight = self.data_source.get_factor_data(self.genFactorReturnArgs['回归权重因子'], start_date=start_date,
                                                  end_date=end_date, ids=ids, dates=dates, idx=idx)
        weight = np.sqrt(weight).astype('float32')
        weight_sum_perdate = weight.groupby(level=0).sum()
        weight = weight / weight_sum_perdate
        quantile_weight_perdate = weight.groupby(level=0).quantile(percentile)
        weight, quantile_weight_perdate = weight.align(quantile_weight_perdate, axis=0, level=0)
        weight.loc[quantile_weight_perdate.iloc[:, 0] < weight.iloc[:, 0]] = quantile_weight_perdate
        return weight

    def getIndustryMarketValue(self, industry_dummy):
        """计算行业市值"""
        float_mkt_mv = self.data_source.get_factor_data(
            self.genFactorReturnArgs['回归权重因子'], idx=self.estu_fw1d)
        industry_mv = pd.DataFrame(industry_dummy.values * float_mkt_mv.values, index=industry_dummy.index, columns=industry_dummy.columns)
        industry_mv = industry_mv.sum(level=0).stack()
        return industry_mv

    def getMarketReturn(self, stock_return, weight):
        """计算市场收益率"""
        data = pd.concat([stock_return, weight], axis=1)
        data['weighted_return'] = data.prod(axis=1, skipna=False)
        mask = pd.notnull(data['weighted_return'])
        market_return = data['weighted_return'].sum(level=0) / data.iloc[mask.values, 1].sum(level=0)
        return market_return.to_frame('market_return')

    def addSample(self, factor_data, industry_dummy, stock_return, weight, market_return):
        """补充代样本"""
        proxy_id = []
        proxy_date = []
        proxy_weight = []
        proxy_industry = []
        proxy_ret = []

        for i, industry in enumerate(industry_dummy.columns):
            sample = factor_data.loc[industry_dummy[industry] == 1, :]
            if sample.empty:
                continue
            iweight = weight.loc[industry_dummy[industry]==1, :]
            itotal_weight = iweight.sum(level=0)
            nsample = itotal_weight ** 2 / (iweight ** 2).sum(level=0)
            nsample_bigger_than_5 = nsample[(nsample.iloc[:, 0] < 5) & (nsample.iloc[:, 0] > 0)]
            if not nsample_bigger_than_5.empty:
                proxy_id += ['P'+str(industry)] * len(nsample_bigger_than_5)
                proxy_date += list(nsample_bigger_than_5.index.get_level_values(0))
                proxy_weight += ((5 - nsample_bigger_than_5) * itotal_weight.loc[nsample_bigger_than_5.index, :] / nsample_bigger_than_5).iloc[:, 0].tolist()
                proxy_industry += [industry_dummy.loc[industry_dummy[industry].argmax(), :].values.tolist()] * len(nsample_bigger_than_5)
                proxy_ret += market_return.loc[nsample_bigger_than_5.index].iloc[:, 0].tolist()
        nproxy = len(proxy_id)
        if nproxy > 0:
            idx = pd.MultiIndex.from_arrays([proxy_date, proxy_id], names=['date', 'IDs'])
            new_weight = pd.DataFrame(proxy_weight, index=idx, columns=weight.columns)
            new_industry = pd.DataFrame(proxy_industry, index=idx, columns=industry_dummy.columns)
            new_ret = pd.DataFrame(proxy_ret, index=idx, columns=stock_return.columns)
            new_data = pd.DataFrame(np.zeros((nproxy, factor_data.shape[1])), index=idx, columns=factor_data.columns)

            factor_data = factor_data.append(new_data).sort_index()
            weight = weight.append(new_weight).sort_index()
            industry_dummy = industry_dummy.append(new_industry).sort_index()
            stock_return = stock_return.append(new_ret).sort_index()
        return factor_data, industry_dummy, stock_return, weight

    def checkIndustryExist(self, exog, cons, nfactor, nindustry):
        """检验每个行业是否存在股票"""
        nstocks = exog[:, nfactor:].sum(axis=0)
        nzero_industry_ind = np.ones(nstocks.shape)
        if not np.all(nstocks):
            nzero_industry_ind = nstocks != 0
            ind = np.hstack((np.ones(nfactor), nzero_industry_ind)).astype('bool')
            exog = exog[:, ind]
            cons = np.array(cons)[ind]
            return exog, cons, nzero_industry_ind
        return exog, cons, nzero_industry_ind

    def regress(self, factor_data, stock_return, weight, industry_cap, nindustry):
        """截面回归
        数据结构如下:
        factor_data: n * k
        stock_return: n * 1
        weight: n * 1
        industry_cap: (n, )
        nindustry: 1 * 1
        """
        X = factor_data.values
        X = sm.add_constant(X)
        Y = stock_return.values
        weight = weight.values
        sqrt_weight = np.sqrt(weight)
        X = X * sqrt_weight
        Y_Data = Y * sqrt_weight
        industry_cap = industry_cap.values
        nfactor = X.shape[1] - nindustry
        cons = [0] * nfactor + list(industry_cap)
        X, cons, ind = self.checkIndustryExist(X, cons, nfactor, nindustry)
        rls_result = RLS(Y_Data, X, cons).fit()

        # 计算残余收益的鲁棒标准差
        resid = rls_result.resid / np.squeeze(sqrt_weight)
        robust_std = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        Y[Y > 4 * robust_std] = 4 * robust_std
        Y[Y < -4 * robust_std] = -4 * robust_std
        # 第二次最小二乘
        Y_Data = Y * sqrt_weight
        vif = self.vif(X[:, 1:nfactor])
        rls_result = RLS(Y_Data, X, cons).fit()
        return rls_result, ind, vif

    def vif(self, X):
        vifs = []
        for i in np.arange(X.shape[1]):
            vifs.append(variance_inflation_factor(X, i))
        return np.array(vifs)

    @DateRange2Dates
    def getFactorReturn(self, start_date=None, end_date=None, dates=None):
        factor_data = self.getStyleFactorData(dates=dates)
        industry_dummy = self.getIndustryDummy(dates=dates)
        industry_cap = self.getIndustryMarketValue(industry_dummy)
        stock_return = self.getStockReturn(idx=self.estu_fw1d)
        weight = self.getResgressWeight(idx=self.estu_fw1d)
        market_return = self.getMarketReturn(stock_return, weight)
        stock_return_1dayfwd = move_dtindex(stock_return, -1, freq='1d')
        weight_1dayfwd = move_dtindex(weight, -1, freq='1d')
        market_return_1dayfwd = move_dtindex(market_return, -1, freq='1d')
        factor_data,industry_dummy,stock_return,weight = self.addSample(factor_data,
                                                                        industry_dummy.reindex(factor_data.index),
                                                                        stock_return_1dayfwd.reindex(factor_data.index),
                                                                        weight_1dayfwd.reindex(factor_data.index),
                                                                        market_return_1dayfwd)
        nindustry = industry_dummy.shape[1]
        nfactor = factor_data.shape[1] + 1
        self.prepareFactorReturnResults(dates)
        for idate in dates:
            ifactor_data = pd.concat([factor_data.loc[idate], industry_dummy.loc[idate]], axis=1).sort_index()
            if np.any(ifactor_data.isnull()):
                warn("因子数据存在缺失值，请检查！")
                ifactor_data = ifactor_data.dropna()
            ids = [x for x in ifactor_data.index.get_level_values(0) if x[0] != 'P']
            istock_return = stock_return.loc[idate].loc[ifactor_data.index]
            iweight = weight.loc[idate].loc[ifactor_data.index]
            isqrt_weight = np.sqrt(iweight)
            iindustry_cap = industry_cap.loc[idate]
            if not (len(ifactor_data)==len(istock_return)==len(iweight)):
                raise KeyError("回归数据在%s索引不一致"%idate.strftime("%Y%m%d"))
            result, ind, vif = self.regress(ifactor_data, istock_return, iweight, iindustry_cap, nindustry)

            # 存储数据
            # 存储统计指标
            params = result.params
            if ind.sum() < nindustry:
                industry_return = np.ones(nindustry) * market_return_1dayfwd.loc[idate].iloc[0]
                industry_return[ind] = params[nfactor:]
                params = np.hstack((params[:nfactor], industry_return))
            self.factor_return.loc[idate] = params
            resid = result.resid / np.squeeze(isqrt_weight)
            self.resid_return.loc[idate, ids] = resid[:len(ids)]
            ind = np.hstack((np.ones(nfactor), ind)).astype('bool')
            self.tvalue.loc[idate, ind] = result.tvalues
            self.vifvalue.loc[idate] = vif
            try:
                self.fvalue.loc[idate] = result.fvalue
                self.rsquared.loc[idate] = result.rsquared
                self.adjust_rsquared.loc[idate] = result.rsquared_adj
            except:
                self.fvalue.loc[idate] = 0
                self.rsquared.loc[idate] = 0
                self.adjust_rsquared.loc[idate] = 0
        return 1

    def save_regress_results(self):
        self.riskdb.save_data(tvalue=self.tvalue, fvalue=self.fvalue, rsquared=self.rsquared,
                              adjust_rsquared=self.adjust_rsquared, vifs=self.vifvalue,
                              factor_return=self.factor_return, resid_return=self.resid_return,
                              factor_riskmatrix=self.factor_riskmatrix, specific_riskmatrix=self.specific_riskmatrix)