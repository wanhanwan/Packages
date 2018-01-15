"""因子构建方式
每一个因子是一个包含三个元素的tuple:
    第一个元素代表因子名称
    第二个元素代表在因子数据库中的字段名
    第三个元素代表因子方向:1 代表正向,-1代表负向

"""
# 估值类因子
bp = ('bp','/stock_value/', 1)
ep = ('ep', '/stock_value/', 1)
bp_standard = ('bp_standard', '/stock_value/', 1)
bp_divide_median = ('bp_divide_median', '/stock_value/', 1)
ep_divide_median = ('ep_divide_median', '/stock_value/', 1)
epttm_divide_median = ('epttm_divide_median', '/stock_value/', 1)
relative_pe = ('relative_pe', '/lg_factor/', -1)
float_mkt_value = ('float_mkt_value', '/stocks/', -1)

# 反转类因子
six_month_highest_returns = ('six_month_highest_returns', '/stock_reversal/', 1)
return_60d = ('return_60d', '/stock_momentum/', -1)
return_25d = ('return_25d', '/stock_momentum/', -1)

# 流动性因子
turnover_adjust_total_mkt_value = ('turnover_adjust_total_mkt_value', '/stock_liquidity/', -1)
turn_60d = ('turn_60d', '/stock_liquidity/', -1)

# 其他因子
iffr = ('iffr', '/stock_alternative/', 1)
ic_weighted = ('ic_weighted_five_factors_combined', '/stock_alternative/', 1)
ths_click_ratio_orthogonalized = ('ths_click_ratio_orthogonalized', '/stock_alternative/', -1)
ths_click_ratio_orthogonalized_amount = ('ths_click_ratio_orthogonalized_amount', '/stock_alternative/', -1)
ths_click_ratio = ('ths_click_ratio', '/stock_alternative/', -1)
snowball = ('xueqiu', '/stock_alternative/', -1)
snowball_orthogonalized = ('xueqiu_orthogonalized', '/stock_alternative/', -1)
StyleTechnicalFactor_AmountAvg_1M_orthogonalized = ('StyleTechnicalFactor_AmountAvg_1M_orthogonalized', '/stock_alternative/', -1)
CGO = ('CGO', '/stock_alternative/', -1)
PB_LRJ = ('PB_LRJ', '/stock_alternative/', -1)
PE_GL = ('PE_GL', '/stock_alternative', -1)
SPE_CAP = ('SPE_CAP', '/stock_alternative', -1)

# 动量类因子
six_month_highest_returns_plus = ('six_month_highest_returns', '/stock_reversal/', -1, 'six_month_highest_returns_plus')
return_60d_plus = ('return_60d_plus', '/stock_momentum/', 1)

# 成长性因子
oper_rev_ttm_csindustry_quantile_diff = ('oper_rev_ttm_csindustry_quantile_diff', '/stock_growth/', 1)



# 兴业金工因子库
# Growth
StyleGrowthFactor_Earnings_LTG = ('StyleGrowthFactor_Earnings_LTG', '/XYData/Growth/', 1)
StyleGrowthFactor_Earnings_SFG = ('StyleGrowthFactor_Earnings_LTG', '/XYData/Growth/', 1)
StyleGrowthFactor_Earnings_SQ_YoY = ('StyleGrowthFactor_Earnings_SQ_YoY', '/XYData/Growth/', 1)
StyleGrowthFactor_SaleEarnings_SQ_YoY = ('StyleGrowthFactor_SaleEarnings_SQ_YoY', '/XYData/Growth/', 1)
StyleGrowthFactor_Sales_LTG = ('StyleGrowthFactor_Sales_LTG', '/XYData/Growth/', 1)
StyleGrowthFactor_Sales_SFG = ('StyleGrowthFactor_Sales_SFG', '/XYData/Growth/', 1)
StyleGrowthFactor_Sales_SQ_YoY = ('StyleGrowthFactor_Sales_SQ_YoY', '/XYData/Growth/', 1)

# Quality
StyleQualityFactor_AssetTurnover = ('StyleQualityFactor_AssetTurnover', '/XYData/Quality/', 1)
StyleQualityFactor_CurrentRatio = ('StyleQualityFactor_CurrentRatio', '/XYData/Quality/', 1)
StyleQualityFactor_Debt2Equity_LR = ('StyleQualityFactor_Debt2Equity_LR', '/XYData/Quality/', 1)
StyleQualityFactor_GrossMargin_TTM = ('StyleQualityFactor_GrossMargin_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_OperatingCashFlows2OperatingProfits_TTM = ('StyleQualityFactor_OperatingCashFlows2OperatingProfits_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_OperatingProfitMargin_TTM = ('StyleQualityFactor_OperatingProfitMargin_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_ROA_TTM = ('StyleQualityFactor_ROA_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_ROE_TTM = ('StyleQualityFactor_ROE_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_ThreeCosts2Sales_TTM = ('StyleQualityFactor_ThreeCosts2Sales_TTM', '/XYData/Quality/', 1)
StyleQualityFactor_ROE_TTM_after_drop_outlier = ('StyleQualityFactor_ROE_TTM_after_drop_outlier', '/XYData/Quality/', 1)

# Sentiment
StyleSentimentFactor_EPSChange_FY0_1M = ('StyleSentimentFactor_EPSChange_FY0_1M', '/XYData/Sentiment', 1)
StyleSentimentFactor_EPSChange_FY0_3M = ('StyleSentimentFactor_EPSChange_FY0_3M', '/XYData/Sentiment', 1)
StyleSentimentFactor_RatingChange_1M = ('StyleSentimentFactor_RatingChange_1M', '/XYData/Sentiment', 1)
StyleSentimentFactor_RatingChange_3M = ('StyleSentimentFactor_RatingChange_3M', '/XYData/Sentiment', 1)
StyleSentimentFactor_SalesChange_FY0_1M = ('StyleSentimentFactor_SalesChange_FY0_1M', '/XYData/Sentiment', 1)
StyleSentimentFactor_SalesChange_FY0_3M = ('StyleSentimentFactor_SalesChange_FY0_3M', '/XYData/Sentiment', 1)
StyleSentimentFactor_TargetReturn = ('StyleSentimentFactor_TargetReturn', '/XYData/Sentiment', 1)

# Technical
StyleTechnicalFactor_AmountAvg_1M = ('StyleTechnicalFactor_AmountAvg_1M', '/XYData/Technical', -1)
StyleTechnicalFactor_ILLIQ_20DAvg = ('StyleTechnicalFactor_ILLIQ_20DAvg', '/XYData/Technical', 1)
StyleTechnicalFactor_LnFloatCap = ('StyleTechnicalFactor_LnFloatCap', '/XYData/Technical', -1)
StyleTechnicalFactor_LnFloatCap_Big = ('StyleTechnicalFactor_LnFloatCap', '/XYData/Technical', 1)
StyleTechnicalFactor_RealizedVolatility_1Y = ('StyleTechnicalFactor_RealizedVolatility_1Y', '/XYData/Technical', -1)
StyleTechnicalFactor_SmallTradeFlow = ('StyleTechnicalFactor_SmallTradeFlow', '/XYData/Technical', -1)
StyleTechnicalFactor_TSKEW_20D = ('StyleTechnicalFactor_TSKEW_20D', '/XYData/Technical', -1)
StyleTechnicalFactor_TurnoverAvg_1M = ('StyleTechnicalFactor_TurnoverAvg_1M', '/XYData/Technical', -1)
StyleTechnicalFactor_Volume20D_240D = ('StyleTechnicalFactor_Volume20D_240D', '/XYData/Technical', -1)
StyleTechnicalFactor_VolumeCV_20D = ('StyleTechnicalFactor_VolumeCV_20D', '/XYData/Technical', -1)

# Value
StyleValueFactor_BP_LR = ('StyleValueFactor_BP_LR', '/XYData/Value/', 1)
StyleValueFactor_CFP_TTM = ('StyleValueFactor_CFP_TTM', '/XYData/Value/', 1)
StyleValueFactor_EP_Fwd12M = ('StyleValueFactor_EP_Fwd12M', '/XYData/Value/', 1)
StyleValueFactor_EP_LYR = ('StyleValueFactor_EP_LYR', '/XYData/Value/', 1)
StyleValueFactor_EP_SQ = ('StyleValueFactor_EP_SQ', '/XYData/Value/', 1)
StyleValueFactor_EP_TTM = ('StyleValueFactor_EP_TTM', '/XYData/Value/', 1)
StyleValueFactor_Sales2EV = ('StyleValueFactor_Sales2EV', '/XYData/Value/', 1)
StyleValueFactor_SP_TTM = ('StyleValueFactor_SP_TTM', '/XYData/Value/', 1)

# Momentum
StyleMomentumFactor_LotteryMomentum_1M = ('StyleMomentumFactor_LotteryMomentum_1M', '/XYData/Momentum', -1)
StyleMomentumFactor_Momentum_1M = ('StyleMomentumFactor_Momentum_1M', '/XYData/Momentum', -1)
StyleMomentumFactor_Momentum_3M = ('StyleMomentumFactor_Momentum_3M', '/XYData/Momentum', -1)
StyleMomentumFactor_Momentum_60M = ('StyleMomentumFactor_Momentum_60M', '/XYData/Momentum', -1)

# StyleFactor
StyleFactor_GM = ('StyleFactor_GM', '/XYData/StyleFactor/', 1)
StyleFactor_GrowthFactor = ('StyleFactor_GrowthFactor', '/XYData/StyleFactor/', 1)
StyleFactor_MomentumFactor = ('StyleFactor_MomentumFactor', '/XYData/StyleFactor/', 1)
StyleFactor_SentimentFactor = ('StyleFactor_SentimentFactor', '/XYData/StyleFactor/', 1)
StyleFactor_TB_adjM = ('StyleFactor_TB_adjM', '/XYData/StyleFactor/', 1)
StyleFactor_TradingBehaviorFactor = ('StyleFactor_TradingBehaviorFactor', '/XYData/StyleFactor/', 1)
StyleFactor_ValueFactor = ('StyleFactor_ValueFactor', '/XYData/StyleFactor/', 1)
StyleFactor_VG = ('StyleFactor_VG', '/XYData/StyleFactor/', 1)
StyleFactor_VGS = ('StyleFactor_VGS', '/XYData/StyleFactor/', 1)
StyleFactor_VGS_TB = ('StyleFactor_VGS_TB', '/XYData/StyleFactor/', 1)
StyleFactor_VGS_TBadjM = ('StyleFactor_VGS_TBadjM', '/XYData/StyleFactor/', 1)
StyleFactor_VS = ('StyleFactor_VS', '/XYData/StyleFactor/', 1)

# Others
FamaFrenchFactor_IVR = ('FamaFrenchFactor_IVR', '/XYData/Others/', -1)
RVFactor_RV = ('RVFactor_RV', '/XYData/Others/', 1)
SpreadBiasFactor_SpreadBias_120D = ('SpreadBiasFactor_SpreadBias_120D', '/XYData/Others/', 1)
ValueBiasFactor_EP_DR = ('ValueBiasFactor_EP_DR', '/XYData/Others/', 1)
ValueBiasFactor_SP_DR = ('ValueBiasFactor_SP_DR', '/XYData/Others/', 1)
ValueBiasFactor_BP_DR = ('ValueBiasFactor_BP_DR', '/XYData/Others/', 1)
VWAPPFactor_VWAPP_OLS = ('VWAPPFactor_VWAPP_OLS', '/XYData/Others/', 1)

# 临时因子
Earning_YOY = ('EARNING_YOY', '/temp/', 1)
SALES_YOY = ('SALES_YOY', '/temp/', 1)
ROE = ('ROE', '/temp/', 1)
BP = ('BP', '/temp/', 1)
EP = ('EP', '/temp/', 1)
TURN = ('TURN', '/temp/', -1)

factor_list_growth = [StyleGrowthFactor_Earnings_LTG, StyleGrowthFactor_Earnings_SFG, StyleGrowthFactor_Earnings_SQ_YoY,
               StyleGrowthFactor_SaleEarnings_SQ_YoY, StyleGrowthFactor_Sales_LTG, StyleGrowthFactor_Sales_SFG,
               StyleGrowthFactor_Sales_SQ_YoY]

factor_list_quality = [StyleQualityFactor_AssetTurnover, StyleQualityFactor_CurrentRatio,
               StyleQualityFactor_Debt2Equity_LR, StyleQualityFactor_GrossMargin_TTM,
               StyleQualityFactor_OperatingCashFlows2OperatingProfits_TTM,
               StyleQualityFactor_OperatingProfitMargin_TTM, StyleQualityFactor_ROA_TTM,
               StyleQualityFactor_ROE_TTM,
               # StyleQualityFactor_ROE_TTM_after_drop_outlier,
               StyleQualityFactor_ThreeCosts2Sales_TTM]

factor_list_sentiment = [StyleSentimentFactor_EPSChange_FY0_1M, StyleSentimentFactor_EPSChange_FY0_3M,
               StyleSentimentFactor_RatingChange_1M, StyleSentimentFactor_RatingChange_3M,
               StyleSentimentFactor_SalesChange_FY0_1M, StyleSentimentFactor_SalesChange_FY0_3M,
               StyleSentimentFactor_TargetReturn]

factor_list_technical = [StyleTechnicalFactor_AmountAvg_1M, StyleTechnicalFactor_ILLIQ_20DAvg, StyleTechnicalFactor_LnFloatCap,
               StyleTechnicalFactor_RealizedVolatility_1Y,StyleTechnicalFactor_TSKEW_20D, StyleTechnicalFactor_TurnoverAvg_1M,
               StyleTechnicalFactor_Volume20D_240D,StyleTechnicalFactor_VolumeCV_20D]

factor_list_value = [StyleValueFactor_BP_LR, StyleValueFactor_CFP_TTM,
               StyleValueFactor_EP_Fwd12M, StyleValueFactor_EP_LYR,
               StyleValueFactor_EP_SQ, StyleValueFactor_EP_TTM,
               StyleValueFactor_Sales2EV, StyleValueFactor_SP_TTM]

factor_list_momentum = [StyleMomentumFactor_LotteryMomentum_1M, StyleMomentumFactor_Momentum_1M,
               StyleMomentumFactor_Momentum_3M, StyleMomentumFactor_Momentum_60M]

factor_list_style = [StyleFactor_GM, StyleFactor_GrowthFactor, StyleFactor_MomentumFactor, StyleFactor_SentimentFactor,
               StyleFactor_TB_adjM, StyleFactor_TradingBehaviorFactor, StyleFactor_ValueFactor,StyleFactor_VG,
               StyleFactor_VGS, StyleFactor_VGS_TB, StyleFactor_VGS_TBadjM, StyleFactor_VS]

self_defined = [bp_divide_median, ep_divide_median, float_mkt_value, six_month_highest_returns, six_month_highest_returns_plus,
                iffr, return_60d, turnover_adjust_total_mkt_value]

factor_list_others = [FamaFrenchFactor_IVR, RVFactor_RV, SpreadBiasFactor_SpreadBias_120D, ValueBiasFactor_EP_DR,
                      ValueBiasFactor_BP_DR, ValueBiasFactor_SP_DR, VWAPPFactor_VWAPP_OLS]

lrj = [CGO, PB_LRJ, PE_GL, SPE_CAP]

# BARRA因子
BETA = ('BETA', '/barra/descriptors/', 1)
BLEV = ('BLEV', '/barra/descriptors/', 1)
BTOP = ('BTOP', '/barra/descriptors/', 1)
CETOP = ('CETOP', '/barra/descriptors/', 1)
CMRA = ('CMRA', '/barra/descriptors/', 1)
DASTD = ('DASTD', '/barra/descriptors/', 1)
DTOA = ('DTOA', '/barra/descriptors/', 1)
EGRLF = ('EGRLF', '/barra/descriptors/', 1)
EGRO = ('EGRO', '/barra/descriptors/', 1)
EGRSF = ('EGRSF', '/barra/descriptors/', 1)
EPFWD = ('EPFWD', '/barra/descriptors/', 1)
ETOP = ('ETOP', '/barra/descriptors/', 1)
HSIGMA = ('HSIGMA', '/barra/descriptors/', 1)
LNCAP = ('LNCAP', '/barra/descriptors/', 1)
MLEV = ('MLEV', '/barra/descriptors/', 1)
RSTR = ('RSTR', '/barra/descriptors/', 1)
SGRO = ('SGRO', '/barra/descriptors/', 1)
STOA = ('STOA', '/barra/descriptors/', 1)
STOQ = ('STOQ', '/barra/descriptors/', 1)
STOM = ('STOM', '/barra/descriptors/', 1)



# 行业因子
cs_level_1 = ('cs_level_1', '/indexes/', 1)
cs_level_2 = ('cs_level_2', '/indexes/', 1)
sw_level_1 = ('sw_level_1', '/indexes/', 1)
sw_level_2 = ('sw_level_2', '/indexes/', 1)
wind_level_1 = ('wind_level_1', '/indexes/', 1)


BARRA = [BETA, BLEV, BTOP, CETOP, CMRA,
         DASTD, DTOA, EGRLF, EGRO, EGRSF,
         EPFWD, ETOP, HSIGMA, LNCAP, MLEV,
         RSTR, SGRO, STOA, STOQ, STOM]


def get_factors(factors):
    all_factors = []
    if isinstance(factors, str):
        all_factors += (globals()[factors])
    else:
        for f in factors:
            try:
                factor = globals()[f]
            except KeyError:
                continue
            all_factors.append(factor)
    return all_factors


factor_list = factor_list_growth + factor_list_momentum + factor_list_quality + factor_list_sentiment + \
    factor_list_style + factor_list_technical + factor_list_value + self_defined

factor_direction = {x[0]: x[2] for x in factor_list}