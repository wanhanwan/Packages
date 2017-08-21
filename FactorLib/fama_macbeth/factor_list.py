"""因子构建方式
每一个因子是一个包含三个元素的tuple:
    第一个元素代表因子名称
    第二个元素代表在因子数据库中的字段名
    第三个元素代表因子方向:1 代表正向,-1代表负向

"""
# 估值类因子
bp = ('bp','/stock_value/', 1)
ep = ('ep', '/stock_value/', 1)
bp_minus_median = ('bp_minus_median', '/stock_value/', 1)
bp_standard = ('bp_standard', '/stock_value/', 1)
bp_div_median = ('bp_divide_median', '/stock_value/', 1)
relative_pe = ('relative_pe', '/lg_factor/', -1)
float_mkt_value = ('float_mkt_value', '/stocks/', -1)

# 反转类因子
six_month_highest_returns = ('six_month_highest_returns', '/stock_reversal/', -1)
return_60d = ('return_60d', '/stock_momentum/', -1)

# 流动性因子
turnover_adjust_total_mkt_value = ('turnover_adjust_total_mkt_value', '/stock_liquidity/', -1)

# 其他因子
iffr = ('iffr', '/stock_alternative/', 1)
ic_weighted = ('ic_weighted_five_factors_combined', '/stock_alternative/', 1)
ths_click_ratio_orthogonalized = ('ths_click_ratio_orthogonalized', '/stock_alternative/', -1)
ths_click_ratio_orthogonalized_amount = ('ths_click_ratio_orthogonalized_amount', '/stock_alternative/', -1)
ths_click_ratio = ('ths_click_ratio', '/stock_alternative/', -1)
snowball = ('xueqiu', '/stock_alternative/', -1)

# 动量类因子
six_month_highest_returns_plus = ('six_month_highest_returns_plus', '/stock_reversal/', 1)
return_60d_plus = ('return_60d_plus', '/stock_momentum/', 1)

# 成长性因子
oper_rev_ttm_csindustry_quantile_diff = ('oper_rev_ttm_csindustry_quantile_diff', '/stock_growth/', 1)

# 流动性因子
StyleTechnicalFactor_AmountAvg_1M = ('StyleTechnicalFactor_AmountAvg_1M', '/XYData/Technical', -1)

factor_list = [ths_click_ratio_orthogonalized_amount, StyleTechnicalFactor_AmountAvg_1M]