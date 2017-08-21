# 自有因子的定义
bp_div_median = ('bp_divide_median', '/stock_value/', 1)
ep_div_median = ('ep_divide_median', '/stock_value/', 1)
float_mkt_value = ('float_mkt_value', '/stocks/', -1)
six_month_highest_returns = ('six_month_highest_returns', '/stock_reversal/', 1)
iffr = ('iffr', '/stock_alternative/', 1)
return_60d = ('return_60d', '/stock_momentum/', -1)
turnover_adjust_total_mkt_value = ('turnover_adjust_total_mkt_value', '/stock_liquidity/', -1)
snowball = ('xueqiu', '/stock_alternative/', -1)
ths_click_ratio = ('ths_click_ratio', '/stock_alternative/', -1)


factor_list1 = [bp_div_median, ep_div_median, float_mkt_value, six_month_highest_returns,
               iffr, return_60d, turnover_adjust_total_mkt_value]

factor_list2 = [ths_click_ratio]

factor_list = factor_list2