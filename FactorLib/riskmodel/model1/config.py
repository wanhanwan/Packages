# 风险模型设置文件

ID = 'barra'

# 风险描述子
risk_descriptors = 'BARRA'
others = ['wind_level_1', 'float_mkt_value']

# 两个股票池(CU、ESTU)
CU = '全A'
ESTU = 'barramodel1'

# 风险描述子的横截面清洗
funcs = []

func_arg = {'descriptors': ['BETA','BLEV','BTOP','CETOP','CMRA','DASTD','DTOA','EGRLF','EGRO','EGRSF','EPFWD','ETOP',
    'HSIGMA','LNCAP','MLEV','RSTR','SGRO','STOA','STOQ','STOM'], 'mean_weight': 'float_mkt_value', 'std_weight': None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'size': 'LNCAP', 'new_name': 'NLSIZE'}
func_description = {'func_name': 'nonlinear_size', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['NLSIZE'], 'mean_weight': 'float_mkt_value', 'std_weight': None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['BETA','BLEV','BTOP','CETOP','CMRA','DASTD','DTOA','EGRLF','EGRO','EGRSF','EPFWD','ETOP',
    'HSIGMA','LNCAP','MLEV','RSTR','SGRO','STOA','STOQ','STOM','NLSIZE'], 'method': 'Barra', 'drop_ratio':0.1, 'drop_mode':'截断',
    'alpha': 0.3}
func_description = {'func_name': 'drop_outlier', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['BETA','BLEV','BTOP','CETOP','CMRA','DASTD','DTOA','EGRLF','EGRO','EGRSF','EPFWD','ETOP',
    'HSIGMA','LNCAP','MLEV','RSTR','SGRO','STOA','STOQ','STOM','NLSIZE'], 'mean_weight': 'float_mkt_value', 'std_weight':None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['DASTD','CMRA','HSIGMA'], 'new_factor': 'VOLATILITY', 'weight': [0.74, 0.16, 0.10]}
func_description = {'func_name': 'merge', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['STOA','STOQ','STOM'], 'new_factor': 'LIQUIDITY', 'weight': [0.30, 0.35, 0.35]}
func_description = {'func_name': 'merge', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['EPFWD','CETOP','ETOP'], 'new_factor': 'EARNING', 'weight': [0.68, 0.21, 0.11]}
func_description = {'func_name': 'merge', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['MLEV','DTOA','BLEV'], 'new_factor': 'LEVERAGE', 'weight': [0.38, 0.35, 0.27]}
func_description = {'func_name': 'merge', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['EGRLF','EGRSF','EGRO','SGRO'], 'new_factor': 'GROWTH', 'weight': [0.18, 0.11, 0.24, 0.47]}
func_description = {'func_name': 'merge', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['VOLATILITY','LIQUIDITY','EARNING','LEVERAGE','GROWTH'], 'mean_weight': 'float_mkt_value',
            'std_weight':None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'dependent': 'VOLATILITY', 'independents': ['BETA','LNCAP']}
func_description = {'func_name': 'orthogonalize', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'dependent': 'LIQUIDITY', 'independents': ['LNCAP']}
func_description = {'func_name': 'orthogonalize', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['VOLATILITY','LIQUIDITY'], 'mean_weight': 'float_mkt_value',
            'std_weight':None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['BETA','RSTR','VOLATILITY','BTOP','LIQUIDITY','EARNING','GROWTH','LEVERAGE','NLSIZE'],
            'classify': ['wind_level_1'], 'refs': ['float_mkt_value']}
func_description = {'func_name': 'fillna_barra', 'arg': func_arg}
funcs.append(func_description)

func_arg = {'descriptors': ['BETA','RSTR','VOLATILITY','BTOP','LIQUIDITY','EARNING','GROWTH','LEVERAGE','LNCAP','NLSIZE'],
            'mean_weight': 'float_mkt_value', 'std_weight':None}
func_description = {'func_name': 'standard', 'arg': func_arg}
funcs.append(func_description)


# 存储信息
save_info = {
    'factor_to_save': ['BETA','RSTR','VOLATILITY','BTOP','LIQUIDITY','EARNING','GROWTH','LEVERAGE','LNCAP','NLSIZE'],
    'industry_factor': 'wind_level_1',
    'regress_weight_factor': 'float_mkt_value',
    'factor_save_path': '/barra/factors/',
    'run_type': 'M',
    'split_length': 75,
    'cpu_use': 4
}






