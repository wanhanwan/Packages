# coding: utf-8
# 存储常量
SW_INDUSTRY_DICT = {'801010':'农林牧渔','801020':'采掘','801030':'化工','801040':'钢铁','801050':'有色金属',
                    '801080':'电子','801110':'家用电器','801120':'食品饮料','801130':'纺织服装','801140':'轻工制造',
                    '801150':'医药生物','801160':'公用事业','801170':'交通运输','801180':'房地产','801200':'商业贸易',
                    '801210':'休闲服务','801230':'综合','801710':'建筑材料','801720':'建筑装饰','801730':'电气设备',
                    '801740':'国防军工','801750':'计算机','801760':'传媒','801770':'通信','801780':'银行','801790':'非银金融',
                    '801880':'汽车','801890':'机械设备','801060':'建筑建材','801070':'机械设备','801090':'交运设备',
                    '801190':'金融服务','801100':'信息设备','801220':'信息服务'}
SW_INDUSTRY_DICT_REVERSE = {SW_INDUSTRY_DICT[x]: x for x in SW_INDUSTRY_DICT}
SW_INDUSTRY_CODES = [x+".WI" for x in SW_INDUSTRY_DICT]

CS_INDUSTRY_DICT = {'CI005001':'石油石化','CI005002':'煤炭','CI005003':'有色金属','CI005004':'电力及公用事业','CI005005':'钢铁',
                    'CI005006':'基础化工','CI005007':'建筑','CI005008':'建材','CI005009':'轻工制造','CI005010':'机械',
                    'CI005011':'电力设备','CI005012':'国防军工','CI005013':'汽车','CI005014':'商贸零售','CI005015':'餐饮旅游',
                    'CI005016':'家电','CI005017':'纺织服装','CI005018':'医药','CI005019':'食品饮料','CI005020':'农林牧渔',
                    'CI005021':'银行','CI005022':'非银行金融','CI005023':'房地产','CI005024':'交通运输','CI005025':'电子元器件',
                    'CI005026':'通信','CI005027':'计算机','CI005028':'传媒','CI005029':'综合'}
CS_INDUSTRY_DICT_REVERSE = {CS_INDUSTRY_DICT[x]: x for x in CS_INDUSTRY_DICT}
CS_INDUSTRY_CODES = [x+".WI" for x in CS_INDUSTRY_DICT]

WIND_INDUSTRY_DICT = {'882002':'材料', '882001':'能源','882003':'工业','882004':'可选消费','882005':'日常消费',
                      '882006':'医疗保健', '882007':'金融', '882008':'信息技术', '882009':'电信服务',
                      '882010':'公用事业', '882011':'房地产'}
WIND_INDUSTRY_DICT_REVERSE = {WIND_INDUSTRY_DICT[x]: x for x in WIND_INDUSTRY_DICT}

MARKET_INDEX_DICT = {'000905':'中证500','000300':'沪深300','000906':'中证800','881001':'万得全A','000001':'上证综指',
                     '000016':'上证50','399102':'创业板综', '399974':'中证国企改革指数', '000991': '中证全指医药',
                     '000808':'申万医药生物', '101005': '全A(剔除银行券商)', '100001':'盈利性筛选1', '000852': '中证1000'}

USER_INDEX_DICT = {'merge_acc': '并购指数', '_100003': '去壳指数', 'risky_stocks': '风险预警指数',
                   '_100004': '风险壳指数', 'analyst_recommand_20d': '分析师推荐指数', 'msci_china': 'MSCI中国指数'}

INDUSTRY_NAME_DICT = {'中信一级':'cs_level_1','申万一级':'sw_level_1', '中信二级': 'cs_level_2', '申万二级': 'sw_level_2',
                      '万得一级':'wind_level_1', '中信细分非银':'diversified_finance_cs', '申万细分非银':'diversified_finance_sw'}

MARKET_INDEX_WINDCODE = {"中证500": "000905.SH",
                         "沪深300": "000300.SH",
                         "上证综指": "000001.SH",
                         "上证50": "000016.SH",
                         "中证800": "000906.SH",
                         "创业板综": "399102.SZ",
                         "万得全A": "881001.WI",
                         "中证全指医药指数": "000991.SH",
                         "中证申万医药生物指数": "000808.SH",
                         "中证国企改革指数": "399974.SZ",
                         "中证1000": "000852.SH"
                         }
MARKET_INDEX_WINDCODE_REVERSE = {MARKET_INDEX_WINDCODE[x]: x for x in MARKET_INDEX_WINDCODE}

INDEX_WEIGHT_DICT = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证800": "000906",
    "沪深300非银行券商": "000300_dropbrkbank",
    "中证500非银行券商": "000905_dropbrkbank",
    "中证800非银行券商": "000906_dropbrkbank",
    "中证全指医药": "000991",
    "中证国企改革": "399974"
}

DATEMULTIPLIER = {'m':20,'w':5,'Y':252}


# iFind接口相关
THS_USERID = 'xyjj096'
THS_PASSWORD = '1991822929'
THS_BAR_DEFAULT_PARAMS = "period:D,pricetype:6,rptcategory:0,fqdate:1900-01-01,hb:YSHB,fill:Previous"