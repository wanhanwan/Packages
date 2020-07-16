# Packages
基于Python的一站式量化研究工具箱
## 数据部分
* DataSource  支持A股行情数据、财务及衍生数据、一致预期数据的提取、加工。几乎所有的数据都可以通过统一的API(DataSource.load_factor)来提取。
* StockUniverse  支持市场上主流指数的成分股及其权重的提取，由Wind底层数据库驱动。支持交、并、补等集合运算，
例如StockUniverse('000300 + 000905')代表沪深300与中证500指数的合集。
* 集成多个财务数据处理函数，包括但不限于TTM、LatestPeriod、LastNPeriods
## 模型部分
* SingleFactorTest模块提供了单因子测试的完成流程，包括数据清洗、分组测试、绩效统计、风险分析等。支持设置文件方式启动，批量因子测试。
