from .stocklist_manager import StockListManager
from ..single_factor_test.config import parse_config
from ..utils import AttrDict
from datetime import datetime
from ..data_source.base_data_source_h5 import tc, h5
from ..data_source.wind_plugin import realtime_quote, get_history_bar
from ..utils.tool_funcs import windcode_to_tradecode, import_module
from ..factor_performance.analyzer import Analyzer
import pandas as pd
import numpy as np
import os
import shutil


class StrategyManager(object):
    fields = ['id', 'name', 'researcher', 'latest_rebalance_date', 'stocklist_name', 'stocklist_id','stockpool',
              'benchmark', 'first_rebalance_date', 'rebalance_frequence', 'industry_neutral', 'industry_class']

    def __init__(self, strategy_path, stocklist_path):
        self._strategy_path = strategy_path
        self._stocklist_path = stocklist_path
        self._strategy_dict = None
        self._stocklist_manager = StockListManager(self._stocklist_path)
        self._init()

    # 初始化
    def _init(self):
        if not os.path.isdir(self._strategy_path):
            os.mkdir(self._strategy_path)
            self._strategy_dict = pd.DataFrame(columns=self.fields)
            self._strategy_dict.to_csv(os.path.join(self._strategy_path, 'summary.csv'))
        if not os.path.isfile(os.path.join(self._strategy_path, 'summary.csv')):
            self._strategy_dict = pd.DataFrame(columns=self.fields)
            self._strategy_dict.to_csv(os.path.join(self._strategy_path, 'summary.csv'), index=False)
        self._strategy_dict = pd.read_csv(os.path.join(self._strategy_path, 'summary.csv'), encoding='GBK',
                                          converters={'benchmark': str})

    # 保存信息
    def _save(self):
        self._strategy_dict.to_csv(os.path.join(self._strategy_path, 'summary.csv'), index=False, quoting=3)

    def performance_analyser(self, strategy_name=None, strategy_id=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        benchmark_name = self.get_attribute('benchmark', strategy_name=strategy_name).zfill(6)
        pkl_file = os.path.join(self._strategy_path, strategy_name+'/backtest/BTresult.pkl')
        if os.path.isfile(pkl_file):
            return Analyzer(pkl_file, benchmark_name)
        else:
            return

    # 最大策略ID
    @property
    def _maxid(self):
        if not self._strategy_dict.empty:
            return self._strategy_dict['id'].max()
        else:
            return 0

    # 策略对应的股票列表名称和ID
    def strategy_stocklist(self, strategy_id=None, strategy_name=None):
        if strategy_id is not None:
            return self._strategy_dict.loc[self._strategy_dict.id == strategy_id,
                                           ['stocklist_name', 'stocklist_id']].iloc[0]
        elif strategy_name is not None:
            return self._strategy_dict.loc[self._strategy_dict.name == strategy_name,
                                           ['stocklist_name', 'stocklist_id']].iloc[0]
        else:
            raise KeyError("No strategy identifier is provided")

    # 策略ID对应的策略名称
    def strategy_name(self, strategy_id=None):
        if strategy_id is not None:
            return self._strategy_dict.loc[self._strategy_dict.id == strategy_id, 'name'].iloc[0]

    def strategy_id(self, strategy_name=None):
        if strategy_name is not None:
            return self._strategy_dict.loc[self._strategy_dict.name == strategy_name, 'id'].iloc[0]

    # 策略是否已存在
    def if_exists(self, name):
        return name in self._strategy_dict['name'].tolist()

    # 从一个文件夹中创建一个策略
    def create_from_directory(self, src, if_exists='error'):
        cwd = os.getcwd()
        os.chdir(src)
        strategy_config = AttrDict(parse_config(os.path.join(src, 'config.yml')))
        if self.if_exists(strategy_config.name):
            if if_exists == 'error':
                raise KeyError("strategy %s already exists"%strategy_config.name)
            elif if_exists == 'replace':
                self.delete(name=strategy_config.name)
        strategy_path = os.path.join(self._strategy_path, strategy_config.name)
        # 创建策略文件夹
        os.mkdir(strategy_path)
        # 复制初始股票列表
        stocklist_filename = strategy_config.stocklist.output
        stocklist_name = stocklist_filename.replace('.csv', '')
        # 策略的调仓日期
        if os.path.isfile(stocklist_filename):
            shutil.copy(stocklist_filename, strategy_path)
            self._stocklist_manager.add_new_one(os.path.abspath(stocklist_filename))
            first_rebalance_date = self._stocklist_manager.min_rebalance_date(stocklist_name)
            latest_rebalance_date = self._stocklist_manager.max_rebalance_date(stocklist_name)
        else:
            first_rebalance_date = np.nan
            latest_rebalance_date = np.nan
        # 复制中间数据
        if os.path.isdir('temp'):
            shutil.copytree('temp', os.path.join(strategy_path, 'temp'))
        # 复制股票列表更新程序
        if os.path.isdir('update'):
            shutil.copytree('update', os.path.join(strategy_path, 'update'))
        # 复制设置文件
        shutil.copy("config.yml", strategy_path)
        # 行业中性
        industry_neutral = '是' if strategy_config.stocklist.industry_neutral else '否'
        industry_class = strategy_config.stocklist.industry
        # 添加新的记录
        self._add_record(stocklist_name=stocklist_name, first_rebalance_date=first_rebalance_date,
                         latest_rebalance_date=latest_rebalance_date, benchmark=strategy_config.stocklist.benchmark,
                         industry_neutral=industry_neutral, industry_class=industry_class, **strategy_config.__dict__)
        os.chdir(cwd)

    # 更新策略股票持仓
    def update_stocks(self, start, end, strategy_name=None, strategy_id=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        module_path = os.path.join(self._strategy_path, strategy_name+'/update/update.py')
        update = import_module('update', module_path)
        update.update(start, end)
        self.refresh_stocks(strategy_name)
        return

    # 刷新股票列表
    def refresh_stocks(self, strategy_name=None, strategy_id=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        stocklistname = self.strategy_stocklist(strategy_name=strategy_name)['stocklist_name']
        src = os.path.join(self._strategy_path, strategy_name+'/%s.csv'%stocklistname)
        shutil.copy(src, os.path.join(self._stocklist_path, stocklistname+'.csv'))
        max_rebalance_date = self._stocklist_manager.max_rebalance_date(stocklistname)
        min_rebalance_date = self._stocklist_manager.min_rebalance_date(stocklistname)
        if strategy_id is None:
            strategy_id = self.strategy_id(strategy_name)
        self.modify_attributes(strategy_id, latest_rebalance_date=max_rebalance_date,
                               first_rebalance_date=min_rebalance_date)

    # 更改策略属性
    def modify_attributes(self, strategy_id, **kwargs):
        for k, v in kwargs.items():
            if k in self._strategy_dict.columns.values:
                self._strategy_dict.loc[self._strategy_dict.id == strategy_id, k] = v
        self._save()

    # 获取属性值
    def get_attribute(self, attr, strategy_name=None, strategy_id=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        return self._strategy_dict.loc[self._strategy_dict.name==strategy_name, attr].iloc[0]

    # 添加一条记录
    def _add_record(self, **kwargs):
        record = pd.DataFrame([[None]*len(self.fields)], columns=self.fields)
        record['id'] = self._maxid + 1
        record['stocklist_id'] = record['id']
        for k, v in kwargs.items():
            if k in self.fields:
                record[k] = v
        self._strategy_dict = self._strategy_dict.append(record)
        self._save()

    # 删除一个策略
    def delete(self, name=None, strategy_id=None):
        if name is not None:
            shutil.rmtree(os.path.join(self._strategy_path, name))
            self._strategy_dict = self._strategy_dict[self._strategy_dict.name != name]
            self._stocklist_manager.delete_stocklist(self.strategy_stocklist(strategy_name=name))
        elif strategy_id is not None:
            name = self.strategy_name(strategy_id)
            shutil.rmtree(os.path.join(self._stocklist_path, name))
            self._strategy_dict = self._strategy_dict[self._strategy_dict.name != name]
            self._stocklist_manager.delete_stocklist(self.strategy_stocklist(strategy_id=strategy_id))
        else:
            self._save()
            raise KeyError("No strategy identifier is provided")
        self._save()

    # 策略最近的股票
    def latest_position(self, strategy_name=None, strategy_id=None):
        stocklist_info = self.strategy_stocklist(strategy_id, strategy_name)
        if strategy_name is not None:
            max_date = self._strategy_dict[self._strategy_dict.name==strategy_name]['latest_rebalance_date']
        else:
            max_date = self._strategy_dict[self._strategy_dict.id==strategy_id]['latest_rebalance_date']
        return self._stocklist_manager.get_position(stocklist_info['stocklist_name'], max_date)

    # 生成交易指令
    def generate_tradeorder(self, strategy_id, capital, realtime=False):
        idx = pd.IndexSlice
        today = tc.get_latest_trade_days(datetime.today().strftime('%Y%m%d'))
        stocks = self.latest_position(strategy_id=strategy_id)
        stocks.index = stocks.index.set_levels([pd.to_datetime(datetime.today().date())], level=0)
        stocks.index = stocks.index.set_levels([windcode_to_tradecode(x) for x in stocks.index.get_level_values(1)], level=1)
        stock_ids = stocks.index.get_level_values(1).tolist()
        last_close = get_history_bar(['收盘价'], start_date=today, end_date=today, **{'复权方式': '前复权'})
        last_close.index = last_close.index.set_levels([pd.to_datetime(datetime.today().date())], level=0)
        """如果当前是交易时间，需要区分停牌和非停牌股票。停牌股票取昨日前复权收盘价，
        非停牌股票取最新成交价。若非交易时间统一使用最新前复权收盘价。"""
        if tc.is_trading_time(datetime.now()) and not realtime:
            data = realtime_quote(['rt_last', 'rt_susp_flag'], ids=stock_ids)
            tradeprice = data['rt_last'].where(data['rt_susp_flag']!=1, last_close['close'])
        else:
            tradeprice = last_close.loc[idx[:, stock_ids], 'close']
        tradeorders = (stocks['Weight'] * capital / tradeprice / 100).reset_index().rename(columns={'IDs': '股票代码',
                                                                                                    0: '手数'})
        strategy_name = self.strategy_name(strategy_id)
        cwd = os.getcwd()
        os.chdir(os.path.join(self._strategy_path, strategy_name))
        tradeorders[['股票代码', '手数']].to_excel('权重文件.xlsx', index=False, float_format='%.4f')
        os.chdir(cwd)
        return

    # 运行回测
    def run_backtest(self, start, end, strategy_id=None, strategy_name=None):
        self.backup()
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        cwd = os.getcwd()
        os.chdir(os.path.join(self._strategy_path, strategy_name))
        if not os.path.isdir('backtest'):
            from scripts import strategy_bttest_templates
            src = strategy_bttest_templates.__path__.__dict__['_path'][-1]
            shutil.copytree(src, os.getcwd()+'/backtest')
            # os.rename('strategy_bttest_templates', 'backtest')
        stocklist_path = self._stocklist_manager.get_path(
            self.strategy_stocklist(strategy_name=strategy_name)['stocklist_name'])
        script = os.path.abspath('./backtest/run.py')
        start = datetime.strptime(start, '%Y%m%d').strftime('%Y-%m-%d')
        end = datetime.strptime(end, '%Y%m%d').strftime('%Y-%m-%d')
        os.system("python %s -s %s -e %s -f %s" % (script, start, end, stocklist_path))
        self.analyze_return(strategy_name)
        os.chdir(cwd)

    # 收益分析
    def analyze_return(self, strategy_name=None, strategy_id=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        cwd = os.getcwd()
        os.chdir(os.path.join(self._strategy_path, strategy_name+'/backtest'))
        analyzer = self.performance_analyser(strategy_name=strategy_name)
        max_date = self.latest_nav_date(strategy_name=strategy_name)
        if analyzer is not None:
            analyzer.returns_sheet(max_date).to_csv("returns_sheet.csv", index=False, float_format='%.4f')
        os.chdir(cwd)

    # back up
    def backup(self):
        from filemanager import zip_dir
        mtime = datetime.today().strftime("%Y%m%d")
        cwd = os.getcwd()
        os.chdir(os.path.abspath(self._strategy_path+'/../strategy_backup'))
        zip_dir(self._strategy_path, "copy_of_%s_%s.zip"%(os.path.split(self._strategy_path)[1], mtime))
        os.chdir(cwd)

    # 最新净值日期
    def latest_nav_date(self, strategy_id=None, strategy_name=None):
        if strategy_id is not None:
            strategy_name = self.strategy_name(strategy_id)
        if os.path.isfile(os.path.join(self._strategy_path, strategy_name+'/backtest/BTresult.pkl')):
            pf = pd.read_pickle(os.path.join(self._strategy_path, strategy_name+'/backtest/BTresult.pkl'))
            return pf['portfolio'].index.max()
        else:
            return


def update_nav(start, end):
    sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
    sm.backup()
    for i, f in sm._strategy_dict['name'].iteritems():
        sm.run_backtest(start, end, strategy_name=f)
    return


def collect_nav(mailling=False):
    from ..const import CS_INDUSTRY_DICT, MARKET_INDEX_DICT
    from .excel_io import write_xlsx
    from .tool_funcs import ensure_dir_exists
    df = pd.DataFrame()
    sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
    for i, f in sm._strategy_dict['name'].iteritems():
        if os.path.isfile(os.path.join(sm._strategy_path, f+'/backtest/returns_sheet.csv')):
            date = sm.latest_nav_date(strategy_name=f)
            ff = open(os.path.join(sm._strategy_path, f+'/backtest/returns_sheet.csv'))
            returns = pd.read_csv(ff)
            returns['最新日期'] = date
            returns.insert(0, '策略名称', f)
            df = df.append(returns)
    df = df.set_index('最新日期')
    maxdate = df.index.max().strftime("%Y%m%d")
    indexreturns = (h5.load_factor('daily_returns_%', '/indexprices/', dates=[maxdate]) / 100).reset_index()
    indexreturns.insert(0, 'name', indexreturns['IDs'].map(MARKET_INDEX_DICT))
    indexreturns = indexreturns.set_index(['date', 'IDs'])
    industry_returns = (h5.load_factor('pct_chg', '/indexprices/cs_level_1/', dates=[maxdate]) / 100).reset_index()
    industry_returns.insert(0, 'name', industry_returns['IDs'].map(CS_INDUSTRY_DICT))
    industry_returns = industry_returns.set_index(['date', 'IDs'])
    ensure_dir_exists("D:/data/strategy_performance/%s"%maxdate)
    write_xlsx("D:/data/strategy_performance/%s/returns_analysis_%s.xlsx"%(maxdate, maxdate),
               **{'returns': df, 'market index':indexreturns, 'citic industry index':industry_returns})
    if mailling:
        from filemanager import zip_dir
        from QuantLib import mymail
        mymail.connect()
        mymail.login()
        zip_dir("D:/data/strategy_performance/%s"%maxdate, 'D:/data/strategy_performance/%s.zip'%maxdate)
        content = 'hello everyone, this is strategy report on %s'%maxdate
        attachment = 'D:/data/strategy_performance/%s.zip'%maxdate
        try:
            mymail.send_mail("strategy daily report on %s"%maxdate, content, {attachment})
        except:
            mymail.connect()
            mymail.send_mail("strategy daily report on %s" % maxdate, content, {attachment})
        mymail.quit()
    return df


if __name__ == '__main__':
    sm = StrategyManager('D:/data/factor_investment_strategies', 'D:/data/factor_investment_stocklists')
    # sm.delete(name="GMTB")
    # sm.create_from_directory('D:/data/factor_investment_temp_strategies/GMTB')
    # sm.generate_tradeorder(1, 1000000000)
    sm.create_from_directory("D:/data/factor_investment_temp_strategies/兴基反转_25D")
    sm.update_stocks('20070101', '20170731', strategy_name='兴基反转_25D')
    sm.run_backtest('20070131', '20170817', strategy_name='兴基反转_25D')
    # sm.run_backtest('20070131', '20170815', strategy_name='兴基VG_逆向')
    # sm.modify_attributes(1, first_rebalance_date=datetime(2007,1,31))
    # sm.analyze_return(strategy_name='兴业风格_成长')