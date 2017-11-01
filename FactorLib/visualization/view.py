# -*- coding=utf-8 -*-

from bokeh.io import output_file, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable, PreText, TableColumn, Select, NumberFormatter, Panel, Tabs, DatePicker
from bokeh.transform import dodge
from bokeh.core.properties import value
from FactorLib.visualization.data_provider import StrategyPerformanceResultProvider, SingleStrategyResultProvider
from fastcache import clru_cache
from datetime import datetime, timedelta
from FactorLib.data_source.trade_calendar import as_timestamp
from FactorLib.riskmodel.riskmodel_data_source import RiskDataSource
from FactorLib.data_source.update_data import index_weights
from FactorLib.const import MARKET_INDEX_WINDCODE_REVERSE

output_file("viwer.html")

data_provider = StrategyPerformanceResultProvider(r"D:\data\strategy_performance")
strategy_data_provider = SingleStrategyResultProvider(r"D:\data\factor_investment_strategies")


# Tab One

table_source = ColumnDataSource(data=data_provider.load_data(data_provider.max_date))
columns = []
for k, v in data_provider.load_data(data_provider.max_date).items():
    if isinstance(v[0], str):
        columns.append(TableColumn(field=k, title=k))
    else:
        columns.append(TableColumn(field=k, title=k, formatter=NumberFormatter(format='0.00%')))


def update():
    table_source.data = data_provider.load_data(select.value)


select = Select(title='Date:', value=data_provider.max_date, options=data_provider.all_dates)
select.on_change('value', lambda attr, old, new: update())
data_table = DataTable(source=table_source, columns=columns, width=1200, height=1000)

controls = widgetbox(select)
table = widgetbox(data_table)
tab1 = Panel(child=column(controls, table), title='SUMMARY')


# Tab Two
# --------PLOT NAV------------------
@clru_cache()
def get_data(s):
    return strategy_data_provider.load_return(s)
source = ColumnDataSource(data=dict(date=[], absolute=[], benchmark=[], relative=[]))
tools = 'pan,wheel_zoom,xbox_select,reset'

ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1.line('date', 'absolute', legend='P', source=source)
ts1.line('date', 'benchmark', legend='B', color='orange', source=source)
ts1.legend.location = 'top_left'

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2.line('date', 'relative', legend='E', source=source)
ts2.legend.location = 'top_left'

strategy_select = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                         options=strategy_data_provider.all_strategies)
stats = PreText(text='', width=500)
performance_stats = PreText(text='', width=500)
yr_performance_stats = PreText(text='', width=500)


def update_stats():
    stats.text = strategy_data_provider.load_info(strategy_select.value)
    performance_stats.text = strategy_data_provider.load_strategy_performance(strategy_select.value)
    yr_performance_stats.text = strategy_data_provider.load_strategy_rel_yr_performance(strategy_select.value)

def update_data():
    new_data = get_data(strategy_select.value)
    source.data = source.from_df(new_data)


def strategy_change(attr, old, new):
    update_data()
    update_stats()

strategy_select.on_change('value', strategy_change)

widgets = row(strategy_select, stats)
stats_widgets = column(performance_stats, yr_performance_stats)
layout = row(column(widgets, ts1, ts2), stats_widgets)
tab2 = Panel(child=layout, title='PLOT NAV')
update_data()
update_stats()


# Tab Three
def max_date():
    return as_timestamp(data_provider.max_date)


datepicker = DatePicker(title='Date', min_date=datetime(2007, 1, 1), max_date=datetime.now(), value=max_date().date())
strategy_select_tb3 = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                             options=strategy_data_provider.all_strategies)
positions = ColumnDataSource(data=dict(
    IDs=[], Weight=[], cs_level_1=[], wind_level_1=[], name=[], list_date=[], daily_return=[]))

columns_tb3 = []
for c in positions.column_names:
    if c in ['IDs', 'cs_level_1', 'wind_level_1', 'name', 'list_date']:
        columns_tb3.append(TableColumn(field=c, title=c))
    else:
        columns_tb3.append(TableColumn(field=c, title=c, formatter=NumberFormatter(format='0.00%')))
table_tb3 = DataTable(source=positions, columns=columns_tb3, width=1000, height=600)


def update_data_tb3():
    strategy = strategy_select_tb3.value
    date = as_timestamp(datepicker.value)
    data = strategy_data_provider.load_positions(date, strategy)
    # print(data)
    positions.data = positions.from_df(data)
datepicker.on_change('value', lambda attr, old, new: update_data_tb3())
strategy_select_tb3.on_change('value', lambda attr, old, new: update_data_tb3())
update_data_tb3()

controls_tb3 = widgetbox(datepicker, strategy_select_tb3)
tab3 = Panel(child=column(controls_tb3, widgetbox(table_tb3)), title='POSITIONS')


# Tab Four 风险控制
all_benchmarks = {MARKET_INDEX_WINDCODE_REVERSE[x]: x[:6] for x in index_weights}

def max_date_tab4():
    risk_ds = RiskDataSource('xy')
    return as_timestamp(risk_ds.max_date_of_factor)


def update_risk_expo_single_date():
    strategy = strategy_select_tb4.value
    benchmark = all_benchmarks[benchmark_select_tb4.value]
    date = as_timestamp(datepicker_tb4.value)
    barra, indu = strategy_data_provider.load_risk_expo_single_date(strategy, date, bchrk_name=benchmark)
    barra_expo.data = barra
    indu_expo.data = indu


barra_expo = ColumnDataSource(data=dict(barra_style=[], portfolio=[], benchmark=[], expo=[]))
indu_expo = ColumnDataSource(data=dict(industry=[], portfolio=[], benchmark=[], expo=[]))
datepicker_tb4 = DatePicker(title='Date', min_date=datetime(2007, 1, 1), max_date=datetime.now(),
                            value=max_date_tab4().date())
strategy_select_tb4 = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                             options=strategy_data_provider.all_strategies)
benchmark_select_tb4 = Select(title='benchmark', value=list(all_benchmarks)[0],
                              options=list(all_benchmarks))
update_risk_expo_single_date()
barra_names = list(barra_expo.to_df()['barra_style'])
barra_fig = figure(x_range=barra_names, y_range=(-3, 3), plot_height=350, plot_width=900, title="Style Expo",
                   toolbar_location=None, tools="")
barra_fig.vbar(x=dodge('barra_style', -0.1, barra_fig.x_range), top='portfolio', width=0.2, source=barra_expo,
               color="#e84d60", legend=value("portfolio"))
barra_fig.vbar(x=dodge('barra_style', 0.1, barra_fig.x_range), top='benchmark', width=0.2, source=barra_expo,
               color="#718dbf", legend=value("benchmark"))
datepicker_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
strategy_select_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
benchmark_select_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
controls_tb4 = row(strategy_select_tb4, datepicker_tb4, benchmark_select_tb4)
tab4 = Panel(child=column(controls_tb4, barra_fig), title="RISK EXPO")

# set layout
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
tabs = Tabs(tabs=[tab4])
curdoc().add_root(tabs)


