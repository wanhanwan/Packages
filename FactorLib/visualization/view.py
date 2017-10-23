# -*- coding=utf-8 -*-

from bokeh.io import output_file, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable, PreText, TableColumn, Select, NumberFormatter, Panel, Tabs, DatePicker
from FactorLib.visualization.data_provider import StrategyPerformanceResultProvider, SingleStrategyResultProvider
from fastcache import clru_cache
from datetime import datetime, timedelta
from FactorLib.data_source.trade_calendar import as_timestamp

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
def yesterday():
    return datetime.now() - timedelta(days=1)
datepicker = DatePicker(title='Date', min_date=datetime(2007, 1, 1), max_date=datetime.now(), value=yesterday().date())
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

# set layout
tabs = Tabs(tabs=[tab1, tab2, tab3])
curdoc().add_root(tabs)


