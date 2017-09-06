# -*- coding=utf-8 -*-

from bokeh.io import output_file, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable, PreText, TableColumn, Select, NumberFormatter, Panel, Tabs
from FactorLib.visualization.data_provider import StrategyPerformanceResultProvider, SingleStrategyResultProvider
from fastcache import clru_cache

output_file("viwer.html")

# data source
data_provider = StrategyPerformanceResultProvider(r"D:\data\strategy_performance")
strategy_data_provider = SingleStrategyResultProvider(r"D:\data\factor_investment_strategies")

# TAB1 SUMMARY
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


# TAB2 NAV PLOT
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
ts2.line('date', 'relative', legend='excess', source=source)
ts2.legend.location = 'top_left'

strategy_select = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                         options=strategy_data_provider.all_strategies)
stats = PreText(text='', width=500)


def update_stats():
    stats.text = strategy_data_provider.load_info(strategy_select.value)


def update_data():
    new_data = get_data(strategy_select.value)
    source.data = source.from_df(new_data)


def strategy_change(attr, old, new):
    update_data()
    update_stats()

strategy_select.on_change('value', strategy_change)

widgets = row(strategy_select, stats)
layout = column(widgets, ts1, ts2)

tab2 = Panel(child=layout, title="NAV PLOT")
update_data()
update_stats()

# set layout
tabs = Tabs(tabs=[tab1, tab2])
curdoc().add_root(tabs)
