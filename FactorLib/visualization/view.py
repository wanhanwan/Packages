# -*- coding=utf-8 -*-

from bokeh.io import output_file, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.ranges import FactorRange
from bokeh.models.widgets import DataTable, PreText, TableColumn, Select, NumberFormatter, Panel, Tabs, DatePicker, Button
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


# Tab Four
all_benchmarks = {MARKET_INDEX_WINDCODE_REVERSE[x]: x[:6] for x in index_weights}

def max_date_tab4():
    risk_ds = RiskDataSource('xy')
    return as_timestamp(risk_ds.max_date_of_factor)


def update_risk_expo_single_date():
    strategy = strategy_select_tb4.value
    benchmark = all_benchmarks[benchmark_select_tb4.value]
    date = as_timestamp(datepicker_tb4.value)
    barra, indu = strategy_data_provider.load_risk_expo_single_date(strategy, date, bchmrk_name=benchmark)
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
indu_names = indu_expo.data['industry']

barra_fig = figure(x_range=barra_names, y_range=(-3, 3), plot_height=350, plot_width=900, title="Style Expo",
                   toolbar_location=None, tools="")
barra_fig.vbar(x=dodge('barra_style', -0.1, barra_fig.x_range), top='portfolio', width=0.2, source=barra_expo,
               color="#e84d60", legend=value("portfolio"))
barra_fig.vbar(x=dodge('barra_style', 0.1, barra_fig.x_range), top='benchmark', width=0.2, source=barra_expo,
               color="#718dbf", legend=value("benchmark"))
# 行业因子展示
indu_fig = figure(x_range=indu_names, y_range=(-0.5, 0.5), plot_height=600, title="Industry Expo", plot_width=1900,
                  toolbar_location=None, tools="")
indu_fig.vbar(x=dodge('industry', -0.1, indu_fig.x_range), top='portfolio', width=0.1, source=indu_expo, color="#e84d60",
              legend=value('portfolio'))
indu_fig.vbar(x=dodge('industry', 0.1, indu_fig.x_range), top='benchmark', width=0.1, source=indu_expo, color="#718dbf",
              legend=value('benchmark'))
indu_fig.xaxis.major_label_orientation = 'vertical'

datepicker_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
strategy_select_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
benchmark_select_tb4.on_change('value', lambda attr, old, new: update_risk_expo_single_date())
controls_tb4 = row(strategy_select_tb4, datepicker_tb4, benchmark_select_tb4)
tab4 = Panel(child=column(controls_tb4, barra_fig, indu_fig), title="RISK EXPO")


# Tab Five
def max_date_tab5():
    risk_ds = RiskDataSource('xy')
    return as_timestamp(risk_ds.max_date_of_factor_return)


def min_date_tab5():
    from FactorLib.data_source.base_data_source_h5 import tc
    date = max_date_tab5()
    return tc.tradeDayOffset(date, -10, retstr=None)


def update_data_tb5():
    start = as_timestamp(startdate_tb5.value)
    end = as_timestamp(enddate_tb5.value)
    strategy = strategy_select_tb5.value
    benchmark = all_benchmarks[benchmark_select_tb5.value]
    bchmrk_ret, tot_ac_ret, style_attr, indu_attr_ = strategy_data_provider.\
        load_range_attribution(strategy, start, end, benchmark)
    barra_attr.data = style_attr.to_dict(orient='list')
    indu_attr.data = indu_attr_.to_dict(orient='list')
    text_summary.text = "Date Range: %s To %s \n Benchmark Return: %f \n Total Active Return: %f \n Total Return: %f"%(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), bchmrk_ret, tot_ac_ret, bchmrk_ret+tot_ac_ret)


startdate_tb5 = DatePicker(title='Start', min_date=datetime(2010, 1, 1), max_date=datetime.now(),
                           value=min_date_tab5().date())
enddate_tb5 = DatePicker(title='End', min_date=datetime(2010, 1, 1), max_date=datetime.now(),
                         value=max_date_tab5().date())
strategy_select_tb5 = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                             options=strategy_data_provider.all_strategies)
benchmark_select_tb5 = Select(title='benchmark', value=list(all_benchmarks)[0],
                              options=list(all_benchmarks))
text_summary = PreText(text='', width=300)
calculate_button = Button(label="Calculate")
calculate_button.on_click(update_data_tb5)
barra_attr = ColumnDataSource(data=dict(barra_style=[], attr=[]))
indu_attr = ColumnDataSource(data=dict(indu=[], attr=[]))
update_data_tb5()
all_barra_names_tb5 = list(barra_attr.data['barra_style'])
barra_fig_tb5 = figure(y_range=all_barra_names_tb5, plot_height=500, plot_width=900, title="Style Attr",  # x_range=(-0.1, 0.1),
                       toolbar_location=None, tools="")
barra_fig_tb5.hbar(y=dodge('barra_style', 0, barra_fig_tb5.y_range), right='attr', height=0.2, color="#718dbf",
                   source=barra_attr, legend=value('attr'))
indu_names_tb5 = list(indu_attr.data['indu'])
indu_fig_tb5 = figure(y_range=indu_names_tb5, plot_height=1900, plot_width=900, title="Industry Attr", toolbar_location=None, tools="")
indu_fig_tb5.hbar(y=dodge('indu', 0, indu_fig_tb5.y_range), right='attr', height=0.2, color="#718dbf", source=indu_attr,
                  legend=value('attr'))
widgets_tb5 = row(column(startdate_tb5, enddate_tb5), column(strategy_select_tb5, benchmark_select_tb5), calculate_button)
tab5 = Panel(child=column(widgets_tb5, row(column(barra_fig_tb5, indu_fig_tb5), text_summary)), title="RETURN ATTR")


# Tab six
def max_date_tab6():
    risk_ds = RiskDataSource('xy')
    return as_timestamp(risk_ds.max_date_of_factor_return)


def min_date_tab6():
    return as_timestamp('20100101')


def update_data_tb6():
    strategy = strategy_select_tb6.value
    start_date = as_timestamp(startdate_tb6.value)
    end_date = as_timestamp(enddate_tb6.value)
    factor = risk_select_tb6.value
    attr = strategy_data_provider.single_risk_attr(start_date, end_date, strategy, risk_factor=factor)
    attr['left'] = attr.index - timedelta(days=0.5)
    attr['right'] = attr.index + timedelta(days=0.5)
    attr = attr.rename(columns={factor: 'data'})
    cum_attr.data = cum_attr.from_df(attr)

strategy_select_tb6 = Select(title='strategy', value=strategy_data_provider.all_strategies[0],
                             options=strategy_data_provider.all_strategies)
risk_select_tb6 = Select(title='risk_factor', value=strategy_data_provider.all_risk_factors()[0],
                         options=strategy_data_provider.all_risk_factors())
startdate_tb6 = DatePicker(title='Start', min_date=datetime(2010, 1, 1), max_date=datetime.now(),
                           value=min_date_tab6().date())
enddate_tb6 = DatePicker(title='End', min_date=datetime(2010, 1, 1), max_date=datetime.now(),
                         value=max_date_tab6().date())
calculate_button_tb6 = Button(label="Calculate")
calculate_button_tb6.on_click(update_data_tb6)
cum_attr = ColumnDataSource()
cum_attr_fig = figure(x_axis_type="datetime", plot_width=1400, tools="", toolbar_location=None)
update_data_tb6()
cum_attr_fig.vbar(x='date', top='data', width=0.1, source=cum_attr, color="#e84d60",
                  legend=value('cum_attr'))
widgets_tb6 = row(column(startdate_tb6, enddate_tb6), column(strategy_select_tb6, risk_select_tb6), calculate_button_tb6)
tab6 = Panel(child=column(widgets_tb6, row(column(cum_attr_fig))), title="CUM ATTR")


# set layout
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6])
# tabs = Tabs(tabs=[tab6])
curdoc().add_root(tabs)


