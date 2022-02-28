import random
from typing import Union

import pandas as pd
from bokeh.plotting import figure, output_notebook, output_file, show, save
from bokeh.models import DatetimeTickFormatter
from bokeh.models.tools import HoverTool
from bokeh.colors import named


LINE_WIDTH = 2


def output_to_notebook(plotter):
    """Decorator to output all plotting functions to jupyter notebook"""
    def wrapper(*args, **kwargs):
        output_notebook()
        p = plotter(*args, **kwargs)
        show(p)

    return wrapper


def output_to_file(plotter):
    """Decorator to output all plotting functions to a file"""
    def wrapper(*args, filename='bokeh_plot.html', **kwargs):
        output_file(filename)
        p = plotter(*args, **kwargs)
        save(p)

    return wrapper


def plot_ts(ts: pd.Series, *args, title=''):
    p = _make_time_series_figure(title)

    # plot required time series
    p.line(ts.index, ts.values, line_width=LINE_WIDTH)

    # plot optional, if provided, following colors order below
    colors_order = [
        'orange',
        'darkgreen',
        'peru',
        'maroon',
        'darkblue',
        'indigo',
        'black',
    ]

    for idx, ts in enumerate(args):
        if not isinstance(ts, pd.Series):
            raise TypeError(f'{type(ts)} is not a valid type for plotting')
        x = ts.index
        y = ts.values
        if idx < len(colors_order):
            color = colors_order[idx]
        else:
            color = random.choice(named.__all__)
        p.line(x, y, color=color, line_width=LINE_WIDTH)

    return p


def plot_detection(true, upper, lower, pred=None, history=None, title=''):
    p = _make_time_series_figure(title)

    p.line(true.index, true.values, line_width=LINE_WIDTH)
    p.line(upper.index, upper.values, color='grey', dash='dashed', line_width=LINE_WIDTH)
    p.line(lower.index, lower.values, color='grey', dash='dashed', line_width=LINE_WIDTH)
    if pred is not None:
        p.line(pred.index, pred.values, color='orange', line_width=LINE_WIDTH)
    if history is not None:
        p.line(history.index, history.values, line_width=LINE_WIDTH)

    anoms = true[(true < lower) | (true > upper)]
    p.circle(anoms.index, anoms.values, color='red', size=4)

    return p


def _make_time_series_figure(title):
    formatter = DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )

    hovertool = HoverTool(
        tooltips=[
            ('date', '@x{%d-%m-%YT%H:%M:%S}'),
            ('value', '@y')
        ],
        formatters={
            '@x': 'datetime',
        },
        mode='vline'
    )

    p = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save,box_select",
        x_axis_label='time',
        x_axis_type='datetime',
        y_axis_label='value',
        plot_width=900,
        active_scroll='wheel_zoom'
    )
    p.add_tools(hovertool)
    p.xaxis.formatter = formatter

    return p


plot_ts = output_to_notebook(plot_ts)
plot_detection = output_to_notebook(plot_detection)
