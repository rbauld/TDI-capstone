"""
This test application was built using this tutorial: http://biobits.org/bokeh-flask.html
In addition to the candle plot example on the bokeh site: https://bokeh.pydata.org/en/latest/docs/gallery/candlestick.html
"""


from flask import Flask, render_template, request, redirect
import datetime
import dateutil.relativedelta
import urllib2
import pandas as pd
from math import pi
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
from bokeh.resources import INLINE
import os
from bokeh.embed import components
import dill
from bokeh.layouts import column
from bokeh.models import Range1d

def train_plot():
    train_df = pd.read_csv('static/data/train.csv', header=None)
    p = figure(title="BTC/LTC market training data", plot_width=700, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Index'
    p.yaxis.axis_label = 'Price (BTC)'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    p.line(train_df[0], train_df[1])
    return p

def test_plot():
    train_df = pd.read_csv('static/data/test.csv', header=None)
    p = figure(title="BTC/LTC market testing data", plot_width=700, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Index'
    p.yaxis.axis_label = 'Price (BTC)'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    p.line(train_df[0], train_df[1])
    return p

def model_plot():
    train_results = pd.read_csv('static/data/log_07_01_18_good_model', delim_whitespace=True)
    p = figure(title="Training results", plot_width=700, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Training iteration'
    p.yaxis.axis_label = 'Score'

    p.line(train_results.index, train_results['train'], line_width=2)
    p.line(train_results.index, train_results['test'], line_color='orange', line_width=2)
    p.line(train_results.index, train_results['best'], line_color='green', line_width=2)

    return p


def make_price_plot(run_history, title, index, buy_overlay=False):
    run_df = run_history[index]['run_data']
    price_x = run_history[index]['price_x']
    price_y = run_history[index]['price_y']

    buy_x = run_df['index'][run_df['buy']]
    sell_x = run_df['index'][run_df['sell']]

    p = figure(title=title, plot_width=700, plot_height=400)
    p.grid.grid_line_alpha = 0
    p.yaxis.axis_label = 'Score'
    p.line(price_x, price_y, line_width=2)

    if buy_overlay:
        # Add in buy/sell overlay
        p.rect(x=buy_x, y=[0] * len(buy_x), width=100, height=10, fill_alpha=0.3, line_alpha=0, color='green',
               legend='buy')
        p.rect(x=sell_x, y=[0] * len(sell_x), width=100, height=10, fill_alpha=0.3, line_alpha=0, color='red',
               legend='sell')

        p.y_range = Range1d(min(price_y) * 0.99, max(price_y) * 1.01)

    return p


def make_feat_plot(run_history, title, index, feat_name='value_after'):
    run_df = run_history[index]['run_data']
    data_x = run_df['index']
    data_y = run_df['value_after']

    p = figure(title=title, plot_width=700, plot_height=400)
    p.grid.grid_line_alpha = 0
    p.yaxis.axis_label = 'Score'
    p.line(data_x, data_y, line_width=2)

    return p

def build_p1():
    with open('static/data/run_data_short.pkl', 'rb') as file:
        run_history = dill.load(file)

    index = 0
    p1 = make_price_plot(run_history=run_history, title='price', index=index, buy_overlay=True)
    p2 = make_feat_plot(run_history=run_history, title='Total coin value', index=index)
    return column(p1, p2)

def build_p2():
    with open('static/data/run_data_short.pkl', 'rb') as file:
        run_history = dill.load(file)

    index = 1
    p1 = make_price_plot(run_history=run_history, title='price', index=index, buy_overlay=True)
    p2 = make_feat_plot(run_history=run_history, title='Total coin value', index=index)
    return column(p1, p2)


app = Flask(__name__)


# Index page
@app.route('/')
def index():

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # Build plots
    train_script, train_div = components(train_plot())
    test_script, test_div = components(test_plot())
    model_script, model_div = components(model_plot())

    p1_script, p1_div = components(build_p1())
    p2_script, p2_div = components(build_p2())

    return render_template("index.html", js_resources=js_resources,
                           css_resources=css_resources,
                           train_script=train_script, train_div=train_div,
                           test_script=test_script, test_div=test_div,
                           model_script=model_script, model_div=model_div,
                           p1_script=p1_script, p1_div=p1_div,
                           p2_script=p2_script, p2_div=p2_div)


if __name__ == '__main__':
    app.run(port=33507)
