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

def train_plot():
    train_df = pd.read_csv('static/data/train.csv', header=None)
    p = figure(title="BTC/LTC market training data", plot_width=600, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Index'
    p.yaxis.axis_label = 'Price (BTC)'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    p.line(train_df[0], train_df[1])
    return p

def test_plot():
    train_df = pd.read_csv('static/data/test.csv', header=None)
    p = figure(title="BTC/LTC market testing data", plot_width=600, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Index'
    p.yaxis.axis_label = 'Price (BTC)'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    p.line(train_df[0], train_df[1])
    return p

def model_plot():
    train_results = pd.read_csv('static/data/log_07_01_18_good_model', delim_whitespace=True)
    p = figure(title="Training results", plot_width=600, plot_height=500)
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Training iteration'
    p.yaxis.axis_label = 'Score'

    p.line(train_results.index, train_results['train'], line_width=2)
    p.line(train_results.index, train_results['test'], line_color='orange', line_width=2)
    p.line(train_results.index, train_results['best'], line_color='green', line_width=2)

    return p


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

    return render_template("index.html", js_resources=js_resources,
                           css_resources=css_resources,
                           train_script=train_script, train_div=train_div,
                           test_script=test_script, test_div=test_div,
                           model_script=model_script, model_div=model_div)


if __name__ == '__main__':
    app.run(port=33507)
