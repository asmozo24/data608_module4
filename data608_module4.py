# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 07:48:18 2021

@author: Alexis Mekueko
"""
#from dash.react import dash
import plotly.express as px
#pip install dash
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import seaborn as sns
import tabula
from tabula import read_pdf
#pip install tabula-py
from pandas.api.types import CategoricalDtype
import numpy as np
import json
from plotly import tools


import plotly.plotly as py
import plotly.graph_objs as go

from pyproj import Proj, transform

import matplotlib.pyplot as plt

import datashader as ds
import datashader.transfer_functions as tf
import datashader.glyphs
from datashader import reductions
from datashader.core import bypixel
from datashader.utils import lnglat_to_meters as webm, export_image
from datashader.colors import colormap_select, Greys9, viridis, inferno
from functools import partial

# Assignment:
        # In this module we’ll be looking at data from the New York City tree census:
        
        # https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh
        
        # This data is collected by volunteers across the city, and is meant to catalog information about every single tree in the city.
        
        # Build a dash app for a arborist studying the health of various tree species (as defined by the variable ‘spc_common’) across each borough (defined by the variable ‘borough’).
        # This arborist would like to answer the following two questions for each species and in each borough:
        
        # What proportion of trees are in good, fair, or poor health according to the ‘health’ variable ?
        # Are stewards (steward activity measured by the ‘steward’ variable) having an impact on the health of trees?
        # Please see the accompanying notebook for an introduction and some notes on the Socrata API.
        
        # Deployment:​ Dash deployment is more complicated than deploying shiny apps, so deployment in this case is ​optional ​(and will result in extra credit).
        #  You can read instructions on deploying a dash app to heroku here: ​https://dash.plot.ly/deployment

# Data title: 2015 Street Tree Census - Tree Data
# Updated
# February 7, 2020
# Data Provided by
# Department of Parks and Recreation (DPR)

#Street tree data from the TreesCount! 2015 Street Tree Census, conducted by volunteers and staff organized by NYC Parks & Recreation and partner organizations. 
#Tree data collected includes tree species, diameter and perception of health. Accompanying blockface data is available indicating status of data collection and data release citywide.
# details on the variable ..StreetTreeCensus2015TreesDataDictionary20161102.pdf
# importing dataset

url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)
url1 = 'file:///C:/Users/owner/Downloads/StreetTreeCensus2015TreesDataDictionary20161102.pdf'
details_V = tabula.io.read_pdf(url1, pages ='all')
# Also we can download(although the dataset is heavy) the file in a common file and save in the working directory, then read it with pandas. 
trees1 = pd.read_csv("C:/Users/owner/.spyder-py3/Data608_module4/2015_Street_Tree_CensusNY.csv")
trees.head(5)
trees1.head(5)

# We need to select few variables
trees.columns.values.tolist()
#dataframe.loc[:,['column1','column2']]
#To select one or more columns by name:
trees1a = trees[['tree_id', 'status', 'spc_common','boroname','health', 'steward','boro_ct']]
#df.loc[:, 'Test_1':'Test_3']
trees1a.head(5)
 #Check for NaN under a single DataFrame column: df['your column name'].isnull().values.any()
trees1a.isnull().values.any()
trees1a.size; trees1a.shape
trees1a.isnull().sum()
# there are 1000 rows..so we can delete the one with Na
trees1b = trees1a.dropna()
trees1b.shape

#trees1b.grouby('spc_common').count()
trees1b['spc_common'].value_counts()
trees1b['boroname'].value_counts()
#trees1b['tree_id'].value_counts()

df = pd.DataFrame(trees1b)

#there are 43 trees species 
# so to answer the question, 1-we can have a selection for each borough, then  a selection for eah species, count sum for each species for each measure under variable "health"...histogram/bar
# another option ..2-we can have a selection for each borough, then count sum id under variable health ...Histogram/bar
# in option 2, we don't need to count by id, we can just the sum of each value under variable health and barplot
#We can vue by spcies: we could also do a multiple barplot with borough nameon the x-axis , colorbar the value under variable 'health'(fair, poor , good) with a selection 

# let's see the trees distribution through all borough
# we can do it with map (adding coordonates Longitude/latitude), with color scale (like gradient)
# Let say we want to see the overall distribution or each each tress species in all boroughs, we can use map with color gradient, and bubble.
# The bubble size will tell us about the quantity of the species and darkness will tell us about the dominance/abudance of one/certain species Vs. others
# in addition, with map chart, one can zoom in to see a particular area, this way it is easy to focus on each boroughs individually
# Also, if it is possible, it would be nice to add picture of tree species, this way the bubble can be remove and the tree will represent itself and plot map in 2D or 3D ...this will be nice
  
sns.set(font_scale=1.5)
df['boroname'].value_counts().plot(kind='bar', figsize=(9, 8), rot=0)
plt.xlabel("New York Boroughs", labelpad=15)
plt.ylabel("number of trees", labelpad=15)
plt.title("Trees Distribution through New York boroughs", y=1.02)
plt.show();

#trees1a = trees[['tree_id', 'status', 'spc_common','boroname','health', 'steward','boro_ct']]

# let's see the distribution of trees conditions in Brooklyn
df_b = df.loc[df['boroname'] == 'Brooklyn', ['health']]
sns.set(font_scale=1.5)
#df_b['health'].value_counts().plot.bar(x = 'Brooklyn trees conditions', y = 'number of trees', title ='Trees Helath Distribution in Brooklyn', figsize=(9, 8), rot=0)#, colorbar = True) 
# color= df_b['health'].({"Good": "green", "Fair": "grey", "Poor":"red"}))
df_b['health'].value_counts().plot(kind = 'bar', figsize=(9, 8), rot=0)
plt.xlabel("Brooklyn trees conditions", labelpad=15)
plt.ylabel("number of trees", labelpad=15)
plt.title("Trees Helath Distribution in Brooklyn", y=1.02)
plt.colormaps()
plt.show();

df_c = df[['boroname','health']]
#df_c1 = pd.crosstab(index, columns)
df_c1 = pd.crosstab(df_c['boroname'], df_c['health'])
#df_c1 = df.groupby(['boroname','health']).size().unstack()
print(df_c1)

pd.crosstab(df_c['boroname'], df_c['health']).plot(kind = 'bar', figsize=(9, 8), rot=0)
plt.xlabel("Trees Conditions in each Boroughs", labelpad=15)
plt.ylabel("number of trees", labelpad=15)
plt.title("Trees Helath Distribution through New York Boroughs", y=1.02)
plt.legend(loc ='upper right')
plt.show();

# =============================================================================
# 
# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# 
# 
# # Building my app
# app = dash.Dash(__name__)
# 
# 
# url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
# trees = pd.read_json(url)
# 
# trees2 = trees[['boroname','health']]
# #df.loc[:, 'Test_1':'Test_3']
# 
# trees2a = trees2.dropna()
# 
# 
# #trees1b.grouby('spc_common').count()
# #trees1b['boroname'].value_counts()
# 
# df = pd.DataFrame(trees2a)
# df['Count'] = 1 # adding dummy variable for px.bar call
# 
# #df_c1 = pd.crosstab(index, columns)
# df_1 = pd.crosstab(df['boroname'], df_c['health'])
# 
# colors = {
#     'background': '#121212',
#     'text': '#7FDBFF'
# }
# 
# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# # df = pd.DataFrame({
# #     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
# #     "Amount": [4, 1, 2, 2, 4, 5],
# #     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# # })
# 
# 
# fig = px.bar(df, x="boroname", y="Count", color="health", barmode="group")
# #fig = px.bar(df_c1, x = "Trees Conditions in each Boroughs", y = "number of trees", color="health", barmode= "group")
# 
# #fig = pd.crosstab(df_c['boroname'], df_c['health']).plot.bar(x = "Trees Conditions in each Boroughs", y = "number of trees", loc ="upper right", figsize=(9, 8), rot=0)
# # plt.xlabel("Trees Conditions in each Boroughs", labelpad=15)
# # plt.ylabel("number of trees", labelpad=15)
# # plt.title("Trees Helath Distribution through New York Boroughs", y=1.02)
# # plt.legend(loc ='upper right')
# # plt.show();
# 
# fig.update_layout(
#     plot_bgcolor=colors['background'],
#     paper_bgcolor=colors['background'],
#     font_color=colors['text']
# )
# 
# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Trees Health Distribution through New York Boroughs',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),
# 
#     html.Div(children='Dash for Environmental Study', style={
#         'textAlign': 'center',
#         'color': colors['text']
#     }),
# 
#     dcc.Graph(
#         id='example-graph-2',
#         figure=fig
#     )
# ])
# 
# if __name__ == '__main__':
#     app.run_server(debug=False)
# 
# =============================================================================


#Another way of looking at  tree 
  
#$ python app.py
# dash can be seen at http://127.0.0.1:8050/
url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)
trees3 = trees.dropna()

df2 = pd.DataFrame(trees3)
    
df2['count'] = 1 # adding dummy variable for px.bar call
df2.columns.values.tolist()
#del df2['Count']    

species = df2.spc_common.unique()

app = dash.Dash(__name__)


app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in species],
        value=species[0],
        clearable=False,
    ),
    dcc.Graph(id="bar-chart"),
])


@app.callback(
    Output("bar-chart", "figure"), 
    [Input("dropdown", "value")])
def update_bar_chart(spc_common):
    mask = df2["spc_common"] == spc_common
    fig = px.bar(df2[mask], x="boroname", y="count", 
                 color="health", barmode="group")
    fig.update_layout( title_text ="Trees Health Distribution through New York Boroughs by Species")
    return fig


if __name__ == "__main__":
    app.run_server(debug=False)

#http://127.0.0.1:8050/
#https://dash.plotly.com/introduction
#https://dash.gallery/Portal/
#https://towardsdatascience.com/plotly-dash-or-react-js-plotly-js-b491b3615512
#https://dash.plotly.com/deployment



# Question 2
# Are stewards (steward activity measured by the ‘steward’ variable) having an impact on the health of trees?
# In other words, is there any correlation between the activities of the steward (caretaker or environmentalist) and the health of the trees?

# I have some ideas but not sure how, 
# 1 - we can add two dropdown menu, first select the species, second select action of stewars (none, some and more), then plot trees health
# in all boroughs with grouping. but there is an information hidden here, one cannot appreciate the proportion of steward action (none, some and more ) within
# a single health condition...meaning if one is looking at trees with health = good, how, does tell the influence of stewards.
# it shows something but not fully clear or quantitable ...if we measure influence by quantity. 

# 2- this approach will have one dropdown, select trees species, plot trees health in all boroughs with grouping, then add color grading scale
# or  pattern_shape="steward" , pattern_shape_sequence=[".", "x", "+"] to indicate steward influence

# Let's try option 2
#$ python app.py
# dash can be seen at http://127.0.0.1:8050/
url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)
trees3 = trees.dropna()

df2 = pd.DataFrame(trees3)
    
df2['count'] = 1 # adding dummy variable for px.bar call
#df2.columns.values.tolist()
#del df2['Count']    

species = df2.spc_common.unique()

app = dash.Dash(__name__)


app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in species],
        value=species[0],
        clearable=False,
    ),
    dcc.Graph(id="bar-chart"),
])


@app.callback(
    Output("bar-chart", "figure"), 
    [Input("dropdown", "value")])
def update_bar_chart(spc_common):
    mask = df2["spc_common"] == spc_common
    fig = px.bar(df2[mask], x="boroname", y="count", 
                 color="steward", barmode="group")
    fig.update_layout( title_text ="Trees Health Distribution through New York Boroughs by Species")
    #fig.update_layout(pattern_shape="steward" , pattern_shape_sequence=[".", "x", "+"])
    return fig


if __name__ == "__main__":
    app.run_server(debug=False)


# Another option, add 02 dropdown, select species, select boroname, then plot steward on the x-axis and barmode = group on health



# Let's try it
#$ python app.py
# dash can be seen at http://127.0.0.1:8050/
url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)
trees3 = trees.dropna()

df2 = pd.DataFrame(trees3)
    
df2['count'] = 1 # adding dummy variable for px.bar call
#df2.columns.values.tolist()
#del df2['Count']    

# =============================================================================
# conditions = [df2.steward.eq('none'), df2.steward.eq('1or2'), df2.steward.eq('3or4')]
# choices = ['zero_care', 'some_care', 'more_care']
# df['stewards']=np.select(conditions, choices)
# 
# =============================================================================
# above code not working because comparing string, it will work with number

# =============================================================================
# df2['stewards'] = np.where(
#     df['steward'] == "none", "zero_care", np.where(
#     df2['steward'] == "1or2", "some_care", "more_care"))
# 
# =============================================================================
# Above not working , not sure why

# let's redefine the value of steward, adding new variable "stewards" based on 'steward' value
# =============================================================================
# def f(row):
#     if row['steward'] == 'None':
#         val = 'zero_care'
#     elif row['steward'] > '1or2':
#         val = 'some_care'
#     else:
#         val = 'more_care'
#     return val
# df2['stewards'] = df2.apply(f, axis=1)
# df2['steward']
# Above code work
# =============================================================================

# another way of generating the new variable
stewards = []
for row in df2['steward']:
    if row == 'None': stewards.append('zero_care')
    elif row == '1or2': stewards.append('some_care')
    else: stewards.append('more_care')
    
df2['stewards'] = stewards
       
df2['stewards']

species = df2.spc_common.unique()

# =============================================================================
# 
# fig.update_layout(
#     title="Plot Title",
#     xaxis_title="X Axis Title",
#     yaxis_title="Y Axis Title",
#     legend_title="Legend Title",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="RebeccaPurple"
#     )
# )
# 
# =============================================================================

# I just discovered that the soil of the area migh have an impact as well since we are talking about trees
#first , we will assume that all the soil in new york are the same. so we will plot show influence of steward on each trees
# where(steward_care): zero, some and more) there is more trees indicate the kind of care the species needs.   
app = dash.Dash(__name__)


app.layout = html.Div([html.P(children = 'Please select a tree species'),
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in species],
        value=species[0],
        clearable=False,
    ),
    dcc.Graph(id="bar-chart"),
])


@app.callback(
    Output("bar-chart", "figure"), 
    [Input("dropdown", "value")])
def update_bar_chart(spc_common):
    mask = df2["spc_common"] == spc_common
    fig = px.bar(df2[mask], x="stewards", y="count", 
                 color="health", barmode="group")
    fig.update_xaxes(title_text ='Stewardship', title_font_size = 15, showgrid = False )
    fig.update_yaxes(title_text ='Number of Trees', title_font_size = 15, showgrid = False)
    fig.update_layout(title_text ='<b>The Influence of Stewardship on Trees Health in New York (Brooklyn, Bronx, Manhattan, Queens,Staten Island) by Species</b><br><sup>Where(steward job: zero, some and more) there is more trees indicates the kind of care the species needs</sup>', title_font_size=20)
    #fig.update_layout(pattern_shape="steward" , pattern_shape_sequence=[".", "x", "+"])
    return fig


if __name__ == "__main__":
    app.run_server(debug=False)



#secondly , we will assume that all the soil in new york are not the same. so we will plot show influence of steward on each trees
# we will add a second dropdown menu for boroname and plot stewardship on health trees only in the called boroname, but it would be good to superpose all in figure
# meaning having 05 subplot to appreciate if there is any soil difference from one area/city to another one  

species = df2.spc_common.unique()
boroname = df2.boroname.unique()
app = dash.Dash(__name__)


app.layout = html.Div([
    dcc.Dropdown(
        id="Species Selection",
        options=[{"label": x, "value": x} for x in species],
        #placeholder = "Please select a tree species",
        value=species[0],
        clearable=False,
    ),
    
        dcc.Dropdown(id='New York boroughs',         options=[{"label": x, "value": x} for x in boroname],
        #placeholder = "Please select a tree species",
        value=boroname[0], placeholder="Please select a New York borough"),

    # html.Div(id='display-selected-values')
    
    # html.P(children = 'Please select a New York borough'),
    # dcc.Dropdown(
    #     id="dropdown",
    #     options=[{"label": x, "value": x} for x in boroname],
    #     value=boroname[0],
    #     clearable=False,
    # ),
    
    dcc.Graph(id="bar-chart"),
])


@app.callback(
    Output("bar-chart", "figure"), 
    [Input("dropdown", "value")])
def update_bar_chart(spc_common):
    mask = df2["spc_common"] == spc_common
    fig = px.bar(df2[mask], x="stewards", y="count", 
                 color="health", barmode="group")
    fig.update_xaxes(title_text ='Stewardship', title_font_size = 15, showgrid = False )
    fig.update_yaxes(title_text ='Number of Trees', title_font_size = 15, showgrid = False)
    fig.update_layout(title_text ='<b>The Influence of Stewardship on Trees Health in New York(selected city) by Species</b><br><sup>Where(steward job: zero, some and more) there is more trees indicates the kind of care the species needs</sup>', title_font_size=20)
    #fig.update_layout(pattern_shape="steward" , pattern_shape_sequence=[".", "x", "+"])
    return fig


if __name__ == "__main__":
    app.run_server(debug=False)




























