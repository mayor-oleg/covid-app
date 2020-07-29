# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:19:02 2020

@author: jr
"""
#my imports
from mycovidparser import parsercovid as parsercovid
from datetime import date
from datetime import timedelta

country = []

#base imports
import pandas as pd
 
# dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#viz imports
from plotly import graph_objs as go


#style
PLOTLY_THEME ='simple_white'
STYLE = [dbc.themes.FLATLY]

#building server
app = dash.Dash('covid_predict_app', external_stylesheets=STYLE)
server = app.server

#list of countries
def country():
    today = date.today()
    l = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+((today-timedelta(5)).strftime("%m-%d-%Y"))+'.csv'
    db = pd.read_csv(l)
    coun = db['Country_Region'].unique().tolist()
    coun.append('All')
    return coun
    


#controls
controls =  dbc.Card(
        [
             dbc.Row([
                 dbc.Col(dbc.FormGroup(
                 [
                     dbc.Label("Choose a Country:"),
                     dcc.Dropdown(
                             id = "country-selector",
                             options = [{"label":x, "value":x} for x in country()],
                             value ="All"#, multi = True
                     )
                 ]
                 )),
              ], align = 'center')  
        ],
        body = True
)
                 
   
# inicialisation Graph                 
graph = dcc.Graph (id = 'graph')


#general layout
app.layout = dbc.Container(
    [
     html.H1("Covid19 reserch"),
     html.Hr(),
     dbc.Row([dbc.Col(controls, width = 6)]),
     dbc.Row([dbc.Col(html.Div("Wait a few minutes, we collect data"),width = 6)]),
     dbc.Row([dbc.Col(graph, width = 6)]
              , align =  "center")
     
    ], fluid = True
)


@app.callback(Output (component_id = 'graph', component_property = 'figure'),
              [Input(component_id = 'country-selector', component_property = 'value')])
def update_date_graph (country):
    print (country)
    #old counts
    #infected = past_data(country)['infected']
    #date = past_data(country)['date']
    
    #new way
    perday = []
    s= 0
    date = []
    infected = [0]
    infectedma = [0]
    datema =[0]
    x=1
    for df in parsercovid():
        if country != 'All':
            newdf = df.loc[df['Country'] == country]
            #print (newdf)
            s = newdf['Confirmed'].sum()
        else:
            s = df['Confirmed'].sum()
        perday.append(s)
        date.append((df.index[0])[0])
    
    while x < len(perday):
        if perday[x]>perday[x-1]:
            infected.append(perday[x]-perday[x-1])
        else:
            infected.append(perday[x-1]-perday[x])
        x+=1
    
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = date,
                             y = infected,
                             fill = None, mode = 'lines+markers',
                             name = 'new infected per day', line = {'color':'green'}))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text = "daily infected", xaxis_title = 'date',
                      yaxis_title = 'infected')
    return fig

if __name__ =="__main__":
    app.run_server()
    #(debug = True, host = '127.0.0.1', port = 8050)
                 