

#author: jr


#Import ganeral libs
import pandas as pd
import numpy as np
import dateparser
import os
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from datetime import date


# dash imports based on flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#viz imports
from plotly import graph_objs as go
import seaborn as sns




# Making time series DF
def ts_df(df):
    columns=np.arange(0,8)
    data = []
    for num in range(len(df)):
        if num >= 8:
            data.append(df[num-8:num].tolist())
    ts = pd.DataFrame(data, columns=columns)
    ts.fillna(0, inplace=True)
    return ts


# Blocks for see what we have for work

# time series DF
#print ('========================ts_df====================')
#print (ts_df(df_for_predict(link), choise))
#print ('========================ts_df====================')

# Data according countries
#print ('========================df_for_predict(link)====================')
#print (df_for_predict(link))
#print ('========================df_for_predict(link)====================')

# View Data
#print (sns.pairplot(ts_df(df_for_predict(link)))) 
#-it is look like strong regression


#style
PLOTLY_THEME ='simple_white'
STYLE = [dbc.themes.FLATLY]

#building server
app = dash.Dash('covid_predict_app', external_stylesheets=STYLE)
server = app.server

#list of countries
def country():
    l = 'general_df.csv'
    db = pd.read_csv(l)
    coun = db.drop(['Date', 'Unnamed: 0'], axis = 1).columns.tolist()
    return coun
countries = country()
#end this block

    
  
  
#controls
controls =  dbc.Card(
        [
             dbc.Row([
                 dbc.Col(dbc.FormGroup(
                 [
                     dbc.Label("Choose a Country:"),
                     dcc.Dropdown(
                             id = "country-selector",
                             options = [{"label":x, "value":x} for x in countries],
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
     dbc.Row([dbc.Col(html.Div("Thy will be done!"),width = 6)]),
     dbc.Row([dbc.Col(graph, width = 6)]
              #dbc.Col(qtygraph, width = 6)]
                , align =  "center")
     
    ], fluid = True
)


@app.callback(Output (component_id = 'graph', component_property = 'figure'),
              [Input(component_id = 'country-selector', component_property = 'value')])
def update_date_graph (country):
    print (country)
    general_df = pd.read_csv('general_df.csv')
    print (general_df)
#split
    choise = country
    nday = 7
    fit_df = general_df.drop(['Date'], axis = 1).drop(['Unnamed: 0'], axis = 1)
    pre_country_df = fit_df[choise]
    country_df = ts_df(pre_country_df )
    x = country_df.drop([7],axis=1)
    y = country_df[7]

#fit

    model = LinearRegression().fit(x, y)
    
# Standard deviation
    std = np.std(pre_country_df[-9:])

#General Test but not last

#predict
    gros_test = general_df.iloc[len(general_df)-9:len(general_df)]
    pre_gros_test_country = gros_test[choise]


    d = len(pre_country_df)-1
    new_predicts =[]
    gros_test_country = ts_df(pre_gros_test_country)

    gros_predict = np.round(model.predict(gros_test_country.drop([0],axis=1)),0)

    d+=1
    new_predicts.append(pd.Series(gros_predict).values[0])
    pre_country_df[d] = pd.Series(gros_predict).values[0]
    loop = 0
#refit on predicted and predict for next week     
    while loop < nday:
        gros_test_country = ts_df(pre_country_df)
        gros_x = gros_test_country.drop([7],axis=1)
        gros_y = gros_test_country[7]
        model.fit(gros_x,gros_y)
        print ('score of train = ', model.score(gros_x,gros_y))
        gros_test_country = ts_df(pre_country_df.iloc[len(pre_country_df)-9:])
        gros_predict = np.round(model.predict(gros_test_country.drop([0],axis=1)),0)
        new_predicts.append(pd.Series(gros_predict).values[0]) 
        d+=1
        pre_country_df[d] = pd.Series(gros_predict).values[0]
        loop+=1

# Make range of predict
    new_predicts = tuple(new_predicts)
    new_predicts_max = list(new_predicts)
    new_predicts_min = list(new_predicts)

    for num in range(len(new_predicts)):
        new_predicts_max[num]+=std
        new_predicts_min[num]-=std
        if new_predicts_min[num]<0:
            new_predicts_min[num] = new_predicts[num]

# Vizualisation     
    datepred = pd.date_range(general_df['Date'].tolist()[-1], periods=8).strftime("%m-%d-%Y")
    dataold = general_df['Date'].tolist()
    fig = go.Figure(layout=go.Layout(height=400, width=1024))
# real Infected    
    fig.add_trace(go.Scatter(x = dataold,
                             y = general_df[choise].tolist(),
                             fill = None, mode = 'lines+markers',
                             name = 'new infected per day', line = {'color':'green'}))
# Predicted Infected
    fig.add_trace(go.Scatter(x = datepred,
                             y = new_predicts,
                             fill = None, mode = 'lines+markers',
                             name = 'predict next 7 days', line = {'color':'blue'}))
# Range Predicted Infected    
    fig.add_trace(go.Scatter(x = datepred,
                             y = new_predicts_max,
                             fill = None, mode = 'lines+markers',
                             name = 'max predict next 7 days', line = {'color':'red'}))
    fig.add_trace(go.Scatter(x = datepred,
                             y = new_predicts_min,
                             fill = None, mode = 'lines+markers',
                             name = 'min predict next 7 days', line = {'color':'yellow'}))
    fig.update_traces(opacity=1)
    fig.update_layout(#template=PLOT_THEME,
            title_text = "daily infected", xaxis_title = 'date',
                      yaxis_title = 'infected')
    return fig

if __name__ =="__main__":
    app.run_server()
    
                 
