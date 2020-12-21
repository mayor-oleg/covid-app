# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:57:58 2020

@author: jr
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:19:02 2020

@author: jr
"""

#Import ganeral libs
import pandas as pd
import numpy as np
import dateparser
import os
from sklearn.linear_model import LinearRegression

# dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#viz imports
from plotly import graph_objs as go
import seaborn as sns


# getting name csv and country name

#directory = os.path.join("c:\\","path")
#os.chdir('https://gihttps://github.com/mayor-oleg/covid-app/tree/master/countries')
#mypath = os.getcwd()
#print ('mypath = ', mypath)
#csv_file = []
#for root,dirs,files in os.walk(mypath):
    #for file in files:       
        #if file.endswith(".csv"):           
           #csv_file.append(file)   
#print (csv_file)           
csv_file = ['df_predAfgha.csv', 'df_predAustr.csv', 'df_predBrazi.csv', 'df_predBulga.csv', 'df_predChina.csv', 'df_predCypru.csv', 'df_predDenma.csv', 
        'df_predEgypt.csv', 'df_predGerma.csv', 'df_predGreec.csv', 'df_predHunga.csv', 'df_predIndia.csv', 'df_predIsrae.csv', 'df_predItaly.csv', 
        'df_predJapan.csv', 'df_predKazak.csv', 'df_predKyrgy.csv', 'df_predLatvi.csv', 'df_predLithu.csv', 'df_predMaldi.csv', 'df_predMoldo.csv', 
        'df_predNorwa.csv', 'df_predRussi.csv', 'df_predSlove.csv', 'df_predSpain.csv', 'df_predSwede.csv', 'df_predTurke.csv', 'df_predUkrai.csv', 
        'df_predUnite.csv', 'df_predUS.csv', 'df_predZimba.csv']
link = []
for name in csv_file:
    link.append('https://raw.githubusercontent.com/mayor-oleg/covid-app/master/'+'name')
print (link)

#read data
def get_right_df(link):
    from datetime import date
    df = pd.read_csv(link)
    df_date = dateparser.parse(df['Date'][0], settings={'DATE_ORDER': 'YDM'}).date()
    start_date = date(2020,1,22)
    if df_date != start_date:
        df_copy = pd.DataFrame(np.zeros(df.shape))
        df_copy.index = df.index
        df_copy.columns = df.columns
        df_copy[['Country']] = df[['Country']]
        dates = pd.date_range(start_date, periods=abs(start_date-df_date).days).strftime("%m-%d-%Y")
        df_work=df_copy.loc[:(int(abs(start_date-df_date).days))-1].copy()
        df_work['Date'] = dates
        #print (df_work) #if need to check df_work, by the way it is zeros in all value
    else:
        df_work = pd.DataFrame(columns = (['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'Active','New_infected']))
    df_work = pd.concat([df_work[['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'Active','New_infected']],df[['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'Active', 'New_infected']]], ignore_index=True)
    df_work.reset_index()
    #Next 3 lines use to see result of Data
    #print (df.head(50))  
    #print(df.info())
    #print (df[['Date','Confirmed',  'Recovered',  'Deaths',   'Active',  'New_infected']])
    return df_work
#check the Data
#for l in link:
    #print (l)
    #print(get_right_df(l)) 

# Making scale accoarding Elliot Theory
def smoothing_vector(link, n=2):
    infected = get_right_df(link)['New_infected'].tolist()    
    infectedma = [0]
    x=1
    while x < len(infected):
        if x//n == x/n: #there is term of smoothing: 7 mean that week 
            infectedma.append(int(sum(infected[x-n:x])//n))
        x+=1
    return infectedma

# Making List of dates
def get_date(link,n=2):
    date = get_right_df(link[1])['Date'].tolist()
    datema = []
    for d in range(len(date)):
        if d//n == d/n:
            datema.append(date[d])
    return datema 

#get Dataframe with all countries
    #This block need only for control Data
    # and also this block show Data before smooth
def collect_df(link):
    columns = [] 
    for l in link:
        columns.append(get_right_df(l)['Country'].iloc[0])
    df_countries_alt = pd.DataFrame(columns = columns)
    for l in link:
        df = get_right_df(l)
        c = get_right_df(l)['Country']
        df_countries_alt[c[0]] = df['New_infected']
    return df_countries_alt    

#print ('========================collect_df====================')
#print (collect_df(link))
#print ('========================collect_df====================')


# Making Data according countries after smooth 
def df_for_predict(link):
    df_countries = pd.DataFrame()
    all = []
    for l in link:
        c = get_right_df(l)['Country']
        df_countries[c[0]] = smoothing_vector(l, n=2)
    for num in range(len(df_countries)):
        s = df_countries.iloc[num].sum()
        all.append(s)
    df_countries['All'] = all    
    return df_countries

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
#Next block need to receive all countries, but 
#as so not all countries give the latest info,
# I should to choose a few from all  
#All countries will be able later in new version of this project
#def country():
    #today = date.today()
    #l = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+((today-timedelta(5)).strftime("%m-%d-%Y"))+'.csv'
    #db = pd.read_csv(l)
    #coun = db['Country_Region'].unique().tolist()
    #coun.append('All')
    #return coun
#end this block

    
countries = ['All', 'Afghanistan', 'Bulgaria', 'China', 'Cyprus', 'Denmark', 'Egypt', 'Germany', 'Greece', 
             'Hungary', 'India', 'Israel', 'Italy', 'Japan', 'Kazakhstan', 'Kyrgyzstan',  'Latvia', 'Lithuania', 'Maldives', 
             'Moldova', 'Norway', 'Philippines', 'Russia', 'Slovenia', 'Spain', 'Sweden', 'Turkey', 'US', 'Ukraine',
             'United Kingdom', 'Zimbabwe'  ]    
  
#print (df_for_predict(link))
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
     dbc.Row([dbc.Col(html.Div("Wait a few minutes, we collect data"),width = 6)]),
     dbc.Row([dbc.Col(graph, width = 6)]
              #dbc.Col(qtygraph, width = 6)]
                , align =  "center")
     
    ], fluid = True
)


@app.callback(Output (component_id = 'graph', component_property = 'figure'),
              [Input(component_id = 'country-selector', component_property = 'value')])
def update_date_graph (country):
    print (country)

#split
    choise = country
    nday = 7
    fit_df = df_for_predict(link).iloc[:len(df_for_predict(link))]
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
    gros_test = df_for_predict(link).iloc[len(df_for_predict(link))-9:len(df_for_predict(link))]
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

# Vizualisation     
    datepred = pd.date_range(get_date(link, n=2)[-1], periods=8).strftime("%m-%d-%Y")
    dataold = get_date(link, n=2)
    fig = go.Figure(layout=go.Layout(height=400, width=1024))
# real Infected    
    fig.add_trace(go.Scatter(x = dataold,
                             y = df_for_predict(link)[choise].tolist(),
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
    
                 
